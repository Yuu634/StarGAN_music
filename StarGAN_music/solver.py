import datetime
import os
import re
import sys
import time
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import T5EncoderModel, T5Tokenizer, LlamaConfig
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
import torch.nn as nn
import json
from pathlib import Path

sys.path.append("../Moonbeam-MIDI-Foundation-Model")
from src.llama_recipes.real_finetuning_player_classification import LlamaForSequenceDoubleClassification  # type: ignore
sys.path.append("../Amadeus")

# Lazy import to avoid pydub audio issues
def _load_amadeus_resources():
    from generate import load_resources  # type: ignore
    return load_resources

AMAEDEUS_FIELDS = ["type", "beat", "chord", "tempo", "instrument", "pitch", "duration", "velocity"]

class Solver(object):
    """Solver for training and testing the symbolic StarGAN."""

    def __init__(self, score_loader, config, rank=0, world_size=1):
        self.score_loader = score_loader
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank == 0)

        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        
        # Generator / discriminator specific configs.
        self.g_modelpath = config.g_modelpath
        self.generate_length = config.generate_length
        self.sampling_method = config.sampling_method
        self.threshold = config.threshold
        self.temperature = config.temperature
        self.d_modelpath = config.d_modelpath
        self.d_config_path = getattr(
            config,
            "d_config_path",
            "../Moonbeam-MIDI-Foundation-Model/src/llama_recipes/configs/player_classification_config.json",
        )
        self.d_pretrained_checkpoint = getattr(
            config,
            "d_pretrained_checkpoint",
            "../Moonbeam-MIDI-Foundation-Model/models/pretrained/moonbeam_839M.pt",
        )
        self.text_encoder_model = getattr(config, "text_encoder_model", "google/flan-t5-large")
        self.text_max_length = getattr(config, "text_max_length", 128)
        self.moonbeam_max_length = getattr(config, "moonbeam_max_length", 134)

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs or ["arrangement"]
        self.grad_accum_steps = getattr(config, "grad_accum_steps", 1)
        self.max_grad_norm = getattr(config, "max_grad_norm", 1.0)
        self.g_max_grad_norm = getattr(config, "g_max_grad_norm", self.max_grad_norm)
        self.d_max_grad_norm = getattr(config, "d_max_grad_norm", self.max_grad_norm)
        self.g_weight_decay = getattr(config, "g_weight_decay", 0.0)
        self.d_weight_decay = getattr(config, "d_weight_decay", 0.0)
        self.d_lora_r = getattr(config, "d_lora_r", 1)  # Reduced to 1 for extreme memory efficiency
        self.d_lora_alpha = getattr(config, "d_lora_alpha", 16)
        self.d_lora_dropout = getattr(config, "d_lora_dropout", 0.05)
        self.use_mixed_precision = getattr(config, "use_mixed_precision", False)
        self.d_freeze_until = getattr(config, "d_freeze_until", 500)  # Freeze D until this iteration to reduce peak memory

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        # text-to-score embedding.
        self.context_proj = nn.Linear(
            1024, 768, bias=False
        ).to(self.device)
        
        # Directories.
        self.vocab_path = config.vocab_path
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        self.tokenizer = None
        self.text_encoder = None

        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        device = self.device
        load_resources = _load_amadeus_resources()
        _, self.G, _ = load_resources(self.g_modelpath, device)
        self.G.to(device)
        self.G.train()

        self.D = self._build_discriminator()
        self.D.to(device)
        self.D.train()
        self.classification_token = torch.tensor(self.classification_token, device=device)
        self.pad_token = torch.tensor(self.pad_token, device=device)

        # Use memory-efficient AdamW with fused=True if available, otherwise fall back to default
        optimizer_kwargs = {
            'lr': self.g_lr,
            'betas': (self.beta1, self.beta2),
            'weight_decay': self.g_weight_decay,
        }
        try:
            # Try to use fused AdamW (CUDA kernel optimization)
            optimizer_kwargs['fused'] = True
        except:
            pass  # Fallback to non-fused if not available
            
        self.g_optimizer = torch.optim.AdamW(self.G.parameters(), **optimizer_kwargs)
        
        d_optimizer_kwargs = {
            'lr': self.d_lr,
            'betas': (self.beta1, self.beta2),
            'weight_decay': self.d_weight_decay,
        }
        try:
            d_optimizer_kwargs['fused'] = True
        except:
            pass
            
        self.d_optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.D.parameters()),
            **d_optimizer_kwargs
        )
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.d_step = 0
        self.g_step = 0
        self._init_text_encoder()
        #self.print_network(self.G, "G")
        #self.print_network(self.D, "D")

    def _load_moonbeam_config(self) -> LlamaConfig:
        llama_config = LlamaConfig.from_pretrained(self.d_config_path)
        llama_config.use_cache = False
        llama_config.num_labels = len(self.selected_attrs)
        llama_config.pretraining_tp = 1  # Avoid tensor parallelism issues
        return llama_config

    def _build_discriminator(self) -> LlamaForSequenceDoubleClassification:
        llama_config = self._load_moonbeam_config()
        base_model = LlamaForSequenceDoubleClassification(llama_config)
        
        # DISABLED: Gradient checkpointing causes CUDA memory corruption with Llama 839M
        # base_model.gradient_checkpointing_enable()  # Reentrant checkpointing

        checkpoint = torch.load(self.d_pretrained_checkpoint, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        cleaned_state_dict = {}
        for key, val in state_dict.items():
            if key.startswith("module."):
                cleaned_state_dict[key[7:]] = val
            else:
                cleaned_state_dict[key] = val

        base_model.load_state_dict(cleaned_state_dict, strict=False)

        adapter_path = Path(self.d_modelpath)
        if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
        else:
            lora_config = LoraConfig(
                r=self.d_lora_r,
                lora_alpha=self.d_lora_alpha,
                lora_dropout=self.d_lora_dropout,
                bias="none",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                task_type=TaskType.SEQ_CLS,
            )
            model = get_peft_model(base_model, lora_config)

        # DISABLED: Gradient checkpointing causes CUDA memory corruption
        # try:
        #     model.gradient_checkpointing_enable()
        # except:
        #     # If method doesn't exist, try enabling on base_model
        #     try:
        #         model.module.gradient_checkpointing_enable()
        #     except:
        #         pass  # Silently fail if not supported

        self.classification_token = llama_config.classification_token
        self.pad_token = llama_config.pad_token
        return model

    def _autocast_ctx(self):
        if self.use_mixed_precision and self.device.type == "cuda":
            # Use bfloat16 for better numerical stability (RTX 2080 Ti supports it)
            return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        return nullcontext()

    def _init_text_encoder(self):
        self.tokenizer = T5Tokenizer.from_pretrained(self.text_encoder_model)
        # Load T5 encoder in float32 but with reduced memory footprint
        self.text_encoder = T5EncoderModel.from_pretrained(
            self.text_encoder_model,
            torch_dtype=torch.float32  # Use float32 for stability
        ).to(self.device)
        # Convert to evaluation mode and disable gradients
        self.text_encoder.eval()
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        # DISABLED: Gradient checkpointing disabled to prevent CUDA errors
        # if hasattr(self.text_encoder, 'gradient_checkpointing_enable'):
        #     try:
        #         self.text_encoder.gradient_checkpointing_enable()
        #     except:
        #         pass

    def print_network(self, model, name):
        num_params = sum(p.numel() for p in model.parameters())
        print(model)
        print(name)
        print(f"The number of parameters: {num_params}")

    def restore_model(self, resume_iters):
        print(f"Loading the trained models from step {resume_iters}...")
        G_path = os.path.join(self.model_save_dir, f"{resume_iters}-G.ckpt")
        D_path = os.path.join(self.model_save_dir, f"{resume_iters}-D.ckpt")
        self.G.load_state_dict(torch.load(G_path, map_location=self.device))
        self.D.load_state_dict(torch.load(D_path, map_location=self.device))

    def build_tensorboard(self):
        from logger import Logger

        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group["lr"] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group["lr"] = d_lr

    def _set_requires_grad(self, module: nn.Module, flag: bool):
        for param in module.parameters():
            param.requires_grad = flag

    def _accumulate_and_step(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer, is_discriminator: bool):
        loss = loss / max(self.grad_accum_steps, 1)
        loss.backward()
        if is_discriminator:
            self.d_step += 1
            step_now = self.d_step
            clip_val = self.d_max_grad_norm
            opt = self.d_optimizer
        else:
            self.g_step += 1
            step_now = self.g_step
            clip_val = self.g_max_grad_norm
            opt = self.g_optimizer

        if step_now % self.grad_accum_steps == 0:
            if clip_val and clip_val > 0:
                params = []
                for group in opt.param_groups:
                    params.extend([p for p in group["params"] if p.grad is not None])
                if params:
                    torch.nn.utils.clip_grad_norm_(params, clip_val)
            opt.step()
            opt.zero_grad()

    def _to_tensor(self, scores):
        if isinstance(scores, torch.Tensor):
            tensor = scores
        else:
            tensor = torch.tensor(scores)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)

    def _prepare_label_tensor(self, labels):
        if isinstance(labels, torch.Tensor):
            tensor = labels.float()
        else:
            tensor = torch.tensor(labels, dtype=torch.float32)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)

    def _select_target_attributes(self, labels: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        label_trg = labels.clone()
        flipped = []
        for row in label_trg:
            false_idx = (row < 0.5).nonzero(as_tuple=False)
            if false_idx.numel() == 0:
                idx = torch.randint(row.numel(), (1,), device=row.device).item()
            else:
                rand = torch.randint(false_idx.size(0), (1,), device=row.device).item()
                idx = int(false_idx[rand])
            row[idx] = 1.0
            flipped.append(idx)
        return label_trg, flipped

    def _select_origin_attributes(self, labels: torch.Tensor) -> List[int]:
        indices = []
        for row in labels:
            positives = (row > 0.5).nonzero(as_tuple=False)
            if positives.numel() == 0:
                indices.append(0)
            else:
                indices.append(int(positives[0]))
        return indices

    def _encode_prompts(self, attr_indices: List[int]) -> torch.Tensor:
        if not attr_indices:
            return torch.empty(0, self.text_max_length, self.text_encoder.config.d_model, device=self.device)
        prompts = [self.selected_attrs[idx % len(self.selected_attrs)] for idx in attr_indices]
        tokenized = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.text_max_length,
        )
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        with torch.no_grad():
            encoded = self.text_encoder(**tokenized).last_hidden_state
        return self.context_proj(encoded)

    def _pad_or_trim(self, seq: torch.Tensor, target_len: int) -> torch.Tensor:
        if seq.size(0) >= target_len:
            return seq[:target_len]
        pad_len = target_len - seq.size(0)
        pad = torch.zeros(pad_len, seq.size(1), device=seq.device, dtype=seq.dtype)
        pad[:, 0] = self.pad_token
        return torch.cat([seq, pad], dim=0)

    def _logits_to_tokens(self, logits_dict: dict) -> torch.Tensor:
        logits_dict = self._ensure_feature_logits(logits_dict)
        feature_list = self.G.decoder.net.vocab.feature_list
        preds = [torch.argmax(logits_dict[feat], dim=-1) for feat in feature_list]
        return torch.stack(preds, dim=-1)

    def _generator_ce_loss(self, logits_dict: dict, target_tokens: torch.Tensor) -> torch.Tensor:
        logits_dict = self._ensure_feature_logits(logits_dict)
        feature_list = self.G.decoder.net.vocab.feature_list
        losses = []
        for idx, feature in enumerate(feature_list):
            logits = logits_dict[feature]
            target = target_tokens[..., idx].long().to(logits.device)
            losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1)))
        return sum(losses) / max(len(losses), 1)

    def _ensure_feature_logits(self, logits_like: dict | list | tuple) -> dict:
        """Normalize logits into a feature-keyed dict produced by Amadeus."""
        if isinstance(logits_like, dict):
            return logits_like
        feature_list = self.G.decoder.net.vocab.feature_list
        if isinstance(logits_like, (list, tuple)):
            # Some sub-decoders return (logits, aux) where logits may already be a dict.
            if len(logits_like) == 2 and isinstance(logits_like[0], (dict, list, tuple)):
                return self._ensure_feature_logits(logits_like[0])
            if len(logits_like) != len(feature_list):
                raise ValueError(
                    f"Logits length {len(logits_like)} does not match feature list {len(feature_list)}"
                )
            return {feat: logits_like[idx] for idx, feat in enumerate(feature_list)}
        raise TypeError(f"Unsupported logits type: {type(logits_like)}")

    def _run_generator_round(
        self,
        x_real: torch.Tensor,
        target_context: torch.Tensor,
        origin_context: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], dict, Optional[dict]]:
        batch_size = x_real.size(0)
        processed_real = []
        for idx in range(batch_size):
            seq = x_real[idx].long()
            seq = self._pad_or_trim(seq, self.generate_length)
            processed_real.append(seq)

        real_batch = torch.stack(processed_real, dim=0)
        context_for_g = target_context if target_context.numel() > 0 else None
        logits_dict, _ = self.G(real_batch, real_batch, context=context_for_g, input_note=None)
        fake_tokens = self._logits_to_tokens(logits_dict)

        real_sequences: List[torch.Tensor] = [seq.detach() for seq in real_batch]
        fake_sequences: List[torch.Tensor] = [fake_tokens[idx].detach() for idx in range(batch_size)]

        recon_sequences: List[torch.Tensor] = []
        recon_logits_dict: Optional[dict] = None
        if origin_context is not None and origin_context.size(0) > 0:
            recon_logits_dict, _ = self.G(fake_tokens, fake_tokens, context=origin_context, input_note=None)
            recon_tokens = self._logits_to_tokens(recon_logits_dict)
            recon_sequences = [recon_tokens[idx].detach() for idx in range(batch_size)]

        return real_sequences, fake_sequences, recon_sequences, logits_dict, recon_logits_dict

    def _pad_sequence(self, seq: torch.Tensor, target_len: int) -> torch.Tensor:
        if seq.size(0) >= target_len:
            return seq[:target_len]
        pad_len = target_len - seq.size(0)
        pad = torch.zeros(pad_len, seq.size(1), device=seq.device, dtype=seq.dtype)
        return torch.cat([seq, pad], dim=0)

    def _stack_and_pad(self, sequences: List[torch.Tensor]) -> torch.Tensor:
        max_len = max(seq.size(0) for seq in sequences)
        padded = [self._pad_sequence(seq, max_len) for seq in sequences]
        return torch.stack(padded, dim=0)

    def _pad_moonbeam_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        if isinstance(seq, np.ndarray):
            seq = torch.from_numpy(seq)
        seq = seq.long().to(self.device)
        cls_row = torch.zeros(1, seq.size(1), device=self.device, dtype=seq.dtype)
        cls_row[0, 0] = self.classification_token
        seq = torch.cat([seq, cls_row], dim=0)
        max_len = self.moonbeam_max_length
        if seq.size(0) < max_len:
            pad_len = max_len - seq.size(0)
            pad = torch.zeros(pad_len, seq.size(1), device=self.device, dtype=seq.dtype)
            pad[:, 0] = self.pad_token
            seq = torch.cat([seq, pad], dim=0)
        else:
            seq = torch.cat([seq[: max_len - 1], cls_row], dim=0)
        return seq

    def _prepare_discriminator_batch(self, sequences: List[torch.Tensor]) -> dict:
        """
        Prepare batch for Moonbeam discriminator.
        
        Process:
        1. Convert Amadeus sequences to Moonbeam representation [T, 6]
        2. Pad to moonbeam_max_length with pad tokens
        3. Flatten features to [T*F,] for Llama embedding lookup
        4. Create attention mask: 1 for valid tokens, 0 for padding
        5. Generate sequential position_ids for rotary embeddings
        
        Returns:
            dict with 'input_ids', 'attention_mask', 'position_ids', all shape [B, seq_len]
        """
        input_ids_list = []
        attn_masks_list = []
        position_ids_list = []
        max_seq_len = 0
        
        # First pass: convert and pad, track max length
        moonbeam_seqs = []
        for seq in sequences:
            vocabs = self.amadeus_to_vocab(seq, self.vocab_path)
            seq_moonbeam = self.vocab_to_moonbeam(vocabs)  # [T, 6]
            seq_moonbeam = self._pad_moonbeam_sequence(seq_moonbeam)  # [T_padded, 6]
            moonbeam_seqs.append(seq_moonbeam)
            max_seq_len = max(max_seq_len, seq_moonbeam.shape[0])
        
        # Second pass: create batch with attention masks
        # When flattened, each sequence position becomes 6 tokens (one per feature)
        max_flat_len = max_seq_len * 6
        
        for seq_moonbeam in moonbeam_seqs:
            # Clamp token values to valid range [0, vocab_size-1]
            # Moonbeam might produce values outside this range, so we clamp to [0, 127]
            # (typical MIDI value range before vocab expansion)
            seq_moonbeam_clamped = torch.clamp(seq_moonbeam, min=0, max=127)
            
            # Flatten features: [T, 6] -> [T*6]
            flat_seq = seq_moonbeam_clamped.reshape(-1)  # [T*6]
            
            # Pad to max length if needed
            if flat_seq.shape[0] < max_flat_len:
                pad_len = max_flat_len - flat_seq.shape[0]
                flat_seq = torch.cat([
                    flat_seq,
                    torch.zeros(pad_len, dtype=flat_seq.dtype, device=flat_seq.device)
                ], dim=0)
            
            input_ids_list.append(flat_seq)
            
            # Attention mask: 1 for valid positions, 0 for padding
            # Valid positions = where tokens are non-zero or pad_token
            # For simplicity, mark everything that's not pure zero padding as valid
            attn_mask = torch.ones_like(flat_seq, dtype=torch.long)
            
            # Determine actual valid length by looking at original moonbeam_seq
            # Assume last non-zero row is the valid boundary
            original_seq_len = seq_moonbeam.shape[0]
            # Mark padding positions in attention mask
            valid_len = original_seq_len * 6
            if valid_len < max_flat_len:
                attn_mask[valid_len:] = 0
            
            attn_masks_list.append(attn_mask)
            
            # Position IDs: sequential 0, 1, 2, ..., seq_len-1
            pos_ids = torch.arange(max_flat_len, dtype=torch.long, device=flat_seq.device)
            position_ids_list.append(pos_ids)
        
        # Stack and convert to device
        input_ids = torch.stack(input_ids_list, dim=0).long().contiguous().to(self.device)
        attention_mask = torch.stack(attn_masks_list, dim=0).long().contiguous().to(self.device)
        position_ids = torch.stack(position_ids_list, dim=0).long().contiguous().to(self.device)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids
        }

    def _update_discriminator(
        self,
        real_sequences: List[torch.Tensor],
        fake_sequences: List[torch.Tensor],
        label_org: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        moonbeam_real = self._prepare_discriminator_batch(real_sequences)
        moonbeam_fake = self._prepare_discriminator_batch(fake_sequences)
        real_rf_labels = torch.ones(moonbeam_real["input_ids"].size(0), device=self.device, dtype=torch.long)
        fake_rf_labels = torch.zeros(moonbeam_fake["input_ids"].size(0), device=self.device, dtype=torch.long)

        self.D.train()
        with self._autocast_ctx():
            outputs_real = self.D(**moonbeam_real)
            outputs_fake = self.D(**moonbeam_fake)

            d_loss_real = F.cross_entropy(outputs_real.real_fake_logits, real_rf_labels)
            d_loss_fake = F.cross_entropy(outputs_fake.real_fake_logits, fake_rf_labels)
            d_loss_cls = self.classification_loss(outputs_real.logits, label_org, "MidiCaps")
            d_loss_gp = torch.tensor(0.0, device=self.device)
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp

        self._accumulate_and_step(d_loss, self.d_optimizer, is_discriminator=True)
        metrics = {
            "D/loss_real": d_loss_real.item(),
            "D/loss_fake": d_loss_fake.item(),
            "D/loss_cls": d_loss_cls.item(),
            "D/loss_gp": d_loss_gp.item(),
        }
        return d_loss.detach(), metrics

    def _update_generator(
        self,
        fake_sequences: List[torch.Tensor],
        label_trg: torch.Tensor,
        real_sequences: List[torch.Tensor],
        recon_sequences: List[torch.Tensor],
        gen_logits: dict,
    ) -> dict:
        self._set_requires_grad(self.D, False)
        self.D.eval()
        with self._autocast_ctx():
            moonbeam_fake = self._prepare_discriminator_batch(fake_sequences)
            outputs_fake = self.D(**moonbeam_fake)
            rf_labels = torch.ones(moonbeam_fake["input_ids"].size(0), device=self.device, dtype=torch.long)
            g_loss_fake = F.cross_entropy(outputs_fake.real_fake_logits, rf_labels)
            g_loss_cls = self.classification_loss(outputs_fake.logits, label_trg, "MidiCaps")

        if recon_sequences:
            real_stack = self._stack_and_pad(real_sequences)
            recon_stack = self._stack_and_pad(recon_sequences)
            g_loss_rec = torch.mean(torch.abs(real_stack.float() - recon_stack.float()))
        else:
            g_loss_rec = torch.tensor(0.0, device=self.device)

        target_tokens = torch.stack(real_sequences, dim=0).to(self.device)
        g_loss_rec_ce = self._generator_ce_loss(gen_logits, target_tokens)

        g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls + g_loss_rec_ce
        self._accumulate_and_step(g_loss, self.g_optimizer, is_discriminator=False)
        self._set_requires_grad(self.D, True)
        return {
            "G/loss_fake": g_loss_fake.item(),
            "G/loss_rec": g_loss_rec.item(),
            "G/loss_cls": g_loss_cls.item(),
            "G/loss_rec_ce": g_loss_rec_ce.item(),
        }

    def train(self):
        data_loader = self.score_loader
        data_iter = iter(data_loader)
        g_lr = self.g_lr
        d_lr = self.d_lr
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        print("Start training...")
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            try:
                x_real, label_org = next(data_iter)
            except (StopIteration, FileNotFoundError):
                data_iter = iter(data_loader)
                continue

            # Step 2.6: Freeze/Unfreeze Discriminator for memory management
            if (i + 1) <= self.d_freeze_until:
                # Frozen phase: only Generator training, no D gradient computation
                for param in self.D.parameters():
                    param.requires_grad = False
                skip_d_update = True
            else:
                # Normal phase: both Generator and Discriminator training
                for param in self.D.parameters():
                    param.requires_grad = True
                skip_d_update = False

            x_real = self._to_tensor(x_real)
            label_org = self._prepare_label_tensor(label_org)
            label_trg, flipped_indices = self._select_target_attributes(label_org)
            origin_indices = self._select_origin_attributes(label_org)
            target_context = self._encode_prompts(flipped_indices)
            origin_context = self._encode_prompts(origin_indices)

            real_sequences, fake_sequences, recon_sequences, gen_logits, _ = self._run_generator_round(
                x_real, target_context, origin_context
            )

            # Skip Discriminator update during frozen phase
            if not skip_d_update:
                _, loss = self._update_discriminator(real_sequences, fake_sequences, label_org)
            else:
                loss = {"D/loss_real": 0.0, "D/loss_fake": 0.0, "D/loss_cls": 0.0, "D/loss_gp": 0.0}

            if (i + 1) % self.n_critic == 0:
                g_loss_metrics = self._update_generator(
                    fake_sequences, label_trg, real_sequences, recon_sequences, gen_logits
                )
                loss.update(g_loss_metrics)

            # Step 2.4: Clear cache every iteration to prevent memory fragmentation and CUDA allocation failures
            torch.cuda.empty_cache()

            if (i + 1) % self.log_step == 0:
                elapsed = str(datetime.timedelta(seconds=time.time() - start_time))[:-7]
                log = f"Elapsed [{elapsed}], Iteration [{i + 1}/{self.num_iters}]"
                for tag, value in loss.items():
                    log += f", {tag}: {value:.4f}"
                print(log)
                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i + 1)

            if (i + 1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, f"{i + 1}-G.ckpt")
                D_path = os.path.join(self.model_save_dir, f"{i + 1}-D.ckpt")
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print(f"Saved model checkpoints into {self.model_save_dir}...")

            if (i + 1) % self.lr_update_step == 0 and (i + 1) > (self.num_iters - self.num_iters_decay):
                g_lr -= self.g_lr / float(self.num_iters_decay)
                d_lr -= self.d_lr / float(self.num_iters_decay)
                self.update_lr(g_lr, d_lr)
                print(f"Decayed learning rates, g_lr: {g_lr}, d_lr: {d_lr}.")

    def train_multi(self):
        raise NotImplementedError("Multi-dataset training is not supported in the symbolic pipeline.")

    def test(self):
        self.restore_model(self.test_iters)
        self.G.eval()
        data_loader = self.score_loader
        os.makedirs(self.result_dir, exist_ok=True)
        with torch.no_grad():
            for idx, (x_real, label_org) in enumerate(data_loader, start=1):
                x_real = self._to_tensor(x_real)
                label_org = self._prepare_label_tensor(label_org)
                label_trg, flipped_indices = self._select_target_attributes(label_org)
                target_context = self._encode_prompts(flipped_indices)
                _, fake_sequences, _, _, _ = self._run_generator_round(x_real, target_context, None)
                for b, seq in enumerate(fake_sequences):
                    save_path = os.path.join(self.result_dir, f"sample_{idx:05d}_{b}.npz")
                    np.savez(save_path, seq.cpu().numpy())
                    print(f"Saved translated score to {save_path}")
        self.G.train()

    def test_multi(self):
        raise NotImplementedError("Multi-dataset testing is not supported in the symbolic pipeline.")

    def classification_loss(self, logit, target, dataset="CelebA"):
        if dataset in ("CelebA", "MidiCaps"):
            return F.binary_cross_entropy_with_logits(logit, target, reduction="mean")
        if dataset == "RaFD":
            return F.cross_entropy(logit, target)
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    def _build_lookup_table(self, field_dict: dict[str, str]) -> np.ndarray:
        max_idx = max(int(k) for k in field_dict.keys())
        table = ["" for _ in range(max_idx + 1)]
        for k, v in field_dict.items():
            table[int(k)] = v
        return np.array(table, dtype=object)

    def amadeus_to_vocab(self, amadeus_tokens: torch.Tensor, vocab_path: str) -> np.ndarray:
        """Amadeusのトークン列を語彙に変換し、amadeus_to_moonbeam利用可能な形式に"""
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        tokens_np = amadeus_tokens.detach().cpu().numpy().astype(np.int64)
        decoded = np.empty(tokens_np.shape, dtype=object)

        for axis, field in enumerate(AMAEDEUS_FIELDS):
            lookup = self._build_lookup_table(vocab[field])
            decoded[:, axis] = lookup[tokens_np[:, axis]]

        return decoded
    
    def vocab_to_moonbeam(self, amadeus_vocabs, time_resolution=10, default_tempo=120, in_beat_resolution=4):
        # (Beat + 小節数) * Tempo → onset(ms)
        # duration * Tempo → duration(ms)
        # pitch → octave*12 + pitch_class
        
        # Amadeus表現：(type, beat, chord, tempo, instrument, pitch, duration, velocity)
        """
        Type：拍子の変化や継続の組み合わせの違いをそれぞれ表すもの。
        Beat：同一小節内における音符の相対的な位置。
        Chord：現在の音符が属している和音。
        Tempo：音符の再生速度。一般に、テンポが高いほど楽曲は速くなる。
        Instrument：現在の音符を演奏している楽器。
        Pitch：音符の高さ。MIDI仕様に基づく128段階の離散値で表される。
        Duration：音符が演奏される長さ（継続時間）。
        Velocity：音符がどの強さで演奏されるかを表す値で、音量の大きさを決定する。
        例）{'type': 'NNN_time_signature_4/4', 'beat': 'Beat_2', 'chord': 'Chord_N_N', 'tempo': 'Tempo_121', 'instrument': 'Instrument_114', 'pitch': 'Note_Pitch_48', 'duration': 'Note_Duration_2', 'velocity': 'Note_Velocity_100'}
        """
        # Moonbeam表現：(onset, duration, octave, pitch_class, instrument, velocity) 
        """
        onset・durationの最小単位は10ms
        onset：音の開始位置　4097トークン(M-model), 1024トークン(S-model)
        duration：音の継続時間　4097トークン(M-model), 1024トークン(S-model)
        octave：11トークン
        pitch_class：12トークン(1オクターブの音階分)
        instrument：129トークン(MIDI楽器全般)
        velocity：128トークン
        """
        
        # Convert to numpy for easier processing
        if isinstance(amadeus_vocabs, torch.Tensor):
            amadeus_np = amadeus_vocabs.cpu().numpy()
            use_torch = True
        elif isinstance(amadeus_vocabs, list):
            amadeus_np = np.array(amadeus_vocabs)
            use_torch = False
        else:
            amadeus_np = np.array(amadeus_vocabs)
            use_torch = False
        
        # Check shape and transpose if needed
        if amadeus_np.shape[0] == 8:
            amadeus_np = amadeus_np.T  # [num_notes, 8]
        
        num_notes = amadeus_np.shape[0]
        moonbeam_np = np.zeros((num_notes, 6), dtype=np.int32)
        
        # State tracking
        current_bar = -1
        current_tempo = default_tempo
        current_time_signature = (4, 4)  # (numerator, denominator)
        
        # First note
        for k in range(6):
            moonbeam_np[0, k] = 0
        
        for i in range(1, num_notes):
            type_token = amadeus_np[i, 0]
            beat_token = amadeus_np[i, 1]
            chord_token = amadeus_np[i, 2]
            tempo_token = amadeus_np[i, 3]
            instrument_token = amadeus_np[i, 4]
            pitch_token = amadeus_np[i, 5]
            duration_token = amadeus_np[i, 6]
            velocity_token = amadeus_np[i, 7]
            
            # Parse type token (NNN/SNN/SSN/SSS format)
            if isinstance(type_token, str):
                # Check for time signature change (NNN format)
                if type_token.startswith('NNN_time_signature_'):
                    time_sig_match = re.search(r'time_signature_(\d+)/(\d+)', type_token)
                    if time_sig_match:
                        current_time_signature = (int(time_sig_match.group(1)), int(time_sig_match.group(2)))
                    current_bar += 1
                elif type_token == 'SNN':  # Same time sig, new bar, new beat
                    current_bar += 1
                # SSN and SSS don't change bar
            
            # Extract tempo
            if isinstance(tempo_token, str):
                tempo_match = re.search(r'Tempo_(\d+)', tempo_token)
                if tempo_match:
                    current_tempo = int(tempo_match.group(1))
            else:
                current_tempo = int(tempo_token) if tempo_token else default_tempo
            
            # Extract beat position within bar (0-15 for 4/4 time with in_beat_resolution=4)
            if isinstance(beat_token, str):
                beat_match = re.search(r'Beat_(\d+)', beat_token)
                beat_index = int(beat_match.group(1)) if beat_match else 0
            else:
                beat_index = int(beat_token)
            
            # Calculate onset
            # Beat index represents position in bar (0-15 for 4/4, in_beat_resolution=4)
            numerator, denominator = current_time_signature
            
            # Total subdivisions per bar
            subdivisions_per_bar = numerator * (4 / denominator) * in_beat_resolution
            # For 4/4: 4 * 1 * 4 = 16 subdivisions per bar
            
            # Calculate onset in subdivisions (16th notes for in_beat_resolution=4)
            onset_in_subdivisions = current_bar * subdivisions_per_bar + beat_index
            
            # Convert to quarter notes
            onset_in_quarter_notes = onset_in_subdivisions / in_beat_resolution
            
            # Convert to milliseconds
            ms_per_quarter_note = 60000.0 / current_tempo
            onset_ms = onset_in_quarter_notes * ms_per_quarter_note
            
            # Convert to 10ms resolution
            onset_in_10ms = int(round(onset_ms / time_resolution))
            
            # Extract duration (in subdivisions)
            if isinstance(duration_token, str):
                dur_match = re.search(r'Note_Duration_([\d.]+)', duration_token)
                duration_in_subdivisions = float(dur_match.group(1)) if dur_match else 1.0
            else:
                duration_in_subdivisions = float(duration_token)
            
            # Convert duration to quarter notes
            duration_in_quarter_notes = duration_in_subdivisions / in_beat_resolution
            
            # Convert to milliseconds
            duration_ms = duration_in_quarter_notes * ms_per_quarter_note
            duration_in_10ms = int(round(duration_ms / time_resolution))
            duration_in_10ms = max(1, min(duration_in_10ms, 1024))
            
            # Extract pitch and convert to octave + pitch_class
            if isinstance(pitch_token, str):
                pitch_match = re.search(r'Note_Pitch_(\d+)', pitch_token)
                pitch = int(pitch_match.group(1)) if pitch_match else 60
            else:
                pitch = int(pitch_token)
            
            octave = pitch // 12
            pitch_class = pitch % 12
            octave = max(0, min(octave, 10))
            
            # Extract instrument
            if isinstance(instrument_token, str):
                inst_match = re.search(r'Instrument_(\d+)', instrument_token)
                instrument = int(inst_match.group(1)) if inst_match else 0
            else:
                instrument = int(instrument_token)
            
            instrument = max(0, min(instrument, 128))
            
            # Extract velocity
            if isinstance(velocity_token, str):
                vel_match = re.search(r'Note_Velocity_(\d+)', velocity_token)
                velocity = int(vel_match.group(1)) if vel_match else 64
            else:
                velocity = int(velocity_token)
            
            velocity = max(0, min(velocity, 127))
            
            # Store in Moonbeam format
            moonbeam_np[i, 0] = onset_in_10ms
            moonbeam_np[i, 1] = duration_in_10ms
            moonbeam_np[i, 2] = octave
            moonbeam_np[i, 3] = pitch_class
            moonbeam_np[i, 4] = instrument
            moonbeam_np[i, 5] = velocity
        
        # Convert back to torch if input was torch
        if use_torch:
            moonbeam_tokens = torch.from_numpy(moonbeam_np).long()
        else:
            moonbeam_tokens = moonbeam_np
        
        return moonbeam_tokens