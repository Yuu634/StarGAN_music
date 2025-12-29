import datetime
import os
import re
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import T5EncoderModel, T5Tokenizer
import torch.nn as nn
import json

sys.path.append("../Moonbeam-MIDI-Foundation-Model")
from inference import ScoreArrangeDomainClassifier  # type: ignore
sys.path.append("../Amadeus")
from generate import load_resources  # type: ignore
AMAEDEUS_FIELDS = ["type", "beat", "chord", "tempo", "instrument", "pitch", "duration", "velocity"]

class Solver(object):
    """Solver for training and testing the symbolic StarGAN."""

    def __init__(self, score_loader, config):
        self.score_loader = score_loader

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

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        _, self.G, _ = load_resources(self.g_modelpath, device)
        self.G.to(device)
        self.G.train()

        self.moonbeam = ScoreArrangeDomainClassifier(
            pretrained_checkpoint="../Moonbeam-MIDI-Foundation-Model/models/pretrained/moonbeam_839M.pt",
            lora_adapter_path=self.d_modelpath,
            config_path="../Moonbeam-MIDI-Foundation-Model/src/llama_recipes/configs/player_classification_config.json",
            device="cuda" if torch.cuda.is_available() else "cpu",
            selected_attr=self.selected_attrs,
        )
        self.D = self.moonbeam.model
        self.D.to(device)
        self.D.train()
        self.classification_token = torch.tensor(self.moonbeam.classification_token, device=device)
        self.pad_token = torch.tensor(self.moonbeam.pad_token, device=device)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

        self._init_text_encoder()
        self.print_network(self.G, "G")
        self.print_network(self.D, "D")

    def _init_text_encoder(self):
        self.tokenizer = T5Tokenizer.from_pretrained(self.text_encoder_model)
        self.text_encoder = T5EncoderModel.from_pretrained(self.text_encoder_model).to(self.device)
        self.text_encoder.eval()
        for param in self.text_encoder.parameters():
            param.requires_grad = False

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

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

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

    def _run_generator_round(
        self,
        x_real: torch.Tensor,
        target_context: torch.Tensor,
        origin_context: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        batch_size = x_real.size(0)
        real_sequences: List[torch.Tensor] = []
        fake_sequences: List[torch.Tensor] = []
        recon_sequences: List[torch.Tensor] = []
        for idx in range(batch_size):
            seq = x_real[idx].long()
            length = min(self.generate_length, seq.size(0))
            seq = seq[:length]
            real_sequences.append(seq)
            context_slice = target_context[idx : idx + 1] if target_context.size(0) > 0 else None
            fake = self.G.generate(
                manual_seed=0,
                max_seq_len=length,
                sampling_method=self.sampling_method,
                threshold=self.threshold,
                temperature=self.temperature,
                context=context_slice,
                input_note=seq,
            ).squeeze(0)
            fake_sequences.append(fake)
            if origin_context is not None and origin_context.size(0) > 0:
                origin_slice = origin_context[idx : idx + 1]
                recon = self.G.generate(
                    manual_seed=0,
                    max_seq_len=length,
                    sampling_method=self.sampling_method,
                    threshold=self.threshold,
                    temperature=self.temperature,
                    context=origin_slice,
                    input_note=fake,
                ).squeeze(0)
                recon_sequences.append(recon)
        return real_sequences, fake_sequences, recon_sequences

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

    def _prepare_discriminator_batch(self, sequences: List[torch.Tensor]) -> torch.Tensor:
        #padded = [self._pad_moonbeam_sequence(self.amadeus_to_moonbeam(seq)) for seq in sequences]
        sequences_moonbeam = []
        for seq in sequences:
            vocabs = self.amadeus_to_vocab(seq, self.vocab_path)
            seq_moonbeam = self.vocab_to_moonbeam(vocabs)
            seq_moonbeam = torch.as_tensor(seq_moonbeam)#.reshape(-1)
            sequences_moonbeam.append(seq_moonbeam)
        sequences_moonbeam = torch.stack(sequences_moonbeam, dim=0)
        return sequences_moonbeam #torch.stack(cls_rows, dim=0)

    def _update_discriminator(
        self,
        real_sequences: List[torch.Tensor],
        fake_sequences: List[torch.Tensor],
        label_org: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        moonbeam_real = self._prepare_discriminator_batch(real_sequences)
        moonbeam_fake = self._prepare_discriminator_batch(fake_sequences)
        moonbeam_real = moonbeam_real.to(self.device)
        moonbeam_fake = moonbeam_fake.to(self.device)
        outputs_real = self.D(input_ids=moonbeam_real)
        outputs_fake = self.D(input_ids=moonbeam_fake)
        out_src_real = outputs_real.real_fake_logits[:, 1]
        out_src_fake = outputs_fake.real_fake_logits[:, 1]
        d_loss_real = -torch.mean(out_src_real)
        d_loss_fake = torch.mean(out_src_fake)
        d_loss_cls = self.classification_loss(outputs_real.logits, label_org, "MidiCaps")
        d_loss_gp = torch.tensor(0.0, device=self.device)
        d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
        self.reset_grad()
        d_loss.backward()
        self.d_optimizer.step()
        metrics = {
            "D/loss_real": d_loss_real.item(),
            "D/loss_fake": d_loss_fake.item(),
            "D/loss_cls": d_loss_cls.item(),
            "D/loss_gp": d_loss_gp.item(),
        }
        return d_loss, metrics

    def _update_generator(
        self,
        fake_sequences: List[torch.Tensor],
        label_trg: torch.Tensor,
        real_sequences: List[torch.Tensor],
        recon_sequences: List[torch.Tensor],
    ) -> dict:
        moonbeam_fake = self._prepare_discriminator_batch(fake_sequences)
        outputs_fake = self.D(input_ids=moonbeam_fake)
        out_src_fake = outputs_fake.real_fake_logits[:, 1]
        g_loss_fake = -torch.mean(out_src_fake)
        g_loss_cls = self.classification_loss(outputs_fake.logits, label_trg, "MidiCaps")
        if recon_sequences:
            real_stack = self._stack_and_pad(real_sequences)
            recon_stack = self._stack_and_pad(recon_sequences)
            g_loss_rec = torch.mean(torch.abs(real_stack.float() - recon_stack.float()))
        else:
            g_loss_rec = torch.tensor(0.0, device=self.device)
        g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
        self.reset_grad()
        g_loss.backward()
        self.g_optimizer.step()
        return {
            "G/loss_fake": g_loss_fake.item(),
            "G/loss_rec": g_loss_rec.item(),
            "G/loss_cls": g_loss_cls.item(),
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
                continue
                #data_iter = iter(data_loader)
                #x_real, label_org = next(data_iter)

            x_real = self._to_tensor(x_real)
            label_org = self._prepare_label_tensor(label_org)
            label_trg, flipped_indices = self._select_target_attributes(label_org)
            origin_indices = self._select_origin_attributes(label_org)
            target_context = self._encode_prompts(flipped_indices)
            origin_context = self._encode_prompts(origin_indices)

            real_sequences, fake_sequences, recon_sequences = self._run_generator_round(
                x_real, target_context, origin_context
            )

            _, loss = self._update_discriminator(real_sequences, fake_sequences, label_org)

            if (i + 1) % self.n_critic == 0:
                g_loss_metrics = self._update_generator(
                    fake_sequences, label_trg, real_sequences, recon_sequences
                )
                loss.update(g_loss_metrics)

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
                _, fake_sequences, _ = self._run_generator_round(x_real, target_context, None)
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