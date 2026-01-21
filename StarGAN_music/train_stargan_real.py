"""
StarGAN Training Script with Real Amadeus and Moonbeam Models
Uses actual pre-trained models instead of dummy implementations
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
import json
import time
import datetime
from torch.nn.parallel import DataParallel
from datetime import datetime

# Add Amadeus and Moonbeam to path
sys.path.append("../Amadeus")
sys.path.append("../Moonbeam-MIDI-Foundation-Model")

# Import original model classes
from Amadeus import model_zoo
from Amadeus.train_utils import adjust_prediction_order
from Amadeus.evaluation_utils import wandb_style_config_to_omega_config
from data_representation import vocab_utils
from transformers import LlamaForSequenceClassification, LlamaConfig, T5Tokenizer, T5EncoderModel
from src.llama_recipes.real_finetuning_player_classification import LlamaForSequenceDoubleClassification
from scheduler import AdaptiveHyperparameterScheduler
from utils import split_sequence_with_sliding_window, aggregate_window_outputs

# Import loss functions
from stargan_losses import compute_discriminator_loss, compute_generator_loss
from data_loader import get_loader


class StarGANTrainer:
    """
    StarGAN Trainer with real Amadeus Generator and Moonbeam Discriminator
    """
    
    def __init__(self, args):
        """
        Args:
            args: Configuration object containing all training parameters
        """
        self.g_modelpath = args.g_modelpath
        self.amadeus_checkpoint = args.amadeus_checkpoint
        self.vocab_path = args.vocab_path
        self.d_configpath = args.d_configpath
        self.d_modelpath = args.d_modelpath
        self.num_domains = len(args.selected_attrs)
        self.selected_attrs = args.selected_attrs
        self.g_lr = args.g_lr
        self.d_lr = args.d_lr
        self.lambda_cls = args.lambda_cls
        self.lambda_rec = args.lambda_rec
        self.lambda_gp_init = args.lambda_gp  # Store initial value
        self.lambda_gp = args.lambda_gp
        self.g_interval = args.g_interval
        self.d_interval = args.d_interval
        self.temperature = args.temperature
        self.device = args.device
        self.window_size = args.window_size
        self.window_stride = args.window_stride if args.window_stride is not None else args.window_size
        self.window_aggregation = args.window_aggregation
        self.use_multi_gpu = True  # Default value
        self.Is_AmadeusRegressive = args.Is_AmadeusRegressive
        
        # Multi-GPU device setup
        if self.use_multi_gpu and torch.cuda.device_count() >= 2:
            self.device_g = 'cuda:0'  # Generator GPU 0
            self.device_d = 'cuda:1'  # Discriminator GPU 1
            self.device_t5 = 'cuda:0'  # T5 encoder on GPU 0
            print(f"\n[Multi-GPU Setup]")
            print(f"  Generator (Amadeus): GPU 0")
            print(f"  Discriminator (Moonbeam): GPU 1")
            print(f"  T5 Encoder: GPU 0")
            print(f"  Total GPUs available: {torch.cuda.device_count()}")
        else:
            self.device_g = self.device
            self.device_d = self.device
            self.device_t5 = self.device
            print(f"\n[Single GPU Setup] Using device: {self.device}")
        
        # OOM handling state
        self.current_batch_size = None
        self.oom_count = 0
        self.max_oom_retries = 1
        
        # Load Amadeus Generator (from model_zoo.py)
        print("Loading Amadeus Generator...")
        self.G, self.vocab, self.nn_params = self._load_amadeus_model(
            config_path=self.g_modelpath,
            checkpoint_path=self.amadeus_checkpoint,
            vocab_path=self.vocab_path,
            device=self.device_g
        )
        
        # Load Moonbeam Discriminator (LlamaForSequenceClassification)
        print("Loading Moonbeam Discriminator...")
        self.D = self._load_moonbeam_model(
            config_path=self.d_configpath,
            checkpoint_path=self.d_modelpath,
            num_domains=self.num_domains,
            device=self.device_d
        )
        
        # Apply multi-GPU distribution for Generator if enabled
        #if self.use_multi_gpu and torch.cuda.device_count() >= 3:
        #    print("\nApplying DataParallel to Generator (GPU 0 + GPU 1)...")
        #    self.G = DataParallel(self.G, device_ids=[0, 1])
        #    print(f"  Generator successfully distributed across GPUs 0 and 1")
        
        # Set to training mode
        self.G.train()
        self.D.train()
        
        # Load T5 encoder for text prompts (for teacher-forcing context)
        print("Loading T5 Encoder for text context...")
        self.t5_tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
        self.t5_encoder = T5EncoderModel.from_pretrained('google/flan-t5-base').to(self.device_t5)
        self.t5_encoder.eval()
        
        # Create projection layer: sum of all vocab_sizes → Discriminator hidden_size
        # Single layer that processes all Amadeus features at once
        print("Creating projection layer for soft embeddings...")
        hidden_size = self.D.config.hidden_size  # Discriminator hidden size
        
        # Get vocabulary sizes for each feature
        vocab_sizes = self.vocab.get_vocab_size()
        
        # Amadeus features: 8 discrete token types
        amadeus_fields = ['type', 'beat', 'chord', 'tempo', 'instrument', 'pitch', 'duration', 'velocity']
        total_vocab_size = 0
        vocab_size_list = []
        
        for feature_name in amadeus_fields:
            if feature_name in vocab_sizes:
                vocab_size = vocab_sizes[feature_name]
            else:
                # Fallback for different vocab structure
                vocab_size = len(self.vocab.idx2event.get(feature_name, {}))
            
            vocab_size_list.append(vocab_size)
            total_vocab_size += vocab_size
        
        print(f"  Total vocab size: {total_vocab_size} (sum of {vocab_size_list})")
        
        # Single projection layer: total_vocab_size → hidden_size
        # Place on Discriminator GPU for efficient input to D
        self.projection_layer = nn.Linear(total_vocab_size, hidden_size).to(self.device_d)
        print(f"  Created projection layer: {total_vocab_size} → {hidden_size} (on {self.device_d})")
        
        # Store vocab sizes for later use in loss functions
        self.vocab_size_list = vocab_size_list
        self.amadeus_fields = amadeus_fields
        
        # Create embedding layers for logits_to_embedded_input function
        # Reference: MultiEmbedding._make_emb_layers() from transformer_utils.py
        print("Creating embedding layers for soft embedding conversion...")
        self.embedding_layers = []
        emb_size = self.nn_params.main_decoder.dim_model  # Use same embedding size as Amadeus model
        
        for vocab_size in vocab_size_list:
            if emb_size != 0:
                # Place embedding layers on Generator GPU 0 for faster access
                self.embedding_layers.append(nn.Embedding(vocab_size, emb_size).to(self.device_g))
        
        self.embedding_layers = nn.ModuleList(self.embedding_layers)
        self.emb_size = emb_size
        print(f"  Created {len(self.embedding_layers)} embedding layers with size {emb_size} (on {self.device_g})")
        
        # Optimizers - include projection layer and embedding layers in generator optimizer for better gradient flow
        # For DataParallel, use self.G.module.parameters() to access actual parameters
        if self.use_multi_gpu and torch.cuda.device_count() >= 3:
            #g_params = list(self.G.module.parameters()) + list(self.embedding_layers.parameters())
            g_params = list(self.G.parameters()) + list(self.embedding_layers.parameters())
        else:
            g_params = list(self.G.parameters()) + list(self.embedding_layers.parameters())
        d_params = list(self.D.parameters()) + list(self.projection_layer.parameters())
        
        self.g_optimizer = optim.Adam(g_params, lr=self.g_lr, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(d_params, lr=self.d_lr, betas=(0.5, 0.999))
        
        # Initialize adaptive hyperparameter scheduler
        print("Initializing adaptive hyperparameter scheduler...")
        self.hp_scheduler = AdaptiveHyperparameterScheduler(
            initial_g_lr=self.g_lr,
            initial_d_lr=self.d_lr,
            initial_lambda_gp=self.lambda_gp,
            initial_lambda_cls=self.lambda_cls,
            initial_lambda_rec=self.lambda_rec,
            ideal_d_loss=0,  
            ideal_balance_ratio=1.5,   # Target D_loss / G_loss
            ema_decay=0.95,
            stability_threshold=0.5,
            warmup_steps=0,
        )
        print(f"  Ideal D_loss: {self.hp_scheduler.ideal_d_loss:.4f}")
        print(f"  Ideal balance ratio (D_loss/G_loss): {self.hp_scheduler.ideal_balance_ratio:.4f}")
        print(f"  Warmup steps: {self.hp_scheduler.warmup_steps}")
        
        print("StarGAN Trainer initialized successfully!")
        print(f"Generator parameters: {sum(p.numel() for p in self.G.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.D.parameters()):,}")
    
    def _load_amadeus_model(self, config_path, checkpoint_path, vocab_path, device):
        """Load AmadeusModel from model_zoo.py"""
        # Load config
        config = OmegaConf.load(config_path)
        config = wandb_style_config_to_omega_config(config)
        nn_params = config.nn_params
        
        # Load vocabulary
        encoding_scheme = nn_params.encoding_scheme
        num_features = nn_params.num_features
        vocab_name = {'remi':'LangTokenVocab', 'cp':'MusicTokenVocabCP', 'nb':'MusicTokenVocabNB'}
        selected_vocab_name = vocab_name[encoding_scheme]
        
        vocab = getattr(vocab_utils, selected_vocab_name)(
            in_vocab_file_path=vocab_path,
            event_data=None,
            encoding_scheme=encoding_scheme,
            num_features=num_features
        )
        
        # Get prediction order
        #prediction_order = adjust_prediction_order(
        #    encoding_scheme, num_features, 
        #    config.data_params.first_pred_feature, nn_params
        #)
        prediction_order = ["type", "beat", "chord", "tempo", "instrument", "pitch", "duration", "velocity"]
        
        # Create AmadeusModel
        model = getattr(model_zoo, nn_params.model_name)(
            vocab=vocab,
            input_length=config.train_params.input_length,
            prediction_order=prediction_order,
            input_embedder_name=nn_params.input_embedder_name,
            main_decoder_name=nn_params.main_decoder_name,
            sub_decoder_name=nn_params.sub_decoder_name,
            sub_decoder_depth=nn_params.sub_decoder.num_layer if hasattr(nn_params, 'sub_decoder') else 0,
            sub_decoder_enricher_use=nn_params.sub_decoder.feature_enricher_use 
                if hasattr(nn_params, 'sub_decoder') and hasattr(nn_params.sub_decoder, 'feature_enricher_use') else False,
            dim=nn_params.main_decoder.dim_model,
            heads=nn_params.main_decoder.num_head,
            depth=nn_params.main_decoder.num_layer,
            dropout=nn_params.model_dropout,
            is_regressive=self.Is_AmadeusRegressive
        )
        
        # Load checkpoint if provided
        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt['model'], strict=False)
            print(f"Loaded checkpoint from {checkpoint_path}")
        
        model.to(device)
        model.train()
        
        print(f"Amadeus Generator loaded successfully!")
        print(f"  Config: {config_path}")
        print(f"  Vocab: {vocab_path}")
        print(f"  Input length: {config.train_params.input_length}")
        print(f"  Dim: {nn_params.main_decoder.dim_model}, Heads: {nn_params.main_decoder.num_head}, Depth: {nn_params.main_decoder.num_layer}")
        print(f"  Encoding: {encoding_scheme}, Features: {num_features}")
        
        return model, vocab, nn_params
    
    def _load_moonbeam_model(self, config_path, checkpoint_path, num_domains, device):
        """Load LlamaForSequenceClassification (real_finetuning_player_classification.py style)"""
        # Load config
        llama_config = LlamaConfig.from_pretrained(config_path)
        llama_config.use_cache = False
        llama_config.num_labels = num_domains
        
        # Create model
        #model = LlamaForSequenceClassification(llama_config)
        model = LlamaForSequenceDoubleClassification(llama_config)
        
        # Load checkpoint
        if (checkpoint_path is not None) and (os.path.isfile(checkpoint_path)):
            model_checkpoint = torch.load(checkpoint_path, map_location=device)
            checkpoint = model_checkpoint.get('model_state_dict', model_checkpoint)
            
            # Remove 'module.' prefix if exists
            new_state_dict = {}
            for k, v in checkpoint.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            # Load state dict
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
        
        model.to(device)
        # Ensure all parameters are float32 for consistent dtype throughout the model
        model = model.float()
        model.train()
        
        print(f"Moonbeam Discriminator loaded successfully!")
        print(f"  Config: {config_path}")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Hidden size: {llama_config.hidden_size}")
        print(f"  Num layers: {llama_config.num_hidden_layers}")
        print(f"  Num classes: {llama_config.num_labels if hasattr(llama_config, 'num_labels') else num_domains}")
        print(f"  Model dtype: {next(model.parameters()).dtype}")
        
        return model
    
    def update_lambda_gp(self, current_step: int, total_steps: int, schedule: str = 'linear'):
        """
        Update gradient penalty weight based on training schedule
        
        Args:
            current_step: Current training step
            total_steps: Total training steps
            schedule: Schedule type ('linear', 'constant', 'warmup', 'cosine')
        
        Typical schedules:
        - 'linear': Linear increase from 0 to lambda_gp_init
        - 'constant': Keep at lambda_gp_init
        - 'warmup': Low value initially, ramp to lambda_gp_init at 20% of training
        - 'cosine': Cosine annealing from lambda_gp_init to 0
        """
        progress = current_step / max(total_steps, 1)
        
        if schedule == 'linear':
            # Gradually increase GP weight from 0 to initial value over first 50% of training
            if progress < 0.5:
                self.lambda_gp = self.lambda_gp_init * (progress * 2)  # 0 to 1 over first 50%
            else:
                self.lambda_gp = self.lambda_gp_init
        
        elif schedule == 'warmup':
            # Keep low during first 20%, then ramp to full
            if progress < 0.2:
                self.lambda_gp = self.lambda_gp_init * 0.1  # 10% of initial
            else:
                # Linear ramp from 10% to 100% over 20% to 40% of training
                ramp_progress = min((progress - 0.2) / 0.2, 1.0)
                self.lambda_gp = self.lambda_gp_init * (0.1 + 0.9 * ramp_progress)
        
        elif schedule == 'cosine':
            # Cosine annealing: high initially, decrease to 0
            import math
            self.lambda_gp = self.lambda_gp_init * 0.5 * (1 + math.cos(math.pi * progress))
        
        else:  # 'constant'
            self.lambda_gp = self.lambda_gp_init
    
    def train_step(
        self,
        real_scores: torch.Tensor,  # [B, T, 8] Amadeus format
        target_context: torch.Tensor,  # [B, T_text, H] Target text embedding
        original_context: torch.Tensor,  # [B, T_text, H] Original text embedding
        real_labels: torch.Tensor,  # [B, 108] Original domain labels
        target_labels: torch.Tensor,  # [B, 108] Target domain labels
        n_success: int = 0
    ):
        """
        Single training step
        
        Args:
            real_scores: Real Amadeus sequences [B, T, 8]
            target_context: Target domain text embedding context [B, T_text, H]
            original_context: Original domain text embedding context [B, T_text, H]
            real_labels: Original domain labels [B, 108] (for discriminator classification loss)
            
        Returns:
            d_loss_val: Discriminator loss value
            g_loss_val: Generator loss value
            logs: Dictionary of loss components
        """
        B, T, _ = real_scores.shape
        
        # ==================== Train Discriminator ====================
        d_loss_val = 0.0
        d_logs = {}
        
        if (n_success + 1) % self.d_interval == 0:
            self.d_optimizer.zero_grad()
            
            d_loss, d_logs = compute_discriminator_loss(
                G=self.G,
                D=self.D,
                real_scores=real_scores,
                context=target_context,
                real_labels=real_labels,
                projection_layer=self.projection_layer,
                vocab_size_list=self.vocab_size_list,
                hidden_size=self.D.config.hidden_size,
                vocab_path=self.vocab_path,
                lambda_cls=self.lambda_cls,
                lambda_gp=self.lambda_gp,  # Use current value (may be updated by schedule)
                temperature=self.temperature,
                device=self.device_d
            )
        
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.D.parameters(), max_norm=1.0)
            self.d_optimizer.step()
            
            d_loss_val = d_loss.item()
        
        # ==================== Train Generator (every g_interval steps) ====================
        g_loss_val = 0.0
        g_logs = {}
        
        if (n_success + 1) % self.g_interval == 0:
            self.g_optimizer.zero_grad()
            
            g_loss, g_logs = compute_generator_loss(
                G=self.G,
                D=self.D,
                real_scores=real_scores,
                context=target_context,
                target_labels=target_labels,
                original_context=original_context,
                projection_layer=self.projection_layer,
                vocab_size_list=self.vocab_size_list,
                hidden_size=self.D.config.hidden_size,
                embedding_layers=self.embedding_layers,
                emb_size=self.emb_size,
                lambda_cls=self.lambda_cls,
                lambda_rec=self.lambda_rec,
                temperature=self.temperature,
                device=self.device_g
            )
            
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=1.0)
            self.g_optimizer.step()
            
            g_loss_val = g_loss.item()
        
        # Combine logs
        logs = {**d_logs, **g_logs}
        
        return d_loss_val, g_loss_val, logs
    
    def train(self, args):
        """
        Full training loop (solver.py style with iter/next)
        
        Args:
            args: Configuration object containing training parameters
        """
        # Extract parameters from args
        dataloader = args.dataloader
        num_epochs = args.num_epochs
        num_iters = args.num_iters
        save_dir = args.model_save_dir
        log_dir = args.log_dir
        log_interval = args.log_step
        save_interval = args.model_save_step
        
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file
        log_file = os.path.join(log_dir, 'training_log.txt')
        log_csv = os.path.join(log_dir, 'training_log.csv')
        
        # Initialize data iterator (solver.py style)
        data_iter = iter(dataloader)
        
        # Write header to CSV file
        csv_header = None
        csv_file_initialized = False
        
        # Initialize progress tracking
        success_count = 0
        failure_count = 0
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            data_iter = iter(dataloader)
            
            for i in range(num_iters):
                # ===== Update lambda_gp based on schedule =====
                # Use 'warmup' schedule: low GP weight initially, increase during training
                self.update_lambda_gp(i, num_iters, schedule='warmup')
                
                # Fetch batch data with error handling (solver.py style)
                try:
                    score, label = next(data_iter)
                except FileNotFoundError as e:
                    #print(f"\n[Warning] FileNotFoundError at iteration {i}: {e}")
                    #print(f"Skipping this batch and continuing...")
                    failure_count += 1
                    continue
                except Exception as e:
                    #print(f"\n[Warning] Error at iteration {i}: {type(e).__name__}: {e}")
                    failure_count += 1
                    continue
                
                # Extract batch data
                real_scores = score.to(self.device_g)  # [B, T, 8] -> move to Generator GPU
                original_labels = label.to(self.device_d)  # [B, 108] -> move to Discriminator GPU
                
                # Handle sequences longer than Amadeus max input length using sliding windows
                seq_len = real_scores.shape[1]
                
                if seq_len > self.window_size:
                    # Split the sequence into windows
                    windows, positions = split_sequence_with_sliding_window(
                        real_scores.squeeze(0),  # Remove batch dim [B, T, 8] -> [T, 8]
                        window_size=self.window_size,
                        stride=self.window_stride
                    )
                else:
                    # For shorter sequences, treat as single window
                    windows = [real_scores.squeeze(0)]
                    positions = [0]
                
                # Generate text prompts for original and target domains
                # For simplicity, use domain indices as text prompts
                B = original_labels.size(0)
                self.current_batch_size = B  # Track batch size for OOM recovery
                
                # Get original domain index (first domain with label=1, since labels are multi-hot binary)
                original_domain_idx = torch.where(original_labels == 1)[1].tolist()  # [B]
                
                # Generate random target domain (different from original)
                target_domain_idx = torch.randint(0, self.num_domains, (B,), device=self.device).tolist()
                target_labels = torch.zeros(B, self.num_domains, dtype=torch.bool)
                target_labels.scatter_(1, torch.tensor(target_domain_idx).unsqueeze(1), True)
                target_labels = target_labels.to(self.device)
                
                # Create text prompts from domain indices with selected_attrs
                # Convert tensor indices to list for indexing self.selected_attrs
                original_domain_vocab_list = [self.selected_attrs[idx] for idx in original_domain_idx]
                target_domain_vocab_list = [self.selected_attrs[idx] for idx in target_domain_idx]
                
                # Get selected_attrs for each domain in batch
                original_prompts = " and ".join(original_domain_vocab_list)
                target_prompts = " and ".join(target_domain_vocab_list)
                
                # Encode prompts with T5 (keep tokenizer output as context)
                # Original context - use tokenizer output dict (input_ids, attention_mask)
                original_input = self.t5_tokenizer(
                    original_prompts,
                    return_tensors='pt',
                    padding='max_length',
                    truncation=True,
                    max_length=128
                ).to(self.device_t5)
                original_context = dict(original_input)  # Keep as dict for Amadeus context: {'input_ids': [1, 128], 'attention_mask': [1, 128]}
                
                # Target context - use tokenizer output dict (input_ids, attention_mask)
                target_input = self.t5_tokenizer(
                    target_prompts,
                    return_tensors='pt',
                    padding='max_length',
                    truncation=True,
                    max_length=128
                ).to(self.device_t5)
                target_context = dict(target_input)  # Keep as dict for Amadeus context: {'input_ids': [1, 128], 'attention_mask': [1, 128]}
                
                # Training step with OOM handling
                train_success = False
                current_oom_retry = 0
                
                while not train_success and current_oom_retry < self.max_oom_retries:
                    try:
                        # Clear cache before attempting training
                        torch.cuda.empty_cache()
                        
                        # Process sliding windows: infer on each window and aggregate results
                        window_outputs = []
                        for window_idx, window in enumerate(windows):
                            # Add batch dimension back: [T, 8] -> [B, T, 8]
                            window_batch = window.unsqueeze(0)
                            
                            #if window_idx == 0:
                            #    print(f"  Processing window {window_idx + 1}/{len(windows)} (length: {window.shape[0]})")
                            
                            # Training step for this window
                            d_loss_w, g_loss_w, logs_w = self.train_step(
                                real_scores=window_batch,
                                target_context=target_context,
                                original_context=original_context,
                                real_labels=original_labels,
                                target_labels=target_labels,
                                n_success=success_count
                            )
                            
                            # Store window output (we'll aggregate after all windows)
                            window_outputs.append((d_loss_w, g_loss_w, logs_w))
                        
                        # Aggregate results from all windows
                        if len(windows) > 1:
                            # Merge logs
                            logs = {}
                            for key in window_outputs[0][2].keys():
                                logs[key] = sum(out[2][key] for out in window_outputs) / len(window_outputs)
                        else:
                            d_loss, g_loss, logs = window_outputs[0]
                        
                        # Extract adversarial losses only (not including classification and reconstruction losses)
                        # D adversarial loss = loss_real + loss_fake
                        d_adversarial_loss = logs.get('D/loss_real', 0.0) + logs.get('D/loss_fake', 0.0)
                        # G adversarial loss = loss_fake (generator tries to fool discriminator)
                        g_adversarial_loss = logs.get('G/loss_fake', 0.0)
                        
                        # Update adaptive hyperparameter scheduler with adversarial losses only
                        d_grad_norm = logs.get('D/grad_norm', None)
                        g_grad_norm = logs.get('G/grad_norm', None)
                        
                        metrics = self.hp_scheduler.update(
                            d_loss=d_adversarial_loss,
                            g_loss=g_adversarial_loss,
                            d_grad_norm=d_grad_norm,
                            g_grad_norm=g_grad_norm,
                        )
                        
                        # Update optimizer learning rates based on scheduler
                        for param_group in self.g_optimizer.param_groups:
                            param_group['lr'] = self.hp_scheduler.current_g_lr
                        for param_group in self.d_optimizer.param_groups:
                            param_group['lr'] = self.hp_scheduler.current_d_lr
                        
                        # Update lambda values for loss computation
                        self.lambda_cls = self.hp_scheduler.current_lambda_cls
                        self.lambda_rec = self.hp_scheduler.current_lambda_rec
                        self.lambda_gp = self.hp_scheduler.current_lambda_gp
                        
                        # Store metrics for logging
                        self._last_metrics = metrics
                        
                        train_success = True
                        self.oom_count = 0  # Reset OOM counter on success
                        
                    except RuntimeError as e:
                        if 'out of memory' in str(e).lower():
                            current_oom_retry += 1
                            self.oom_count += 1
                            print(f"\n[OOM Error] Iteration {i}: CUDA out of memory (Attempt {current_oom_retry}/{self.max_oom_retries})")
                            print(f"  Batch size: {B}, Seq length: {real_scores.shape[1]}")
                            
                            # Try to reduce sequence length or batch size
                            if current_oom_retry < self.max_oom_retries:
                                # For now, skip this batch and retry next iteration
                                # In production, implement gradual batch size reduction
                                torch.cuda.empty_cache()
                                time.sleep(2)  # Wait before retry
                                print(f"  Retrying after clearing cache...")
                            else:
                                print(f"  Max OOM retries exceeded. Skipping batch.")
                                train_success = False  # Mark as failed to skip logging
                                break
                        else:
                            # Re-raise non-OOM runtime errors
                            raise
                
                # Update progress counters
                if train_success:
                    success_count += 1
                else:
                    failure_count += 1
                
                # Print progress on the same line (carriage return)
                progress_percent = ((i + 1) / num_iters) * 100
                progress_str = f"Progress: {i+1}/{num_iters} ({progress_percent:.1f}%) | Success: {success_count} | Failure: {failure_count}"
                print(progress_str, end='\r')
                
                if (success_count % log_interval == 0) and (success_count > 0):
                    # Prepare log data
                    log_data = {
                        'epoch': epoch + 1,
                        'iteration': i + 1,
                        'step': self.hp_scheduler.step_count,
                        'lambda_gp': self.lambda_gp,  # Add current GP weight
                        'lambda_cls': self.lambda_cls,
                        'lambda_rec': self.lambda_rec,
                        'g_lr': self.hp_scheduler.current_g_lr,
                        'd_lr': self.hp_scheduler.current_d_lr,
                        **logs
                    }
                    
                    if self.oom_count > 0:
                        log_data['oom_events'] = self.oom_count
                    
                    # Add monitoring metrics if available (after first update)
                    if hasattr(self, '_last_metrics'):
                        log_data['balance_ratio'] = self._last_metrics.balance_ratio
                        log_data['d_ideal_gap'] = self._last_metrics.d_ideal_gap
                        log_data['stability_score'] = self._last_metrics.stability_score
                        log_data['d_loss_ema'] = self._last_metrics.d_loss_ema
                        log_data['g_loss_ema'] = self._last_metrics.g_loss_ema
                    
                    # Write to text log file with gradient monitoring
                    log_str = f"Epoch [{log_data['epoch']}/{num_epochs}] | Iteration [{log_data['iteration']:6}/{num_iters}] | Success: {success_count:6} | Failure: {failure_count:6} | "
                    
                    # Format numeric values
                    numeric_logs = {k: v for k, v in logs.items() if isinstance(v, float)}
                    log_str += ", ".join([f"{k}: {v:10.4f}" for k, v in numeric_logs.items()])
                    
                    # Add scheduler info
                    log_str += f" | G_LR: {self.hp_scheduler.current_g_lr:.2e} | D_LR: {self.hp_scheduler.current_d_lr:.2e}"
                    
                    # Add stability metrics
                    if hasattr(self, '_last_metrics'):
                        log_str += f" | Balance: {self._last_metrics.balance_ratio:.4f} | Stability: {self._last_metrics.stability_score:.4f}"
                    
                    if self.oom_count > 0:
                        log_str += f" | OOM Events: {self.oom_count}"
                    
                    with open(log_file, 'a') as f:
                        f.write(log_str + '\n')
                    
                    # Write to CSV file
                    if not csv_file_initialized:
                        csv_header = list(log_data.keys())
                        with open(log_csv, 'w') as f:
                            f.write(','.join(csv_header) + '\n')
                        csv_file_initialized = True
                    
                    # Write data row to CSV
                    csv_values = [str(log_data.get(key, '')) for key in csv_header]
                    with open(log_csv, 'a') as f:
                        f.write(','.join(csv_values) + '\n')
                
                # Save checkpoint
                if success_count % save_interval == 0:
                    epoch = i // len(dataloader)
                    self.save_checkpoint(
                        save_dir=save_dir,
                        epoch=epoch,
                        step=i + 1
                    )
            
        # Print final newline to complete progress output
        print()  # Newline after progress bar
        
        # Final checkpoint
        final_epoch = num_iters // len(dataloader)
        self.save_checkpoint(save_dir=save_dir, epoch=final_epoch, step=num_iters)
        
        # Write completion message to log file
        completion_msg = f"Training completed! Final checkpoint saved to {save_dir}"
        with open(log_file, 'a') as f:
            f.write(completion_msg + '\n')
    
    def save_checkpoint(self, save_dir: str, epoch: int, step: int):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(save_dir, f'stargan_epoch{epoch}_step{step}.pt')
        
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'generator_state_dict': self.G.state_dict(),
            'discriminator_state_dict': self.D.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'hp_scheduler_state_dict': self.hp_scheduler.get_state_dict(),  # Save scheduler state
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.G.load_state_dict(checkpoint['generator_state_dict'])
        self.D.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        
        # Load scheduler state if available
        if 'hp_scheduler_state_dict' in checkpoint:
            self.hp_scheduler.load_state_dict(checkpoint['hp_scheduler_state_dict'])
            print(f"Loaded scheduler state from checkpoint (step: {self.hp_scheduler.step_count})")
        
        epoch = checkpoint['epoch']
        step = checkpoint['step']
        
        return epoch, step


def main():
    parser = argparse.ArgumentParser(description='StarGAN Training with Real Models')
    
    """<Experimental parameters>"""
    #now_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    #model_name = f'MidiCaps-{now_time}'
    model_name = 'MidiCaps-GenreSmall-ParamsTuned-G_interval=50_logits'
    parser.add_argument('--Is_AmadeusRegressive', type=bool, default=False,
                        help='Whether to use Amadeus in regressive mode (True/False)')
    parser.add_argument('--g_interval', type=int, default=50, help='G update interval iterations')
    parser.add_argument('--d_interval', type=int, default=1, help='D update interval iterations')
    
    
    # Model paths (from main.py)
    parser.add_argument('--g_modelpath', type=str, default="../Amadeus/models/Amadeus-S/files/config.yaml", 
                       help='path to the generator config yaml')
    parser.add_argument('--amadeus_checkpoint', type=str, default="../Amadeus/models/Amadeus-S/files/checkpoints/iter103662_loss-0.2098.pt",
                       help='Path to Amadeus checkpoint (optional)')
    parser.add_argument('--d_modelpath', type=str, default="../Moonbeam-MIDI-Foundation-Model/models/pretrained/moonbeam_309M.pt", 
                       help='path to the discriminator model')
    parser.add_argument('--d_configpath', type=str, default="../Moonbeam-MIDI-Foundation-Model/src/llama_recipes/configs/player_classification_config_small.json", 
                       help='path to the discriminator config')
    
    # Training hyperparameters (from main.py)
    parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size')
    parser.add_argument('--num_epochs', type=int, default=1, help='number of total epochs for training D')
    parser.add_argument('--num_iters', type=int, default=None, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=5e-5, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=5e-5, help='learning rate for D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=2, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--temperature', type=float, default=1.15, help='temperature for sampling method')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    
    # Sliding window parameters for handling sequences longer than Amadeus max input length
    parser.add_argument('--window_size', type=int, default=512,
                       help='sliding window size (default: 3072, Amadeus max input length)')
    parser.add_argument('--window_stride', type=int, default=512,
                       help='sliding window stride (default: None = window_size, no overlap)')
    parser.add_argument('--window_aggregation', type=str, default='mean',
                       choices=['mean', 'max', 'first'],
                       help='aggregation method for window outputs: mean (average), max (maximum), first (first window only)')
    
    # Data and I/O (from main.py)
    parser.add_argument('--score_dir', type=str, default='../Amadeus/dataset/MidiCaps/corpus/tuneidx_', help='Training data directory')
    parser.add_argument('--attr_path', type=str, default='../Dataset/MidiCaps/train.json', help='Attribute path')
    parser.add_argument('--vocab_path', type=str, default='../Amadeus/models/Amadeus-S/files/checkpoints/vocab_LakhALLFined_nb8.json', help='Vocabulary path')
    parser.add_argument('--model_dir', type=str, default=f'output/{model_name}', help='Model directory')
    parser.add_argument('--model_save_dir', type=str, default=f'output/{model_name}/models', help='Checkpoint save directory')
    parser.add_argument('--log_dir', type=str, default=f'output/{model_name}/logs', help='Log directory')
    parser.add_argument('--encoding', type=str, default='nb8', help='Encoding type')
    parser.add_argument('--dataset', type=str, default='MidiCaps', help='Dataset name')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--device', type=str, default='cuda', help='Training device')
    
    # Step sizes (from main.py)
    parser.add_argument('--log_step', type=int, default=10, help='Log step interval')
    #parser.add_argument('--sample_step', type=int, default=1000, help='Sample step interval')
    parser.add_argument('--model_save_step', type=int, default=10000, help='Model save step interval')
    #parser.add_argument('--lr_update_step', type=int, default=1000, help='Learning rate update step')
    
    # Generator configuration (from main.py)
    parser.add_argument('--generate_length', type=int, default=100, help='length of the generated sequence')
    parser.add_argument('--sampling_method', type=str, choices=('top_p', 'top_k'), default="top_k", help='sampling method for generation')
    parser.add_argument('--threshold', type=float, default=0.99, help='threshold for sampling method')
    
    # Domain labels (from main.py)
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for Music dataset',
                        default=["classical", "rock"]
                        #default=['chillout', 'electropop', 'latin', 'popfolk', 'classical', 'metal', 'pop', 'poprock', 'drumnbass', 'celtic', 'newage', 'instrumentalpop', 'jazzfusion', 'alternative', 'lounge', 'funk', 'orchestral', 'blues', 'indie', 'house', 'instrumentalrock', 'world', 'trance', 'ambient', '80s', 'dance', 'experimental', 'rock', 'symphonic', 'reggae', 'punkrock', 'jazz', 'easylistening', 'country', 'soundtrack', 'folk', 'electronic', '90s', 'techno', 'hiphop', 'swing', 'synthpop']
    )
    
    args = parser.parse_args()
    
    # Create directories if not exist
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # Save parsed arguments to args.model_dir for reproducibility
    args_path = Path(args.model_dir) / "args.json"
    with open(args_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    
    # Create trainer
    trainer = StarGANTrainer(args)
    
    print(f"Configuration:")
    print(f"  Generator model: {args.g_modelpath}")
    print(f"  Discriminator model: {args.d_modelpath}")
    print(f"  Discriminator config: {args.d_configpath}")
    print(f"  Batch size: {args.batch_size}")
    #print(f"  Number of iterations: {args.num_iters}")
    print(f"  Generator LR: {args.g_lr}")
    print(f"  Discriminator LR: {args.d_lr}")
    print(f"  Number of domains: {len(args.selected_attrs)}")
    print(f"  Score directory: {args.score_dir}")
    print(f"  Attribute path: {args.attr_path}")
    print(f"  Model save directory: {args.model_save_dir}")
    print(f"  Sliding Window Configuration:")
    print(f"    Window size: {args.window_size}")
    print(f"    Window stride: {args.window_stride if args.window_stride is not None else 'None (default to window_size)'}")
    print(f"    Aggregation method: {args.window_aggregation}")
    
    # Create dataloader from args.score_dir and args.attr_path
    print("\nCreating dataloader...")
    print(f"  Score directory: {args.score_dir}")
    print(f"  Attribute path: {args.attr_path}")
    print(f"  Vocabulary: {args.vocab_path}")
    print(f"  Encoding: {args.encoding}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Mode: {args.mode}")
    print(f"  Num workers: {args.num_workers}")
    
    dataloader = get_loader(
        score_dir=args.score_dir,
        encoding=args.encoding,
        attr_path=args.attr_path,
        selected_attrs=args.selected_attrs,
        batch_size=args.batch_size,
        dataset=args.dataset,
        mode=args.mode,
        num_workers=args.num_workers
    )
    
    if args.num_iters is None:
        args.num_iters = len(dataloader)
    
    # Store dataloader in args
    args.dataloader = dataloader
    
    print(f"Dataloader created successfully!")
    print(f"  Number of batches: {len(dataloader)}")
    print(f"  Training iterations: {args.num_iters}")
    
    # Start training
    trainer.train(args)


if __name__ == '__main__':
    main()
