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

# Import loss functions
from stargan_losses import compute_discriminator_loss, compute_generator_loss
from data_loader import get_loader


def split_sequence_with_sliding_window(sequence, window_size=3072, stride=None):
    """
    Split a sequence longer than window_size into overlapping windows.
    
    Args:
        sequence: Tensor of shape [T, ...] where T is sequence length
        window_size: Maximum length per window (default: Amadeus max input length = 3072)
        stride: Step size for sliding window. If None, stride = window_size (no overlap)
    
    Returns:
        List of sequence windows, each with length <= window_size
        Also returns start positions for each window
    
    Example:
        >>> seq = torch.randn(5000, 8)  # Longer than window_size
        >>> windows, positions = split_sequence_with_sliding_window(seq, window_size=3072, stride=1536)
        >>> # windows[0].shape = [3072, 8]
        >>> # windows[1].shape = [3072, 8]
        >>> # windows[2].shape = [1928, 8]  (remaining)
    """
    seq_len = sequence.shape[0]
    
    # If sequence is shorter than window size, return as is
    if seq_len <= window_size:
        return [sequence], [0]
    
    # Default stride = window_size (no overlap)
    if stride is None:
        stride = window_size
    
    windows = []
    positions = []
    
    # Create sliding windows
    current_pos = 0
    while current_pos < seq_len:
        end_pos = min(current_pos + window_size, seq_len)
        window = sequence[current_pos:end_pos]
        windows.append(window)
        positions.append(current_pos)
        
        # Break if we've reached the end
        if end_pos >= seq_len:
            break
        
        current_pos += stride
    
    return windows, positions


def aggregate_window_outputs(window_outputs, aggregation_method='mean'):
    """
    Aggregate predictions from multiple sliding windows.
    
    Args:
        window_outputs: List of dicts, each containing:
            - 'd_real': Real/fake discrimination score [B, 1]
            - 'd_fake': Real/fake discrimination score [B, 1]
            - 'd_cls': Domain classification logits [B, num_domains]
            - Other optional keys
        aggregation_method: 'mean' (average), 'max' (maximum), or 'first' (first window only)
    
    Returns:
        Aggregated output dict with same structure as input
    """
    if len(window_outputs) == 1:
        return window_outputs[0]
    
    # Stack outputs from all windows
    aggregated = {}
    
    for key in window_outputs[0].keys():
        values = [out[key] for out in window_outputs]
        
        # Stack along a new dimension
        stacked = torch.stack(values, dim=0)  # [num_windows, ...]
        
        if aggregation_method == 'mean':
            aggregated[key] = stacked.mean(dim=0)
        elif aggregation_method == 'max':
            # For classification logits, take max; for scores, take mean
            if 'cls' in key:
                aggregated[key] = stacked.max(dim=0)[0]
            else:
                aggregated[key] = stacked.mean(dim=0)
        elif aggregation_method == 'first':
            aggregated[key] = window_outputs[0][key]
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    
    return aggregated


class StarGANTrainer:
    """
    StarGAN Trainer with real Amadeus Generator and Moonbeam Discriminator
    """
    
    def __init__(
        self,
        amadeus_config_path: str,
        amadeus_checkpoint_path: str,
        amadeus_vocab_path: str,
        moonbeam_config_path: str,
        moonbeam_checkpoint_path: str,
        num_domains: int = 108,
        selected_attrs: list = None,
        g_lr: float = 1e-4,
        d_lr: float = 1e-4,
        lambda_cls: float = 1.0,
        lambda_rec: float = 10.0,
        lambda_gp: float = 10.0,
        n_critic: int = 5,
        temperature: float = 0.5,
        device: str = 'cuda',
        use_multi_gpu: bool = True,
        window_size: int = 3072,
        window_stride: int = None,
        window_aggregation: str = 'mean'
    ):
        """
        Args:
            amadeus_config_path: Path to Amadeus config YAML
            amadeus_checkpoint_path: Path to Amadeus checkpoint
            amadeus_vocab_path: Path to Amadeus vocabulary
            moonbeam_config_path: Path to Moonbeam config JSON
            moonbeam_checkpoint_path: Path to Moonbeam checkpoint
            num_domains: Number of domain labels (default: 108)
            g_lr: Generator learning rate
            d_lr: Discriminator learning rate
            lambda_cls: Domain classification loss weight
            lambda_rec: Cycle consistency loss weight
            lambda_gp: Gradient penalty weight
            n_critic: Discriminator updates per Generator update
            temperature: Gumbel-Softmax temperature
            device: Training device
            use_multi_gpu: Enable multi-GPU model parallelization
            window_size: Sliding window size (default: 3072, Amadeus max input length)
            window_stride: Sliding window stride (default: None = no overlap)
            window_aggregation: Method to aggregate window outputs ('mean', 'max', 'first')
        """
        self.device = device
        self.num_domains = num_domains
        self.selected_attrs = selected_attrs
        self.lambda_cls = lambda_cls
        self.lambda_rec = lambda_rec
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic
        self.temperature = temperature
        self.vocab_path = amadeus_vocab_path  # Save vocab path for loss functions
        self.use_multi_gpu = use_multi_gpu
        self.window_size = window_size
        self.window_stride = window_stride if window_stride is not None else window_size
        self.window_aggregation = window_aggregation
        
        # Multi-GPU device setup
        if use_multi_gpu and torch.cuda.device_count() >= 3:
            self.device_g = 'cuda:0'  # Generator GPU 0
            self.device_d = 'cuda:1'  # Discriminator GPU 1
            self.device_t5 = 'cuda:0'  # T5 encoder on GPU 0
            print(f"\n[Multi-GPU Setup]")
            print(f"  Generator (Amadeus): GPU 0")
            print(f"  Discriminator (Moonbeam): GPU 1")
            print(f"  T5 Encoder: GPU 0")
            print(f"  Total GPUs available: {torch.cuda.device_count()}")
        else:
            self.device_g = device
            self.device_d = device
            self.device_t5 = device
            print(f"\n[Single GPU Setup] Using device: {device}")
        
        # OOM handling state
        self.current_batch_size = None
        self.oom_count = 0
        self.max_oom_retries = 1
        
        # Load Amadeus Generator (from model_zoo.py)
        print("Loading Amadeus Generator...")
        self.G, self.vocab, self.nn_params = self._load_amadeus_model(
            config_path=amadeus_config_path,
            checkpoint_path=amadeus_checkpoint_path,
            vocab_path=amadeus_vocab_path,
            device=self.device_g
        )
        
        # Load Moonbeam Discriminator (LlamaForSequenceClassification)
        print("Loading Moonbeam Discriminator...")
        self.D = self._load_moonbeam_model(
            config_path=moonbeam_config_path,
            checkpoint_path=moonbeam_checkpoint_path,
            num_domains=num_domains,
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
        
        self.g_optimizer = optim.Adam(g_params, lr=g_lr, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(d_params, lr=d_lr, betas=(0.5, 0.999))
        
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
        prediction_order = adjust_prediction_order(
            encoding_scheme, num_features, 
            config.data_params.first_pred_feature, nn_params
        )
        
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
    
    def train_step(
        self,
        real_scores: torch.Tensor,  # [B, T, 8] Amadeus format
        target_context: torch.Tensor,  # [B, T_text, H] Target text embedding
        original_context: torch.Tensor,  # [B, T_text, H] Original text embedding
        real_labels: torch.Tensor  # [B, 108] Original domain labels for discriminator loss
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
            lambda_gp=self.lambda_gp,
            temperature=self.temperature,
            device=self.device_d
        )
        
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.D.parameters(), max_norm=1.0)
        self.d_optimizer.step()
        
        d_loss_val = d_loss.item()
        
        # ==================== Train Generator (every n_critic steps) ====================
        g_loss_val = 0.0
        g_logs = {}
        
        # Note: This simplified version trains G every step
        # In full implementation, use iteration counter to train every n_critic steps
        
        self.g_optimizer.zero_grad()
        
        g_loss, g_logs = compute_generator_loss(
            G=self.G,
            D=self.D,
            real_scores=real_scores,
            context=target_context,
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
    
    def train(
        self,
        dataloader: DataLoader,
        num_epochs: int = 1,
        num_iters: int = 200000,
        save_dir: str = './checkpoints',
        log_dir: str = './logs',
        log_interval: int = 100,
        save_interval: int = 1000
    ):
        """
        Full training loop (solver.py style with iter/next)
        
        Args:
            dataloader: Training data loader
            num_iters: Number of training iterations
            save_dir: Directory to save checkpoints
            log_dir: Directory to save training logs
            log_interval: Steps between logging
            save_interval: Steps between checkpoint saves
        """
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
                                real_labels=original_labels
                            )
                            
                            # Store window output (we'll aggregate after all windows)
                            window_outputs.append((d_loss_w, g_loss_w, logs_w))
                        
                        # Aggregate results from all windows
                        if len(windows) > 1:
                            #print(f"  Aggregating results from {len(windows)} windows using '{self.window_aggregation}' method")
                            # Simple aggregation: average loss across windows
                            d_loss = sum(out[0] for out in window_outputs) / len(window_outputs)
                            g_loss = sum(out[1] for out in window_outputs) / len(window_outputs)
                            # Merge logs
                            logs = {}
                            for key in window_outputs[0][2].keys():
                                logs[key] = sum(out[2][key] for out in window_outputs) / len(window_outputs)
                        else:
                            d_loss, g_loss, logs = window_outputs[0]
                        
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
                #if train_success:
                if success_count % log_interval == 0:
                    # Prepare log data
                    log_data = {
                        'epoch': epoch + 1,
                        'iteration': i + 1,
                        **logs
                    }
                    
                    if self.oom_count > 0:
                        log_data['oom_events'] = self.oom_count
                    
                    # Write to text log file
                    log_str = f"Epoch [{log_data['epoch']}/{num_epochs}] | Iteration [{log_data['iteration']}/{num_iters}] | Success: {success_count} | Failure: {failure_count} | "
                    log_str += ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in logs.items()])
                    if self.oom_count > 0:
                        log_str += f", OOM Events: {self.oom_count}"
                    
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
            'd_optimizer_state_dict': self.d_optimizer.state_dict()
        }
        
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.G.load_state_dict(checkpoint['generator_state_dict'])
        self.D.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        
        epoch = checkpoint['epoch']
        step = checkpoint['step']
        
        return epoch, step


def main():
    parser = argparse.ArgumentParser(description='StarGAN Training with Real Models')
    
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
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--temperature', type=float, default=1.15, help='temperature for sampling method')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    
    # Sliding window parameters for handling sequences longer than Amadeus max input length
    parser.add_argument('--window_size', type=int, default=2048,#3072//3, 
                       help='sliding window size (default: 3072, Amadeus max input length)')
    parser.add_argument('--window_stride', type=int, default=3072//3,
                       help='sliding window stride (default: None = window_size, no overlap)')
    parser.add_argument('--window_aggregation', type=str, default='mean',
                       choices=['mean', 'max', 'first'],
                       help='aggregation method for window outputs: mean (average), max (maximum), first (first window only)')
    
    # Data and I/O (from main.py)
    now_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f'MidiCaps-{now_time}'
    parser.add_argument('--score_dir', type=str, default='../Amadeus/dataset/MidiCaps/corpus/tuneidx_', help='Training data directory')
    parser.add_argument('--attr_path', type=str, default='../Dataset/MidiCaps/train.json', help='Attribute path')
    parser.add_argument('--vocab_path', type=str, default='../Amadeus/models/Amadeus-S/files/checkpoints/vocab_LakhALLFined_nb8.json', help='Vocabulary path')
    parser.add_argument('--model_save_dir', type=str, default=f'output/{model_name}/models', help='Checkpoint save directory')
    parser.add_argument('--log_dir', type=str, default=f'output/{model_name}/logs', help='Log directory')
    parser.add_argument('--sample_dir', type=str, default=f'output/{model_name}/samples', help='Sample directory')
    parser.add_argument('--result_dir', type=str, default=f'output/{model_name}/results', help='Result directory')
    parser.add_argument('--encoding', type=str, default='nb8', help='Encoding type')
    parser.add_argument('--dataset', type=str, default='MidiCaps', help='Dataset name')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--device', type=str, default='cuda', help='Training device')
    
    # Step sizes (from main.py)
    parser.add_argument('--log_step', type=int, default=10, help='Log step interval')
    parser.add_argument('--sample_step', type=int, default=1000, help='Sample step interval')
    parser.add_argument('--model_save_step', type=int, default=10000, help='Model save step interval')
    parser.add_argument('--lr_update_step', type=int, default=1000, help='Learning rate update step')
    
    # Generator configuration (from main.py)
    parser.add_argument('--generate_length', type=int, default=100, help='length of the generated sequence')
    parser.add_argument('--sampling_method', type=str, choices=('top_p', 'top_k'), default="top_k", help='sampling method for generation')
    parser.add_argument('--threshold', type=float, default=0.99, help='threshold for sampling method')
    
    # Domain labels (from main.py)
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for Music dataset',
                        default=['funk', 'celtic', 'instrumentalpop', 'ambient', 'reggae', 'popfolk', 'dance', 'rock', 'classical', 'instrumentalrock', 'folk', 'poprock', 'indie', 'hiphop', 'blues', 'experimental', 'punkrock', 'jazz', 'electronic', 'techno', 'jazzfusion', 'pop', 'alternative', 'electropop', 'soundtrack', 'trance', 'house', 'metal', 'world', 'symphonic', 'lounge', 'easylistening', 'orchestral', 'country', 'newage', 'latin', 'drumnbass', '80s', '90s', 'swing', 'chillout', 'synthpop', 'movie', 'christmas', 'heavy', 'corporate', 'action', 'romantic', 'energetic', 'background', 'children', 'calm', 'adventure', 'motivational', 'summer', 'funny', 'dramatic', 'cool', 'positive', 'emotional', 'holiday', 'deep', 'love', 'dark', 'dream', 'advertising', 'happy', 'soundscape', 'film', 'melodic', 'drama', 'uplifting', 'epic', 'ballad', 'sad', 'relaxing', 'party', 'trailer', 'inspiring', 'soft', 'slow', 'game', 'retro', 'fun', 'meditative', 'sport', 'space', 'commercial', 'documentary', 'upbeat', 'Eb major', 'B major', 'Bb major', 'F# minor', 'F# major', 'G# minor', 'A major', 'B minor', 'E minor', 'D minor', 'F minor', 'G minor', 'F major', 'Eb minor', 'C major', 'A minor', 'G major', 'D major', 'C# major', 'Bb minor', 'Ab major', 'C# minor', 'C minor', 'E major'])
    
    args = parser.parse_args()
    
    # Create directories if not exist
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    
    # Create trainer
    trainer = StarGANTrainer(
        amadeus_config_path=args.g_modelpath,  # Use g_modelpath as config path
        amadeus_checkpoint_path=args.amadeus_checkpoint,
        amadeus_vocab_path=args.vocab_path,  # Pass vocab_path
        moonbeam_config_path=args.d_configpath,
        moonbeam_checkpoint_path=args.d_modelpath,  # Use d_modelpath as checkpoint
        num_domains=len(args.selected_attrs),  # Number of domains from selected_attrs
        selected_attrs=args.selected_attrs,
        g_lr=args.g_lr,
        d_lr=args.d_lr,
        lambda_cls=args.lambda_cls,
        lambda_rec=args.lambda_rec,
        lambda_gp=args.lambda_gp,
        n_critic=args.n_critic,
        temperature=args.temperature,
        device=args.device,
        window_size=args.window_size,
        window_stride=args.window_stride,
        window_aggregation=args.window_aggregation
    )
    
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
    
    print(f"Dataloader created successfully!")
    print(f"  Number of batches: {len(dataloader)}")
    print(f"  Training iterations: {args.num_iters}")
    
    # Start training
    trainer.train(
        dataloader=dataloader,
        num_epochs=args.num_epochs,
        num_iters=args.num_iters,
        save_dir=args.model_save_dir,
        log_dir=args.log_dir,
        log_interval=args.log_step,
        save_interval=args.model_save_step
    )


if __name__ == '__main__':
    main()
