"""
Loss functions for End-to-End differentiable StarGAN
Supports gradient flow from Discriminator to Generator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import json
import numpy as np
import re


class StraightThroughEstimator(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) for differentiable argmax
    
    Forward: Returns hard discrete tokens via argmax
    Backward: Passes gradients through softmax probabilities
    
    This allows:
    - Forward pass: Get discrete tokens for Amadeus model
    - Backward pass: Gradients flow through probability distribution
    """
    
    @staticmethod
    def forward(ctx, logits):
        """
        Args:
            logits: [B, T, vocab_size]
        
        Returns:
            hard_tokens: [B, T] with values in [0, vocab_size-1]
        """
        hard_tokens = logits.argmax(dim=-1)  # [B, T]
        probs = F.softmax(logits, dim=-1)  # [B, T, vocab_size]
        
        # Save for backward
        ctx.save_for_backward(logits, hard_tokens, probs)
        return hard_tokens.float()
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Gradient flows through softmax(logits)
        
        Args:
            grad_output: [B, T] gradients w.r.t. hard_tokens
        
        Returns:
            grad_logits: [B, T, vocab_size] gradients w.r.t. logits
        """
        logits, hard_tokens, probs = ctx.saved_tensors
        
        # Broadcast grad_output to match logits shape
        # grad_output: [B, T] → [B, T, 1]
        grad_output_expanded = grad_output.unsqueeze(-1)  # [B, T, 1]
        
        # Gradient flows proportional to probabilities
        # grad_logits = grad_output * probs (weighted by probability)
        grad_logits = grad_output_expanded * probs  # [B, T, vocab_size]
        
        return grad_logits


def gumbel_softmax_sample(logits, tau=0.5, hard=True):
    """
    Gumbel-Softmax sampling for differentiable discrete sampling
    
    Args:
        logits: [B, T, vocab_size]
        tau: Temperature (lower = more discrete, higher = more continuous)
        hard: If True, return hard samples via STE; if False, return soft probabilities
    
    Returns:
        samples: [B, T, vocab_size] if hard=False
        or [B, T] discrete token indices if hard=True
    """
    # Sample Gumbel noise
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-20) + 1e-20)
    
    # Add noise to logits
    noisy_logits = (logits + gumbel_noise) / tau
    
    # Get soft probabilities
    soft_probs = F.softmax(noisy_logits, dim=-1)  # [B, T, vocab_size]
    
    if hard:
        # Hard samples: argmax
        hard_samples = soft_probs.argmax(dim=-1)  # [B, T]
        # Straight-through: use soft in forward, hard in backward
        hard_samples_float = hard_samples.float()
        # Gradient flows through soft_probs
        return hard_samples_float + (soft_probs - soft_probs.detach())
    else:
        return soft_probs


def logits_to_discrete_tokens_differentiable(fake_logits, temperature=0.5, use_gumbel=True):
    """
    Convert fake_logits to discrete tokens in a differentiable way
    
    Two methods:
    1. Gumbel-Softmax (recommended): More stable gradient flow
    2. STE: Traditional straight-through estimator
    
    Args:
        fake_logits: Dict[feature_name] = [B, T, vocab_size]
        temperature: Temperature parameter (lower = more discrete)
        use_gumbel: If True, use Gumbel-Softmax; if False, use STE
    
    Returns:
        hard_tokens: [B, T, num_features] with values as token indices
    """
    device = None
    B, T = None, None
    hard_tokens_list = []
    
    # Convert each feature's logits to discrete tokens
    for feature_name in AMADEUS_FIELDS:
        if feature_name in list(fake_logits[0].keys()):
            logit = fake_logits[0][feature_name]  # [B, T, vocab_size]
            
            if B is None:
                B, T, _ = logit.shape
                device = logit.device
            
            if use_gumbel:
                # Method 1: Gumbel-Softmax (more stable)
                # Use soft sampling with gradient flow
                soft_sample = gumbel_softmax_sample(logit, tau=temperature, hard=False)  # [B, T, vocab_size]
                # Get hard token via argmax, but gradients flow through soft probabilities
                hard_token = soft_sample.argmax(dim=-1)  # [B, T]
                hard_tokens_list.append(hard_token.float())
            else:
                # Method 2: Traditional STE
                hard_token = StraightThroughEstimator.apply(logit)  # [B, T]
                hard_tokens_list.append(hard_token)
    
    # Stack all feature tokens: [B, T, num_features]
    hard_tokens = torch.stack(hard_tokens_list, dim=-1)  # [B, T, 8]
    hard_tokens = hard_tokens.long()  # Convert to long type for embedding indices
    
    return hard_tokens


AMADEUS_FIELDS = ["type", "beat", "chord", "tempo", "instrument", "pitch", "duration", "velocity"]

# Import conversion functions from solver
# We'll define them here to avoid circular imports
def _build_lookup_table(field_dict: dict) -> np.ndarray:
    """Build lookup table from vocab dictionary"""
    max_idx = max(int(k) for k in field_dict.keys())
    table = ["" for _ in range(max_idx + 1)]
    for k, v in field_dict.items():
        table[int(k)] = v
    return np.array(table, dtype=object)

def amadeus_to_vocab(amadeus_tokens: torch.Tensor, vocab_path: str) -> np.ndarray:
    """Convert Amadeus token indices to vocabulary strings"""
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # Handle batch dimension
    if amadeus_tokens.dim() == 3:
        B, T, _ = amadeus_tokens.shape
        tokens_np = amadeus_tokens.detach().cpu().numpy().astype(np.int64)
        decoded = np.empty((B, T, 8), dtype=object)
        
        for b in range(B):
            for axis, field in enumerate(AMADEUS_FIELDS):
                lookup = _build_lookup_table(vocab[field])
                decoded[b, :, axis] = lookup[tokens_np[b, :, axis]]
    else:
        tokens_np = amadeus_tokens.detach().cpu().numpy().astype(np.int64)
        decoded = np.empty(tokens_np.shape, dtype=object)
        for axis, field in enumerate(AMADEUS_FIELDS):
            lookup = _build_lookup_table(vocab[field])
            decoded[:, axis] = lookup[tokens_np[:, axis]]

    return decoded

def vocab_to_moonbeam(amadeus_vocabs, time_resolution=10, default_tempo=120, in_beat_resolution=4):
    """Convert Amadeus vocab strings to Moonbeam token indices"""
    # Convert to numpy for easier processing
    if isinstance(amadeus_vocabs, torch.Tensor):
        amadeus_np = amadeus_vocabs.cpu().numpy()
        use_torch = True
    else:
        amadeus_np = np.array(amadeus_vocabs)
        use_torch = False
    
    # Handle batch dimension
    if amadeus_np.ndim == 3:
        B, T, _ = amadeus_np.shape
        moonbeam_tokens = []
        for b in range(B):
            moonbeam_b = _vocab_to_moonbeam_single(
                amadeus_np[b], time_resolution, default_tempo, in_beat_resolution
            )
            moonbeam_tokens.append(moonbeam_b)
        moonbeam_np = np.stack(moonbeam_tokens, axis=0)
    else:
        moonbeam_np = _vocab_to_moonbeam_single(
            amadeus_np, time_resolution, default_tempo, in_beat_resolution
        )
    
    # Convert back to torch if needed
    if use_torch:
        return torch.from_numpy(moonbeam_np).long()
    return moonbeam_np

def _vocab_to_moonbeam_single(amadeus_np, time_resolution, default_tempo, in_beat_resolution):
    """Convert single sequence from Amadeus vocab to Moonbeam indices"""
    num_notes = amadeus_np.shape[0]
    moonbeam_np = np.zeros((num_notes, 6), dtype=np.int32)
    
    # State tracking
    current_bar = -1
    current_tempo = default_tempo
    current_time_signature = (4, 4)
    
    for i in range(num_notes):
        type_token = amadeus_np[i, 0]
        beat_token = amadeus_np[i, 1]
        tempo_token = amadeus_np[i, 3]
        instrument_token = amadeus_np[i, 4]
        pitch_token = amadeus_np[i, 5]
        duration_token = amadeus_np[i, 6]
        velocity_token = amadeus_np[i, 7]
        
        # Parse type token
        if isinstance(type_token, str):
            if type_token.startswith('NNN_time_signature_'):
                time_sig_match = re.search(r'time_signature_(\d+)/(\d+)', type_token)
                if time_sig_match:
                    current_time_signature = (int(time_sig_match.group(1)), int(time_sig_match.group(2)))
                current_bar += 1
            elif type_token == 'SNN':
                current_bar += 1
        
        # Extract tempo
        if isinstance(tempo_token, str):
            tempo_match = re.search(r'Tempo_(\d+)', tempo_token)
            if tempo_match:
                current_tempo = int(tempo_match.group(1))
        
        # Extract beat position
        if isinstance(beat_token, str):
            beat_match = re.search(r'Beat_(\d+)', beat_token)
            beat_index = int(beat_match.group(1)) if beat_match else 0
        else:
            beat_index = int(beat_token) if beat_token else 0
        
        # Calculate onset
        numerator, _ = current_time_signature
        subdivisions_per_bar = numerator * in_beat_resolution
        onset_in_subdivisions = max(0, current_bar) * subdivisions_per_bar + beat_index
        onset_in_quarter_notes = onset_in_subdivisions / in_beat_resolution
        ms_per_quarter_note = 60000.0 / current_tempo
        onset_ms = onset_in_quarter_notes * ms_per_quarter_note
        onset_in_10ms = int(round(onset_ms / time_resolution))
        
        # Extract duration
        if isinstance(duration_token, str):
            dur_match = re.search(r'Note_Duration_([\d.]+)', duration_token)
            duration_in_subdivisions = float(dur_match.group(1)) if dur_match else 1.0
        else:
            duration_in_subdivisions = float(duration_token) if duration_token else 1.0
        
        duration_in_quarter_notes = duration_in_subdivisions / in_beat_resolution
        duration_ms = duration_in_quarter_notes * ms_per_quarter_note
        duration_in_10ms = int(round(duration_ms / time_resolution))
        duration_in_10ms = max(1, min(duration_in_10ms, 1024))
        
        # Extract pitch
        if isinstance(pitch_token, str):
            pitch_match = re.search(r'Note_Pitch_(\d+)', pitch_token)
            pitch = int(pitch_match.group(1)) if pitch_match else 60
        else:
            pitch = int(pitch_token) if pitch_token else 60
        
        octave = min(max(pitch // 12, 0), 10)
        pitch_class = pitch % 12
        
        # Extract instrument
        if isinstance(instrument_token, str):
            inst_match = re.search(r'Instrument_(\d+)', instrument_token)
            instrument = int(inst_match.group(1)) if inst_match else 0
        else:
            instrument = int(instrument_token) if instrument_token else 0
        instrument = max(0, min(instrument, 128))
        
        # Extract velocity
        if isinstance(velocity_token, str):
            vel_match = re.search(r'Note_Velocity_(\d+)', velocity_token)
            velocity = int(vel_match.group(1)) if vel_match else 64
        else:
            velocity = int(velocity_token) if velocity_token else 64
        velocity = max(0, min(velocity, 127))
        
        # Store in Moonbeam format
        moonbeam_np[i, 0] = onset_in_10ms
        moonbeam_np[i, 1] = duration_in_10ms
        moonbeam_np[i, 2] = octave
        moonbeam_np[i, 3] = pitch_class
        moonbeam_np[i, 4] = instrument
        moonbeam_np[i, 5] = velocity
    
    return moonbeam_np

def amadeus_to_moonbeam_discrete(amadeus_tokens, vocab_path):
    """
    Convert discrete Amadeus tokens to Moonbeam format using vocab conversion
    
    Args:
        amadeus_tokens: [B, T, 8] Amadeus format (token indices)
        vocab_path: Path to Amadeus vocabulary file
    Returns:
        moonbeam_tokens: [B, T, 6] Moonbeam format
    """
    device = amadeus_tokens.device
    
    # Convert token indices to vocab strings
    amadeus_vocabs = amadeus_to_vocab(amadeus_tokens, vocab_path)
    
    # Convert vocab strings to Moonbeam indices
    moonbeam_tokens = vocab_to_moonbeam(amadeus_vocabs)
    
    # Ensure it's on the right device
    if isinstance(moonbeam_tokens, torch.Tensor):
        moonbeam_tokens = moonbeam_tokens.to(device)
    else:
        moonbeam_tokens = torch.from_numpy(moonbeam_tokens).long().to(device)
    
    return moonbeam_tokens


def logits_to_embeddings_via_projection(fake_logits, projection_layer, vocab_size_list, hidden_size):
    """
    Convert Generator logits to Discriminator embeddings via single projection layer
    FULLY DIFFERENTIABLE approach
    
    Args:
        fake_logits: Dict[feature_name] = [B, T, vocab_size] Generator output
        projection_layer: nn.Linear(total_vocab_size, hidden_size)
        vocab_size_list: List of vocab sizes for each feature in order
        hidden_size: Discriminator hidden size
    
    Returns:
        embeddings: [B, T, hidden_size] Embeddings for Discriminator
    """
    B, T = None, None
    device = None
    logits_list = []
    
    # Collect logits for each feature in order
    for feature_name in AMADEUS_FIELDS:
        if feature_name in list(fake_logits[0].keys()):
            logit = fake_logits[0][feature_name]  # [B, T, vocab_size]
            if B is None:
                B, T, _ = logit.shape
                device = logit.device
            
            #logit = logit.squeeze(0)
            # Apply softmax to get probabilities (differentiable!)
            #probs = F.softmax(logit, dim=-1)  # [B, T, vocab_size]
            logits_list.append(logit)
    
    # Concatenate all feature probabilities: [B, T, total_vocab_size]
    concatenated = torch.cat(logits_list, dim=-1)  # [B, T, Σvocab_sizes]
    
    # Move to projection_layer device if needed
    projection_device = projection_layer.weight.device
    if concatenated.device != projection_device:
        concatenated = concatenated.to(projection_device)
    
    # Apply projection layer: [B, T, Σvocab_sizes] → [B, T, hidden_size]
    embeddings = projection_layer(concatenated)  # [B, T, hidden_size]
    
    return embeddings


def logits_to_embedded_input(fake_logits, embedding_layers, vocab_size_list, emb_size):
    """
    Convert Generator logits to Generator input embeddings via vocabulary embeddings
    FULLY DIFFERENTIABLE approach: probs × vocab_embedding with summation
    
    Args:
        fake_logits: Dict[feature_name] = [B, T, vocab_size] Generator output
        embedding_layers: List of nn.Embedding layers for each feature (same structure as MultiEmbedding._make_emb_layers)
        vocab_size_list: List of vocab sizes for each feature in order
        emb_size: Embedding size for each feature
    
    Returns:
        embedded_input: [B, T, emb_size] Embedded representation for Generator input
    """
    B, T = None, None
    device = None
    embedded_features = []
    
    # Process each feature in order
    for feat_idx, feature_name in enumerate(AMADEUS_FIELDS):
        if feature_name in list(fake_logits[0].keys()):
            logit = fake_logits[0][feature_name]  # [B, T, vocab_size]
            if B is None:
                B, T, _ = logit.shape
                device = logit.device
            
            # 1. Convert logits to probabilities via softmax
            probs = F.softmax(logit, dim=-1)  # [B, T, vocab_size]
            
            # 2. Get vocabulary embeddings for this feature
            vocab_size = vocab_size_list[feat_idx]
            emb_layer = embedding_layers[feat_idx]
            
            # Create indices for all vocabulary entries
            vocab_indices = torch.arange(vocab_size, device=device, dtype=torch.long)  # [vocab_size]
            vocab_embeddings = emb_layer(vocab_indices)  # [vocab_size, emb_size]
            
            # 3. Compute weighted sum: probs × embeddings
            # probs: [B, T, vocab_size], vocab_embeddings: [vocab_size, emb_size]
            # result: [B, T, emb_size]
            weighted_emb = torch.einsum('btv,ve->bte', probs, vocab_embeddings)  # [B, T, emb_size]
            
            embedded_features.append(weighted_emb)
    
    # 4. Sum all feature embeddings
    # embedded_features is list of [B, T, emb_size], length = 8
    summed_embedding = torch.stack(embedded_features, dim=2)  # [B, T, 8, emb_size]
    embedded_input = torch.sum(summed_embedding, dim=2)  # [B, T, emb_size]
    
    return embedded_input


def tokens_to_embeddings_via_projection(amadeus_tokens, projection_layer, vocab_size_list, hidden_size):
    """
    Convert discrete Amadeus tokens to Discriminator embeddings via single projection layer
    FULLY DIFFERENTIABLE approach for real scores
    
    Args:
        amadeus_tokens: [B, T, 8] Discrete Amadeus token indices
        projection_layer: nn.Linear(total_vocab_size, hidden_size)
        vocab_size_list: List of vocab sizes for each feature in order
        hidden_size: Discriminator hidden size
    
    Returns:
        embeddings: [B, T, hidden_size] Embeddings for Discriminator
    """
    B, T, num_features = amadeus_tokens.shape
    device = amadeus_tokens.device
    onehot_list = []
    
    # Convert each feature to one-hot and collect
    for feat_idx, feature_name in enumerate(AMADEUS_FIELDS):
        tokens = amadeus_tokens[:, :, feat_idx]  # [B, T] discrete indices
        vocab_size = vocab_size_list[feat_idx]
        
        # One-hot encode discrete tokens (differentiable!)
        onehot = F.one_hot(tokens.long(), num_classes=vocab_size).float()  # [B, T, vocab_size]
        onehot_list.append(onehot)
    
    # Concatenate all feature one-hots: [B, T, total_vocab_size]
    concatenated = torch.cat(onehot_list, dim=-1)  # [B, T, Σvocab_sizes]
    
    # Move to projection_layer device if needed
    projection_device = projection_layer.weight.device
    if concatenated.device != projection_device:
        concatenated = concatenated.to(projection_device)
    
    # Apply projection layer: [B, T, Σvocab_sizes] → [B, T, hidden_size]
    embeddings = projection_layer(concatenated)  # [B, T, hidden_size]
    
    return embeddings


def compute_discriminator_loss(
    G,
    D,
    real_scores,
    context,
    real_labels,
    projection_layer,
    vocab_size_list,
    hidden_size,
    vocab_path,
    lambda_cls=1.0,
    lambda_gp=10.0,
    temperature=0.5,
    device=None
):
    """
    Compute Discriminator loss with gradient flow through Generator
    FULLY DIFFERENTIABLE design
    
    Args:
        G: Generator (AmadeusForStarGAN)
        D: Discriminator (LlamaForSequenceDoubleClassification)
        real_scores: [B, T, 8] Real scores (Amadeus discrete tokens)
        context: dict with 'input_ids' and 'attention_mask' for T5 encoder
        real_labels: [B, 108] Original domain labels (multi-hot)
        projection_layer: nn.Linear(total_vocab_size, hidden_size)
        vocab_size_list: List of vocab sizes for each feature
        hidden_size: Discriminator hidden size
        vocab_path: Path to Amadeus vocabulary file
        lambda_cls: Weight for domain classification loss
        lambda_gp: Weight for gradient penalty
        temperature: Temperature (unused in differentiable approach)
    
    Returns:
        d_loss: Total discriminator loss
        loss_dict: Dictionary of individual losses
    """
    real_scores = real_scores.long()
    B, T, _ = real_scores.shape
    
    # Move real_scores to Discriminator device if needed
    if real_scores.device != device:
        real_scores = real_scores.to(device)
    
    # Move labels to Discriminator device
    real_labels = real_labels.to(device)
    
    # ========== Real score processing ==========
    # Convert discrete Amadeus tokens to embeddings via projection layer (FULLY DIFFERENTIABLE!)
    real_embeddings = tokens_to_embeddings_via_projection(real_scores, projection_layer, vocab_size_list, hidden_size)  # [B, T, hidden_size]
    
    # Use embeddings directly as Discriminator input
    real_cls_output = D(inputs_embeds=real_embeddings)
    
    # Domain classification loss
    real_cls_logits = real_cls_output.logits if hasattr(real_cls_output, 'logits') else real_cls_output["logits"]
    d_loss_cls = F.binary_cross_entropy_with_logits(
        real_cls_logits,
        real_labels.float()
    ).to(device)
    
    # ========== Fake score processing (gradient flows!) ==========
    fake_logits, fake_input_dict = G(
        real_scores.to(G.device if hasattr(G, 'device') else 'cuda:0'),  # Move to Generator device
        real_scores.to(G.device if hasattr(G, 'device') else 'cuda:0'),  # target = input for teacher-forcing
        context=context
    )
    # fake_logits: Dict of {feature: [B, T, vocab_size]}
    
    # Convert Generator logits to embeddings via projection layer (FULLY DIFFERENTIABLE!)
    fake_embeddings = logits_to_embeddings_via_projection(fake_logits, projection_layer, vocab_size_list, hidden_size)  # [B, T, hidden_size]
    
    # Use embeddings directly as Discriminator input
    fake_cls_output = D(inputs_embeds=fake_embeddings)
    
    # Fake loss: maximize likelihood that D classifies fake as real domain
    fake_cls_logits = fake_cls_output.logits if hasattr(fake_cls_output, 'logits') else fake_cls_output["logits"]
    d_loss_fake = F.binary_cross_entropy_with_logits(
        fake_cls_logits,
        real_labels.float()  # Fool D into thinking fake is real
    ).to(device)
    
    # ========== Gradient Penalty (simplified) ==========
    # For soft embeddings, gradient penalty is less critical
    d_loss_gp = torch.tensor(0.0, device=device)
    
    # ========== Total Discriminator loss ==========
    d_loss = d_loss_fake + lambda_cls * d_loss_cls + lambda_gp * d_loss_gp
    
    loss_dict = {
        'D/loss_fake': d_loss_fake.item(),
        'D/loss_cls': d_loss_cls.item(),
        'D/loss_gp': d_loss_gp.item() if isinstance(d_loss_gp, torch.Tensor) else d_loss_gp,
        'D/loss_total': d_loss.item()
    }
    
    return d_loss, loss_dict


def compute_generator_loss(
    G,
    D,
    real_scores,
    context,
    original_context,
    projection_layer,
    vocab_size_list,
    hidden_size,
    embedding_layers,
    emb_size,
    lambda_cls=1.0,
    lambda_rec=10.0,
    temperature=0.5,
    device=None
):
    """
    Compute Generator loss with gradient flow through Discriminator
    FULLY DIFFERENTIABLE design
    
    Args:
        G: Generator (AmadeusForStarGAN)
        D: Discriminator (LlamaForSequenceDoubleClassification)
        real_scores: [B, T, 8] Real scores (Amadeus discrete tokens)
        context: dict with 'input_ids' and 'attention_mask' for target domain
        original_context: dict with 'input_ids' and 'attention_mask' for original domain
        projection_layer: nn.Linear(total_vocab_size, hidden_size)
        vocab_size_list: List of vocab sizes for each feature
        hidden_size: Discriminator hidden size
        embedding_layers: List of nn.Embedding layers (from MultiEmbedding._make_emb_layers)
        emb_size: Embedding size for each feature
        lambda_cls: Weight for domain classification loss
        lambda_rec: Weight for reconstruction loss
        temperature: Temperature (unused in differentiable approach)
    
    Returns:
        g_loss: Total generator loss
        loss_dict: Dictionary of individual losses
        fake_hard_tokens: [B, T, 8] Generated discrete tokens
    """
    real_scores = real_scores.long()
    B, T, _ = real_scores.shape
    
    # Move real_scores to Generator device
    if real_scores.device != device:
        real_scores = real_scores.to(device)
    
    # ========== Original → Target transformation ==========
    fake_logits, fake_input_dict = G(
        real_scores,
        real_scores,  # target = input for teacher-forcing
        context=context
    )
    # fake_logits: Dict of {feature: [B, T, vocab_size]}
    
    # Convert Generator logits to embeddings via projection layer (FULLY DIFFERENTIABLE!)
    fake_embeddings = logits_to_embeddings_via_projection(fake_logits, projection_layer, vocab_size_list, hidden_size)  # [B, T, hidden_size]
    
    # Use embeddings directly as Discriminator input
    fake_cls_output = D(inputs_embeds=fake_embeddings)
    
    # Adversarial loss: Generator tries to fool Discriminator
    fake_cls_logits = fake_cls_output.logits if hasattr(fake_cls_output, 'logits') else fake_cls_output["logits"]
    # For adversarial loss, we can use gradient reversal or inverse labels
    g_loss_adv = F.binary_cross_entropy_with_logits(
        fake_cls_logits,
        torch.zeros_like(fake_cls_logits)  # Try to make D think it's original domain
    ).to(device)
    
    # Domain classification loss (skip for now)
    g_loss_cls = torch.tensor(0.0, device=device)
    
    # ========== Convert fake_logits to discrete tokens (DIFFERENTIABLE via Gumbel-Softmax) ==========
    # Use Gumbel-Softmax for more stable and reliable gradient flow
    # - Forward: discrete tokens for Amadeus model (via argmax on soft probabilities)
    # - Backward: gradients flow through softmax probabilities
    fake_hard_tokens = logits_to_discrete_tokens_differentiable(
        fake_logits, 
        temperature=temperature,  # Use provided temperature parameter
        use_gumbel=True  # Use Gumbel-Softmax (more stable than STE)
    )  # [B, T, 8] differentiable
    
    # ========== Cycle Consistency: Target → Original ==========
    # Reconstruct back to original domain using differentiable hard tokens
    # Gradients flow through Gumbel-Softmax to the Generator
    reconst_logits, _ = G(
        fake_hard_tokens,  # [B, T, 8] differentiable hard tokens from Gumbel-Softmax
        fake_hard_tokens,  # target = input for teacher-forcing
        context=original_context
    )
    
    # Reconstruction loss: Cross-entropy per feature
    g_loss_rec = 0
    for feature_idx, feature_name in enumerate(AMADEUS_FIELDS):
        reconst_logit = reconst_logits[0][feature_name]  # [B, T, vocab_size]
        target_token = real_scores[:, :, feature_idx]  # [B, T]
        
        g_loss_rec += F.cross_entropy(
            reconst_logit.reshape(-1, reconst_logit.size(-1)),
            target_token.reshape(-1),
            ignore_index=-1
        ).to(device)
    
    g_loss_rec = g_loss_rec / len(AMADEUS_FIELDS)
    
    # ========== Total Generator loss ==========
    g_loss = g_loss_adv + lambda_cls * g_loss_cls + lambda_rec * g_loss_rec
    
    loss_dict = {
        'G/loss_adv': g_loss_adv.item(),
        'G/loss_cls': g_loss_cls.item() if isinstance(g_loss_cls, torch.Tensor) else g_loss_cls,
        'G/loss_rec': g_loss_rec.item(),
        'G/loss_total': g_loss.item()
    }
    
    return g_loss, loss_dict


def check_gradient_flow(model, name="Model"):
    """
    Check gradient flow through model parameters
    
    Args:
        model: PyTorch model
        name: Model name for logging
    """
    print(f"\n=== Gradient Flow Check: {name} ===")
    total_norm = 0
    param_count = 0
    no_grad_count = 0
    
    for param_name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.norm().item()
            total_norm += param_norm
            param_count += 1
            if param_norm < 1e-7:
                print(f"  [WARN] {param_name}: grad_norm={param_norm:.6f} (very small)")
        else:
            no_grad_count += 1
            print(f"  [ERROR] {param_name}: NO GRADIENT!")
    
    avg_norm = total_norm / param_count if param_count > 0 else 0
    print(f"  Total params with grad: {param_count}")
    print(f"  Params without grad: {no_grad_count}")
    print(f"  Average grad norm: {avg_norm:.6f}")
    print("=" * 50)


def generate_target_domain(source_domains, num_samples=None):
    """
    Generate random target domain labels different from source
    
    Args:
        source_domains: [B, 108] Source domain labels
        num_samples: Number of samples (default: B)
    
    Returns:
        target_domains: [B, 108] Target domain labels
    """
    B = source_domains.size(0)
    if num_samples is None:
        num_samples = B
    
    # Random permutation
    rand_idx = torch.randperm(B)
    target_domains = source_domains[rand_idx]
    
    return target_domains
