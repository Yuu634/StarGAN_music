"""
Loss functions for End-to-End differentiable StarGAN
Supports gradient flow from Discriminator to Generator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import json
import numpy as np
import re
import sys

sys.path.append("../Amadeus/Amadeus")
from train_utils import dispersive_loss, NLLLoss4CompoundToken
from utils import logits_to_moonbeam_embeddings

AMADEUS_FIELDS = ["type", "beat", "chord", "tempo", "instrument", "pitch", "duration", "velocity"]


class MoonbeamProjectionAdapter(nn.Module):
    """Project 6×hidden Moonbeam embeddings to discriminator hidden size."""

    def __init__(self, moonbeam_combined_dim: int, hidden_size: int):
        super().__init__()
        self.projection = nn.Linear(moonbeam_combined_dim, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, moonbeam_embeddings: torch.Tensor) -> torch.Tensor:
        projected = self.projection(moonbeam_embeddings)
        return self.layer_norm(projected)

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
        if feature_name in list(fake_logits.keys()):
            logit = fake_logits[feature_name]  # [B, T, vocab_size]
            
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


# Import conversion functions from solver
# We'll define them here to avoid circular imports
def _build_lookup_table(field_dict: dict) -> np.ndarray:
    """Build lookup table from vocab dictionary"""
    max_idx = max(int(k) for k in field_dict.keys())
    table = ["" for _ in range(max_idx + 1)]
    for k, v in field_dict.items():
        table[int(k)] = v
    return np.array(table, dtype=object)
"""
def amadeus_to_vocab(amadeus_tokens: torch.Tensor, vocab_path: str) -> np.ndarray:
    Convert Amadeus token indices to vocabulary string
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
    Convert Amadeus vocab strings to Moonbeam token indices
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
    """

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
        if feature_name in list(fake_logits.keys()):
            logit = fake_logits[feature_name]  # [B, T, vocab_size]
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
        if feature_name in list(fake_logits.keys()):
            logit = fake_logits[feature_name]  # [B, T, vocab_size]
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
    D_config,
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
        moonbeam_converter: Optional LogitsToMoonbeamEmbedding instance
        moonbeam_adapter: Optional MoonbeamProjectionAdapter instance
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
    #real_embeddings = tokens_to_embeddings_via_projection(real_scores, projection_layer, vocab_size_list, hidden_size)  # [B, T, hidden_size]
    vocab = amadeus_to_vocab(real_scores, vocab_path)
    real_tokens = vocab_to_moonbeam(vocab)
    real_tokens = torch.as_tensor(real_tokens, device=device, dtype=torch.long)
    """sos_tokens = torch.full((B,6),D_config.sos_token, device=device)
    eos_tokens = torch.full((B,6),D_config.eos_token, device=device)
    cls_tokens = torch.full((B,6),D_config.classification_token, device=device)
    real_tokens = torch.cat([
        sos_tokens,
        real_tokens,
        eos_tokens,
        cls_tokens
    ], dim=-2)"""
    #real_tokens = real_tokens.unsqueeze(0)
    
    #print("real_tokens")
    #print(real_tokens.shape)
    #print(real_tokens)
    # Use embeddings directly as Discriminator input
    real_cls_output = D(input_ids=real_tokens, Is_real=True)
    
    # Calculation real loss
    d_loss_real = - torch.mean(real_cls_output.real_fake_logits)
    
    real_cls_logits = real_cls_output.logits if hasattr(real_cls_output, 'logits') else real_cls_output["logits"]
    print(real_cls_logits)
    print(real_labels)
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
    
    fake_embeddings = logits_to_moonbeam_embeddings(fake_logits, D, vocab_path)
    
    # Use embeddings directly as Discriminator input
    fake_cls_output = D(inputs_embeds=fake_embeddings, Is_real=False)
    
    # Calculation fake loss
    d_loss_fake = torch.mean(fake_cls_output.real_fake_logits)
    
    # ========== Gradient Penalty (WGAN-GP) ==========
    # Interpolate between real and fake embeddings for gradient penalty
    # This stabilizes the discriminator and prevents gradient explosion
    batch_size = real_embeddings.size(0)
    seq_len = real_embeddings.size(1)
    
    # Generate random interpolation coefficients [B, 1, 1]
    alpha = torch.rand(batch_size, 1, 1, device=device)
    alpha = alpha.expand(batch_size, seq_len, -1)  # [B, T, 1] for broadcasting
    
    # Interpolate: hat_embeddings = alpha * real + (1 - alpha) * fake
    hat_embeddings = (alpha * real_embeddings.detach() + (1 - alpha) * fake_embeddings.detach()).requires_grad_(True)
    
    # Forward pass through discriminator with interpolated embeddings
    hat_cls_output = D(inputs_embeds=hat_embeddings, Is_real=True)
    hat_real_fake_loss = - hat_cls_output.real_fake_logits  # [B*T,]
    
    # Compute gradient of discriminator output w.r.t. interpolated embeddings
    # This measures the Lipschitz constant of the discriminator
    gradients = torch.autograd.grad(
        outputs=hat_real_fake_loss.sum(),
        inputs=hat_embeddings,
        create_graph=True,
        retain_graph=True,
    )[0]  # [B, T, hidden_size]
    
    # Compute L2 norm of gradients for each sample
    gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=[1, 2]) + 1e-8)  # [B]
    
    # Gradient penalty: (||grad|| - 1)^2
    # Target is 1-Lipschitz constraint
    d_loss_gp = torch.mean((gradients_norm - 1.0)**2)
    
    # ========== Total Discriminator loss ==========
    d_loss = d_loss_real + d_loss_fake + lambda_cls * d_loss_cls + lambda_gp * d_loss_gp
    
    # Compute gradient norm for monitoring
    grad_norm_d = 0
    for p in D.parameters():
        if p.grad is not None:
            grad_norm_d += (p.grad.data ** 2).sum().item()
    grad_norm_d = grad_norm_d ** 0.5 if grad_norm_d > 0 else 0
    
    loss_dict = {
        'D/loss_real': d_loss_real.item(),
        'D/loss_fake': d_loss_fake.item(),
        'D/loss_cls': d_loss_cls.item(),
        'D/loss_gp': d_loss_gp.item() if isinstance(d_loss_gp, torch.Tensor) else d_loss_gp,
        'D/loss_total': d_loss.item(),
        #'D/grad_norm': grad_norm_d,
        #'D/grad_penalty_norm': gradients_norm.mean().item() if 'gradients_norm' in locals() else 0
    }
    
    return d_loss, loss_dict


def compute_generator_loss(
    G,
    D,
    real_scores,
    context,
    target_labels,
    original_context,
    projection_layer,
    vocab_size_list,
    hidden_size,
    embedding_layers,
    emb_size,
    lambda_cls=1.0,
    lambda_rec=10.0,
    temperature=0.5,
    device=None,
    lambda_amadeus=1.0,
    lambda_dispersive=0.5,
):
    """
    Compute Generator loss with gradient flow through Discriminator
    FULLY DIFFERENTIABLE design
    
    Now includes Amadeus model losses:
    - Note Decoder Loss: Per-feature cross-entropy for generated tokens
    - Music Latent Space Discriminability Enhancement Loss: Promotes diversity in latent space
    
    Args:
        G: Generator (AmadeusForStarGAN)
        D: Discriminator (LlamaForSequenceDoubleClassification)
        real_scores: [B, T, 8] Real scores (Amadeus discrete tokens)
        context: dict with 'input_ids' and 'attention_mask' for target domain
        target_labels: [B, 108] Target domain labels
        original_context: dict with 'input_ids' and 'attention_mask' for original domain
        projection_layer: nn.Linear(total_vocab_size, hidden_size)
        vocab_size_list: List of vocab sizes for each feature
        hidden_size: Discriminator hidden size
        embedding_layers: List of nn.Embedding layers (from MultiEmbedding._make_emb_layers)
        emb_size: Embedding size for each feature
        moonbeam_converter: Optional LogitsToMoonbeamEmbedding instance
        moonbeam_adapter: Optional MoonbeamProjectionAdapter instance
        lambda_cls: Weight for domain classification loss
        lambda_rec: Weight for reconstruction loss
        temperature: Temperature for Gumbel-Softmax sampling
        device: Device for computation
        lambda_amadeus: Weight for Amadeus Note Decoder Loss
        lambda_dispersive: Weight for Discriminability Enhancement Loss (dispersive loss)
    
    Returns:
        g_loss: Total generator loss
        loss_dict: Dictionary of individual losses
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
    
    # Convert Generator logits to embeddings via Moonbeam converter if provided
    if moonbeam_converter is not None and moonbeam_adapter is not None:
        moonbeam_device = next(moonbeam_converter.parameters()).device
        fake_logits_moonbeam = {k: v.to(moonbeam_device) for k, v in fake_logits.items()}
        moonbeam_out = moonbeam_converter(fake_logits_moonbeam)
        moonbeam_combined = moonbeam_out["combined_embedding"]  # [B, T, 6*hidden_size]
        fake_embeddings = moonbeam_adapter(moonbeam_combined)
    else:
        fake_embeddings = logits_to_embeddings_via_projection(fake_logits, projection_layer, vocab_size_list, hidden_size)  # [B, T, hidden_size]

    if fake_embeddings.device != device:
        fake_embeddings = fake_embeddings.to(device)
    
    # Use embeddings directly as Discriminator input
    fake_cls_output = D(inputs_embeds=fake_embeddings, Is_real=True)
    
    # Calculation fake loss
    g_loss_fake = - torch.mean(fake_cls_output.real_fake_logits)
    
    fake_cls_logits = fake_cls_output.logits if hasattr(fake_cls_output, 'logits') else fake_cls_output["logits"]
    g_loss_cls = F.binary_cross_entropy_with_logits(
        fake_cls_logits,
        target_labels.float()
    ).to(device)
    
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
        reconst_logit = reconst_logits[feature_name]  # [B, T, vocab_size]
        target_token = real_scores[:, :, feature_idx]  # [B, T]
        
        g_loss_rec += F.cross_entropy(
            reconst_logit.reshape(-1, reconst_logit.size(-1)),
            target_token.reshape(-1),
            ignore_index=-1
        ).to(device)
    
    g_loss_rec = g_loss_rec / len(AMADEUS_FIELDS)
    
    # ========== Amadeus Model Losses ==========
    g_loss_amadeus = torch.tensor(0.0, device=device)
    g_loss_dispersive = torch.tensor(0.0, device=device)
    
    amadeus_loss_fn = setup_amadeus_losses()
    
    if amadeus_loss_fn is not None:
        # ========== Note Decoder Loss ==========
        # Use fake_logits as predictions and real_scores as targets
        # Create mask for valid tokens (non-padding)
        valid_mask = torch.ones((B, T), dtype=torch.bool, device=device)
        
        g_loss_amadeus, amadeus_log_dict = amadeus_loss_fn(
            logits_dict=fake_logits,  # Dict of {feature: [B, T, vocab_size]}
            shifted_tgt=real_scores,  # [B, T, 8] target tokens
            mask=valid_mask,  # [B, T] valid mask
            valid=False  # No per-feature logging needed
        )
        
        # ========== Music Latent Space Discriminability Enhancement Loss ==========
        # Promote diversity in the latent space representation
        if fake_input_dict is not None and 'hidden_vec' in fake_input_dict:
            hidden_vec = fake_input_dict['hidden_vec']  # [B, T, hidden_dim]
            # Average over sequence dimension to get batch representation
            feat = hidden_vec.mean(dim=1)  # [B, hidden_dim]
            
            # Import dispersive_loss if not already available
            g_loss_dispersive = dispersive_loss(feat, tau=0.5)  # scalar
    
    # ========== Total Generator loss ==========
    # Weighted combination of all losses:
    # 1. Adversarial loss (fool discriminator)
    # 2. Domain classification loss (match target domain)
    # 3. Cycle consistency reconstruction loss
    # 4. Amadeus Note Decoder loss (generate realistic sequences)
    # 5. Dispersive loss (promote latent space diversity)
    g_loss = (
        g_loss_fake 
        + lambda_cls * g_loss_cls 
        + lambda_rec * g_loss_rec
        + lambda_amadeus * g_loss_amadeus
        #+ lambda_dispersive * g_loss_dispersive
    )
    
    loss_dict = {
        'G/loss_fake': g_loss_fake.item(),
        'G/loss_cls': g_loss_cls.item() if isinstance(g_loss_cls, torch.Tensor) else g_loss_cls,
        'G/loss_rec': g_loss_rec.item(),
        'G/loss_amadeus': g_loss_amadeus.item(),
        #'G/loss_dispersive': g_loss_dispersive.item(),
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


def check_gradient_flow_for_moonbeam(generator, discriminator, embeddings):
    """Verify gradients propagate through Moonbeam path."""
    loss = discriminator(inputs_embeds=embeddings, Is_real=True).real_fake_logits.mean()
    loss.backward(retain_graph=True)
    for name, param in generator.named_parameters():
        if param.grad is None:
            return False
    return True


def check_numerical_stability(amadeus_logits: Dict[str, torch.Tensor], embeddings: torch.Tensor) -> bool:
    """Detect NaN/Inf in logits and embeddings."""
    for feat, logit in amadeus_logits.items():
        if torch.isnan(logit).any() or torch.isinf(logit).any():
            return False
    if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
        return False
    return True


def test_discriminator_compatibility(discriminator, embedding_shape: Tuple[int, int, int]) -> bool:
    """Check if discriminator accepts the given embedding shape."""
    B, T, hidden_size = embedding_shape
    dummy = torch.randn(B, T, hidden_size, device=next(discriminator.parameters()).device)
    try:
        _ = discriminator(inputs_embeds=dummy, Is_real=True)
        return True
    except Exception:
        return False


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


def setup_amadeus_losses(encoding_scheme='nb', feature_list=None, focal_alpha=1.0, focal_gamma=2.0):
    """
    Setup Amadeus loss functions for StarGAN training
    
    Args:
        encoding_scheme: 'cp' (Compound Token) or 'nb' (Narrow-Band)
        feature_list: List of features (default: ["type", "beat", "chord", "tempo", "instrument", "pitch", "duration", "velocity"])
        focal_alpha: Focal loss alpha parameter
        focal_gamma: Focal loss gamma parameter
    
    Returns:
        amadeus_loss_fn: Instantiated loss function
    
    Example:
        from train_utils import NLLLoss4CompoundToken, DiffusionLoss4CompoundToken
        
        # For regular decoders
        amadeus_loss = NLLLoss4CompoundToken(
            feature_list=["type", "beat", "chord", "tempo", "instrument", "pitch", "duration", "velocity"],
            focal_alpha=1.0,
            focal_gamma=2.0
        )
        
        # For Diffusion decoders
        amadeus_loss = DiffusionLoss4CompoundToken(
            feature_list=["type", "beat", "chord", "tempo", "instrument", "pitch", "duration", "velocity"],
            focal_alpha=1.0,
            focal_gamma=2.0
        )
    """
    if feature_list is None:
        feature_list = ["type", "beat", "chord", "tempo", "instrument", "pitch", "duration", "velocity"]
    
    amadeus_loss_fn = NLLLoss4CompoundToken(
        feature_list=feature_list,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma
    )
    
    return amadeus_loss_fn


AMAEDEUS_FIELDS = ["type", "beat", "chord", "tempo", "instrument", "pitch", "duration", "velocity"]
def _build_lookup_table(field_dict: dict[str, str]) -> np.ndarray:
    max_idx = max(int(k) for k in field_dict.keys())
    table = ["" for _ in range(max_idx + 1)]
    for k, v in field_dict.items():
        table[int(k)] = v
    return np.array(table, dtype=object)

def amadeus_to_vocab(amadeus_tokens: torch.Tensor, vocab_path: str) -> np.ndarray:
    """Amadeusのトークン列を語彙に変換し、amadeus_to_moonbeam利用可能な形式に"""
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    tokens_np = amadeus_tokens.detach().cpu().numpy().astype(np.int64)
    decoded = np.empty(tokens_np.shape, dtype=object)

    for axis, field in enumerate(AMAEDEUS_FIELDS):
        lookup = _build_lookup_table(vocab[field])
        decoded[:, :, axis] = lookup[tokens_np[:, :, axis]]

    return decoded


def vocab_to_moonbeam(amadeus_vocabs, time_resolution=10, default_tempo=120, in_beat_resolution=4):
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
    
    num_notes = amadeus_np.shape[1]
    moonbeam_np = np.zeros((num_notes, 6), dtype=np.int32)
    amadeus_np = amadeus_np[0]
    
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

def vocab_to_moonbeam2(amadeus_vocabs, time_resolution=10, default_tempo=120, in_beat_resolution=4):
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
    
    moonbeam_array = []
    batch_size = amadeus_np.shape[0]
    for b in range(batch_size):
        num_notes = amadeus_np[b].shape[0]
        moonbeam_np = np.zeros((num_notes*6), dtype=np.int32)
    
        # State tracking
        current_bar = -1
        current_tempo = default_tempo
        current_time_signature = (4, 4)  # (numerator, denominator)
        
        # First note
        for k in range(6):
            moonbeam_np[k] = 0
        
        for i in range(1, num_notes):
            type_token = amadeus_np[b, i, 0]
            beat_token = amadeus_np[b, i, 1]
            chord_token = amadeus_np[b, i, 2]
            tempo_token = amadeus_np[b, i, 3]
            instrument_token = amadeus_np[b, i, 4]
            pitch_token = amadeus_np[b, i, 5]
            duration_token = amadeus_np[b, i, 6]
            velocity_token = amadeus_np[b, i, 7]
            
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
            moonbeam_np[6*i+0] = onset_in_10ms
            moonbeam_np[6*i+1] = duration_in_10ms
            moonbeam_np[6*i+2] = octave
            moonbeam_np[6*i+3] = pitch_class
            moonbeam_np[6*i+4] = instrument
            moonbeam_np[6*i+5] = velocity
        
        moonbeam_array.append(moonbeam_np)
    
    # Convert back to torch if input was torch
    if use_torch:
        moonbeam_tokens = torch.from_numpy(moonbeam_array).long()
    else:
        moonbeam_tokens = moonbeam_array
    
    return moonbeam_tokens