"""
LlamaForSequenceDoubleClassification: Discriminator for StarGAN
Supports both discrete tokens (Real) and soft embeddings (Fake)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaPreTrainedModel, LlamaModel, LlamaConfig
from typing import Optional, Tuple


class LlamaForSequenceDoubleClassification(LlamaPreTrainedModel):
    """
    Llama-based Discriminator with dual classification heads
    
    Features:
    - Dual input modes: discrete tokens (Real) and soft embeddings (Fake)
    - Real/Fake binary classification
    - Domain multi-label classification (108 dimensions)
    - End-to-end differentiable for gradient flow from Generator
    """
    
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.num_labels = getattr(config, 'num_labels', 108)
        self.config = config
        
        # Llama backbone
        self.model = LlamaModel(config)
        
        # Discrete token embeddings (for Real scores - Moonbeam format)
        # Moonbeam: (onset, duration, octave, pitch_class, instrument, velocity)
        self.embed_onset = nn.Embedding(
            getattr(config, 'onset_vocab_size', 1024), 
            config.hidden_size
        )
        self.embed_duration = nn.Embedding(
            getattr(config, 'dur_vocab_size', 1024),
            config.hidden_size
        )
        self.embed_octave = nn.Embedding(
            getattr(config, 'octave_vocab_size', 11),
            config.hidden_size
        )
        self.embed_pitch_class = nn.Embedding(
            getattr(config, 'pitch_class_vocab_size', 12),
            config.hidden_size
        )
        self.embed_instrument = nn.Embedding(
            getattr(config, 'instrument_vocab_size', 129),
            config.hidden_size
        )
        self.embed_velocity = nn.Embedding(
            getattr(config, 'velocity_vocab_size', 128),
            config.hidden_size
        )
        
        # Soft embedding projection (for Fake scores from Generator)
        # Amadeus dim → Moonbeam hidden_size
        amadeus_dim = getattr(config, 'amadeus_dim', 512)
        self.soft_proj = nn.Linear(amadeus_dim, config.hidden_size)
        self.soft_layer_norm = nn.LayerNorm(config.hidden_size)
        
        # Classification heads
        self.score_classifier = nn.Linear(config.hidden_size, 2)  # Real/Fake
        self.domain_classifier = nn.Linear(config.hidden_size, self.num_labels)  # 108 domains
        
        # Dropout for regularization
        self.dropout = nn.Dropout(getattr(config, 'classifier_dropout', 0.1))
        
        # Initialize weights
        self.post_init()
    
    def embed_discrete_tokens(self, input_ids):
        """
        Embed discrete Moonbeam tokens (for Real scores)
        
        Args:
            input_ids: [B, T, 6] Moonbeam format tokens
                       (onset, duration, octave, pitch_class, instrument, velocity)
        Returns:
            embeddings: [B, T, hidden_size]
        """
        onset_emb = self.embed_onset(input_ids[:, :, 0])           # [B, T, H]
        duration_emb = self.embed_duration(input_ids[:, :, 1])     # [B, T, H]
        octave_emb = self.embed_octave(input_ids[:, :, 2])         # [B, T, H]
        pitch_class_emb = self.embed_pitch_class(input_ids[:, :, 3])  # [B, T, H]
        instrument_emb = self.embed_instrument(input_ids[:, :, 4])  # [B, T, H]
        velocity_emb = self.embed_velocity(input_ids[:, :, 5])     # [B, T, H]
        
        # Aggregate embeddings (sum)
        embeddings = (
            onset_emb + duration_emb + octave_emb + 
            pitch_class_emb + instrument_emb + velocity_emb
        )
        
        return embeddings
    
    def embed_soft_embeddings(self, soft_embeddings):
        """
        Project soft embeddings from Generator (for Fake scores)
        
        Args:
            soft_embeddings: [B, T, amadeus_dim] Soft embeddings from AmadeusForStarGAN
        Returns:
            embeddings: [B, T, hidden_size]
        """
        embeddings = self.soft_proj(soft_embeddings)
        embeddings = self.soft_layer_norm(embeddings)
        return embeddings
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        soft_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass supporting both discrete and soft inputs
        
        Args:
            input_ids: [B, T, 6] Discrete Moonbeam tokens (for Real scores)
            soft_embeddings: [B, T, amadeus_dim] Soft embeddings (for Fake scores)
            attention_mask: [B, T] Attention mask
            position_ids: [B, T] Position IDs
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return dict
        
        Returns:
            real_fake_logits: [B, T, 2] Real/Fake classification logits
            domain_logits: [B, T, 108] Domain classification logits
        """
        # Input embedding (discrete or soft)
        if input_ids is not None:
            # Real score: discrete tokens
            inputs_embeds = self.embed_discrete_tokens(input_ids)
        elif soft_embeddings is not None:
            # Fake score: soft embeddings from Generator
            inputs_embeds = self.embed_soft_embeddings(soft_embeddings)
        else:
            raise ValueError("Either input_ids or soft_embeddings must be provided")
        
        # Llama backbone forward
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs.last_hidden_state  # [B, T, hidden_size]
        hidden_states = self.dropout(hidden_states)
        
        # Dual classification heads
        real_fake_logits = self.score_classifier(hidden_states)  # [B, T, 2]
        domain_logits = self.domain_classifier(hidden_states)     # [B, T, 108]
        
        return real_fake_logits, domain_logits
    
    def get_input_embeddings(self):
        """Return input embeddings (for compatibility)"""
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        """Set input embeddings (for compatibility)"""
        self.model.embed_tokens = value


def create_discriminator_config(
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16,
    intermediate_size=4096,
    max_position_embeddings=2048,
    num_labels=108,
    onset_vocab_size=1024,
    dur_vocab_size=1024,
    octave_vocab_size=11,
    pitch_class_vocab_size=12,
    instrument_vocab_size=129,
    velocity_vocab_size=128,
    amadeus_dim=512,
    **kwargs
):
    """
    Create LlamaConfig for Discriminator
    
    Args:
        hidden_size: Hidden dimension
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        intermediate_size: FFN intermediate size
        max_position_embeddings: Maximum sequence length
        num_labels: Number of domain labels (default: 108)
        onset_vocab_size: Onset vocabulary size
        dur_vocab_size: Duration vocabulary size
        octave_vocab_size: Octave vocabulary size (default: 11)
        pitch_class_vocab_size: Pitch class vocabulary size (default: 12)
        instrument_vocab_size: Instrument vocabulary size (default: 129)
        velocity_vocab_size: Velocity vocabulary size (default: 128)
        amadeus_dim: Amadeus hidden dimension for soft projection
        **kwargs: Additional config parameters
    
    Returns:
        config: LlamaConfig instance
    """
    config = LlamaConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        **kwargs
    )
    
    # Add custom attributes
    config.num_labels = num_labels
    config.onset_vocab_size = onset_vocab_size
    config.dur_vocab_size = dur_vocab_size
    config.octave_vocab_size = octave_vocab_size
    config.pitch_class_vocab_size = pitch_class_vocab_size
    config.instrument_vocab_size = instrument_vocab_size
    config.velocity_vocab_size = velocity_vocab_size
    config.amadeus_dim = amadeus_dim
    config.use_cache = False  # Disable KV cache for training
    
    return config
