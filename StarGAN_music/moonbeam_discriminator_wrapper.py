"""
Moonbeam Discriminator Wrapper for StarGAN
Wraps LlamaForSequenceClassification from Moonbeam for dual classification
"""

import sys
import os
import torch
import torch.nn as nn

# Add Moonbeam path to system path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../Moonbeam-MIDI-Foundation-Model/src'))

from transformers import LlamaConfig, LlamaForSequenceClassification
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel, LlamaModel


class MoonbeamDiscriminatorForStarGAN(LlamaPreTrainedModel):
    """
    Moonbeam Discriminator for End-to-End differentiable StarGAN.
    
    Dual classification:
    1. Real/Fake classification (2 classes)
    2. Domain classification (108 classes for MidiCaps)
    
    Dual input support:
    - Discrete tokens (for Real samples)
    - Soft embeddings (for Fake samples from Generator)
    """
    
    def __init__(self, config: LlamaConfig, num_domains: int = 108):
        """
        Args:
            config: LlamaConfig from Moonbeam
            num_domains: Number of domain classes (default: 108 for MidiCaps)
        """
        super().__init__(config)
        self.num_domains = num_domains
        self.config = config
        
        # Llama backbone model
        self.model = LlamaModel(config)
        
        # Dual classification heads
        self.score_classifier = nn.Linear(config.hidden_size, 2, bias=False)  # Real/Fake
        self.domain_classifier = nn.Linear(config.hidden_size, num_domains, bias=False)  # Domain
        
        # Soft embedding projection layer (for Fake samples)
        # This converts soft embeddings from Generator to Moonbeam's hidden dimension
        self.soft_projection = nn.Linear(
            self._get_amadeus_embed_dim(),  # Amadeus total embedding dim
            config.hidden_size,
            bias=False
        )
        
        # Initialize weights
        self.post_init()
    
    def _get_amadeus_embed_dim(self):
        """
        Calculate total Amadeus embedding dimension
        Amadeus has 8 features, each with different vocab sizes
        Typical dims: type(64) + beat(64) + chord(64) + tempo(64) + instrument(128) + pitch(128) + duration(64) + velocity(64)
        """
        # This is an approximation - adjust based on actual Amadeus config
        return 640  # Sum of all feature embedding dimensions
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    
    def embed_discrete_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed discrete Moonbeam tokens (for Real samples)
        
        Args:
            input_ids: Discrete tokens [B, T, 6]
            
        Returns:
            embeddings: [B, T, hidden_size]
        """
        # Moonbeam uses custom embedding for each feature
        # This should use the actual Moonbeam embedding logic
        # For now, we use the standard Llama embedding
        
        # Extract features from input_ids [B, T, 6]
        # Moonbeam features: [onset, duration, octave, pitch_class, instrument, velocity]
        
        # NOTE: This is a simplified version. 
        # Actual Moonbeam uses FME (Fourier Music Embedding) for most features
        # You may need to implement the actual Moonbeam embedding logic here
        
        embeddings = self.model.embed_tokens(input_ids)  # Simplified
        return embeddings
    
    def embed_soft_embeddings(self, soft_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project soft embeddings from Generator (for Fake samples)
        
        Args:
            soft_embeddings: Soft embeddings from Amadeus Generator [B, T, amadeus_dim]
            
        Returns:
            projected: [B, T, hidden_size]
        """
        return self.soft_projection(soft_embeddings)
    
    def forward(
        self,
        input_ids: torch.Tensor = None,  # For Real samples [B, T, 6]
        soft_embeddings: torch.Tensor = None,  # For Fake samples [B, T, amadeus_dim]
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        return_dict: bool = True
    ):
        """
        Forward pass with dual input support
        
        Args:
            input_ids: Discrete tokens for Real samples [B, T, 6] (optional)
            soft_embeddings: Soft embeddings for Fake samples [B, T, amadeus_dim] (optional)
            attention_mask: Attention mask [B, T]
            position_ids: Position IDs [B, T]
            return_dict: Whether to return dict
            
        Returns:
            real_fake_logits: Real/Fake classification logits [B, T, 2]
            domain_logits: Domain classification logits [B, T, num_domains]
        """
        # Choose input type
        if input_ids is not None:
            # Real samples: use discrete tokens
            inputs_embeds = self.embed_discrete_tokens(input_ids)
        elif soft_embeddings is not None:
            # Fake samples: use soft embeddings
            inputs_embeds = self.embed_soft_embeddings(soft_embeddings)
        else:
            raise ValueError("Either input_ids or soft_embeddings must be provided")
        
        # Forward through Llama backbone
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True
        )
        
        hidden_states = outputs.last_hidden_state  # [B, T, hidden_size]
        
        # Dual classification
        real_fake_logits = self.score_classifier(hidden_states)  # [B, T, 2]
        domain_logits = self.domain_classifier(hidden_states)  # [B, T, num_domains]
        
        return real_fake_logits, domain_logits


def load_moonbeam_discriminator(
    config_path: str,
    checkpoint_path: str,
    num_domains: int = 108,
    device: str = 'cuda'
) -> MoonbeamDiscriminatorForStarGAN:
    """
    Load pre-trained Moonbeam model and wrap for StarGAN Discriminator
    
    Args:
        config_path: Path to Moonbeam config JSON file
        checkpoint_path: Path to pre-trained checkpoint
        num_domains: Number of domain classes
        device: Device to load model on
        
    Returns:
        MoonbeamDiscriminatorForStarGAN instance
    """
    import json
    
    # Load config
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create LlamaConfig
    config = LlamaConfig(**config_dict)
    config.use_cache = False
    
    # Create Discriminator
    discriminator = MoonbeamDiscriminatorForStarGAN(config, num_domains=num_domains)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if exists (from DDP/FSDP)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    # Load state dict (some keys may not match due to new classification heads)
    missing_keys, unexpected_keys = discriminator.load_state_dict(new_state_dict, strict=False)
    
    if missing_keys:
        print(f"Missing keys (expected for new classification heads): {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    
    print(f"Loaded Moonbeam checkpoint from {checkpoint_path}")
    
    discriminator.to(device)
    discriminator.eval()
    
    return discriminator


# Moonbeam-specific embedding utilities
def create_moonbeam_embeddings(config: LlamaConfig):
    """
    Create Moonbeam-specific embeddings (FME + WE)
    
    This function should implement Fourier Music Embedding (FME) and Word Embedding (WE)
    as used in the original Moonbeam model.
    
    Currently simplified - needs full implementation based on Moonbeam paper/code.
    """
    # TODO: Implement FME and WE according to Moonbeam specifications
    pass
