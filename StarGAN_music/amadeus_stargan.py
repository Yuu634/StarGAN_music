"""
AmadeusForStarGAN: End-to-End微分可能なGenerator実装
Gumbel-Softmax samplingとSoft embeddings生成をサポート
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

AMADEUS_FIELDS = ["type", "beat", "chord", "tempo", "instrument", "pitch", "duration", "velocity"]


class AmadeusForStarGAN(nn.Module):
    """
    End-to-End微分可能なAmadeus Generator
    
    Features:
    - Gumbel-Softmax sampling for differentiable discrete sampling
    - Soft embeddings generation for gradient flow to Discriminator
    - Domain conditioning via projection layer
    """
    
    def __init__(self, amadeus_model, vocab, hidden_dim=512):
        """
        Args:
            amadeus_model: Pre-trained AmadeusModel instance
            vocab: Vocabulary object for token information
            hidden_dim: Hidden dimension size (default: 512)
        """
        super().__init__()
        self.amadeus = amadeus_model.decoder.net  # AmadeusModelWrapper
        self.vocab = vocab
        self.dim = hidden_dim
        
        # Domain label → context embedding
        self.domain_proj = nn.Linear(108, self.dim)
        
        # Soft embedding layers: vocab_size → dim for each feature
        self.soft_embedders = nn.ModuleDict()
        for feature_name in AMADEUS_FIELDS:
            vocab_size = self._get_feature_vocab_size(feature_name)
            self.soft_embedders[feature_name] = nn.Linear(vocab_size, self.dim)
    
    def _get_feature_vocab_size(self, feature_name):
        """Get vocabulary size for each feature"""
        try:
            if hasattr(self.vocab, 'get_feature_vocab_size'):
                return self.vocab.get_feature_vocab_size(feature_name)
            elif hasattr(self.vocab, f'{feature_name}_vocab_size'):
                return getattr(self.vocab, f'{feature_name}_vocab_size')
            else:
                # Default fallback values
                default_sizes = {
                    'type': 128, 'beat': 128, 'chord': 128, 'tempo': 128,
                    'instrument': 129, 'pitch': 128, 'duration': 128, 'velocity': 128
                }
                return default_sizes.get(feature_name, 128)
        except:
            return 128  # Safe fallback
    
    def encode_domain(self, domain_labels):
        """
        Convert domain labels to context embeddings
        
        Args:
            domain_labels: [B, 108] multi-hot domain labels
        Returns:
            context: [B, dim] context embeddings
        """
        return self.domain_proj(domain_labels.float())
    
    def forward(self, input_seq, target_domain, temperature=1.0, hard=False):
        """
        Forward pass with Gumbel-Softmax sampling
        
        Args:
            input_seq: [B, T, 8] Input score (discrete Amadeus tokens)
            target_domain: [B, 108] Target domain labels
            temperature: Gumbel-Softmax temperature (default: 1.0)
            hard: If True, use Straight-Through Estimator
        
        Returns:
            logits_dict: Dict of {feature: [B, T, vocab_size]}
            soft_embeddings: [B, T, dim] Differentiable embeddings for Discriminator
        """
        context = self.encode_domain(target_domain)  # [B, dim]
        
        # Amadeus encoding
        embedding = self.amadeus.input_embedder(input_seq)  # [B, T, dim]
        embedding = embedding + self.amadeus.pos_enc(input_seq)
        embedding = self.amadeus.emb_dropout(embedding)
        
        # Main decoder with context conditioning
        hidden_vec, _ = self.amadeus.main_decoder(
            embedding,
            train=True,
            context=context.unsqueeze(1)  # [B, 1, dim]
        )
        hidden_vec = self.amadeus.main_norm(hidden_vec)
        
        # Sub-decoder: Generate logits for each feature
        input_dict = {
            'hidden_vec': hidden_vec,
            'input_seq': input_seq,
            'target': input_seq,
            'bos_token_hidden': None
        }
        logits_dict = self.amadeus.sub_decoder(input_dict)
        # logits_dict = {feature: [B, T, vocab_size]}
        
        # Gumbel-Softmax sampling for differentiable discrete sampling
        soft_embeddings_list = []
        for feature_name in AMADEUS_FIELDS:
            logits = logits_dict[feature_name]  # [B, T, vocab_size]
            
            # Gumbel-Softmax: discrete → continuous
            # hard=True: forward pass uses one-hot, backward pass uses soft (Straight-Through)
            soft_probs = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)
            # [B, T, vocab_size]
            
            # Soft embedding: weighted sum of embeddings
            soft_emb = self.soft_embedders[feature_name](soft_probs)
            # [B, T, vocab_size] @ [vocab_size, dim] → [B, T, dim]
            
            soft_embeddings_list.append(soft_emb)
        
        # Aggregate embeddings from all features
        soft_embeddings = sum(soft_embeddings_list)  # [B, T, dim]
        
        return logits_dict, soft_embeddings
    
    def get_hard_tokens(self, logits_dict):
        """
        Extract discrete tokens from logits (for evaluation/saving)
        
        Args:
            logits_dict: Dict of {feature: [B, T, vocab_size]}
        Returns:
            tokens: [B, T, 8] Discrete tokens
        """
        tokens = []
        for feature_name in AMADEUS_FIELDS:
            logits = logits_dict[feature_name]
            token = torch.argmax(logits, dim=-1)  # [B, T]
            tokens.append(token)
        return torch.stack(tokens, dim=-1)  # [B, T, 8]
    
    @torch.no_grad()
    def generate(self, input_seq, target_domain, max_len=512, temperature=1.0):
        """
        Autoregressive generation for inference
        
        Args:
            input_seq: [B, T, 8] or [T, 8] Input condition
            target_domain: [B, 108] Target domain labels
            max_len: Maximum generation length
            temperature: Sampling temperature
        
        Returns:
            generated: [B, T_gen, 8] Generated tokens
        """
        context = self.encode_domain(target_domain)
        
        # Use Amadeus's original generate method
        if input_seq.dim() == 2:
            input_seq = input_seq.unsqueeze(0)
        
        condition = input_seq[0] if input_seq.size(0) > 0 else None
        
        generated = self.amadeus.generate(
            manual_seed=42,
            max_seq_len=max_len,
            condition=condition,
            context=context.unsqueeze(1),
            sampling_method='top_p',
            threshold=0.9,
            temperature=temperature
        )
        
        return generated


class SoftEmbeddingProjector(nn.Module):
    """
    Project soft embeddings from Amadeus to Moonbeam dimension
    Used in Discriminator for processing Generator outputs
    """
    
    def __init__(self, amadeus_dim, moonbeam_dim):
        """
        Args:
            amadeus_dim: Amadeus hidden dimension
            moonbeam_dim: Moonbeam (Llama) hidden dimension
        """
        super().__init__()
        self.projection = nn.Linear(amadeus_dim, moonbeam_dim)
        self.layer_norm = nn.LayerNorm(moonbeam_dim)
    
    def forward(self, soft_embeddings):
        """
        Args:
            soft_embeddings: [B, T, amadeus_dim]
        Returns:
            projected: [B, T, moonbeam_dim]
        """
        projected = self.projection(soft_embeddings)
        projected = self.layer_norm(projected)
        return projected
