"""
Amadeus Generator Wrapper for StarGAN
Wraps the Amadeus model from Amadeus/Amadeus/model_zoo.py for StarGAN usage
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add Amadeus path to system path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../Amadeus'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../Amadeus/Amadeus'))

from Amadeus.model_zoo import AmadeusModelWrapper, AmadeusModelAutoregressiveWrapper
from data_representation.vocab_utils import LangTokenVocab


class AmadeusGeneratorForStarGAN(nn.Module):
    """
    Amadeus Generator wrapper for End-to-End differentiable StarGAN.
    Uses the actual Amadeus model from Amadeus/Amadeus/model_zoo.py
    Adds Gumbel-Softmax sampling for differentiable generation
    """
    
    def __init__(
        self, 
        amadeus_model: AmadeusModelWrapper,
        vocab: LangTokenVocab,
        domain_dim: int = 108,
        temperature: float = 0.5,
        dim: int = 768  # Model dimension
    ):
        """
        Args:
            amadeus_model: Pre-trained AmadeusModelWrapper instance
            vocab: Vocabulary object
            domain_dim: Number of domain labels (default: 108 for MidiCaps)
            temperature: Temperature for Gumbel-Softmax (default: 0.5)
            dim: Model hidden dimension (default: 768)
        """
        super().__init__()
        
        self.amadeus_model = amadeus_model
        self.vocab = vocab
        self.domain_dim = domain_dim
        self.temperature = temperature
        
        # Domain conditioning layer
        self.domain_embedding = nn.Linear(domain_dim, dim)
        
        # Soft embedding projection layers for each feature
        self.feature_list = vocab.feature_list
        self.soft_embedders = nn.ModuleDict()
        
        for idx, feature_name in enumerate(self.feature_list):
            vocab_size = len(vocab.event2idx[feature_name]) if isinstance(vocab.event2idx, dict) else len(vocab.event2idx)
            # Get embedding dimension from amadeus_model's input_embedder
            # For SummationEmbedder, all features have the same dim_model
            embed_dim = dim  # Use the model dimension
            # Use 'feat_' prefix to avoid reserved keywords like 'type'
            safe_key = f'feat_{feature_name}'
            self.soft_embedders[safe_key] = nn.Linear(vocab_size, embed_dim, bias=False)
            
            # Initialize with embedding weights for better convergence
            with torch.no_grad():
                # Get the corresponding embedding layer from amadeus_model
                if hasattr(amadeus_model.input_embedder, 'layers') and idx < len(amadeus_model.input_embedder.layers):
                    emb_weight = amadeus_model.input_embedder.layers[idx].weight
                    # Linear layer weight is [out_features, in_features]
                    # Embedding layer weight is [num_embeddings, embedding_dim]
                    # We need to transpose for Linear layer
                    if emb_weight.shape[0] == vocab_size and emb_weight.shape[1] == embed_dim:
                        # Transpose: [vocab_size, embed_dim] -> [embed_dim, vocab_size]
                        self.soft_embedders[safe_key].weight.copy_(emb_weight.T)
    
    def forward(
        self, 
        input_seq: torch.Tensor,  # [B, T, 8] for Amadeus
        target_domain: torch.Tensor,  # [B, 108]
        temperature: float = None,
        hard: bool = True
    ):
        """
        Forward pass with Gumbel-Softmax sampling
        
        Args:
            input_seq: Input token sequence [B, T, num_features]
            target_domain: Target domain labels [B, domain_dim]
            temperature: Gumbel-Softmax temperature (None = use default)
            hard: If True, use Straight-Through Estimator
            
        Returns:
            logits_dict: Dictionary of logits for each feature {feature_name: [B, T, vocab_size]}
            soft_embeddings: Soft embeddings for Discriminator [B, T, total_embed_dim]
        """
        B, T, num_features = input_seq.shape
        temp = temperature if temperature is not None else self.temperature
        
        # Prepare domain conditioning as context
        domain_emb = self.domain_embedding(target_domain)  # [B, dim]
        domain_emb = domain_emb.unsqueeze(1)  # [B, 1, dim] for broadcasting
        
        # Create dummy target (same as input for translation task)
        target = input_seq.clone()
        
        # Forward through Amadeus model
        logits_dict, _ = self.amadeus_model(input_seq, target, context=domain_emb)
        
        # Gumbel-Softmax sampling for each feature
        soft_embeddings_list = []
        sampled_tokens_dict = {}
        
        for feature_name in self.feature_list:
            logits = logits_dict[feature_name]  # [B, T, vocab_size]
            
            # Apply Gumbel-Softmax
            soft_probs = F.gumbel_softmax(logits, tau=temp, hard=hard, dim=-1)  # [B, T, vocab_size]
            sampled_tokens_dict[feature_name] = soft_probs
            
            # Project to soft embeddings (use safe_key)
            safe_key = f'feat_{feature_name}'
            soft_emb = self.soft_embedders[safe_key](soft_probs)  # [B, T, embed_dim]
            soft_embeddings_list.append(soft_emb)
        
        # Concatenate all feature embeddings
        soft_embeddings = torch.cat(soft_embeddings_list, dim=-1)  # [B, T, total_embed_dim]
        
        return logits_dict, soft_embeddings, sampled_tokens_dict
    
    def get_discrete_tokens(self, logits_dict):
        """
        Convert logits to discrete tokens (for evaluation/inference)
        
        Args:
            logits_dict: Dictionary of logits {feature_name: [B, T, vocab_size]}
            
        Returns:
            tokens: Discrete tokens [B, T, num_features]
        """
        tokens_list = []
        for feature_name in self.feature_list:
            logits = logits_dict[feature_name]
            tokens = torch.argmax(logits, dim=-1)  # [B, T]
            tokens_list.append(tokens.unsqueeze(-1))
        
        tokens = torch.cat(tokens_list, dim=-1)  # [B, T, num_features]
        return tokens


def load_amadeus_generator(
    config_path: str,
    checkpoint_path: str = None,
    vocab_path: str = None,
    device: str = 'cuda'
) -> AmadeusGeneratorForStarGAN:
    """
    Load pre-trained Amadeus model and wrap for StarGAN
    
    Args:
        config_path: Path to Amadeus config YAML file
        checkpoint_path: Path to pre-trained checkpoint (optional)
        vocab_path: Path to vocabulary file (required)
        device: Device to load model on
        
    Returns:
        AmadeusGeneratorForStarGAN instance
    """
    import yaml
    
    # Load config
    with open(config_path, 'r') as f:
        config_raw = yaml.safe_load(f)
    
    # Extract nested config values
    nn_params = config_raw.get('nn_params', {}).get('value', {})
    train_params = config_raw.get('train_params', {}).get('value', {})
    data_params = config_raw.get('data_params', {}).get('value', {})
    
    # Extract model parameters (needed for vocab loading)
    encoding_scheme = nn_params.get('encoding_scheme', 'nb')
    num_features = nn_params.get('num_features', 8)
    
    # Load vocabulary
    if vocab_path is None:
        raise ValueError("vocab_path is required. Please specify --vocab_path argument.")
    
    vocab = LangTokenVocab(
        in_vocab_file_path=vocab_path,
        event_data=None,
        encoding_scheme=encoding_scheme,
        num_features=num_features
    )
    
    # Continue extracting other parameters
    input_length = train_params.get('input_length', 3072)
    
    # Main decoder params
    main_decoder_config = nn_params.get('main_decoder', {})
    dim = main_decoder_config.get('dim_model', 768)
    heads = main_decoder_config.get('num_head', 12)
    depth = main_decoder_config.get('num_layer', 16)
    
    # Sub decoder params
    sub_decoder_config = nn_params.get('sub_decoder', {})
    sub_decoder_depth = sub_decoder_config.get('num_layer', 1)
    sub_decoder_enricher_use = sub_decoder_config.get('feature_enricher_use', False)
    
    # Other params
    dropout = nn_params.get('model_dropout', 0.2)
    input_embedder_name = nn_params.get('input_embedder_name', 'SummationEmbedder')
    main_decoder_name = nn_params.get('main_decoder_name', 'XtransformerCrossAttendDecoder')
    sub_decoder_name = nn_params.get('sub_decoder_name', 'DiffusionDecoder')
    
    # Prediction order (default for nb encoding)
    first_pred_feature = data_params.get('first_pred_feature', 'pitch')
    prediction_order = [first_pred_feature] + [f for f in vocab.feature_list if f != first_pred_feature]
    
    # Create Amadeus model
    amadeus_model = AmadeusModelWrapper(
        vocab=vocab,
        input_length=input_length,
        prediction_order=prediction_order,
        input_embedder_name=input_embedder_name,
        main_decoder_name=main_decoder_name,
        sub_decoder_name=sub_decoder_name,
        sub_decoder_depth=sub_decoder_depth,
        sub_decoder_enricher_use=sub_decoder_enricher_use,
        dim=dim,
        heads=heads,
        depth=depth,
        dropout=dropout
    )
    
    # Load checkpoint if provided
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            amadeus_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            amadeus_model.load_state_dict(checkpoint)
        print(f"Loaded Amadeus checkpoint from {checkpoint_path}")
    
    # Wrap for StarGAN
    generator = AmadeusGeneratorForStarGAN(
        amadeus_model=amadeus_model,
        vocab=vocab,
        domain_dim=108,
        temperature=0.5,
        dim=dim  # Pass the dim parameter
    )
    
    generator.to(device)
    generator.eval()
    
    print(f"Amadeus Generator loaded successfully!")
    print(f"  Config: {config_path}")
    print(f"  Vocab: {vocab_path}")
    print(f"  Input length: {input_length}")
    print(f"  Dim: {dim}, Heads: {heads}, Depth: {depth}")
    print(f"  Encoding: {encoding_scheme}, Features: {num_features}")
    
    return generator


# Token conversion utilities
def amadeus_to_moonbeam_discrete(amadeus_tokens: torch.Tensor) -> torch.Tensor:
    """
    Convert Amadeus tokens [B, T, 8] to Moonbeam tokens [B, T, 6]
    
    Amadeus features: [type, beat, chord, tempo, instrument, pitch, duration, velocity]
    Moonbeam features: [onset, duration, octave, pitch_class, instrument, velocity]
    
    Args:
        amadeus_tokens: [B, T, 8]
        
    Returns:
        moonbeam_tokens: [B, T, 6]
    """
    B, T, _ = amadeus_tokens.shape
    
    # Extract features
    beat = amadeus_tokens[:, :, 1]
    instrument = amadeus_tokens[:, :, 4]
    pitch = amadeus_tokens[:, :, 5]
    duration = amadeus_tokens[:, :, 6]
    velocity = amadeus_tokens[:, :, 7]
    
    # Convert pitch to octave and pitch_class
    octave = pitch // 12
    pitch_class = pitch % 12
    
    # Use beat as onset (simplified conversion)
    onset = beat
    
    # Stack features
    moonbeam_tokens = torch.stack([
        onset,
        duration,
        octave,
        pitch_class,
        instrument,
        velocity
    ], dim=-1)  # [B, T, 6]
    
    return moonbeam_tokens
