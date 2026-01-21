#!/usr/bin/env python3
"""
Arrangement Score Data Generation Script
Generates arrangement score data using StarGAN-trained Amadeus Generator.

This script loads a pre-trained Amadeus model from a StarGAN checkpoint and
generates arrangement score sequences for specified target domains.

Usage:
    python generate.py \
        --model_path path/to/stargan_checkpoint.pt \
        --amadeus_config path/to/amadeus_config.yaml \
        --vocab_path path/to/vocab.pkl \
        --num_samples 10 \
        --sequence_length 2048 \
        --target_domains "domain1" "domain2" "domain3" \
        --input_score_path path/to/input_score.npy \
        --output_dir output \
        --temperature 1.15 \
        --threshold 0.99 \
        --sampling_method top_p
"""

import torch
import argparse
import os
import json
import numpy as np
from pathlib import Path
import random
import sys
from typing import Optional, Tuple, Dict, Any
from omegaconf import OmegaConf, DictConfig
from transformers import T5Tokenizer, T5EncoderModel

sys.path.append("../Amadeus")
from generate import generate_with_textANDscore_prompt, load_resources

#sys.path.append("../Amadeus")
from Amadeus import model_zoo
from Amadeus.symbolic_encoding import decoding_utils
from Amadeus.train_utils import adjust_prediction_order
from Amadeus.evaluation_utils import wandb_style_config_to_omega_config
from data_representation import vocab_utils
from Amadeus.symbolic_encoding.compile_utils import reverse_shift_and_pad_for_tensor


def get_argument_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate arrangement score data using StarGAN-trained Amadeus Generator'
    )
    
    model_name = "MidiCaps-GenreSmall-ParamsTuned-G_interval=10"
    ver_name = "stargan_epoch0_step56614.pt"
    
    # Model and checkpoint paths
    parser.add_argument(
        '--model_path',
        type=str,
        default=f"output/{model_name}/models/{ver_name}",
        help='Path to StarGAN checkpoint containing trained Amadeus Generator'
    )
    parser.add_argument(
        '--model_id',
        type=str,
        default="../Amadeus/models/Amadeus-S",
        help='Model identifier for Amadeus configuration and resources'
    )
    parser.add_argument(
        "-text_encoder_model",
        type=str,
        default='google/flan-t5-base',
        help="pretrained text encoder model",
    )
    
    # Generation parameters
    parser.add_argument(
        '--num_samples',
        type=int,
        default=10,
        help='Number of arrangement samples to generate (default: 10)'
    )
    parser.add_argument(
        '--sequence_length',
        type=int,
        default=1024,
        help='Length of generated sequences in tokens (default: 1024)'
    )
    parser.add_argument(
        '--target_domains',
        nargs='+',
        default=["classical", "rock"],
        help='List of target arrangement domains (e.g., "pop" "jazz" "classical")'
    )
    
    # Input score data
    parser.add_argument(
        '--input_score_path',
        type=str,
        default="generated_scores/dataset/0a0a2b0e4d3b7bf4c5383ba025c4683e.npz",
        help='Path to input score data (numpy format) or MIDI file. If specified, will use as input context'
    )
    parser.add_argument(
        '--input_score_length',
        type=int,
        default=1024,
        help='Maximum length of input score to use as context (default: 1024)'
    )
    
    # Output settings
    parser.add_argument(
        '--output_dir',
        type=str,
        default=f"generated_scores/{model_name}",
        help='Directory to save generated arrangements (default: generated_scores)'
    )
    parser.add_argument(
        '--save_format',
        type=str,
        choices=['midi', 'npy', 'both'],
        default='midi',
        help='Format to save generated sequences (default: midi)'
    )
    parser.add_argument(
        '--save_metadata',
        action='store_true',
        default=True,
        help='Save metadata JSON for each generation'
    )
    
    # Sampling parameters
    parser.add_argument(
        '--sampling_method',
        type=str,
        choices=['top_p', 'top_k'],
        default='top_p',
        help='Sampling method (default: top_p)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.99,
        help='Threshold for top_p or top_k sampling (default: 0.99)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.2,
        help='Temperature for sampling (default: 1.2)'
    )
    
    # Misc
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (default: cuda if available, else cpu)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def load_stargan_checkpoint(model_path: str, device: str) -> Dict[str, Any]:
    """
    Load StarGAN checkpoint containing trained Amadeus Generator.
    
    Args:
        model_path: Path to StarGAN checkpoint (.pt file)
        device: Device to load model on
        
    Returns:
        Dictionary containing model state and metadata
    """
    print(f"Loading StarGAN checkpoint from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract generator state dict
    if 'generator_state_dict' in checkpoint:
        generator_state = checkpoint['generator_state_dict']
        print(f"  Extracted generator_state_dict from checkpoint")
    elif 'G_state_dict' in checkpoint:
        generator_state = checkpoint['G_state_dict']
        print(f"  Extracted G_state_dict from checkpoint")
    else:
        # Try to use checkpoint directly if it only contains generator weights
        generator_state = checkpoint
        print(f"  Using checkpoint as generator state directly")
    
    return {
        'state_dict': generator_state,
        'full_checkpoint': checkpoint
    }


def prepare_amadeus_model(
    config: DictConfig,
    vocab: Any,
    device: str
) -> Tuple[Any, Any, Any]:
    """
    Prepare Amadeus model architecture and vocabulary.
    
    Args:
        config_path: Path to Amadeus configuration YAML
        vocab_path: Path to vocabulary file
        device: Device to load on
        
    Returns:
        Tuple of (model, vocab, config)
    """
    
    nn_params = config.nn_params
    encoding_scheme = nn_params.encoding_scheme
    num_features = nn_params.num_features
    
    # Create model
    print("Building Amadeus model architecture...")
    prediction_order = adjust_prediction_order(
        encoding_scheme,
        num_features,
        config.data_params.first_pred_feature,
        nn_params
    )
    
    model = getattr(model_zoo, nn_params.model_name)(
        vocab=vocab,
        input_length=config.train_params.input_length,
        prediction_order=prediction_order,
        input_embedder_name=nn_params.input_embedder_name,
        main_decoder_name=nn_params.main_decoder_name,
        sub_decoder_name=nn_params.sub_decoder_name,
        sub_decoder_depth=nn_params.sub_decoder.num_layer if hasattr(nn_params, 'sub_decoder') else 0,
        sub_decoder_enricher_use=nn_params.sub_decoder.feature_enricher_use
        if hasattr(nn_params, 'sub_decoder') and hasattr(nn_params.sub_decoder, 'feature_enricher_use')
        else False,
        dim=nn_params.main_decoder.dim_model,
        heads=nn_params.main_decoder.num_head,
        depth=nn_params.main_decoder.num_layer,
        dropout=nn_params.model_dropout,
        is_regressive=True
    )
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Encoding: {encoding_scheme}, Features: {num_features}")
    print(f"  Input length: {config.train_params.input_length}")
    
    model.to(device)
    model.eval()
    
    return model


def load_amadeus_generator(
    model_path: str,
    amadeus_config: DictConfig,
    vocab: Any,
    device: str
) -> Tuple[Any, Any, Any]:
    """
    Load trained Amadeus Generator from StarGAN checkpoint.
    
    Args:
        model_path: Path to StarGAN checkpoint
        amadeus_config: Amadeus configuration
        vocab: Vocabulary object
        device: Device to load on
        
    Returns:
        Tuple of (model, vocab, config)
    """
    # Load model architecture and vocab
    model = prepare_amadeus_model(amadeus_config, vocab, device)
    
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['generator_state_dict'], strict=False)
    print(f"Loaded checkpoint from {model_path}")
    
    return model


def load_input_score(
    input_path: str,
    input_length: int,
    device: str
) -> Optional[torch.Tensor]:
    """
    Load input score data.
    
    Args:
        input_path: Path to input score (numpy or npz format)
        input_length: Maximum length to use
        device: Device to load on
        
    Returns:
        Input tensor of shape (B, T, num_features) or None if not available
    """
    if input_path is None:
        return None
    
    print(f"Loading input score from: {input_path}")
    
    if input_path.endswith('.npz'):
        # Load npz file and extract 'score'
        score = np.load(input_path)['arr_0']
    
    return score


def create_domain_label(
    target_domain: str,
    num_domains: int
) -> torch.Tensor:
    """
    Create domain label one-hot tensor.
    
    Args:
        target_domain: Target domain name/index
        num_domains: Total number of domains
        
    Returns:
        One-hot domain label tensor
    """
    try:
        domain_idx = int(target_domain)
    except ValueError:
        # Hash the string to get a consistent index
        domain_idx = hash(target_domain) % num_domains
    
    label = torch.zeros(num_domains, dtype=torch.long)
    label[domain_idx] = 1
    
    return label

def save_generated_score(
    sequence: np.ndarray,
    save_dir: Path,
    sample_idx: int,
    target_domain: str,
    vocab: Any,
    config: DictConfig,
    save_format: str = 'both',
    save_metadata: bool = True
) -> Dict[str, str]:
    """
    Save generated arrangement sequence.
    
    Args:
        sequence: Generated token sequence
        save_dir: Directory to save to
        sample_idx: Sample index
        target_domain: Target domain used
        vocab: Vocabulary object
        config: Model configuration
        save_format: 'midi', 'npy', or 'both'
        save_metadata: Whether to save metadata JSON
        
    Returns:
        Dictionary of saved file paths
    """
    saved_files = {}
    
    # Prepare filename
    filename_base = f"arrangement_{sample_idx:04d}_domain_{target_domain}"
    
    # Save as NumPy array
    if save_format in ['npy', 'both']:
        npy_path = save_dir / f"{filename_base}.npy"
        np.save(str(npy_path), sequence)
        saved_files['npy'] = str(npy_path)
        print(f"    ✓ Saved: {npy_path.name}")
    
    # Convert to MIDI and save
    if save_format in ['midi', 'both']:
        try:
            # Prepare MIDI decoder
            encoding_scheme = config.nn_params.encoding_scheme
            in_beat_resolution_dict = {'Pop1k7': 4, 'Pop909': 4, 'SOD': 12, 'LakhClean': 4}
            in_beat_resolution = in_beat_resolution_dict.get(config.dataset, 4)
            
            midi_decoder_dict = {
                'remi': 'MidiDecoder4REMI',
                'cp': 'MidiDecoder4CP',
                'nb': 'MidiDecoder4NB'
            }
            decoder_name = midi_decoder_dict[encoding_scheme]
            
            decoder = getattr(decoding_utils, decoder_name)(
                vocab=vocab,
                in_beat_resolution=in_beat_resolution,
                dataset_name=config.dataset
            )
            
            # Convert sequence to tensor
            seq_tensor = torch.from_numpy(sequence).unsqueeze(0)
            
            # Decode to MIDI
            midi_path = save_dir / f"{filename_base}.mid"
            decoder(seq_tensor, output_path=str(midi_path))
            saved_files['midi'] = str(midi_path)
            print(f"    ✓ Saved: {midi_path.name}")
        except Exception as e:
            print(f"    ⚠ MIDI conversion failed: {e}")
    
    # Save metadata
    if save_metadata:
        metadata = {
            'sample_idx': sample_idx,
            'target_domain': target_domain,
            'sequence_length': int(sequence.size),
            'saved_files': saved_files
        }
        
        json_path = save_dir / f"{filename_base}_metadata.json"
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        saved_files['metadata'] = str(json_path)
        print(f"    ✓ Saved: {json_path.name}")
    
    return saved_files


def main():
    """Main generation script."""
    # Parse arguments
    parser = get_argument_parser()
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load model and vocabulary
    device = torch.device(args.device)
    print(f"Using device: {device}\n")
     
    config, model_dummy, vocab = load_resources(args.model_id, device)
    config.text_encoder_model = args.text_encoder_model
    first_pred_feature = config.data_params.first_pred_feature
    
    model = load_amadeus_generator(
        args.model_path,
        config,
        vocab,
        device
    )
    
    # Load input score if provided
    input_note = None
    if args.input_score_path:
        input_note = load_input_score(
            args.input_score_path,
            args.input_score_length,
            device
        )
    if isinstance(input_note, np.ndarray):
        input_note = torch.from_numpy(input_note).to(device)
    
    # Generate arrangements
    print(f"\nGenerating {args.num_samples} arrangement(s)...")
    print(f"  Sequence length: {args.sequence_length} tokens")
    print(f"  Target domains: {args.target_domains}")
    print(f"  Sampling: {args.sampling_method} (threshold={args.threshold}, temperature={args.temperature})")
    print()
    
    generation_log = []
    for sample_idx in range(args.num_samples):
        # Select random target domain
        target_domain = random.choice(args.target_domains)
        
        print(f"[{sample_idx + 1}/{args.num_samples}] Generating arrangement...")
        print(f"  Target domain: {target_domain}")
        
        try:
            # Generate sequence
            generated_seq = generate_with_textANDscore_prompt(
                config=config,
                vocab=vocab,
                model=model,
                device=device,
                prompt=target_domain,
                input_note=input_note,
                save_dir=output_dir,
                first_pred_feature=first_pred_feature,
                sampling_method=args.sampling_method,
                threshold=args.threshold,
                temperature=args.temperature,
                generation_length=args.sequence_length
            )
            
            # Save generated sequence
            """saved_files = save_generated_score(
                sequence=generated_seq,
                save_dir=output_dir,
                sample_idx=sample_idx,
                target_domain=target_domain,
                vocab=vocab,
                config=config,
                save_format=args.save_format,
                save_metadata=args.save_metadata
            )"""
            
            # Log generation
            generation_log.append({
                'sample_idx': sample_idx,
                'target_domain': target_domain,
                'sequence_length': int(generated_seq.size),
                'files': f"{args.output_dir}/{target_domain}"
            })
            
        except Exception as e:
            print(f"  ✗ Generation failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Save generation log
    log_path = output_dir / 'generation_log.json'
    with open(log_path, 'w') as f:
        json.dump({
            'total_samples': args.num_samples,
            'target_domains': args.target_domains,
            'sampling_params': {
                'method': args.sampling_method,
                'threshold': args.threshold,
                'temperature': args.temperature
            },
            'generations': generation_log
        }, f, indent=2)
    
    print(f"\n✓ Generation complete!")
    print(f"  Generated {len(generation_log)} samples")
    print(f"  Log saved to: {log_path}")


if __name__ == '__main__':
    main()
