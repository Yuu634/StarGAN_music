"""
Testing and debugging utilities for End-to-End StarGAN
"""

import torch
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import os


def test_gradient_flow(G, D, dummy_scores, dummy_labels):
    """
    Test if gradients flow properly from D to G
    
    Args:
        G: Generator model
        D: Discriminator model
        dummy_scores: [B, T, 8] Dummy input scores
        dummy_labels: [B, 108] Dummy domain labels
    """
    print("\n=== Testing Gradient Flow ===")
    
    # Set to training mode
    G.train()
    D.train()
    
    # Forward through Generator
    fake_logits, fake_soft_embeddings = G(
        dummy_scores,
        dummy_labels,
        temperature=0.5,
        hard=True
    )
    
    print(f"Generator output shape: {fake_soft_embeddings.shape}")
    print(f"Requires grad: {fake_soft_embeddings.requires_grad}")
    
    # Forward through Discriminator
    fake_src, fake_cls = D(soft_embeddings=fake_soft_embeddings)
    
    print(f"Discriminator output shapes: real_fake={fake_src.shape}, domain={fake_cls.shape}")
    
    # Compute dummy loss
    dummy_loss = fake_src.mean() + fake_cls.mean()
    
    # Backward
    dummy_loss.backward()
    
    # Check gradients in Generator
    print("\n--- Generator Gradients ---")
    g_grad_count = 0
    g_no_grad_count = 0
    for name, param in G.named_parameters():
        if param.grad is not None:
            g_grad_count += 1
            grad_norm = param.grad.norm().item()
            if grad_norm > 1e-7:
                print(f"  {name}: grad_norm={grad_norm:.6f} ✓")
        else:
            g_no_grad_count += 1
            print(f"  {name}: NO GRADIENT ✗")
    
    print(f"\nGenerator: {g_grad_count} params with grad, {g_no_grad_count} without")
    
    # Check gradients in Discriminator
    print("\n--- Discriminator Gradients ---")
    d_grad_count = 0
    d_no_grad_count = 0
    for name, param in D.named_parameters():
        if param.grad is not None:
            d_grad_count += 1
        else:
            d_no_grad_count += 1
    
    print(f"Discriminator: {d_grad_count} params with grad, {d_no_grad_count} without")
    
    # Verdict
    if g_grad_count > 0 and d_grad_count > 0:
        print("\n✓ Gradient flow test PASSED: Both G and D have gradients")
        return True
    else:
        print("\n✗ Gradient flow test FAILED")
        return False


def test_gumbel_softmax_temperature(G, dummy_scores, dummy_labels, temperatures=[0.1, 0.5, 1.0, 2.0]):
    """
    Test Gumbel-Softmax behavior at different temperatures
    
    Args:
        G: Generator model
        dummy_scores: [B, T, 8] Dummy input
        dummy_labels: [B, 108] Dummy labels
        temperatures: List of temperatures to test
    """
    print("\n=== Testing Gumbel-Softmax Temperatures ===")
    
    G.eval()
    
    results = {}
    for temp in temperatures:
        with torch.no_grad():
            fake_logits, fake_soft_embeddings = G(
                dummy_scores,
                dummy_labels,
                temperature=temp,
                hard=True
            )
            
            # Get hard tokens
            fake_hard = G.get_hard_tokens(fake_logits)
            
            # Calculate entropy (measure of uncertainty)
            probs = torch.softmax(fake_logits['type'], dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
            
            results[temp] = {
                'entropy': entropy,
                'unique_tokens': torch.unique(fake_hard).numel()
            }
            
            print(f"Temperature {temp}: entropy={entropy:.4f}, unique_tokens={results[temp]['unique_tokens']}")
    
    return results


def validate_token_conversion(amadeus_tokens):
    """
    Validate Amadeus to Moonbeam conversion
    
    Args:
        amadeus_tokens: [B, T, 8] Amadeus format tokens
    """
    print("\n=== Validating Token Conversion ===")
    
    from stargan_losses import amadeus_to_moonbeam_discrete
    
    moonbeam_tokens = amadeus_to_moonbeam_discrete(amadeus_tokens)
    
    print(f"Input shape: {amadeus_tokens.shape}")
    print(f"Output shape: {moonbeam_tokens.shape}")
    
    # Check value ranges
    print("\nMoonbeam token ranges:")
    for i, feature in enumerate(['onset', 'duration', 'octave', 'pitch_class', 'instrument', 'velocity']):
        min_val = moonbeam_tokens[:, :, i].min().item()
        max_val = moonbeam_tokens[:, :, i].max().item()
        print(f"  {feature}: min={min_val}, max={max_val}")
    
    # Check for invalid values
    octave = moonbeam_tokens[:, :, 2]
    pitch_class = moonbeam_tokens[:, :, 3]
    
    invalid_octave = (octave < 0) | (octave > 10)
    invalid_pitch = (pitch_class < 0) | (pitch_class > 11)
    
    if invalid_octave.any():
        print(f"  ✗ WARNING: {invalid_octave.sum().item()} invalid octave values")
    else:
        print(f"  ✓ All octave values valid")
    
    if invalid_pitch.any():
        print(f"  ✗ WARNING: {invalid_pitch.sum().item()} invalid pitch_class values")
    else:
        print(f"  ✓ All pitch_class values valid")
    
    return moonbeam_tokens


def visualize_loss_history(loss_history: Dict[str, List[float]], save_path='loss_curves.png'):
    """
    Visualize training loss curves
    
    Args:
        loss_history: Dictionary of loss lists
        save_path: Path to save plot
    """
    print(f"\n=== Saving Loss Curves to {save_path} ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Plot Discriminator losses
    if 'D/loss_real' in loss_history:
        axes[0].plot(loss_history['D/loss_real'], label='D Real')
        axes[0].plot(loss_history['D/loss_fake'], label='D Fake')
        axes[0].set_title('Discriminator Real/Fake Loss')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
    
    if 'D/loss_cls' in loss_history:
        axes[1].plot(loss_history['D/loss_cls'], label='D Classification')
        axes[1].set_title('Discriminator Domain Classification Loss')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
    
    # Plot Generator losses
    if 'G/loss_adv' in loss_history:
        axes[2].plot(loss_history['G/loss_adv'], label='G Adversarial')
        axes[2].plot(loss_history['G/loss_cls'], label='G Classification')
        axes[2].set_title('Generator Adversarial & Classification Loss')
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('Loss')
        axes[2].legend()
        axes[2].grid(True)
    
    if 'G/loss_rec' in loss_history:
        axes[3].plot(loss_history['G/loss_rec'], label='G Reconstruction')
        axes[3].set_title('Generator Reconstruction Loss')
        axes[3].set_xlabel('Iteration')
        axes[3].set_ylabel('Loss')
        axes[3].legend()
        axes[3].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved to {save_path}")


def run_sanity_checks(G, D, device='cuda'):
    """
    Run all sanity checks before training
    
    Args:
        G: Generator model
        D: Discriminator model
        device: Device to run on
    """
    print("\n" + "="*60)
    print("RUNNING SANITY CHECKS")
    print("="*60)
    
    # Create dummy data
    B, T = 2, 128
    dummy_scores = torch.randint(0, 100, (B, T, 8), device=device)
    dummy_labels = torch.randint(0, 2, (B, 108), device=device).float()
    
    all_passed = True
    
    # Test 1: Gradient flow
    try:
        passed = test_gradient_flow(G, D, dummy_scores, dummy_labels)
        if not passed:
            all_passed = False
    except Exception as e:
        print(f"✗ Gradient flow test FAILED with error: {e}")
        all_passed = False
    
    # Test 2: Gumbel-Softmax temperatures
    try:
        test_gumbel_softmax_temperature(G, dummy_scores, dummy_labels)
    except Exception as e:
        print(f"✗ Gumbel-Softmax test FAILED with error: {e}")
        all_passed = False
    
    # Test 3: Token conversion
    try:
        validate_token_conversion(dummy_scores)
    except Exception as e:
        print(f"✗ Token conversion test FAILED with error: {e}")
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL SANITY CHECKS PASSED")
    else:
        print("✗ SOME SANITY CHECKS FAILED - Please fix before training")
    print("="*60 + "\n")
    
    return all_passed


def create_small_test_dataset(num_samples=10, seq_len=64, save_dir='./test_data'):
    """
    Create a small synthetic dataset for testing
    
    Args:
        num_samples: Number of samples
        seq_len: Sequence length
        save_dir: Directory to save data
    """
    print(f"\n=== Creating Test Dataset ===")
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(num_samples):
        # Random Amadeus tokens
        score = np.random.randint(0, 100, size=(seq_len, 8), dtype=np.int64)
        
        # Random multi-hot labels (108 dimensions)
        label = np.random.randint(0, 2, size=108, dtype=np.float32)
        
        # Ensure at least one domain is active
        if label.sum() == 0:
            label[np.random.randint(0, 108)] = 1
        
        # Save
        np.savez(
            os.path.join(save_dir, f'sample_{i:03d}.npz'),
            score=score,
            label=label
        )
    
    print(f"Created {num_samples} test samples in {save_dir}")
    return save_dir
