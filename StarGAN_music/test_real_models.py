"""
Test script for StarGAN with real Amadeus and Moonbeam models
Verifies that models load correctly and forward pass works
"""

import torch
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../Amadeus'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../Moonbeam-MIDI-Foundation-Model/src'))

from amadeus_generator_wrapper import load_amadeus_generator, amadeus_to_moonbeam_discrete
from moonbeam_discriminator_wrapper import load_moonbeam_discriminator


def test_model_loading():
    """Test 1: Load pre-trained models"""
    print("=" * 60)
    print("Test 1: Loading Pre-trained Models")
    print("=" * 60)
    
    # Paths - UPDATE THESE TO YOUR ACTUAL PATHS
    amadeus_config = "/mnt/kiso-qnap5/obara/Amadeus/path/to/config.yaml"
    amadeus_checkpoint = "/mnt/kiso-qnap5/obara/Amadeus/models/checkpoint.pt"
    moonbeam_config = "/mnt/kiso-qnap5/obara/Moonbeam-MIDI-Foundation-Model/src/llama_recipes/configs/player_classification_config.json"
    moonbeam_checkpoint = "/mnt/kiso-qnap5/obara/Moonbeam-MIDI-Foundation-Model/path/to/checkpoint.pt"
    
    print(f"Amadeus config: {amadeus_config}")
    print(f"Amadeus checkpoint: {amadeus_checkpoint}")
    print(f"Moonbeam config: {moonbeam_config}")
    print(f"Moonbeam checkpoint: {moonbeam_checkpoint}")
    
    # Check if files exist
    if not os.path.exists(amadeus_config):
        print(f"WARNING: Amadeus config not found: {amadeus_config}")
        print("Please update the path in this script")
        return False
    
    if not os.path.exists(moonbeam_config):
        print(f"WARNING: Moonbeam config not found: {moonbeam_config}")
        return False
    
    try:
        # Load Generator
        print("\nLoading Amadeus Generator...")
        G = load_amadeus_generator(
            config_path=amadeus_config,
            checkpoint_path=amadeus_checkpoint if os.path.exists(amadeus_checkpoint) else None,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"✓ Generator loaded successfully!")
        print(f"  Parameters: {sum(p.numel() for p in G.parameters()):,}")
        
        # Load Discriminator
        print("\nLoading Moonbeam Discriminator...")
        D = load_moonbeam_discriminator(
            config_path=moonbeam_config,
            checkpoint_path=moonbeam_checkpoint if os.path.exists(moonbeam_checkpoint) else None,
            num_domains=108,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"✓ Discriminator loaded successfully!")
        print(f"  Parameters: {sum(p.numel() for p in D.parameters()):,}")
        
        return G, D
        
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_forward_pass(G, D):
    """Test 2: Forward pass through both models"""
    print("\n" + "=" * 60)
    print("Test 2: Forward Pass")
    print("=" * 60)
    
    device = next(G.parameters()).device
    
    try:
        # Create dummy input
        B, T = 2, 100  # Batch size=2, Sequence length=100
        num_features = 8  # Amadeus features
        
        print(f"\nCreating dummy input: [B={B}, T={T}, features={num_features}]")
        
        # Random Amadeus tokens
        real_scores = torch.randint(0, 50, (B, T, num_features)).to(device)
        target_labels = torch.randn(B, 108).to(device)  # 108 domain labels
        
        print("✓ Dummy input created")
        
        # Forward through Generator
        print("\nRunning Generator forward pass...")
        logits_dict, soft_embeddings, sampled_tokens_dict = G(
            input_seq=real_scores,
            target_domain=target_labels,
            temperature=0.5,
            hard=True
        )
        
        print(f"✓ Generator output:")
        print(f"  Logits dict keys: {list(logits_dict.keys())}")
        print(f"  Soft embeddings shape: {soft_embeddings.shape}")
        
        # Forward through Discriminator (with soft embeddings)
        print("\nRunning Discriminator forward pass (Fake)...")
        fake_src, fake_cls = D(soft_embeddings=soft_embeddings)
        
        print(f"✓ Discriminator output (Fake):")
        print(f"  Real/Fake logits shape: {fake_src.shape}")
        print(f"  Domain logits shape: {fake_cls.shape}")
        
        # Forward through Discriminator (with discrete tokens)
        print("\nConverting to Moonbeam tokens and running Discriminator (Real)...")
        
        # Get discrete tokens from Generator
        discrete_tokens = G.get_discrete_tokens(logits_dict)
        print(f"  Discrete tokens shape: {discrete_tokens.shape}")
        
        # Convert Amadeus to Moonbeam format
        moonbeam_tokens = amadeus_to_moonbeam_discrete(discrete_tokens)
        print(f"  Moonbeam tokens shape: {moonbeam_tokens.shape}")
        
        real_src, real_cls = D(input_ids=moonbeam_tokens)
        
        print(f"✓ Discriminator output (Real):")
        print(f"  Real/Fake logits shape: {real_src.shape}")
        print(f"  Domain logits shape: {real_cls.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_flow(G, D):
    """Test 3: Gradient flow from D to G"""
    print("\n" + "=" * 60)
    print("Test 3: Gradient Flow")
    print("=" * 60)
    
    device = next(G.parameters()).device
    
    try:
        # Create dummy input
        B, T = 2, 100
        real_scores = torch.randint(0, 50, (B, T, 8)).to(device)
        target_labels = torch.randn(B, 108).to(device)
        
        # Zero gradients
        G.zero_grad()
        D.zero_grad()
        
        print("\nForward pass through G and D...")
        
        # Forward through Generator
        logits_dict, soft_embeddings, _ = G(
            input_seq=real_scores,
            target_domain=target_labels,
            temperature=0.5,
            hard=True
        )
        
        # Forward through Discriminator (NO DETACH - this is key!)
        fake_src, fake_cls = D(soft_embeddings=soft_embeddings)
        
        # Compute simple loss
        loss = fake_src.mean() + fake_cls.mean()
        
        print(f"Loss value: {loss.item():.4f}")
        
        # Backward pass
        print("\nBackward pass...")
        loss.backward()
        
        # Check gradients in both models
        g_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in G.parameters())
        d_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in D.parameters())
        
        print(f"\n✓ Gradient check:")
        print(f"  Generator has gradients: {g_has_grad}")
        print(f"  Discriminator has gradients: {d_has_grad}")
        
        if g_has_grad and d_has_grad:
            print("\n✓✓✓ SUCCESS: Gradient flows from D to G!")
            return True
        else:
            print("\n✗ FAILED: Gradient flow not working properly")
            return False
        
    except Exception as e:
        print(f"✗ Error in gradient flow test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("StarGAN Real Models Test Suite")
    print("=" * 60)
    
    # Test 1: Load models
    result = test_model_loading()
    if result is None:
        print("\n✗ Test 1 FAILED: Could not load models")
        print("\nPlease check:")
        print("1. Amadeus config and checkpoint paths are correct")
        print("2. Moonbeam config and checkpoint paths are correct")
        print("3. All dependencies are installed")
        return
    
    G, D = result
    print("\n✓ Test 1 PASSED: Models loaded successfully")
    
    # Test 2: Forward pass
    if not test_forward_pass(G, D):
        print("\n✗ Test 2 FAILED: Forward pass error")
        return
    
    print("\n✓ Test 2 PASSED: Forward pass works")
    
    # Test 3: Gradient flow
    if not test_gradient_flow(G, D):
        print("\n✗ Test 3 FAILED: Gradient flow error")
        return
    
    print("\n✓ Test 3 PASSED: Gradient flow verified")
    
    print("\n" + "=" * 60)
    print("✓✓✓ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nYou can now proceed to full training with train_stargan_real.py")


if __name__ == '__main__':
    main()
