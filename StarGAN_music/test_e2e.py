"""
Quick test script for End-to-End StarGAN implementation
Run this to verify all components work before full training
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from amadeus_stargan import AmadeusForStarGAN
from llama_discriminator import LlamaForSequenceDoubleClassification, create_discriminator_config
from stargan_losses import compute_discriminator_loss, compute_generator_loss
from test_utils import run_sanity_checks, create_small_test_dataset


def create_dummy_amadeus():
    """Create a dummy Amadeus model for testing"""
    import torch.nn as nn
    
    class DummyAmadeus(nn.Module):
        def __init__(self):
            super().__init__()
            self.dim = 512
            self.input_embedder = nn.Embedding(128, self.dim)
            self.pos_enc = nn.Embedding(512, self.dim)
            self.emb_dropout = nn.Dropout(0.1)
            self.main_decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(self.dim, 8, dim_feedforward=2048),
                num_layers=6
            )
            self.main_norm = nn.LayerNorm(self.dim)
            self.sub_decoder = DummySubDecoder()
    
    class DummySubDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = nn.ModuleDict({
                'type': nn.Linear(512, 128),
                'beat': nn.Linear(512, 128),
                'chord': nn.Linear(512, 128),
                'tempo': nn.Linear(512, 128),
                'instrument': nn.Linear(512, 129),
                'pitch': nn.Linear(512, 128),
                'duration': nn.Linear(512, 128),
                'velocity': nn.Linear(512, 128),
            })
        
        def forward(self, input_dict):
            hidden = input_dict['hidden_vec']  # [B, T, 512]
            outputs = {}
            for name, head in self.heads.items():
                outputs[name] = head(hidden)
            return outputs
    
    class DummyAmadeusWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = nn.Module()
            self.decoder.net = DummyAmadeus()
    
    return DummyAmadeusWrapper()


class DummyVocab:
    """Dummy vocabulary for testing"""
    def get_feature_vocab_size(self, feature_name):
        sizes = {
            'type': 128, 'beat': 128, 'chord': 128, 'tempo': 128,
            'instrument': 129, 'pitch': 128, 'duration': 128, 'velocity': 128
        }
        return sizes.get(feature_name, 128)


def test_basic_forward():
    """Test basic forward pass"""
    print("\n" + "="*60)
    print("TEST 1: Basic Forward Pass")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create models
    print("\nCreating models...")
    dummy_amadeus = create_dummy_amadeus()
    vocab = DummyVocab()
    
    G = AmadeusForStarGAN(dummy_amadeus, vocab, hidden_dim=512).to(device)
    
    config = create_discriminator_config(
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=512,
        amadeus_dim=512
    )
    D = LlamaForSequenceDoubleClassification(config).to(device)
    
    print(f"Generator params: {sum(p.numel() for p in G.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in D.parameters()):,}")
    
    # Create dummy input
    B, T = 2, 64
    dummy_scores = torch.randint(0, 100, (B, T, 8), device=device)
    dummy_labels = torch.randint(0, 2, (B, 108), device=device).float()
    
    print(f"\nInput shapes: scores={dummy_scores.shape}, labels={dummy_labels.shape}")
    
    # Forward through Generator
    print("\nForward through Generator...")
    fake_logits, fake_soft_embeddings = G(
        dummy_scores,
        dummy_labels,
        temperature=0.5,
        hard=True
    )
    print(f"✓ Generator output: soft_embeddings={fake_soft_embeddings.shape}")
    
    # Forward through Discriminator (soft)
    print("\nForward through Discriminator (soft embeddings)...")
    fake_src, fake_cls = D(soft_embeddings=fake_soft_embeddings)
    print(f"✓ Discriminator output: real_fake={fake_src.shape}, domain={fake_cls.shape}")
    
    # Forward through Discriminator (discrete)
    print("\nForward through Discriminator (discrete tokens)...")
    from stargan_losses import amadeus_to_moonbeam_discrete
    moonbeam_tokens = amadeus_to_moonbeam_discrete(dummy_scores)
    real_src, real_cls = D(input_ids=moonbeam_tokens)
    print(f"✓ Discriminator output: real_fake={real_src.shape}, domain={real_cls.shape}")
    
    print("\n✓ TEST 1 PASSED")
    return G, D


def test_loss_computation(G, D):
    """Test loss computation"""
    print("\n" + "="*60)
    print("TEST 2: Loss Computation")
    print("="*60)
    
    device = next(G.parameters()).device
    
    # Create dummy input
    B, T = 2, 64
    real_scores = torch.randint(0, 100, (B, T, 8), device=device)
    real_labels = torch.randint(0, 2, (B, 108), device=device).float()
    target_labels = torch.randint(0, 2, (B, 108), device=device).float()
    
    # Discriminator loss
    print("\nComputing Discriminator loss...")
    d_loss, d_logs = compute_discriminator_loss(
        G, D, real_scores, target_labels, real_labels,
        lambda_cls=1.0, lambda_gp=10.0, temperature=0.5
    )
    print(f"✓ D loss: {d_loss.item():.4f}")
    for key, value in d_logs.items():
        print(f"  {key}: {value:.4f}")
    
    # Generator loss
    print("\nComputing Generator loss...")
    g_loss, g_logs, fake_tokens = compute_generator_loss(
        G, D, real_scores, target_labels, real_labels,
        lambda_cls=1.0, lambda_rec=10.0, temperature=0.5
    )
    print(f"✓ G loss: {g_loss.item():.4f}")
    for key, value in g_logs.items():
        print(f"  {key}: {value:.4f}")
    print(f"✓ Fake tokens shape: {fake_tokens.shape}")
    
    print("\n✓ TEST 2 PASSED")


def test_backward_pass(G, D):
    """Test backward pass and gradient flow"""
    print("\n" + "="*60)
    print("TEST 3: Backward Pass & Gradient Flow")
    print("="*60)
    
    device = next(G.parameters()).device
    
    # Create dummy input
    B, T = 2, 64
    real_scores = torch.randint(0, 100, (B, T, 8), device=device)
    real_labels = torch.randint(0, 2, (B, 108), device=device).float()
    target_labels = torch.randint(0, 2, (B, 108), device=device).float()
    
    # Discriminator backward
    print("\nTesting Discriminator backward...")
    d_loss, _ = compute_discriminator_loss(
        G, D, real_scores, target_labels, real_labels,
        lambda_cls=1.0, lambda_gp=10.0, temperature=0.5
    )
    
    G.zero_grad()
    D.zero_grad()
    d_loss.backward()
    
    # Check if G has gradients (should have, because of gradient flow!)
    g_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in G.parameters())
    d_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in D.parameters())
    
    print(f"G has gradients: {g_has_grad}")
    print(f"D has gradients: {d_has_grad}")
    
    if g_has_grad and d_has_grad:
        print("✓ Gradient flow from D to G is working!")
    else:
        print("✗ Gradient flow issue detected")
        return False
    
    # Generator backward
    print("\nTesting Generator backward...")
    G.zero_grad()
    D.zero_grad()
    
    g_loss, _, _ = compute_generator_loss(
        G, D, real_scores, target_labels, real_labels,
        lambda_cls=1.0, lambda_rec=10.0, temperature=0.5
    )
    g_loss.backward()
    
    g_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in G.parameters())
    d_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in D.parameters())
    
    print(f"G has gradients: {g_has_grad}")
    print(f"D has gradients: {d_has_grad}")
    
    print("\n✓ TEST 3 PASSED")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("STARTING END-TO-END STARGAN TESTS")
    print("="*60)
    
    try:
        # Test 1: Basic forward
        G, D = test_basic_forward()
        
        # Test 2: Loss computation
        test_loss_computation(G, D)
        
        # Test 3: Backward pass
        test_backward_pass(G, D)
        
        # Test 4: Sanity checks
        run_sanity_checks(G, D, device=next(G.parameters()).device)
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nYou can now proceed with training:")
        print("  1. Prepare your dataset")
        print("  2. Configure main.py to use train_stargan_e2e()")
        print("  3. Run training")
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"✗ TEST FAILED WITH ERROR: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
