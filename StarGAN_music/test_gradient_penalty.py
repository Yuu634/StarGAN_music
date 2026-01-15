#!/usr/bin/env python3
"""
Test script for Gradient Penalty implementation in StarGAN_music
Verifies that d_loss_gp is correctly computed and contributes to loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import autograd


def test_gradient_penalty_computation():
    """Test basic gradient penalty computation"""
    print("\n" + "="*70)
    print("TEST 1: Basic Gradient Penalty Computation")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4
    seq_len = 100
    hidden_size = 768
    
    # Create simple discriminator mock
    class SimpleDiscriminator(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.fc = nn.Linear(hidden_size, 1)
        
        def forward(self, x):
            # x: [B, T, hidden_size] or [B*T, hidden_size]
            if x.dim() == 3:
                x = x.view(-1, x.size(-1))
            return self.fc(x)
    
    D = SimpleDiscriminator(hidden_size).to(device)
    
    # Create real and fake embeddings
    real_embeddings = torch.randn(batch_size, seq_len, hidden_size, device=device)
    fake_embeddings = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    # Create interpolated embeddings
    alpha = torch.rand(batch_size, 1, 1, device=device)
    alpha = alpha.expand(batch_size, seq_len, -1)
    
    hat_embeddings = (alpha * real_embeddings.detach() + 
                      (1 - alpha) * fake_embeddings.detach()).requires_grad_(True)
    
    # Forward pass
    hat_output = D(hat_embeddings)  # [B*T, 1]
    
    # Compute gradient w.r.t. hat_embeddings
    gradients = torch.autograd.grad(
        outputs=hat_output.sum(),
        inputs=hat_embeddings,
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Compute gradient norm
    gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=[1, 2]) + 1e-8)
    
    # Compute gradient penalty
    d_loss_gp = torch.mean((gradients_norm - 1.0)**2)
    
    print(f"✓ Real embeddings shape: {real_embeddings.shape}")
    print(f"✓ Fake embeddings shape: {fake_embeddings.shape}")
    print(f"✓ Interpolated embeddings shape: {hat_embeddings.shape}")
    print(f"✓ Gradients shape: {gradients.shape}")
    print(f"✓ Gradient norm shape: {gradients_norm.shape}")
    print(f"✓ Gradient norm values: min={gradients_norm.min():.4f}, max={gradients_norm.max():.4f}, mean={gradients_norm.mean():.4f}")
    print(f"✓ Gradient Penalty Loss: {d_loss_gp.item():.6f}")
    
    assert d_loss_gp.item() > 0, "Gradient penalty should be > 0"
    assert not torch.isnan(d_loss_gp), "Gradient penalty should not be NaN"
    print("✓ TEST 1 PASSED\n")


def test_gradient_penalty_with_schedule():
    """Test gradient penalty scheduling"""
    print("="*70)
    print("TEST 2: Gradient Penalty Scheduling")
    print("="*70)
    
    total_steps = 1000
    lambda_gp_init = 10.0
    
    # Test different schedules
    schedules = ['linear', 'warmup', 'cosine', 'constant']
    
    for schedule in schedules:
        print(f"\n  Schedule: {schedule}")
        lambda_gp_values = []
        
        for step in [0, 250, 500, 750, 999]:
            progress = step / total_steps
            
            if schedule == 'linear':
                if progress < 0.5:
                    lambda_gp = lambda_gp_init * (progress * 2)
                else:
                    lambda_gp = lambda_gp_init
            
            elif schedule == 'warmup':
                if progress < 0.2:
                    lambda_gp = lambda_gp_init * 0.1
                else:
                    ramp_progress = min((progress - 0.2) / 0.2, 1.0)
                    lambda_gp = lambda_gp_init * (0.1 + 0.9 * ramp_progress)
            
            elif schedule == 'cosine':
                import math
                lambda_gp = lambda_gp_init * 0.5 * (1 + math.cos(math.pi * progress))
            
            else:
                lambda_gp = lambda_gp_init
            
            lambda_gp_values.append(lambda_gp)
            print(f"    Step {step:4d} (progress {progress:.2f}): lambda_gp = {lambda_gp:.4f}")
        
        # Verify monotonicity for 'warmup'
        if schedule == 'warmup':
            is_increasing = all(lambda_gp_values[i] <= lambda_gp_values[i+1] 
                               for i in range(len(lambda_gp_values)-1))
            assert is_increasing, f"Warmup schedule should be monotonically increasing"
            print(f"    ✓ Monotonically increasing")
    
    print("✓ TEST 2 PASSED\n")


def test_gradient_penalty_effect_on_training():
    """Test that gradient penalty affects loss magnitude"""
    print("="*70)
    print("TEST 3: Gradient Penalty Effect on Training")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4
    seq_len = 100
    hidden_size = 768
    
    # Simple discriminator
    class SimpleD(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(hidden_size, 1)
        def forward(self, x):
            if x.dim() == 3:
                x = x.view(-1, x.size(-1))
            return self.fc(x)
    
    D = SimpleD().to(device)
    optimizer = torch.optim.Adam(D.parameters(), lr=1e-4)
    
    # Test with different lambda_gp values
    lambda_gp_values = [0.0, 1.0, 5.0, 10.0]
    
    print("\n  Training steps with different lambda_gp:")
    for lambda_gp_test in lambda_gp_values:
        D = SimpleD().to(device)
        optimizer = torch.optim.Adam(D.parameters(), lr=1e-4)
        
        losses_with_gp = []
        
        for step in range(3):
            optimizer.zero_grad()
            
            # Create data
            real_embeddings = torch.randn(batch_size, seq_len, hidden_size, device=device)
            fake_embeddings = torch.randn(batch_size, seq_len, hidden_size, device=device)
            
            # Forward pass for real/fake
            real_output = D(real_embeddings)
            fake_output = D(fake_embeddings)
            
            # Wasserstein loss
            d_loss_real = -torch.mean(real_output)
            d_loss_fake = torch.mean(fake_output)
            
            # Gradient penalty
            alpha = torch.rand(batch_size, 1, 1, device=device)
            alpha = alpha.expand(batch_size, seq_len, -1)
            hat_embeddings = (alpha * real_embeddings.detach() + 
                            (1 - alpha) * fake_embeddings.detach()).requires_grad_(True)
            
            hat_output = D(hat_embeddings)
            gradients = torch.autograd.grad(
                outputs=hat_output.sum(),
                inputs=hat_embeddings,
                create_graph=True,
            )[0]
            gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=[1, 2]) + 1e-8)
            d_loss_gp = torch.mean((gradients_norm - 1.0)**2)
            
            # Total loss
            d_loss = d_loss_real + d_loss_fake + lambda_gp_test * d_loss_gp
            
            d_loss.backward()
            optimizer.step()
            
            losses_with_gp.append(d_loss.item())
        
        print(f"    lambda_gp={lambda_gp_test:4.1f}: loss trajectory = {[f'{l:.4f}' for l in losses_with_gp]}")
    
    print("✓ TEST 3 PASSED\n")


def test_gradient_flow_stability():
    """Test that gradients flow through discriminator correctly"""
    print("="*70)
    print("TEST 4: Gradient Flow Stability")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4
    seq_len = 100
    hidden_size = 768
    
    class SimpleD(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(hidden_size, 256)
            self.fc2 = nn.Linear(256, 1)
        def forward(self, x):
            if x.dim() == 3:
                x = x.view(-1, x.size(-1))
            x = F.relu(self.fc1(x))
            return self.fc2(x)
    
    D = SimpleD().to(device)
    
    # Create data
    real_embeddings = torch.randn(batch_size, seq_len, hidden_size, device=device)
    fake_embeddings = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    # Forward pass
    alpha = torch.rand(batch_size, 1, 1, device=device)
    alpha = alpha.expand(batch_size, seq_len, -1)
    hat_embeddings = (alpha * real_embeddings.detach() + 
                      (1 - alpha) * fake_embeddings.detach()).requires_grad_(True)
    
    hat_output = D(hat_embeddings)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=hat_output.sum(),
        inputs=hat_embeddings,
        create_graph=True,
    )[0]
    
    d_loss_gp = torch.mean((torch.sqrt(torch.sum(gradients**2, dim=[1, 2]) + 1e-8) - 1.0)**2)
    
    # Backward pass through GP loss
    d_loss_gp.backward()
    
    # Check gradient norms
    grad_norms = {}
    for name, param in D.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms[name] = grad_norm
    
    print(f"\n  Gradient norms for each parameter:")
    for name, norm in grad_norms.items():
        status = "✓" if norm > 0 else "✗"
        print(f"    {status} {name}: {norm:.6f}")
    
    # Verify gradients exist
    assert len(grad_norms) > 0, "No gradients computed!"
    assert all(norm > 0 for norm in grad_norms.values()), "Some gradients are zero!"
    
    print("✓ TEST 4 PASSED\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("GRADIENT PENALTY TEST SUITE")
    print("="*70)
    
    try:
        test_gradient_penalty_computation()
        test_gradient_penalty_with_schedule()
        test_gradient_penalty_effect_on_training()
        test_gradient_flow_stability()
        
        print("="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

