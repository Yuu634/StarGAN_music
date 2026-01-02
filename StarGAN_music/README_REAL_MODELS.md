# StarGAN with Real Amadeus and Moonbeam Models

End-to-End differentiable StarGAN implementation using:
- **Generator**: Amadeus model from `Amadeus/Amadeus/model_zoo.py`
- **Discriminator**: Moonbeam model from `Moonbeam-MIDI-Foundation-Model/src/llama_recipes/`

## Architecture Overview

### Generator: Amadeus
- Input: [B, T, 8] (Amadeus token format)
- Features: type, beat, chord, tempo, instrument, pitch, duration, velocity
- Gumbel-Softmax sampling for differentiable discrete token generation
- Outputs soft embeddings for Discriminator input

### Discriminator: Moonbeam (LlamaForSequenceClassification)
- Dual classification:
  1. Real/Fake (2 classes)
  2. Domain classification (108 classes for MidiCaps)
- Dual input support:
  - Discrete tokens (Real samples)
  - Soft embeddings (Fake samples from Generator)

## Key Features

1. **End-to-End Gradient Flow**: Gradients flow from Discriminator to Generator through soft embeddings
2. **Gumbel-Softmax with Straight-Through Estimator**: Enables differentiable discrete sampling
3. **Pre-trained Model Integration**: Uses actual pre-trained Amadeus and Moonbeam checkpoints
4. **Dual Classification**: Discriminator performs both real/fake and domain classification

## File Structure

```
StarGAN_music/
├── amadeus_generator_wrapper.py      # Amadeus Generator wrapper
├── moonbeam_discriminator_wrapper.py # Moonbeam Discriminator wrapper
├── stargan_losses.py                 # Loss functions (reused from previous implementation)
├── train_stargan_real.py             # Training script
├── test_real_models.py               # Model loading and testing
└── README_REAL_MODELS.md             # This file
```

## Setup

### 1. Prerequisites

Install dependencies:
```bash
# Amadeus dependencies
cd ../../Amadeus
pip install -r requirements.txt

# Moonbeam dependencies
cd ../Moonbeam-MIDI-Foundation-Model
pip install -r requirements.txt

# StarGAN dependencies
cd ../StarGAN_music/StarGAN_music
pip install torch transformers pyyaml tqdm
```

### 2. Prepare Model Checkpoints

You need:
1. **Amadeus**:
   - Config YAML file (e.g., `Amadeus/symbolic_yamls/your_config.yaml`)
   - Checkpoint file (optional, can train from scratch)

2. **Moonbeam**:
   - Config JSON file: `Moonbeam-MIDI-Foundation-Model/src/llama_recipes/configs/player_classification_config.json`
   - Checkpoint file: Pre-trained Moonbeam model

### 3. Update Paths

Edit `test_real_models.py` line 19-22:
```python
amadeus_config = "/path/to/Amadeus/config.yaml"
amadeus_checkpoint = "/path/to/Amadeus/checkpoint.pt"
moonbeam_config = "/path/to/Moonbeam/config.json"
moonbeam_checkpoint = "/path/to/Moonbeam/checkpoint.pt"
```

## Usage

### Step 1: Test Model Loading

First, verify that models load correctly:

```bash
python test_real_models.py
```

Expected output:
```
✓ Test 1 PASSED: Models loaded successfully
✓ Test 2 PASSED: Forward pass works
✓ Test 3 PASSED: Gradient flow verified
✓✓✓ ALL TESTS PASSED!
```

### Step 2: Prepare Training Data

Create a dataset loader that provides:
- `scores`: Amadeus tokens [B, T, 8]
- `target_labels`: Target domain labels [B, 108]
- `original_labels`: Original domain labels [B, 108]

Example dataset structure:
```python
{
    'scores': torch.LongTensor,  # [B, T, 8]
    'target_labels': torch.FloatTensor,  # [B, 108]
    'original_labels': torch.FloatTensor  # [B, 108]
}
```

### Step 3: Training

Run training with real models:

```bash
python train_stargan_real.py \
    --amadeus_config /path/to/Amadeus/config.yaml \
    --amadeus_checkpoint /path/to/Amadeus/checkpoint.pt \
    --moonbeam_config /path/to/Moonbeam/config.json \
    --moonbeam_checkpoint /path/to/Moonbeam/checkpoint.pt \
    --data_dir /path/to/training/data \
    --batch_size 16 \
    --num_epochs 10 \
    --g_lr 1e-4 \
    --d_lr 1e-4 \
    --lambda_cls 1.0 \
    --lambda_rec 10.0 \
    --lambda_gp 10.0 \
    --n_critic 5 \
    --temperature 0.5 \
    --save_dir ./checkpoints \
    --device cuda
```

## Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_size` | 16 | Batch size |
| `--num_epochs` | 10 | Number of training epochs |
| `--g_lr` | 1e-4 | Generator learning rate |
| `--d_lr` | 1e-4 | Discriminator learning rate |
| `--lambda_cls` | 1.0 | Domain classification loss weight |
| `--lambda_rec` | 10.0 | Cycle consistency loss weight |
| `--lambda_gp` | 10.0 | Gradient penalty weight (WGAN-GP) |
| `--n_critic` | 5 | Discriminator updates per Generator update |
| `--temperature` | 0.5 | Gumbel-Softmax temperature |

## Loss Functions

### Discriminator Loss
```
L_D = L_real + L_fake + λ_cls * L_cls_real + λ_gp * L_gp
```
- `L_real`: Real samples classified as real
- `L_fake`: Fake samples classified as fake
- `L_cls_real`: Domain classification on real samples
- `L_gp`: Gradient penalty (WGAN-GP)

### Generator Loss
```
L_G = L_adv + λ_cls * L_cls_fake + λ_rec * L_rec
```
- `L_adv`: Adversarial loss (fool Discriminator)
- `L_cls_fake`: Domain classification on fake samples
- `L_rec`: Cycle consistency (reconstruction loss)

## Token Format Conversion

### Amadeus → Moonbeam

Amadeus tokens [8 features]:
```
[type, beat, chord, tempo, instrument, pitch, duration, velocity]
```

Moonbeam tokens [6 features]:
```
[onset, duration, octave, pitch_class, instrument, velocity]
```

Conversion:
- `onset = beat` (simplified, can use tempo for better accuracy)
- `duration = duration`
- `octave = pitch // 12`
- `pitch_class = pitch % 12`
- `instrument = instrument`
- `velocity = velocity`

## Implementation Details

### 1. Gumbel-Softmax Sampling

```python
soft_probs = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)
```
- Forward pass: Discrete (one-hot via argmax)
- Backward pass: Continuous (soft probabilities)
- Enables gradient flow through discrete sampling

### 2. Soft Embeddings Projection

Generator outputs soft embeddings that Discriminator can process:
```python
soft_emb = self.soft_embedders[feature_name](soft_probs)
soft_embeddings = torch.cat(soft_embeddings_list, dim=-1)
```

Discriminator projects them to Moonbeam's hidden dimension:
```python
projected = self.soft_projection(soft_embeddings)
```

### 3. Dual Input Support

Discriminator handles both discrete tokens (Real) and soft embeddings (Fake):
```python
if input_ids is not None:
    inputs_embeds = self.embed_discrete_tokens(input_ids)
elif soft_embeddings is not None:
    inputs_embeds = self.embed_soft_embeddings(soft_embeddings)
```

## Troubleshooting

### Issue 1: Import Errors
**Problem**: `ModuleNotFoundError: No module named 'Amadeus'`

**Solution**: Check sys.path additions in wrapper files:
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../Amadeus'))
```

### Issue 2: Checkpoint Loading Errors
**Problem**: `Missing keys` or `Unexpected keys` when loading checkpoints

**Solution**: This is expected when adding new classification heads to Moonbeam. The wrapper handles this with `strict=False`.

### Issue 3: Dimension Mismatch
**Problem**: `RuntimeError: mat1 and mat2 shapes cannot be multiplied`

**Solution**: Verify `_get_amadeus_embed_dim()` matches your actual Amadeus configuration. Update the value based on your Amadeus embedding dimensions.

### Issue 4: CUDA Out of Memory
**Solution**: Reduce batch size or sequence length:
```bash
python train_stargan_real.py --batch_size 8 ...
```

## Model Compatibility

### Amadeus Configuration
Tested with Amadeus models using:
- Encoding scheme: CP (Compound Word)
- Features: 8 (type, beat, chord, tempo, instrument, pitch, duration, velocity)
- Input length: 512-2048 tokens

### Moonbeam Configuration
Tested with Moonbeam models using:
- Architecture: LlamaForSequenceClassification
- Features: 6 (onset, duration, octave, pitch_class, instrument, velocity)
- Hidden size: 1920
- Layers: 15

## Next Steps

1. **Implement Dataset Loader**: Create `StarGANDataset` class for MidiCaps data
2. **Tune Hyperparameters**: Experiment with learning rates, lambda weights, temperature
3. **Add Evaluation Metrics**: Implement FID, domain classification accuracy, reconstruction quality
4. **Multi-GPU Training**: Add DDP/FSDP support for large-scale training
5. **Inference Pipeline**: Create generation script for music arrangement transfer

## References

- **StarGAN**: [Choi et al., 2018 - StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/abs/1711.09020)
- **Gumbel-Softmax**: [Jang et al., 2017 - Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144)
- **Amadeus**: Your Amadeus model repository
- **Moonbeam**: Your Moonbeam model repository

## Citation

If you use this code, please cite:
```
@misc{stargan-music,
  title={StarGAN for Music Arrangement with Amadeus and Moonbeam},
  author={Your Name},
  year={2026}
}
```

## Contact

For questions or issues, please open an issue in the repository.
