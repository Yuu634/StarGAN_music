# End-to-End Differentiable StarGAN for Music Arrangement

改訂版計画書に基づいた、Generator-Discriminator間で勾配が流れるStarGAN実装

## 📁 実装ファイル

### Phase 1: 基盤実装
- ✅ `amadeus_stargan.py` - AmadeusForStarGAN Generator
  - Gumbel-Softmax sampling実装
  - Soft embeddings生成
  - ドメイン条件付け機能
  
- ✅ `llama_discriminator.py` - LlamaForSequenceDoubleClassification
  - Discrete/Soft両入力対応
  - Real/Fake + Domain分類
  - Soft embedding projection

### Phase 2: 損失関数
- ✅ `stargan_losses.py` - 損失関数群
  - `compute_discriminator_loss()` - 勾配が流れる版
  - `compute_generator_loss()` - Adversarial + Cycle loss
  - `check_gradient_flow()` - 勾配フロー検証
  - トークン変換関数

### Phase 3: 学習ループ
- ✅ `solver.py` - 改訂版学習ループ
  - `train_stargan_e2e()` メソッド追加
  - End-to-End微分可能学習
  - 勾配クリッピング・ログ機能

### Phase 4: デバッグ・検証
- ✅ `test_utils.py` - テストユーティリティ
  - 勾配フローテスト
  - Gumbel-Softmax温度テスト
  - トークン変換検証
  - サニティチェック

- ✅ `test_e2e.py` - 統合テストスクリプト
  - 基本forward pass
  - 損失計算
  - 勾配フロー検証

## 🚀 使い方

### 1. テストの実行

```bash
cd /mnt/kiso-qnap5/obara/StarGAN_music/StarGAN_music
python test_e2e.py
```

**期待される出力:**
```
TEST 1: Basic Forward Pass
✓ Generator output: soft_embeddings=[2, 64, 512]
✓ Discriminator output: real_fake=[2, 64, 2], domain=[2, 64, 108]

TEST 2: Loss Computation
✓ D loss: 15.2341
✓ G loss: 23.4567

TEST 3: Backward Pass & Gradient Flow
✓ Gradient flow from D to G is working!

✓ ALL TESTS PASSED!
```

### 2. 学習の開始

main.pyを修正:
```python
from solver import Solver

# Solver初期化
solver = Solver(score_loader, config)

# End-to-End学習を実行
solver.train_stargan_e2e()
```

### 3. 設定パラメータ

重要なハイパーパラメータ:

```python
# 学習率
g_lr = 1e-4  # Generator
d_lr = 1e-4  # Discriminator

# 損失重み
lambda_cls = 1.0   # Domain classification
lambda_rec = 10.0  # Cycle consistency
lambda_gp = 10.0   # Gradient penalty

# Gumbel-Softmax
temperature = 0.5  # 温度パラメータ (0.1~2.0)

# 学習設定
n_critic = 5  # D更新頻度
batch_size = 16
num_iters = 200000
```

## 🔑 重要な技術的特徴

### 1. Gumbel-Softmax Sampling
```python
soft_probs = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)
```
- `hard=True`: Straight-Through Estimator
- forward: one-hot (離散)
- backward: soft gradient (連続)

### 2. 勾配フロー
```
Real Score → G(target) → Soft Embeddings → D → Loss
                ↑                             ↑
                └───────── 勾配が流れる ────────┘
```

### 3. Dual Classification
- **Real/Fake判定**: `[B, T, 2]`
- **Domain分類**: `[B, T, 108]`

### 4. Cycle Consistency
```
Original → G(target) → Fake → G(original) → Reconstruct
Loss = CrossEntropy(Reconstruct, Original)
```

## 📊 データフォーマット

### Generator入力/出力
- 入力: `[B, T, 8]` Amadeus format (discrete)
- 出力: `[B, T, dim]` Soft embeddings (continuous)
- ドメイン: `[B, 108]` Multi-hot labels

### Discriminator入力
- Real: `[B, T, 6]` Moonbeam format (discrete)
- Fake: `[B, T, dim]` Soft embeddings (continuous)

### トークン変換
Amadeus → Moonbeam:
```
(type, beat, chord, tempo, instrument, pitch, duration, velocity)
           ↓
(onset, duration, octave, pitch_class, instrument, velocity)
```

## 🐛 トラブルシューティング

### 勾配が流れない
```python
# test_e2e.pyで確認
python test_e2e.py

# 勾配ノルムをチェック
from stargan_losses import check_gradient_flow
check_gradient_flow(G, "Generator")
check_gradient_flow(D, "Discriminator")
```

### NaN/Inf損失
- 勾配クリッピング有効化 (実装済み)
- 学習率を下げる (`1e-5` ~ `1e-4`)
- Gumbel-Softmax温度を上げる (`1.0` ~ `2.0`)

### メモリ不足
- Batch sizeを減らす
- Gradient checkpointing有効化
- Mixed precision (AMP) 使用

## 📈 期待される効果

### 従来版 (detach使用)
- ❌ Gの勾配がD経由で流れない
- ❌ DはFakeの「結果」のみ評価
- ❌ 情報共有が限定的

### End-to-End版 (改訂版)
- ✅ Gの勾配がD経由で流れる
- ✅ DはFakeの「生成過程」も考慮
- ✅ G-D協調学習
- ✅ より高品質な編曲生成

## 📝 次のステップ

1. **小規模テスト**: 10サンプルで動作確認
2. **勾配監視**: TensorBoardで勾配ノルム可視化
3. **温度調整**: Gumbel-Softmax温度の最適値探索
4. **本格学習**: 全データセットで学習開始

## 🔗 関連ファイル

- 計画書: `StarGAN音楽編曲モデル 実装計画書（改訂版）.md`
- 元のStarGAN: `/mnt/kiso-qnap5/obara/StarGAN/solver.py`
- Amadeus: `/mnt/kiso-qnap5/obara/Amadeus/Amadeus/model_zoo.py`
- Moonbeam: `/mnt/kiso-qnap5/obara/Moonbeam-MIDI-Foundation-Model/`

---

**実装完了日**: 2026年1月2日  
**Status**: ✅ Phase 1-4 全完了、テスト準備完了
