# StarGAN Implementation Summary

## 完了した実装

### 1. モデルラッパー

#### Amadeus Generator Wrapper (`amadeus_generator_wrapper.py`)
- ✅ `AmadeusModelWrapper`の統合
- ✅ Gumbel-Softmaxサンプリング
- ✅ Soft Embeddings生成
- ✅ ドメイン条件付け (domain_embedding)
- ✅ チェックポイントローダー (`load_amadeus_generator`)
- ✅ トークン変換ユーティリティ (`amadeus_to_moonbeam_discrete`)

**モデルパス**: `Amadeus/Amadeus/model_zoo.py`

#### Moonbeam Discriminator Wrapper (`moonbeam_discriminator_wrapper.py`)
- ✅ `LlamaForSequenceClassification`の統合
- ✅ デュアル分類ヘッド (Real/Fake + Domain)
- ✅ デュアル入力サポート (discrete tokens + soft embeddings)
- ✅ Soft Embeddings投影層
- ✅ チェックポイントローダー (`load_moonbeam_discriminator`)

**モデルパス**: `Moonbeam-MIDI-Foundation-Model/src/llama_recipes/`

### 2. トレーニングスクリプト

#### メイントレーニング (`train_stargan_real.py`)
- ✅ `StarGANTrainer`クラス
- ✅ モデルロード統合
- ✅ トレーニングループ実装
- ✅ チェックポイント保存/ロード
- ✅ コマンドライン引数パーサー

#### 損失関数 (`stargan_losses.py`) - 既存使用
- ✅ Discriminator損失
- ✅ Generator損失 (Adversarial + Domain + Cycle)
- ✅ Gradient Penalty (WGAN-GP)

### 3. テストスクリプト

#### モデルテスト (`test_real_models.py`)
- ✅ モデルロードテスト
- ✅ Forward pass検証
- ✅ Gradient flow検証

#### 統合テスト (`test_e2e.py`) - 既存
- ダミーモデルでの動作確認済み

### 4. ドキュメント

- ✅ `README_REAL_MODELS.md` - 完全な使用方法ガイド
- ✅ `quick_start.sh` - クイックスタートスクリプト

## 実装の特徴

### End-to-End勾配流れ
```python
# Generator → Discriminator (勾配流れる!)
logits_dict, soft_embeddings, _ = G(input_seq, target_domain)
fake_src, fake_cls = D(soft_embeddings=soft_embeddings)  # detach()なし
loss.backward()  # DとGの両方に勾配が流れる
```

### Gumbel-Softmax with Straight-Through Estimator
```python
soft_probs = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)
# Forward: discrete (argmax)
# Backward: continuous (gradient flows)
```

### デュアル入力サポート
```python
# Real samples
real_src, real_cls = D(input_ids=moonbeam_tokens)

# Fake samples
fake_src, fake_cls = D(soft_embeddings=soft_embeddings)
```

## 使用方法

### 1. 準備

```bash
# 依存関係インストール
cd /mnt/kiso-qnap5/obara/Amadeus
pip install -r requirements.txt

cd /mnt/kiso-qnap5/obara/Moonbeam-MIDI-Foundation-Model
pip install -r requirements.txt

cd /mnt/kiso-qnap5/obara/StarGAN_music/StarGAN_music
pip install torch transformers pyyaml tqdm
```

### 2. モデルパスの設定

`test_real_models.py`の19-22行目を編集:
```python
amadeus_config = "/path/to/Amadeus/config.yaml"
amadeus_checkpoint = "/path/to/Amadeus/checkpoint.pt"
moonbeam_config = "/path/to/Moonbeam/config.json"
moonbeam_checkpoint = "/path/to/Moonbeam/checkpoint.pt"
```

### 3. テスト実行

```bash
cd /mnt/kiso-qnap5/obara/StarGAN_music/StarGAN_music
python test_real_models.py
```

### 4. トレーニング実行

```bash
python train_stargan_real.py \
    --amadeus_config /path/to/config.yaml \
    --amadeus_checkpoint /path/to/checkpoint.pt \
    --moonbeam_config /path/to/config.json \
    --moonbeam_checkpoint /path/to/checkpoint.pt \
    --data_dir /path/to/data \
    --batch_size 16 \
    --num_epochs 10 \
    --save_dir ./checkpoints
```

## トークンフォーマット

### Amadeus [8 features]
```
[type, beat, chord, tempo, instrument, pitch, duration, velocity]
```

### Moonbeam [6 features]
```
[onset, duration, octave, pitch_class, instrument, velocity]
```

### 変換
```python
onset = beat
octave = pitch // 12
pitch_class = pitch % 12
```

## ハイパーパラメータ

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| g_lr | 1e-4 | Generator学習率 |
| d_lr | 1e-4 | Discriminator学習率 |
| lambda_cls | 1.0 | ドメイン分類損失の重み |
| lambda_rec | 10.0 | サイクル一貫性損失の重み |
| lambda_gp | 10.0 | 勾配ペナルティの重み |
| n_critic | 5 | Generator更新前のDiscriminator更新回数 |
| temperature | 0.5 | Gumbel-Softmax温度 |

## TODO (次のステップ)

1. **データセットローダー実装**
   - MidiCapsデータをロードする`StarGANDataset`クラス
   - データ形式: `[scores, target_labels, original_labels]`

2. **実際のモデルパスを確認**
   - Amadeusのconfig YAMLファイルパス
   - Amadeusのチェックポイントパス
   - Moonbeamのチェックポイントパス

3. **Embedding次元の調整**
   - `moonbeam_discriminator_wrapper.py`の`_get_amadeus_embed_dim()`
   - 実際のAmadeus設定に合わせて更新

4. **Moonbeam Embedding実装**
   - FME (Fourier Music Embedding)の実装
   - WE (Word Embedding)の実装
   - 現在はシンプルなembeddingを使用

5. **評価指標の追加**
   - FID score
   - Domain classification accuracy
   - Reconstruction quality

## ファイル構成

```
StarGAN_music/StarGAN_music/
├── amadeus_generator_wrapper.py       # Amadeus Generatorラッパー
├── moonbeam_discriminator_wrapper.py  # Moonbeam Discriminatorラッパー
├── stargan_losses.py                  # 損失関数
├── train_stargan_real.py              # トレーニングスクリプト
├── test_real_models.py                # モデルテスト
├── README_REAL_MODELS.md              # 詳細ドキュメント
├── quick_start.sh                     # クイックスタート
└── IMPLEMENTATION_SUMMARY.md          # このファイル
```

## 技術詳細

### 勾配フロー
```
Input → Generator (Amadeus)
         ↓ (soft embeddings)
         Discriminator (Moonbeam)
         ↓ (loss)
         Backward
         ↓
         ∇L/∂G ← ∇L/∂D (勾配が両方に流れる!)
```

### モデルサイズ
- **Amadeus Generator**: 約50-100M parameters (configによる)
- **Moonbeam Discriminator**: 約200-300M parameters

### メモリ要件
- 16GB VRAM (batch_size=16, seq_len=512)
- 32GB VRAM推奨 (batch_size=32)

## トラブルシューティング

### Import Error
```python
# sys.pathが正しく設定されているか確認
sys.path.insert(0, '/mnt/kiso-qnap5/obara/Amadeus')
sys.path.insert(0, '/mnt/kiso-qnap5/obara/Moonbeam-MIDI-Foundation-Model/src')
```

### CUDA Out of Memory
```bash
# バッチサイズを減らす
python train_stargan_real.py --batch_size 8 ...
```

### Checkpoint Loading Error
```python
# strict=Falseで新しい分類ヘッドを許可
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
```

## 参考文献

1. **StarGAN**: Choi et al., 2018
2. **Gumbel-Softmax**: Jang et al., 2017
3. **WGAN-GP**: Gulrajani et al., 2017

## 連絡先

問題や質問がある場合は、GitHubでissueを開いてください。
