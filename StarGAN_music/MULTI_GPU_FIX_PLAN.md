# GPU複数枚対応 - メモリ不足エラー解決計画

## 問題分析
- **エラー**: `torch.OutOfMemoryError` on GPU 0
- **原因**: DiffusionDecoderのforward処理でGPU 0メモリが満杯
- **現状**: 単一GPU（GPU 0）で全モデルが実行中

## 修正計画（段階別）

### フェーズ1: マルチGPU初期化
**対象ファイル**: `train_stargan_real.py`

1.1 **GPU設定の自動検出**
   - 利用可能なGPU数を確認
   - GPUメモリ容量を確認
   - 適切なデバイスリストを作成

1.2 **StarGANTrainerクラスへマルチGPU対応パラメータを追加**
   - `use_data_parallel: bool` (デフォルト: True)
   - `gpu_ids: list` (使用するGPU ID)
   - `batch_size_per_gpu: int` (自動調整機能)

### フェーズ2: モデルのマルチGPU化
**対象**: G（AmadeusModel）, D（LlamaForSequenceClassification）, 補助層

2.1 **nn.DataParallelでラップ**
   ```
   G = nn.DataParallel(G, device_ids=gpu_ids)
   D = nn.DataParallel(D, device_ids=gpu_ids)
   ```

2.2 **補助レイヤーの配置**
   ```
   projection_layer → nn.DataParallel
   embedding_layers → nn.DataParallel
   ```

2.3 **T5エンコーダーの処理**
   - T5エンコーダーはGPU 0に固定（推論用）
   - キャッシング機能を追加

### フェーズ3: 学習ループの修正
**対象**: `train_step()` 方法, `train()` 方法

3.1 **ロス計算の平均化**
   - DataParallelの出力から正しくロスを取得
   - バッチスプリット時のロス平均化

3.2 **グラディエント集約**
   - 複数GPUからのグラディエント同期
   - Optimizerの正しい更新

3.3 **メモリ効率化**
   - gradient checkpointingの導入
   - activation checkpointingの実装

### フェーズ4: バッチサイズと学習パラメータの調整
**対象**: コマンドライン引数, config設定

4.1 **動的バッチサイズ計算**
   ```
   total_batch_size = batch_size_per_gpu * num_gpus
   ```

4.2 **学習率の調整**
   ```
   lr_adjusted = base_lr * (total_batch_size / reference_batch_size)
   ```

### フェーズ5: メモリ監視とロギング
**対象**: `train_step()`, `train()`

5.1 **GPU メモリ使用量の追跡**
   - 各ステップで利用可能なメモリを確認
   - ログに記録

5.2 **エラーハンドリング**
   - CUDA OOM時の自動リカバリ
   - グレースフルフォールバック

## 実装順序
1. ✓ GPU検出・初期化コード（フェーズ1）
2. ✓ モデルのDataParallel化（フェーズ2）
3. ✓ train_step修正（フェーズ3）
4. ✓ パラメータ調整（フェーズ4）
5. ✓ メモリ監視（フェーズ5）

## 期待される改善
- **メモリ効率**: 複数GPU間でメモリ負荷を分散
- **スループット**: 複数GPU並列処理で高速化
- **安定性**: OOMエラーの回避、エラーハンドリング

## 代替案（段階的対応）
- **代替案A**: バッチサイズを減らす（クイックフィックス）
- **代替案B**: Gradient checkpointingのみ導入
- **代替案C**: モデル並列化（DataParallelより複雑）
