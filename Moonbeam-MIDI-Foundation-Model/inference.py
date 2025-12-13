import torch
import numpy as np
from pathlib import Path
from transformers import LlamaConfig, LlamaForSequenceClassification
from peft import PeftModel, LoraConfig
from typing import List, Dict

class MusicEmotionClassifier:
    """Moonbeam事前学習モデル + LoRAで感情分類を行うクラス"""
    
    def __init__(
        self,
        pretrained_checkpoint: str = "models/pretrained/moonbeam_839M.pt",
        lora_adapter_path: str = "models/emotion_classification-v1",
        config_path: str = "src/llama_recipes/configs/player_classification_config.json",
        device: str = "cuda"
    ):
        self.device = device
        self.config_path = config_path
        self.pretrained_checkpoint = pretrained_checkpoint
        self.lora_adapter_path = lora_adapter_path
        
        # モデルとトークナイザーを初期化
        self._load_model()
        self._load_config()
    
    def _load_model(self):
        """モデルの読み込み"""
        print(f"Loading model from {self.config_path}")
        
        # 1. 設定ファイルを読み込み
        llama_config = LlamaConfig.from_pretrained(self.config_path)
        llama_config.use_cache = False
        
        print(f"Model config: {llama_config}")
        print(f"Number of labels: {llama_config.num_labels}")
        
        # 2. 分類モデルを作成
        self.model = LlamaForSequenceClassification(llama_config)
        
        # 3. 事前学習済み重みを読み込み
        print(f"Loading pretrained weights from {self.pretrained_checkpoint}")
        checkpoint = torch.load(self.pretrained_checkpoint, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 'module.'プレフィックスを除去
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        # strict=False: 分類ヘッドは新規なので不一致を許容
        missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
        
        print(f"Missing keys (分類ヘッド): {len(missing_keys)}")
        print(f"Unexpected keys: {len(unexpected_keys)}")
        
        # 4. LoRAアダプターを読み込み
        if Path(self.lora_adapter_path).exists():
            print(f"Loading LoRA adapter from {self.lora_adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, self.lora_adapter_path)
            self.model = self.model.merge_and_unload()
            print("LoRA adapter loaded and merged")
        else:
            print(f"Warning: LoRA adapter not found at {self.lora_adapter_path}")
        
        # 5. 評価モードに設定
        self.model.eval()
        self.model.to(self.device)
        
        print(f"Model loaded on {self.device}")
    
    def _load_config(self):
        """設定ファイルから語彙サイズなどを読み込み"""
        llama_config = LlamaConfig.from_pretrained(self.config_path)
        
        # 各特徴量の語彙サイズ
        self.onset_vocab_size = llama_config.onset_vocab_size  # 128
        self.dur_vocab_size = llama_config.dur_vocab_size  # 128
        self.octave_vocab_size = llama_config.octave_vocab_size  # 11
        self.pitch_class_vocab_size = llama_config.pitch_class_vocab_size  # 12
        self.instrument_vocab_size = llama_config.instrument_vocab_size  # 129
        self.velocity_vocab_size = llama_config.velocity_vocab_size  # 128
        
        self.classification_token = llama_config.classification_token  # 3
        self.pad_token = llama_config.pad_token  # 0
        
        print(f"\n=== Vocabulary Sizes ===")
        print(f"Onset: {self.onset_vocab_size}")
        print(f"Duration: {self.dur_vocab_size}")
        print(f"Octave: {self.octave_vocab_size}")
        print(f"Pitch Class: {self.pitch_class_vocab_size}")
        print(f"Instrument: {self.instrument_vocab_size}")
        print(f"Velocity: {self.velocity_vocab_size}")
        print(f"Classification token: {self.classification_token}")
        print(f"Pad token: {self.pad_token}")
    
    def normalize_onset_tokens(self, tokens: np.ndarray) -> np.ndarray:
        """
        onset値を相対時間（差分）に変換し、語彙サイズ内に収める
        generation.pyの処理と同様に、累積時間を差分に変換
        
        Args:
            tokens: [seq_len, 6] shape のトークン配列
        
        Returns:
            normalized_tokens: 正規化後のトークン配列
        """
        normalized_tokens = tokens.copy()
        
        # onset列（0列目）を差分に変換
        onset_values = tokens[:, 0].astype(np.float32)
        
        if len(onset_values) == 0:
            return normalized_tokens
        
        # 累積時間を差分に変換（generation.pyと同様の処理）
        # previous_onset + onset_diff = current_onset
        # onset_diff = current_onset - previous_onset
        onset_diff = np.diff(onset_values, prepend=onset_values[0])
        
        # 最初のonsetは0からの相対時間とする
        if onset_values[0] > 0:
            onset_diff[0] = onset_values[0]
        
        # 負の差分を0にする（時間は逆戻りしない）
        onset_diff = np.maximum(onset_diff, 0)
        
        # 語彙サイズ内に収める
        # 対数スケールでビニング（大きな時間差をより細かく表現）
        max_onset_diff = onset_diff.max()
        if max_onset_diff > self.onset_vocab_size - 1:
            # 対数スケールでビニング
            onset_diff_log = np.log1p(onset_diff)  # log(1+x)
            onset_diff_log_max = onset_diff_log.max()
            if onset_diff_log_max > 0:
                onset_diff_normalized = (onset_diff_log / onset_diff_log_max * (self.onset_vocab_size - 1))
                normalized_tokens[:, 0] = onset_diff_normalized.astype(np.int32)
            else:
                normalized_tokens[:, 0] = 0
        else:
            # すでに範囲内の場合はそのまま使用
            normalized_tokens[:, 0] = onset_diff.astype(np.int32)
        
        print(f"Onset normalization: [{tokens[:, 0].min()}, {tokens[:, 0].max()}] "
            f"-> [{normalized_tokens[:, 0].min()}, {normalized_tokens[:, 0].max()}]")
        
        return normalized_tokens
    
    def validate_and_clip_tokens(self, tokens: np.ndarray, skip_onset: bool = False) -> np.ndarray:
        """
        トークンの値を検証して語彙サイズ内にクリッピング
        
        Args:
            tokens: [seq_len, 6] shape のトークン配列
            skip_onset: onset列のクリッピングをスキップするか
        
        Returns:
            clipped_tokens: クリッピング後のトークン配列
        """
        vocab_sizes = [
            self.onset_vocab_size,
            self.dur_vocab_size,
            self.octave_vocab_size,
            self.pitch_class_vocab_size,
            self.instrument_vocab_size,
            self.velocity_vocab_size
        ]
        
        feature_names = ["onset", "duration", "octave", "pitch_class", "instrument", "velocity"]
        
        clipped_tokens = tokens.copy()
        
        for i, (vocab_size, feature_name) in enumerate(zip(vocab_sizes, feature_names)):
            # onset列はnormalize_onset_tokensで処理するのでスキップ
            if i == 0 and skip_onset:
                continue
                
            # 各特徴量の最大値・最小値を確認
            max_val = tokens[:, i].max()
            min_val = tokens[:, i].min()
            
            if max_val >= vocab_size or min_val < 0:
                print(f"WARNING: {feature_name} values out of range!")
                print(f"  Range: [{min_val}, {max_val}], Expected: [0, {vocab_size-1}]")
                
                # クリッピング
                clipped_tokens[:, i] = np.clip(tokens[:, i], 0, vocab_size - 1)
                
                n_clipped = np.sum((tokens[:, i] < 0) | (tokens[:, i] >= vocab_size))
                print(f"  Clipped {n_clipped} tokens")
        
        return clipped_tokens
    
    def npy_to_tokens(self, npy_path: str, max_length: int = 1203) -> torch.Tensor:
        """
        npyファイル（前処理済みトークン）を読み込んでテンソルに変換
        
        Args:
            npy_path: npyファイルのパス
            max_length: 最大シーケンス長
        
        Returns:
            tokens: [1, seq_len, 6] shape のトークンテンソル
        """
        # npyファイルを読み込み
        tokens = np.load(npy_path)  # [seq_len, 6]
        
        print(f"\n=== Loading NPY File ===")
        print(f"File: {npy_path}")
        print(f"Original shape: {tokens.shape}")
        print(f"Original onset range: [{tokens[:, 0].min()}, {tokens[:, 0].max()}]")
        
        # データ形式の確認
        if tokens.ndim != 2 or tokens.shape[1] != 6:
            raise ValueError(f"Expected shape [seq_len, 6], got {tokens.shape}")
        
        # onset値を相対時間（差分）に正規化
        tokens = self.normalize_onset_tokens(tokens)
        
        # その他の列をクリッピング（onset列はスキップ）
        tokens = self.validate_and_clip_tokens(tokens, skip_onset=True)
        
        # 分類トークンを追加（シーケンスの最後）
        cls_token = np.array([[
            self.classification_token,  # onset
            0, 0, 0, 0, 0  # dur, octave, pitch_class, instrument, velocity
        ]], dtype=tokens.dtype)
        
        tokens = np.vstack([tokens, cls_token])
        
        # パディングまたはトリミング
        if len(tokens) < max_length:
            # パディング
            pad_length = max_length - len(tokens)
            pad_tokens = np.zeros((pad_length, 6), dtype=tokens.dtype)
            pad_tokens[:, 0] = self.pad_token  # onset列のみpad_token
            tokens = np.vstack([tokens, pad_tokens])
            print(f"\nPadded to {max_length} tokens")
        else:
            # トリミング（分類トークンは保持）
            tokens = np.vstack([tokens[:max_length-1], cls_token])
            print(f"\nTrimmed to {max_length} tokens")
        
        # 最終確認
        print(f"\nFinal token shape: {tokens.shape}")
        print(f"Final onset range: [{tokens[:, 0].min()}, {tokens[:, 0].max()}]")
        
        # Tensorに変換 [1, seq_len, 6]
        tokens = torch.from_numpy(tokens).long().unsqueeze(0)
        
        return tokens
    
    def npy_to_tokens_chunked(self, npy_path: str, chunk_length: int = 1024) -> List[torch.Tensor]:
        """
        npyファイルを非オーバーラップでチャンク化
        onset値を相対時間に正規化してから分割
        
        Args:
            npy_path: npyファイルのパス
            chunk_length: チャンクの最大長
        
        Returns:
            chunks: チャンクのリスト
        """
        tokens = np.load(npy_path)
        
        if tokens.ndim != 2 or tokens.shape[1] != 6:
            raise ValueError(f"Expected shape [seq_len, 6], got {tokens.shape}")
        
        print(f"Original onset range: [{tokens[:, 0].min()}, {tokens[:, 0].max()}]")
        
        # onset値を相対時間（差分）に正規化
        tokens = self.normalize_onset_tokens(tokens)
        
        # その他の列をクリッピング（onset列はスキップ）
        tokens = self.validate_and_clip_tokens(tokens, skip_onset=True)
        
        chunks = []
        seq_len = len(tokens)
        
        print(f"Splitting into chunks (length={chunk_length}, total_length={seq_len})")
        
        # 非オーバーラップで分割（stride = chunk_length）
        for start_idx in range(0, seq_len, chunk_length):
            end_idx = min(start_idx + chunk_length, seq_len)
            chunk = tokens[start_idx:end_idx].copy()
            
            # 各チャンクの先頭onsetを0にリセット（相対時間として扱う）
            if len(chunk) > 0 and chunk[0, 0] > 0:
                first_onset = chunk[0, 0]
                chunk[:, 0] = np.maximum(chunk[:, 0] - first_onset, 0).astype(np.int32)
            
            # 分類トークンを追加
            cls_token = np.array([[self.classification_token, 0, 0, 0, 0, 0]], dtype=chunk.dtype)
            chunk = np.vstack([chunk, cls_token])
            
            # パディング
            if len(chunk) < chunk_length + 1:
                pad_length = chunk_length + 1 - len(chunk)
                pad_tokens = np.zeros((pad_length, 6), dtype=chunk.dtype)
                pad_tokens[:, 0] = self.pad_token
                chunk = np.vstack([chunk, pad_tokens])
            
            chunk_tensor = torch.from_numpy(chunk).long().unsqueeze(0)
            chunks.append(chunk_tensor)
        
        print(f"Created {len(chunks)} chunks")
        
        return chunks
    
    def predict(
        self, 
        npy_path: str, 
        return_probabilities: bool = False,
        chunk_length: int = 1024
    ) -> Dict:
        """
        npyファイルから感情分類を予測（チャンキング対応）
        
        Args:
            npy_path: npyファイルのパス
            return_probabilities: 確率分布を返すか
            chunk_length: チャンクの最大長
        
        Returns:
            result: 予測結果の辞書
        """
        # トークン化
        tokens = np.load(npy_path)
        
        # シーケンス長をチェック
        if len(tokens) <= chunk_length:
            # 短い場合は従来の方法
            tokens_tensor = self.npy_to_tokens(npy_path, max_length=chunk_length+1)
            tokens_tensor = tokens_tensor.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids=tokens_tensor)
                logits = outputs.logits
            
            num_chunks = 1
        else:
            # 長い場合はチャンク化して平均
            chunks = self.npy_to_tokens_chunked(npy_path, chunk_length)
            
            all_logits = []
            for chunk in chunks:
                chunk = chunk.to(self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids=chunk)
                    all_logits.append(outputs.logits)
            
            # logitsの平均を取る
            logits = torch.mean(torch.stack(all_logits), dim=0)
            num_chunks = len(chunks)
        
        # 予測クラス
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        predicted_class = int(torch.argmax(logits, dim=-1).cpu().item())
        confidence = float(probabilities[predicted_class])
        
        emotion_labels = {
            0: "Happy (Q1)",
            1: "Sad (Q2)",
            2: "Angry (Q3)",
            3: "Relaxed (Q4)"
        }
        
        predicted_label = emotion_labels.get(predicted_class, f"Class_{predicted_class}")
        
        result = {
            'predicted_class': predicted_class,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'num_chunks': num_chunks
        }
        
        if return_probabilities:
            result['probabilities'] = probabilities
            result['all_labels'] = emotion_labels
        
        return result
    
    def predict_batch(
        self, 
        npy_paths: List[str], 
        batch_size: int = 8
    ) -> List[Dict]:
        """
        複数のnpyファイルをバッチ処理
        """
        results = []
        
        for i in range(0, len(npy_paths), batch_size):
            batch_paths = npy_paths[i:i+batch_size]
            
            # バッチトークン化
            batch_tokens = []
            for path in batch_paths:
                try:
                    tokens = self.npy_to_tokens(path)
                    batch_tokens.append(tokens)
                except Exception as e:
                    print(f"Error processing {path}: {e}")
                    continue
            
            if len(batch_tokens) == 0:
                continue
            
            batch_tokens = torch.cat(batch_tokens, dim=0).to(self.device)
            
            # バッチ推論
            with torch.no_grad():
                outputs = self.model(input_ids=batch_tokens)
                logits = outputs.logits
            
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            
            # 感情ラベル
            emotion_labels = {
                0: "Happy (Q1)",
                1: "Sad (Q2)",
                2: "Angry (Q3)",
                3: "Relaxed (Q4)"
            }
            
            # 結果を格納
            for j, path in enumerate(batch_paths):
                result = {
                    'npy_path': path,
                    'predicted_class': int(predictions[j]),
                    'predicted_label': emotion_labels.get(int(predictions[j]), f"Class_{predictions[j]}"),
                    'confidence': float(probabilities[j, predictions[j]]),
                    'probabilities': probabilities[j]
                }
                results.append(result)
        
        return results


def main():
    """使用例"""
    import pandas as pd
    from pathlib import Path
    
    # 分類器を初期化
    classifier = MusicEmotionClassifier(
        pretrained_checkpoint="models/pretrained/moonbeam_839M.pt",
        lora_adapter_path="models/emotion_classification-v1",
        config_path="src/llama_recipes/configs/player_classification_config.json",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # === テストデータで評価 ===
    print("\n" + "="*60)
    print("Test Data Evaluation")
    print("="*60)
    
    # CSVファイルを読み込み
    csv_path = "processed_datasets/classification/emopia2.2_1071_clips/train_test_split.csv"
    df = pd.read_csv(csv_path)
    
    # testデータのみを抽出
    test_df = df[df['split'] == 'train'].reset_index(drop=True)
    
    print(f"\nTotal test samples: {len(test_df)}")
    print(f"Label distribution:")
    print(test_df['label'].value_counts().sort_index())
    
    # npyファイルのベースパス
    base_path = Path("processed_datasets/classification/emopia2.2_1071_clips/processed")
    
    # 推論結果を格納
    predictions = []
    true_labels = []
    failed_files = []
    
    # 各テストサンプルに対して推論
    print(f"\n{'='*60}")
    print("Running inference on test samples...")
    print(f"{'='*60}\n")
    
    for idx, row in test_df.iterrows():
        file_name = row['file_base_name']
        true_label = row['label']
        npy_path = base_path / file_name
        
        # ファイルの存在確認
        if not npy_path.exists():
            print(f"[{idx+1}/{len(test_df)}] File not found: {file_name}")
            failed_files.append(file_name)
            continue
        
        try:
            result = classifier.predict(str(npy_path), return_probabilities=False)
            predicted_class = result['predicted_class']
            num_chunks = result.get('num_chunks', 1)

            predictions.append(predicted_class)
            true_labels.append(true_label)

            # 進捗表示
            match_symbol = "✓" if predicted_class == true_label else "✗"
            if (idx + 1) % 10 == 0 or predicted_class != true_label:
                print(f"[{idx+1}/{len(test_df)}] {match_symbol} {file_name[:40]:40s} "
                    f"Pred: {predicted_class}, True: {true_label}, Chunks: {num_chunks}")
    
        except Exception as e:
            print(f"[{idx+1}/{len(test_df)}] Error: {file_name} - {str(e)[:50]}")
            failed_files.append(file_name)
            continue
    
    # === 結果の集計 ===
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}\n")
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # 全体の精度
    accuracy = np.mean(predictions == true_labels)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Correct predictions: {np.sum(predictions == true_labels)}/{len(predictions)}")
    print(f"Failed files: {len(failed_files)}")
    
    # クラスごとの精度
    print(f"\n{'='*60}")
    print("Per-Class Performance")
    print(f"{'='*60}\n")
    
    emotion_labels = {
        0: "Happy (Q1)",
        1: "Sad (Q2)",
        2: "Angry (Q3)",
        3: "Relaxed (Q4)"
    }
    
    for label_id in range(4):
        mask = true_labels == label_id
        if np.sum(mask) == 0:
            continue
        
        class_predictions = predictions[mask]
        class_accuracy = np.mean(class_predictions == label_id)
        n_samples = np.sum(mask)
        n_correct = np.sum(class_predictions == label_id)
        
        print(f"{emotion_labels[label_id]:15s}: {class_accuracy:.4f} "
              f"({n_correct}/{n_samples} correct)")
    
    # 混同行列
    print(f"\n{'='*60}")
    print("Confusion Matrix")
    print(f"{'='*60}\n")
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, predictions)
    
    print("       ", end="")
    for i in range(4):
        print(f"Pred {i:1d}  ", end="")
    print()
    
    for i in range(4):
        print(f"True {i}: ", end="")
        for j in range(4):
            print(f"{cm[i,j]:6d}  ", end="")
        print()
    
    # 失敗したファイルのリスト
    if failed_files:
        print(f"\n{'='*60}")
        print(f"Failed Files ({len(failed_files)} files)")
        print(f"{'='*60}\n")
        for file in failed_files[:10]:  # 最初の10件のみ表示
            print(f"  - {file}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files)-10} more files")


if __name__ == "__main__":
    main()