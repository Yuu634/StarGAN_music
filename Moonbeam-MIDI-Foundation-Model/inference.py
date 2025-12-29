import torch
import numpy as np
from pathlib import Path
from transformers import LlamaConfig, LlamaForSequenceClassification
from peft import PeftModel, LoraConfig
from typing import List, Dict
import sys
from src.llama_recipes.real_finetuning_player_classification import LlamaForSequenceDoubleClassification

class ScoreArrangeDomainClassifier:
    """Moonbeam事前学習モデル + LoRAで編曲ドメイン分類"""
    
    def __init__(
        self,
        pretrained_checkpoint: str = "models/pretrained/moonbeam_839M.pt",
        lora_adapter_path: str = "models/emotion_classification-v1",
        config_path: str = "src/llama_recipes/configs/player_classification_config.json",
        device: str = "cuda",
        selected_attr: List[str] = None,
    ):
        self.device = device
        self.config_path = config_path
        self.pretrained_checkpoint = pretrained_checkpoint
        self.lora_adapter_path = lora_adapter_path
        self.selected_attr = selected_attr
        
        # モデルとトークナイザーを初期化
        self._load_model()
        self._load_config()
    
    def _load_model(self):
        """モデルの読み込み"""
        print(f"Loading model from {self.config_path}")
        
        # 1. 設定ファイルを読み込み
        llama_config = LlamaConfig.from_pretrained(self.config_path)
        llama_config.use_cache = False
        if self.selected_attr:
            llama_config.num_labels = len(self.selected_attr)
        
        print(f"Model config: {llama_config}")
        print(f"Number of labels: {llama_config.num_labels}")
        
        # 2. 分類モデルを作成
        self.model = LlamaForSequenceDoubleClassification(llama_config)
        #self.model = LlamaForSequenceClassification(llama_config)
        
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
    
    def npy_to_tokens(self, tokens: np.ndarray, max_length: int = 1024) -> torch.Tensor:
        """
        npyファイル（前処理済みトークン）を読み込んでテンソルに変換
        
        Args:
            tokens: npyファイル
            max_length: 最大シーケンス長
        
        Returns:
            tokens: [1, seq_len, 6] shape のトークンテンソル
        """
        print(f"\n=== Loading NPY File ===")
        print(f"Original shape: {tokens.shape}")
        print(f"Original onset range: [{tokens[:, 0].min()}, {tokens[:, 0].max()}]")
        
        # データ形式の確認
        if tokens.ndim != 2 or tokens.shape[1] != 6:
            raise ValueError(f"Expected shape [seq_len, 6], got {tokens.shape}")
        
        # onset値を相対時間（差分）に正規化
        #tokens = self.normalize_onset_tokens(tokens)
        
        # その他の列をクリッピング（onset列はスキップ）
        #tokens = self.validate_and_clip_tokens(tokens, skip_onset=True)
        
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
        tokens = torch.from_numpy(tokens).long()#.unsqueeze(0)
        
        return tokens
    
    def npy_to_tokens_chunked(self, tokens: np.ndarray, chunk_length: int = 1024, stride: int = 1024) -> List[torch.Tensor]:
        """
        npyファイルを非オーバーラップでチャンク化
        onset値を相対時間に正規化してから分割
        
        Args:
            tokens: npyファイル
            chunk_length: チャンクの最大長
            stride: スライディングウィンドウのストライド（Noneの場合はchunk_lengthと同じ=非オーバーラップ）
        
        Returns:
            chunks: チャンクのリスト
        """
        if tokens.ndim != 2 or tokens.shape[1] != 6:
            raise ValueError(f"Expected shape [seq_len, 6], got {tokens.shape}")
        
        print(f"Original onset range: [{tokens[:, 0].min()}, {tokens[:, 0].max()}]")
        
        # onset値を相対時間（差分）に正規化
        #tokens = self.normalize_onset_tokens(tokens)
        
        # その他の列をクリッピング（onset列はスキップ）
        #tokens = self.validate_and_clip_tokens(tokens, skip_onset=True)
        
        chunks = []
        seq_len = len(tokens)
        
        print(f"Splitting into chunks (length={chunk_length}, total_length={seq_len})")
        
        # スライディングウィンドウで分割
        start_idx = 0
        while start_idx < seq_len:
            end_idx = min(start_idx + chunk_length, seq_len)
            chunk = tokens[start_idx:end_idx].copy()
            
            # 各チャンクの先頭onsetを0にリセット（相対時間として扱う）
            #if len(chunk) > 0 and chunk[0, 0] > 0:
            #    first_onset = chunk[0, 0]
            #    chunk[:, 0] = np.maximum(chunk[:, 0] - first_onset, 0).astype(np.int32)
            
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
            
            # 次のウィンドウへ移動
            start_idx += stride
            
            # 最後のチャンクに到達したら終了
            if end_idx >= seq_len:
                break
        
        print(f"Created {len(chunks)} chunks")
        
        return chunks
    
    def predict(
        self, 
        tokens: np.ndarray,
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
        # シーケンス長をチェック
        if len(tokens) <= chunk_length:
            # 短い場合は従来の方法
            tokens_tensor = self.npy_to_tokens(tokens, max_length=chunk_length+1)
            tokens_tensor = tokens_tensor.to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids=tokens_tensor)
                logits = outputs.logits
                realfake_logits = outputs.real_fake_logits
            
            num_chunks = 1
        else:
            # 長い場合はチャンク化して平均
            chunks = self.npy_to_tokens_chunked(tokens, chunk_length)
            
            all_logits = []
            all_realfake_logits = []
            for chunk in chunks:
                chunk = chunk.to(self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids=chunk)
                    all_logits.append(outputs.logits)
                    all_realfake_logits.append(outputs.real_fake_logits)
            
            # logitsの平均を取る
            logits = torch.mean(torch.stack(all_logits), dim=0)
            realfake_logits = torch.mean(torch.stack(all_realfake_logits), dim=0)
            num_chunks = len(chunks)
        
        # 予測クラス
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        predicted_class = int(torch.argmax(logits, dim=-1).cpu().item())
        confidence = float(probabilities[predicted_class])
        
        probabilities_realfake = torch.softmax(realfake_logits, dim=-1).cpu().numpy()[0]
        predicted_class_realfake = int(torch.argmax(realfake_logits, dim=-1).cpu().item())
        confidence_realfake = float(probabilities_realfake[predicted_class_realfake])
        
        emotion_labels = {
            0: "Happy (Q1)",
            1: "Sad (Q2)",
            2: "Angry (Q3)",
            3: "Relaxed (Q4)"
        }
        
        predicted_label = emotion_labels.get(predicted_class, f"Class_{predicted_class}")
        predicted_label_realfake = "Real" if predicted_class_realfake == 1 else "Fake"
        
        result = {
            'predicted_class': predicted_class,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'predicted_class_realfake': predicted_class_realfake,
            'predicted_label_realfake': predicted_label_realfake,
            'confidence_label_realfake': confidence_realfake,
            'num_chunks': num_chunks
        }
        
        if return_probabilities:
            result['probabilities'] = probabilities
            result['all_labels'] = emotion_labels
        
        return result
    
    def predict_batch(
        self, 
        tokens_list: List[np.ndarray], 
        batch_size: int = 8
    ) -> List[Dict]:
        """
        複数のnpyファイルをバッチ処理
        """
        results = []
        
        for i in range(0, len(tokens_list), batch_size):
            batch_tokens_list = tokens_list[i:i+batch_size]
            
            # バッチトークン化
            batch_tokens = []
            for tokens in batch_tokens_list:
                try:
                    tokens = self.npy_to_tokens(tokens)
                    batch_tokens.append(tokens)
                except Exception as e:
                    print(f"Error processing {tokens}: {e}")
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
            for j, tokens in enumerate(batch_tokens_list):
                result = {
                    'tokens': tokens,
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
    classifier = ScoreArrangeDomainClassifier(
        pretrained_checkpoint="/mnt/kiso-qnap5/obara/StarGAN_music/Moonbeam-MIDI-Foundation-Model/models/pretrained/moonbeam_839M.pt",
        lora_adapter_path="/mnt/kiso-qnap5/obara/StarGAN_music/Moonbeam-MIDI-Foundation-Model/models/emotion_classification-v3",
        config_path="/mnt/kiso-qnap5/obara/StarGAN_music/Moonbeam-MIDI-Foundation-Model/src/llama_recipes/configs/player_classification_config.json",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # === テストデータで評価 ===
    print("\n" + "="*60)
    print("Test Data Evaluation")
    print("="*60)
    
    # CSVファイルを読み込み
    csv_path = "/mnt/kiso-qnap5/obara/StarGAN_music/Moonbeam-MIDI-Foundation-Model/processed_datasets/classification/emopia2.2_1071_clips/train_test_split.csv"
    df = pd.read_csv(csv_path)
    
    # testデータのみを抽出
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    
    print(f"\nTotal test samples: {len(test_df)}")
    print(f"Label distribution:")
    print(test_df['label'].value_counts().sort_index())
    
    # npyファイルのベースパス
    base_path = Path("/mnt/kiso-qnap5/obara/StarGAN_music/Moonbeam-MIDI-Foundation-Model/processed_datasets/classification/emopia2.2_1071_clips/processed")
    
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
        tokens = np.load(npy_path)  
        
        # ファイルの存在確認
        if not npy_path.exists():
            print(f"[{idx+1}/{len(test_df)}] File not found: {file_name}")
            failed_files.append(file_name)
            continue
        
        try:
            result = classifier.predict(tokens, return_probabilities=False)
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
    """
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
    """


if __name__ == "__main__":
    main()