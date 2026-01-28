import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from typing import Dict, Tuple, List, Optional

def split_sequence_with_sliding_window(sequence, window_size=3072, stride=None):
    """
    Split a sequence longer than window_size into overlapping windows.
    
    Args:
        sequence: Tensor of shape [T, ...] where T is sequence length
        window_size: Maximum length per window (default: Amadeus max input length = 3072)
        stride: Step size for sliding window. If None, stride = window_size (no overlap)
    
    Returns:
        List of sequence windows, each with length <= window_size
        Also returns start positions for each window
    
    Example:
        >>> seq = torch.randn(5000, 8)  # Longer than window_size
        >>> windows, positions = split_sequence_with_sliding_window(seq, window_size=3072, stride=1536)
        >>> # windows[0].shape = [3072, 8]
        >>> # windows[1].shape = [3072, 8]
        >>> # windows[2].shape = [1928, 8]  (remaining)
    """
    seq_len = sequence.shape[0]
    
    # If sequence is shorter than window size, return as is
    if seq_len <= window_size:
        return [sequence], [0]
    
    # Default stride = window_size (no overlap)
    if stride is None:
        stride = window_size
    
    windows = []
    positions = []
    
    # Create sliding windows
    current_pos = 0
    while current_pos < seq_len:
        end_pos = min(current_pos + window_size, seq_len)
        window = sequence[current_pos:end_pos]
        windows.append(window)
        positions.append(current_pos)
        
        # Break if we've reached the end
        if end_pos >= seq_len:
            break
        
        current_pos += stride
    
    return windows, positions


def aggregate_window_outputs(window_outputs, aggregation_method='mean'):
    """
    Aggregate predictions from multiple sliding windows.
    
    Args:
        window_outputs: List of dicts, each containing:
            - 'd_real': Real/fake discrimination score [B, 1]
            - 'd_fake': Real/fake discrimination score [B, 1]
            - 'd_cls': Domain classification logits [B, num_domains]
            - Other optional keys
        aggregation_method: 'mean' (average), 'max' (maximum), or 'first' (first window only)
    
    Returns:
        Aggregated output dict with same structure as input
    """
    if len(window_outputs) == 1:
        return window_outputs[0]
    
    # Stack outputs from all windows
    aggregated = {}
    
    for key in window_outputs[0].keys():
        values = [out[key] for out in window_outputs]
        
        # Stack along a new dimension
        stacked = torch.stack(values, dim=0)  # [num_windows, ...]
        
        if aggregation_method == 'mean':
            aggregated[key] = stacked.mean(dim=0)
        elif aggregation_method == 'max':
            # For classification logits, take max; for scores, take mean
            if 'cls' in key:
                aggregated[key] = stacked.max(dim=0)[0]
            else:
                aggregated[key] = stacked.mean(dim=0)
        elif aggregation_method == 'first':
            aggregated[key] = window_outputs[0][key]
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    
    return aggregated

def amadeus_to_vocab(self, amadeus_tokens: torch.Tensor, vocab_path: str) -> np.ndarray:
    """Amadeusのトークン列を語彙に変換し、amadeus_to_moonbeam利用可能な形式に"""
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    amadeus_tokens = amadeus_tokens.squeeze(0)
    tokens_np = amadeus_tokens.detach().cpu().numpy().astype(np.int64)
    decoded = np.empty(tokens_np.shape, dtype=object)

    for axis, field in enumerate(AMAEDEUS_FIELDS):
        lookup = self._build_lookup_table(vocab[field])
        decoded[:, axis] = lookup[tokens_np[:, axis]]

    return decoded

import json
def logits_to_embed(logits, D, vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    
    # Moonbeamの各埋め込み重み取得
    weight = D.model.embed_tokens.weight  # [32000, 1536]
    start = 0
    embed_list = {}
    for field in ["onset", "dur", "octave", "pitch_class", "instrument", "velocity"]:
        vocab_size = getattr(D.config, f"{field}_vocab_size")
        embed_list[field] = weight[start : start+vocab_size]
        start += vocab_size

    # Generator出力確率取得
    probs = {}
    for key in logits.keys():
        probs[key] = F.softmax(logits[key], dim=-1)
    
    # 最大確率トークン取得
    type_probs = probs['type']  # [B, T, num_type_vocab]
    type_tokens = torch.argmax(type_probs, dim=-1)  # [B, T]
    tempo_probs = probs['tempo']  # [B, T, num_tempo_vocab]
    tempo_tokens = torch.argmax(tempo_probs, dim=-1)  # [B, T]
    
    # 各音符における小節数onset算出
    """
    onset_list = []
    for b in range(type_tokens.shape[0]):
        onset_counts = 0
        onset_seq = []
        numerator, denominator = 4, 4
        for t in range(type_tokens.shape[1]):
            tempo_idx = tempo_tokens[b, t].item()
            tempo_str = vocab['tempo'][str(tempo_idx)]
            if tempo_str.startswith('Tempo_'):
                current_tempo = int(tempo_str.split('Tempo_')[1])
            
            type_idx = type_tokens[b, t].item()
            type_str = vocab['type'][str(type_idx)]
            if type_str.startswith('NNN_time_signature_'):
                current_beat = int(type_str.split('NNN_time_signature_')[1])
                numerator, denominator = int(current_beat.split('/')[0]), int(current_beat.split('/')[1])
                quarter_per_bar = numerator * (4.0 / denominator)
                ms_per_quarter = 60000.0 / current_tempo
                onset_counts += quarter_per_bar * ms_per_quarter / 10
            elif type_str == 'SNN':
                quarter_per_bar = numerator * (4.0 / denominator)
                ms_per_quarter = 60000.0 / current_tempo
                onset_counts += quarter_per_bar * ms_per_quarter / 10
                
            # SSN and SSS do not change onset count
            onset_seq.append(onset_counts)
        onset_list.append(onset_seq)
    """
    
    ### onset 埋め込みの混合埋め込みを算出 ###
    #onset_probs = probs['type']
    #onset_mix = torch.einsum('btv,vh->bth', onset_probs, embed_list['onset'])
    pitch_probs = probs['pitch']
    
    pitch_value_map = []
    for idx in range(len(vocab['pitch'])):
        token = vocab['pitch'][str(idx)]
        if isinstance(token, str) and token.startswith('Note_Pitch_'):
            val = int(token.split('Note_Pitch_')[1])
            val = val // 12
        else:
            val = idx
        # 埋め込み表の範囲に収まるようクリップ
        val = max(0, min(val, embed_list['octave'].shape[0] - 1))
        pitch_value_map.append(val)
    pitch_value_map = torch.tensor(pitch_value_map, device=pitch_probs.device)
    # idx -> pitch_class value で埋め込みテーブルを並べ替えたもの
    octave_table = embed_list['octave'].index_select(0, pitch_value_map)  # [V_pc, H]
    # pitch_class_probs と pitch_class_table の全組み合わせを掛けて idx 方向に総和を取る
    onset_mix = torch.einsum('btv,vh->bth', pitch_probs, octave_table)  # [B, T, H]
   
    
    ### duration 埋め込みの混合埋め込みを算出 ###
    #duration_probs = probs['beat']
    #duration_mix = torch.einsum('btv,vh->bth', duration_probs, embed_list['dur'])
    pitch_probs = probs['pitch']
    
    pitch_value_map = []
    for idx in range(len(vocab['pitch'])):
        token = vocab['pitch'][str(idx)]
        if isinstance(token, str) and token.startswith('Note_Pitch_'):
            val = int(token.split('Note_Pitch_')[1])
            val = val // 12
        else:
            val = idx
        # 埋め込み表の範囲に収まるようクリップ
        val = max(0, min(val, embed_list['octave'].shape[0] - 1))
        pitch_value_map.append(val)
    pitch_value_map = torch.tensor(pitch_value_map, device=pitch_probs.device)
    # idx -> pitch_class value で埋め込みテーブルを並べ替えたもの
    octave_table = embed_list['octave'].index_select(0, pitch_value_map)  # [V_pc, H]
    # pitch_class_probs と pitch_class_table の全組み合わせを掛けて idx 方向に総和を取る
    duration_mix = torch.einsum('btv,vh->bth', pitch_probs, octave_table)  # [B, T, H]


    ### octave 埋め込みの混合埋め込みを算出 ###
    pitch_probs = probs['pitch']
    
    pitch_value_map = []
    for idx in range(len(vocab['pitch'])):
        token = vocab['pitch'][str(idx)]
        if isinstance(token, str) and token.startswith('Note_Pitch_'):
            val = int(token.split('Note_Pitch_')[1])
            val = val // 12
        else:
            val = idx
        # 埋め込み表の範囲に収まるようクリップ
        val = max(0, min(val, embed_list['octave'].shape[0] - 1))
        pitch_value_map.append(val)
    pitch_value_map = torch.tensor(pitch_value_map, device=pitch_probs.device)
    # idx -> pitch_class value で埋め込みテーブルを並べ替えたもの
    octave_table = embed_list['octave'].index_select(0, pitch_value_map)  # [V_pc, H]
    # pitch_class_probs と pitch_class_table の全組み合わせを掛けて idx 方向に総和を取る
    octave_mix = torch.einsum('btv,vh->bth', pitch_probs, octave_table)  # [B, T, H]
    
    
    ### pitch_class 埋め込みの混合埋め込みを算出 ###
    pitch_probs = probs['pitch']
    
    pitch_value_map = []
    for idx in range(len(vocab['pitch'])):
        token = vocab['pitch'][str(idx)]
        if isinstance(token, str) and token.startswith('Note_Pitch_'):
            val = int(token.split('Note_Pitch_')[1])
            val = val % 12
        else:
            val = idx
        # 埋め込み表の範囲に収まるようクリップ
        val = max(0, min(val, embed_list['pitch_class'].shape[0] - 1))
        pitch_value_map.append(val)
    pitch_value_map = torch.tensor(pitch_value_map, device=pitch_probs.device)
    # idx -> pitch_class value で埋め込みテーブルを並べ替えたもの
    pitch_class_table = embed_list['pitch_class'].index_select(0, pitch_value_map)  # [V_pc, H]
    # pitch_class_probs と pitch_class_table の全組み合わせを掛けて idx 方向に総和を取る
    pitch_class_mix = torch.einsum('btv,vh->bth', pitch_probs, pitch_class_table)  # [B, T, H]
    
    
    ### instrument 埋め込みの混合埋め込みを算出 ###
    instrument_probs = probs['instrument']
    
    instrument_value_map = []
    for idx in range(len(vocab['instrument'])):
        token = vocab['instrument'][str(idx)]
        if isinstance(token, str) and token.startswith('Instrument_'):
            val = int(token.split('Instrument_')[1])
        else:
            val = idx
        # 埋め込み表の範囲に収まるようクリップ
        val = max(0, min(val, embed_list['instrument'].shape[0] - 1))
        instrument_value_map.append(val)
    instrument_value_map = torch.tensor(instrument_value_map, device=instrument_probs.device)
    # idx -> instrument value で埋め込みテーブルを並べ替えたもの
    instrument_table = embed_list['instrument'].index_select(0, instrument_value_map)  # [V_inst, H]
    # instrument_probs と instrument_table の全組み合わせを掛けて idx 方向に総和を取る
    instrument_mix = torch.einsum('btv,vh->bth', instrument_probs, instrument_table)  # [B, T, H]
    
    
    ### velocity 埋め込みの混合埋め込みを算出 ###
    velocity_probs = probs['velocity']  # [B, T, V_vel]

    velocity_value_map = []
    for idx in range(len(vocab['velocity'])):
        token = vocab['velocity'][str(idx)]
        if isinstance(token, str) and token.startswith('Note_Velocity_'):
            val = int(token.split('Note_Velocity_')[1])
        else:
            val = idx
        # 埋め込み表の範囲に収まるようクリップ
        val = max(0, min(val, embed_list['velocity'].shape[0] - 1))
        velocity_value_map.append(val)
    velocity_value_map = torch.tensor(velocity_value_map, device=velocity_probs.device)
    # idx -> velocity value で埋め込みテーブルを並べ替えたもの
    velocity_table = embed_list['velocity'].index_select(0, velocity_value_map)  # [V_vel, H]
    # velocity_probs と velocity_table の全組み合わせを掛けて idx 方向に総和を取る
    velocity_mix = torch.einsum('btv,vh->bth', velocity_probs, velocity_table)  # [B, T, H]

    # 4つの mixture を連結して返す
    return torch.cat([onset_mix, duration_mix, octave_mix, pitch_class_mix, instrument_mix, velocity_mix], dim=-1)
        
def _tie_moonbeam_embeddings(model: nn.Module):
    """
    学習中のMoonbeamモデルからEmbeddingを共有して参照する。
    modelに存在しない場合は初期化にフォールバック。
    """
    try:
        # 可能な名称を優先して拾う
        attr_map = {
            'onset_embedding':    ['onset_embedding'],
            'duration_embedding': ['dur_embedding', 'duration_embedding'],
            'octave_embedding':   ['octave_embedding'],
            'pitch_class_embedding': ['pitch_embedding', 'pitch_class_embedding'],
            'instrument_embedding': ['instrument_embedding'],
            'velocity_embedding': ['velocity_embedding'],
        }
        for dst, candidates in attr_map.items():
            found = None
            for name in candidates:
                if hasattr(model, name):
                    found = getattr(model, name)
                    break
                sd_key = f"{name}.weight"
                if sd_key in model.state_dict():
                    weight = model.state_dict()[sd_key]
                    found = nn.Embedding.from_pretrained(weight, freeze=False)
                    break
            if found is None:
                raise AttributeError(f"missing {dst}")
        print("✓ Tied Moonbeam embeddings from running model")
    except Exception as e:
        print(f"⚠ Could not tie embeddings from model: {e}")
        

def _parse_amadeus_vocab_index(vocab_list: List[str], prefix: str) -> Dict[int, str]:
    """
    Extract specific type tokens (e.g., NNN, SNN, SSN, SSS) from vocabulary list.
    
    Args:
        vocab_list: List of token strings from Amadeus vocabulary
        prefix: Prefix to search for (e.g., 'NNN', 'SNN')
    
    Returns:
        Dict mapping index to token string
    """
    result = {}
    for idx, token in enumerate(vocab_list):
        if isinstance(token, str) and token.startswith(prefix):
            result[idx] = token
    return result


def _extract_beats_and_indices(vocab_list: List[str]) -> Tuple[Dict[int, int], Dict[int, str]]:
    """
    Extract beat information and tempos from vocabulary.
    
    Returns:
        beats_dict: Dict mapping vocab_index to beat_value
        tempo_dict: Dict mapping vocab_index to tempo_value
    """
    beats_dict = {}
    tempo_dict = {}
    
    for idx, token in enumerate(vocab_list):
        if isinstance(token, str):
            # Extract Beat_N
            beat_match = re.search(r'Beat_(\d+)', token)
            if beat_match:
                beats_dict[idx] = int(beat_match.group(1))
            
            # Extract Tempo_N
            tempo_match = re.search(r'Tempo_(\d+)', token)
            if tempo_match:
                tempo_dict[idx] = int(tempo_match.group(1))
    
    return beats_dict, tempo_dict


def _extract_pitch_range(vocab_list: List[str]) -> Tuple[int, int]:
    """
    Extract minimum and maximum pitch values from vocabulary.
    
    Returns:
        (min_pitch, max_pitch)
    """
    pitches = []
    for token in vocab_list:
        if isinstance(token, str):
            pitch_match = re.search(r'Note_Pitch_(\d+)', token)
            if pitch_match:
                pitches.append(int(pitch_match.group(1)))
    
    return (min(pitches), max(pitches)) if pitches else (0, 127)


def _calculate_bar_progression_probs(
    type_probs: torch.Tensor,
    nnn_indices: List[int],
    snn_index: Optional[int],
    seq_len: int,
    device: torch.device
) -> torch.Tensor:
    """
    Calculate bar progression based on argmax of type probabilities.
    
    一音符ごとに、type確率の最大値を取るトークンがNNNまたはSNNに対応している場合、
    小節数を+1する。そうでない場合は小節数を維持。
    
    Args:
        type_probs: [B, T, num_type_vocab] - typeトークンの確率
        nnn_indices: NNN_time_signature_*に対応するインデックスリスト
        snn_index: SNNに対応するインデックス（NoneはSNNがない場合）
        seq_len: シーケンス長T
        device: 計算デバイス
    
    Returns:
        bar_counts: [B, T] - 各位置での小節数（整数値）
    """
    batch_size = type_probs.shape[0]
    bar_counts = torch.zeros(batch_size, seq_len, device=device)
    
    # 最初の音符は小節0
    bar_counts[:, 0] = 0.0
    
    # 小節進行をシーケンシャルに計算
    for t in range(1, seq_len):
        # t番目の音符のtype確率の最大値インデックス
        max_type_indices = torch.argmax(type_probs[:, t, :], dim=-1)  # [B]
        
        # NNNまたはSNNに対応しているかチェック
        bar_advance = torch.zeros(batch_size, device=device, dtype=torch.bool)
        
        for b in range(batch_size):
            max_idx = max_type_indices[b].item()
            # NNN indicesまたはSNN indexに該当するかチェック
            if max_idx in nnn_indices or (snn_index is not None and max_idx == snn_index):
                bar_advance[b] = True
        
        # 小節カウント更新
        bar_counts[:, t] = bar_counts[:, t - 1] + bar_advance.float()
    
    return bar_counts


def _compute_onset_probs(
    bar_counts: torch.Tensor,
    beat_probs: torch.Tensor,
    tempo_probs: torch.Tensor,
    beat_vocab_indices: Dict[int, int],
    tempo_vocab_indices: Dict[int, int],
    time_signature: Tuple[int, int] = (4, 4),
    in_beat_resolution: int = 4,
    time_resolution: float = 10.0,
    default_tempo: int = 120,
    max_onset_value: int = 1024
) -> torch.Tensor:
    """
    Compute probability distribution over Moonbeam onset tokens.
    
    Args:
        bar_counts: [B, T] - Expected bar number at each position
        beat_probs: [B, T, num_beat_vocab] - Beat token probabilities
        tempo_probs: [B, T, num_tempo_vocab] - Tempo token probabilities
        beat_vocab_indices: Dict mapping vocab_idx -> beat_value
        tempo_vocab_indices: Dict mapping vocab_idx -> tempo_value
        time_signature: (numerator, denominator)
        in_beat_resolution: Resolution of beats per quarter note
        time_resolution: Time resolution in milliseconds (e.g., 10ms)
        default_tempo: Default tempo in BPM
        max_onset_value: Maximum onset token value
    
    Returns:
        onset_probs: [B, T, max_onset_value] - Probability over onset tokens
    """
    batch_size, seq_len = bar_counts.shape
    device = bar_counts.device
    
    # Onset probability distribution
    onset_probs = torch.zeros(batch_size, seq_len, max_onset_value, device=device)
    
    numerator, denominator = time_signature
    subdivisions_per_bar = numerator * (4.0 / denominator) * in_beat_resolution
    ms_per_quarter_note_default = 60000.0 / default_tempo
    
    # For each time step
    for t in range(seq_len):
        bars_t = bar_counts[:, t]  # [B]
        beat_prob_t = beat_probs[:, t]  # [B, num_beat_vocab]
        tempo_prob_t = tempo_probs[:, t]  # [B, num_tempo_vocab]
        
        # Compute onset values for all beat/tempo combinations
        for beat_vocab_idx, beat_value in beat_vocab_indices.items():
            for tempo_vocab_idx, tempo_value in tempo_vocab_indices.items():
                # Calculate onset
                # onset_in_subdivisions = bar * subdivisions_per_bar + beat_value
                # onset_in_quarter_notes = onset_in_subdivisions / in_beat_resolution
                # ms_per_quarter_note = 60000 / tempo
                # onset_ms = onset_in_quarter_notes * ms_per_quarter_note
                # onset_in_10ms = round(onset_ms / 10)
                
                onset_in_subdivisions = bars_t * subdivisions_per_bar + beat_value  # [B]
                onset_in_quarter_notes = onset_in_subdivisions / in_beat_resolution
                ms_per_quarter_note = 60000.0 / tempo_value
                onset_ms = onset_in_quarter_notes * ms_per_quarter_note
                onset_token = torch.round(onset_ms / time_resolution).long()  # [B]
                
                # Clamp to valid range
                onset_token = torch.clamp(onset_token, 0, max_onset_value - 1)
                
                # Update probability
                # P(onset | beat, tempo) = P(beat) * P(tempo)
                prob_contribution = beat_prob_t[:, beat_vocab_idx] * tempo_prob_t[:, tempo_vocab_idx]  # [B]
                
                # Accumulate
                for b in range(batch_size):
                    onset_probs[b, t, onset_token[b]] += prob_contribution[b]
    
    return onset_probs


def _compute_duration_probs(
    duration_probs: torch.Tensor,
    tempo_probs: torch.Tensor,
    tempo_vocab_indices: Dict[int, int],
    duration_vocab_indices: Dict[int, float],
    in_beat_resolution: int = 4,
    time_resolution: float = 10.0,
    default_tempo: int = 120,
    max_duration_value: int = 1024
) -> torch.Tensor:
    """
    Compute probability distribution over Moonbeam duration tokens.
    
    Args:
        duration_probs: [B, T, num_duration_vocab] - Duration token probabilities
        tempo_probs: [B, T, num_tempo_vocab] - Tempo token probabilities
        tempo_vocab_indices: Dict mapping vocab_idx -> tempo_value
        duration_vocab_indices: Dict mapping vocab_idx -> duration_value (in subdivisions)
        in_beat_resolution: Resolution of beats per quarter note
        time_resolution: Time resolution in milliseconds
        default_tempo: Default tempo in BPM
        max_duration_value: Maximum duration token value
    
    Returns:
        duration_out_probs: [B, T, max_duration_value] - Probability over duration tokens
    """
    batch_size, seq_len = duration_probs.shape[:2]
    device = duration_probs.device
    
    duration_out_probs = torch.zeros(batch_size, seq_len, max_duration_value, device=device)
    
    ms_per_quarter_note_default = 60000.0 / default_tempo
    
    for t in range(seq_len):
        duration_prob_t = duration_probs[:, t]  # [B, num_duration_vocab]
        tempo_prob_t = tempo_probs[:, t]  # [B, num_tempo_vocab]
        
        for duration_vocab_idx, duration_value in duration_vocab_indices.items():
            for tempo_vocab_idx, tempo_value in tempo_vocab_indices.items():
                # Calculate duration
                duration_in_quarter_notes = duration_value / in_beat_resolution
                ms_per_quarter_note = 60000.0 / tempo_value
                duration_ms = duration_in_quarter_notes * ms_per_quarter_note
                duration_token = torch.round(
                    torch.tensor(duration_ms / time_resolution, device=device)
                ).long()
                
                # Clamp to valid range
                duration_token = torch.clamp(duration_token, 1, max_duration_value - 1)
                
                # Update probability
                prob_contribution = duration_prob_t[:, duration_vocab_idx] * tempo_prob_t[:, tempo_vocab_idx]
                
                for b in range(batch_size):
                    duration_out_probs[b, t, duration_token] += prob_contribution[b]
    
    return duration_out_probs


def _compute_octave_pitch_class_probs(
    pitch_probs: torch.Tensor,
    pitch_vocab_indices: Dict[int, int],
    max_octave: int = 11,
    max_pitch_class: int = 12
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute probability distributions for octave and pitch_class.
    
    Args:
        pitch_probs: [B, T, num_pitch_vocab] - Pitch token probabilities
        pitch_vocab_indices: Dict mapping vocab_idx -> pitch_value (0-127)
        max_octave: Maximum octave value
        max_pitch_class: Maximum pitch class value (12)
    
    Returns:
        (octave_probs, pitch_class_probs): Each [B, T, max_value]
    """
    batch_size, seq_len = pitch_probs.shape[:2]
    device = pitch_probs.device
    
    octave_probs = torch.zeros(batch_size, seq_len, max_octave + 1, device=device)
    pitch_class_probs = torch.zeros(batch_size, seq_len, max_pitch_class, device=device)
    
    for t in range(seq_len):
        pitch_prob_t = pitch_probs[:, t]  # [B, num_pitch_vocab]
        
        for pitch_vocab_idx, pitch_value in pitch_vocab_indices.items():
            octave = pitch_value // 12
            pitch_class = pitch_value % 12
            
            # Clamp octave
            octave = min(octave, max_octave)
            
            prob = pitch_prob_t[:, pitch_vocab_idx]  # [B]
            octave_probs[:, t, octave] += prob
            pitch_class_probs[:, t, pitch_class] += prob
    
    return octave_probs, pitch_class_probs


def _compute_instrument_velocity_probs(
    instrument_probs: torch.Tensor,
    velocity_probs: torch.Tensor,
    instrument_vocab_indices: Dict[int, int],
    velocity_vocab_indices: Dict[int, int],
    max_instrument: int = 128,
    max_velocity: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute probability distributions for instrument and velocity.
    (Direct mapping - no transformation needed)
    
    Args:
        instrument_probs: [B, T, num_instrument_vocab]
        velocity_probs: [B, T, num_velocity_vocab]
        instrument_vocab_indices: Dict mapping vocab_idx -> instrument_value
        velocity_vocab_indices: Dict mapping vocab_idx -> velocity_value
        max_instrument: Maximum instrument value
        max_velocity: Maximum velocity value
    
    Returns:
        (instrument_out_probs, velocity_out_probs): Each [B, T, max_value]
    """
    batch_size, seq_len = instrument_probs.shape[:2]
    device = instrument_probs.device
    
    instrument_out_probs = torch.zeros(batch_size, seq_len, max_instrument + 1, device=device)
    velocity_out_probs = torch.zeros(batch_size, seq_len, max_velocity, device=device)
    
    for t in range(seq_len):
        instrument_prob_t = instrument_probs[:, t]  # [B, num_instrument_vocab]
        velocity_prob_t = velocity_probs[:, t]  # [B, num_velocity_vocab]
        
        for inst_vocab_idx, inst_value in instrument_vocab_indices.items():
            inst_value = min(inst_value, max_instrument)
            instrument_out_probs[:, t, inst_value] += instrument_prob_t[:, inst_vocab_idx]
        
        for vel_vocab_idx, vel_value in velocity_vocab_indices.items():
            vel_value = min(vel_value, max_velocity - 1)
            velocity_out_probs[:, t, vel_value] += velocity_prob_t[:, vel_vocab_idx]
    
    return instrument_out_probs, velocity_out_probs
    
    
    

def vocab_to_moonbeam(self, amadeus_vocabs, time_resolution=10, default_tempo=120, in_beat_resolution=4):
    # (Beat + 小節数) * Tempo → onset(ms)
    # duration * Tempo → duration(ms)
    # pitch → octave*12 + pitch_class
    
    # Amadeus表現：(type, beat, chord, tempo, instrument, pitch, duration, velocity)
    """
    Type：拍子の変化や継続の組み合わせの違いをそれぞれ表すもの。
    Beat：同一小節内における音符の相対的な位置。
    Chord：現在の音符が属している和音。
    Tempo：音符の再生速度。一般に、テンポが高いほど楽曲は速くなる。
    Instrument：現在の音符を演奏している楽器。
    Pitch：音符の高さ。MIDI仕様に基づく128段階の離散値で表される。
    Duration：音符が演奏される長さ（継続時間）。
    Velocity：音符がどの強さで演奏されるかを表す値で、音量の大きさを決定する。
    例）{'type': 'NNN_time_signature_4/4', 'beat': 'Beat_2', 'chord': 'Chord_N_N', 'tempo': 'Tempo_121', 'instrument': 'Instrument_114', 'pitch': 'Note_Pitch_48', 'duration': 'Note_Duration_2', 'velocity': 'Note_Velocity_100'}
    """
    # Moonbeam表現：(onset, duration, octave, pitch_class, instrument, velocity) 
    """
    onset・durationの最小単位は10ms
    onset：音の開始位置　4097トークン(M-model), 1024トークン(S-model)
    duration：音の継続時間　4097トークン(M-model), 1024トークン(S-model)
    octave：11トークン
    pitch_class：12トークン(1オクターブの音階分)
    instrument：129トークン(MIDI楽器全般)
    velocity：128トークン
    """
    
    # Convert to numpy for easier processing
    if isinstance(amadeus_vocabs, torch.Tensor):
        amadeus_np = amadeus_vocabs.cpu().numpy()
        use_torch = True
    elif isinstance(amadeus_vocabs, list):
        amadeus_np = np.array(amadeus_vocabs)
        use_torch = False
    else:
        amadeus_np = np.array(amadeus_vocabs)
        use_torch = False
    
    # Check shape and transpose if needed
    if amadeus_np.shape[0] == 8:
        amadeus_np = amadeus_np.T  # [num_notes, 8]
    
    num_notes = amadeus_np.shape[0]
    moonbeam_np = np.zeros((num_notes, 6), dtype=np.int32)
    
    # State tracking
    current_bar = -1
    current_tempo = default_tempo
    current_time_signature = (4, 4)  # (numerator, denominator)
    
    # First note
    for k in range(6):
        moonbeam_np[0, k] = 0
            
    for i in range(1, num_notes):
        type_token = amadeus_np[i, 0]
        beat_token = amadeus_np[i, 1]
        chord_token = amadeus_np[i, 2]
        tempo_token = amadeus_np[i, 3]
        instrument_token = amadeus_np[i, 4]
        pitch_token = amadeus_np[i, 5]
        duration_token = amadeus_np[i, 6]
        velocity_token = amadeus_np[i, 7]
        
        # Parse type token (NNN/SNN/SSN/SSS format)
        if isinstance(type_token, str):
            # Check for time signature change (NNN format)
            if type_token.startswith('NNN_time_signature_'):
                time_sig_match = re.search(r'time_signature_(\d+)/(\d+)', type_token)
                if time_sig_match:
                    current_time_signature = (int(time_sig_match.group(1)), int(time_sig_match.group(2)))
                current_bar += 1
            elif type_token == 'SNN':  # Same time sig, new bar, new beat
                current_bar += 1
            # SSN and SSS don't change bar
        
        # Extract tempo
        if isinstance(tempo_token, str):
            tempo_match = re.search(r'Tempo_(\d+)', tempo_token)
            if tempo_match:
                current_tempo = int(tempo_match.group(1))
        else:
            current_tempo = int(tempo_token) if tempo_token else default_tempo
        
        # Extract beat position within bar (0-15 for 4/4 time with in_beat_resolution=4)
        if isinstance(beat_token, str):
            beat_match = re.search(r'Beat_(\d+)', beat_token)
            beat_index = int(beat_match.group(1)) if beat_match else 0
        else:
            beat_index = int(beat_token)
        
        # Calculate onset
        # Beat index represents position in bar (0-15 for 4/4, in_beat_resolution=4)
        numerator, denominator = current_time_signature
        
        # Total subdivisions per bar
        subdivisions_per_bar = numerator * (4 / denominator) * in_beat_resolution
        # For 4/4: 4 * 1 * 4 = 16 subdivisions per bar
        
        # Calculate onset in subdivisions (16th notes for in_beat_resolution=4)
        onset_in_subdivisions = current_bar * subdivisions_per_bar + beat_index
        
        # Convert to quarter notes
        onset_in_quarter_notes = onset_in_subdivisions / in_beat_resolution
        
        # Convert to milliseconds
        ms_per_quarter_note = 60000.0 / current_tempo
        onset_ms = onset_in_quarter_notes * ms_per_quarter_note
        
        # Convert to 10ms resolution
        onset_in_10ms = int(round(onset_ms / time_resolution))
        
        # Extract duration (in subdivisions)
        if isinstance(duration_token, str):
            dur_match = re.search(r'Note_Duration_([\d.]+)', duration_token)
            duration_in_subdivisions = float(dur_match.group(1)) if dur_match else 1.0
        else:
            duration_in_subdivisions = float(duration_token)
        
        # Convert duration to quarter notes
        duration_in_quarter_notes = duration_in_subdivisions / in_beat_resolution
        
        # Convert to milliseconds
        duration_ms = duration_in_quarter_notes * ms_per_quarter_note
        duration_in_10ms = int(round(duration_ms / time_resolution))
        duration_in_10ms = max(1, min(duration_in_10ms, 1024))
        
        # Extract pitch and convert to octave + pitch_class
        if isinstance(pitch_token, str):
            pitch_match = re.search(r'Note_Pitch_(\d+)', pitch_token)
            pitch = int(pitch_match.group(1)) if pitch_match else 60
        else:
            pitch = int(pitch_token)
        
        octave = pitch // 12
        pitch_class = pitch % 12
        octave = max(0, min(octave, 10))
        
        # Extract instrument
        if isinstance(instrument_token, str):
            inst_match = re.search(r'Instrument_(\d+)', instrument_token)
            instrument = int(inst_match.group(1)) if inst_match else 0
        else:
            instrument = int(instrument_token)
        
        instrument = max(0, min(instrument, 128))
        
        # Extract velocity
        if isinstance(velocity_token, str):
            vel_match = re.search(r'Note_Velocity_(\d+)', velocity_token)
            velocity = int(vel_match.group(1)) if vel_match else 64
        else:
            velocity = int(velocity_token)
        
        velocity = max(0, min(velocity, 127))
        
        # Store in Moonbeam format
        moonbeam_np[i, 0] = onset_in_10ms
        moonbeam_np[i, 1] = duration_in_10ms
        moonbeam_np[i, 2] = octave
        moonbeam_np[i, 3] = pitch_class
        moonbeam_np[i, 4] = instrument
        moonbeam_np[i, 5] = velocity
    
    # Convert back to torch if input was torch
    if use_torch:
        moonbeam_tokens = torch.from_numpy(moonbeam_np).long()
    else:
        moonbeam_tokens = moonbeam_np
    
    return moonbeam_tokens


class LogitsToMoonbeamEmbedding(nn.Module):
    """
    Convert Amadeus logits to Moonbeam embeddings in a differentiable manner.
    
    Moonbeamの学習済みモデルのembedding層を利用して、属性embeddingを計算します。
    
    Args:
        amadeus_vocab_path: Path to vocab_LakhALLFined_nb8.json
        moonbeam_model_path: Path to Moonbeam pretrained model checkpoint
        moonbeam_hidden_size: Hidden size for Moonbeam embeddings (default: 1920)
        time_resolution: Time resolution in milliseconds (default: 10)
        in_beat_resolution: Beat resolution (default: 4)
        default_tempo: Default tempo in BPM (default: 120)
        moonbeam_vocab_sizes: Dict with vocab sizes for each Moonbeam attribute
    """
    
    def __init__(
        self,
        amadeus_vocab_path: str,
        moonbeam_model_path: Optional[str] = None,
        moonbeam_hidden_size: int = 1920,
        time_resolution: float = 10.0,
        in_beat_resolution: int = 4,
        default_tempo: int = 120,
        moonbeam_vocab_sizes: Optional[Dict[str, int]] = None
    ):
        super().__init__()
        
        self.amadeus_hidden_size = moonbeam_hidden_size
        self.time_resolution = time_resolution
        self.in_beat_resolution = in_beat_resolution
        self.default_tempo = default_tempo
        
        # Default Moonbeam vocab sizes
        if moonbeam_vocab_sizes is None:
            moonbeam_vocab_sizes = {
                'onset': 1026,
                'duration': 1026,
                'octave': 13,
                'pitch_class': 14,
                'instrument': 131,
                'velocity': 130
            }
        
        self.moonbeam_vocab_sizes = moonbeam_vocab_sizes
        
        # Load Amadeus vocabulary
        import json
        with open(amadeus_vocab_path, 'r') as f:
            self.vocab_data = json.load(f)
        
        # Build vocab index mappings
        self._build_vocab_indices()
        
        # Load Moonbeam embedding layers
        if moonbeam_model_path is not None:
            self._load_moonbeam_embeddings(moonbeam_model_path)
        else:
            # Fallback: Create new embedding layers if Moonbeam model not available
            self._init_embedding_layers(moonbeam_hidden_size)
    
    def _load_moonbeam_embeddings(self, model_path: str):
        """
        Load embedding layers from pretrained Moonbeam model.
        
        Args:
            model_path: Path to Moonbeam checkpoint (.pt file)
        """
        try:
            # Load Moonbeam checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Extract state dict (handle different checkpoint formats)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Create embedding layers from pretrained weights
            # Moonbeam uses FME (Frequency Music Embedding) and WE (Word Embedding)
            
            # onset_embedding: FME with base=199999
            if 'onset_embedding.weight' in state_dict:
                self.onset_embedding = nn.Embedding.from_pretrained(
                    state_dict['onset_embedding.weight'].clone(),
                    freeze=False
                )
            else:
                self._init_embedding_layer('onset', self.moonbeam_vocab_sizes['onset'])
            
            # duration_embedding: FME with base=1031
            if 'dur_embedding.weight' in state_dict:
                self.duration_embedding = nn.Embedding.from_pretrained(
                    state_dict['dur_embedding.weight'].clone(),
                    freeze=False
                )
            else:
                self._init_embedding_layer('duration', self.moonbeam_vocab_sizes['duration'])
            
            # octave_embedding: FME with base=19
            if 'octave_embedding.weight' in state_dict:
                self.octave_embedding = nn.Embedding.from_pretrained(
                    state_dict['octave_embedding.weight'].clone(),
                    freeze=False
                )
            else:
                self._init_embedding_layer('octave', self.moonbeam_vocab_sizes['octave'])
            
            # pitch_embedding: FME with base=20
            if 'pitch_embedding.weight' in state_dict:
                self.pitch_class_embedding = nn.Embedding.from_pretrained(
                    state_dict['pitch_embedding.weight'].clone(),
                    freeze=False
                )
            else:
                self._init_embedding_layer('pitch_class', self.moonbeam_vocab_sizes['pitch_class'])
            
            # instrument_embedding: Word Embedding (WE)
            if 'instrument_embedding.weight' in state_dict:
                self.instrument_embedding = nn.Embedding.from_pretrained(
                    state_dict['instrument_embedding.weight'].clone(),
                    freeze=False
                )
            else:
                self._init_embedding_layer('instrument', self.moonbeam_vocab_sizes['instrument'])
            
            # velocity_embedding: FME with base=131
            if 'velocity_embedding.weight' in state_dict:
                self.velocity_embedding = nn.Embedding.from_pretrained(
                    state_dict['velocity_embedding.weight'].clone(),
                    freeze=False
                )
            else:
                self._init_embedding_layer('velocity', self.moonbeam_vocab_sizes['velocity'])
            
            print(f"✓ Loaded Moonbeam embedding layers from {model_path}")
            
        except Exception as e:
            print(f"⚠ Could not load Moonbeam embeddings: {e}")
            print("⚠ Falling back to initialized embedding layers")
            self._init_embedding_layers(self.amadeus_hidden_size)
    
    def _init_embedding_layers(self, hidden_size: int):
        """
        Initialize embedding layers with random weights (fallback).
        
        Args:
            hidden_size: Dimension of embedding vectors
        """
        self.onset_embedding = nn.Embedding(
            self.moonbeam_vocab_sizes['onset'], hidden_size
        )
        self.duration_embedding = nn.Embedding(
            self.moonbeam_vocab_sizes['duration'], hidden_size
        )
        self.octave_embedding = nn.Embedding(
            self.moonbeam_vocab_sizes['octave'], hidden_size
        )
        self.pitch_class_embedding = nn.Embedding(
            self.moonbeam_vocab_sizes['pitch_class'], hidden_size
        )
        self.instrument_embedding = nn.Embedding(
            self.moonbeam_vocab_sizes['instrument'], hidden_size
        )
        self.velocity_embedding = nn.Embedding(
            self.moonbeam_vocab_sizes['velocity'], hidden_size
        )
        
        self._init_embeddings()
    
    def _init_embedding_layer(self, attr_name: str, vocab_size: int):
        """
        Initialize a single embedding layer.
        
        Args:
            attr_name: Attribute name ('onset', 'duration', etc.)
            vocab_size: Size of vocabulary for this attribute
        """
        embedding = nn.Embedding(vocab_size, self.amadeus_hidden_size)
        nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
        
        if attr_name == 'onset':
            self.onset_embedding = embedding
        elif attr_name == 'duration':
            self.duration_embedding = embedding
        elif attr_name == 'octave':
            self.octave_embedding = embedding
        elif attr_name == 'pitch_class':
            self.pitch_class_embedding = embedding
        elif attr_name == 'instrument':
            self.instrument_embedding = embedding
        elif attr_name == 'velocity':
            self.velocity_embedding = embedding
    
    def _build_vocab_indices(self):
        """Build mappings from vocab indices to token values."""
        # Type tokens (for bar progression)
        type_list = list(self.vocab_data['type'].values())
        nnn_indices = []
        snn_index = None
        
        for idx, token in enumerate(type_list):
            if isinstance(token, str):
                if token.startswith('NNN_time_signature_'):
                    nnn_indices.append(idx)
                elif token == 'SNN':
                    snn_index = idx
        
        self.nnn_indices = nnn_indices
        self.snn_index = snn_index
        
        # Beat tokens
        beat_list = list(self.vocab_data['beat'].values())
        self.beat_vocab_indices = {}
        for idx, token in enumerate(beat_list):
            if isinstance(token, str):
                beat_match = re.search(r'Beat_(\d+)', token)
                if beat_match:
                    self.beat_vocab_indices[idx] = int(beat_match.group(1))
        
        # Tempo tokens
        tempo_list = list(self.vocab_data['tempo'].values())
        self.tempo_vocab_indices = {}
        for idx, token in enumerate(tempo_list):
            if isinstance(token, str):
                tempo_match = re.search(r'Tempo_(\d+)', token)
                if tempo_match:
                    self.tempo_vocab_indices[idx] = int(tempo_match.group(1))
        
        # Pitch tokens
        pitch_list = list(self.vocab_data['pitch'].values())
        self.pitch_vocab_indices = {}
        for idx, token in enumerate(pitch_list):
            if isinstance(token, str):
                pitch_match = re.search(r'Note_Pitch_(\d+)', token)
                if pitch_match:
                    self.pitch_vocab_indices[idx] = int(pitch_match.group(1))
        
        # Duration tokens
        duration_list = list(self.vocab_data['duration'].values())
        self.duration_vocab_indices = {}
        for idx, token in enumerate(duration_list):
            if isinstance(token, str):
                dur_match = re.search(r'Note_Duration_([\d.]+)', token)
                if dur_match:
                    self.duration_vocab_indices[idx] = float(dur_match.group(1))
        
        # Instrument tokens
        instrument_list = list(self.vocab_data['instrument'].values())
        self.instrument_vocab_indices = {}
        for idx, token in enumerate(instrument_list):
            if isinstance(token, str):
                inst_match = re.search(r'Instrument_(\d+)', token)
                if inst_match:
                    self.instrument_vocab_indices[idx] = int(inst_match.group(1))
        
        # Velocity tokens
        velocity_list = list(self.vocab_data['velocity'].values())
        self.velocity_vocab_indices = {}
        for idx, token in enumerate(velocity_list):
            if isinstance(token, str):
                vel_match = re.search(r'Note_Velocity_(\d+)', token)
                if vel_match:
                    self.velocity_vocab_indices[idx] = int(vel_match.group(1))
    
    def _init_embeddings(self):
        """Initialize embedding layers with normal distribution."""
        for module in [self.onset_embedding, self.duration_embedding, 
                      self.octave_embedding, self.pitch_class_embedding,
                      self.instrument_embedding, self.velocity_embedding]:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        amadeus_logits: Dict[str, torch.Tensor],
        return_attention_weights: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Convert Amadeus logits to Moonbeam embeddings.
        
        Args:
            amadeus_logits: Dict with keys ['type', 'beat', 'chord', 'tempo', 
                                           'instrument', 'pitch', 'duration', 'velocity']
                           Each tensor shape: [B, T, vocab_size]
            return_attention_weights: If True, return probability distributions
        
        Returns:
            Dict with keys:
                - 'onset_emb': [B, T, hidden_size]
                - 'duration_emb': [B, T, hidden_size]
                - 'octave_emb': [B, T, hidden_size]
                - 'pitch_class_emb': [B, T, hidden_size]
                - 'instrument_emb': [B, T, hidden_size]
                - 'velocity_emb': [B, T, hidden_size]
                - 'combined_embedding': [B, T, 6*hidden_size] (concatenated)
                - 'onset_probs': [B, T, onset_vocab_size] (optional)
                - ... (other probs if return_attention_weights=True)
        """
        device = next(self.parameters()).device
        batch_size = amadeus_logits['type'].shape[0]
        seq_len = amadeus_logits['type'].shape[1]
        
        # 1. Apply softmax to get probabilities
        type_probs = F.softmax(amadeus_logits['type'], dim=-1)
        beat_probs = F.softmax(amadeus_logits['beat'], dim=-1)
        tempo_probs = F.softmax(amadeus_logits['tempo'], dim=-1)
        pitch_probs = F.softmax(amadeus_logits['pitch'], dim=-1)
        duration_probs = F.softmax(amadeus_logits['duration'], dim=-1)
        instrument_probs = F.softmax(amadeus_logits['instrument'], dim=-1)
        velocity_probs = F.softmax(amadeus_logits['velocity'], dim=-1)
        
        # 2. Compute bar progression
        bar_counts = _calculate_bar_progression_probs(
            type_probs, self.nnn_indices, self.snn_index,
            seq_len, device
        )
        
        # 3. Compute Moonbeam attribute probabilities
        onset_probs = _compute_onset_probs(
            bar_counts, beat_probs, tempo_probs,
            self.beat_vocab_indices, self.tempo_vocab_indices,
            time_signature=(4, 4),
            in_beat_resolution=self.in_beat_resolution,
            time_resolution=self.time_resolution,
            default_tempo=self.default_tempo,
            max_onset_value=self.moonbeam_vocab_sizes['onset']
        )
        
        duration_out_probs = _compute_duration_probs(
            duration_probs, tempo_probs,
            self.tempo_vocab_indices, self.duration_vocab_indices,
            in_beat_resolution=self.in_beat_resolution,
            time_resolution=self.time_resolution,
            default_tempo=self.default_tempo,
            max_duration_value=self.moonbeam_vocab_sizes['duration']
        )
        
        octave_probs, pitch_class_probs = _compute_octave_pitch_class_probs(
            pitch_probs, self.pitch_vocab_indices,
            max_octave=self.moonbeam_vocab_sizes['octave'] - 1,
            max_pitch_class=self.moonbeam_vocab_sizes['pitch_class']
        )
        
        instrument_out_probs, velocity_out_probs = _compute_instrument_velocity_probs(
            instrument_probs, velocity_probs,
            self.instrument_vocab_indices, self.velocity_vocab_indices,
            max_instrument=self.moonbeam_vocab_sizes['instrument'] - 1,
            max_velocity=self.moonbeam_vocab_sizes['velocity']
        )
        
        # 4. Compute weighted embeddings using einsum
        # onset_embedding: [vocab_size, hidden]
        # onset_probs: [B, T, vocab_size]
        # result: [B, T, hidden]
        onset_emb = torch.einsum('btv,vh->bth', onset_probs, 
                                 self.onset_embedding.weight)
        duration_emb = torch.einsum('btv,vh->bth', duration_out_probs,
                                    self.duration_embedding.weight)
        octave_emb = torch.einsum('btv,vh->bth', octave_probs,
                                  self.octave_embedding.weight)
        pitch_class_emb = torch.einsum('btv,vh->bth', pitch_class_probs,
                                       self.pitch_class_embedding.weight)
        instrument_emb = torch.einsum('btv,vh->bth', instrument_out_probs,
                                      self.instrument_embedding.weight)
        velocity_emb = torch.einsum('btv,vh->bth', velocity_out_probs,
                                    self.velocity_embedding.weight)
        
        # 5. Combine embeddings
        combined_embedding = torch.cat([
            onset_emb, duration_emb, octave_emb,
            pitch_class_emb, instrument_emb, velocity_emb
        ], dim=-1)  # [B, T, 6*hidden_size]
        
        result = {
            'onset_emb': onset_emb,
            'duration_emb': duration_emb,
            'octave_emb': octave_emb,
            'pitch_class_emb': pitch_class_emb,
            'instrument_emb': instrument_emb,
            'velocity_emb': velocity_emb,
            'combined_embedding': combined_embedding,
        }
        
        if return_attention_weights:
            result.update({
                'onset_probs': onset_probs,
                'duration_probs': duration_out_probs,
                'octave_probs': octave_probs,
                'pitch_class_probs': pitch_class_probs,
                'instrument_probs': instrument_out_probs,
                'velocity_probs': velocity_out_probs,
            })
        
        return result