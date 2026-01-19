import torch

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