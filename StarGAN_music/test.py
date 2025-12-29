import torch
import numpy as np
import re

def amadeus_to_moonbeam(amadeus_tokens, time_resolution=10, default_tempo=120, in_beat_resolution=4):
    """
    Convert Amadeus representation to Moonbeam representation.
    
    Args:
        amadeus_tokens: torch.Tensor or np.ndarray or list
            Shape: [8, num_notes] or [num_notes, 8]
            Amadeus format: (type, beat, chord, tempo, instrument, pitch, duration, velocity)
        time_resolution: int
            Time resolution in ms per tick (default: 10ms)
        default_tempo: int
            Default tempo if not specified (default: 120 BPM)
        in_beat_resolution: int
            Number of subdivisions per quarter note (default: 4, meaning 16th note resolution)
    
    Returns:
        moonbeam_tokens: torch.Tensor or np.ndarray
            Shape: [num_notes, 6]
            Moonbeam format: (onset, duration, octave, pitch_class, instrument, velocity)
    """
    
    # Convert to numpy for easier processing
    if isinstance(amadeus_tokens, torch.Tensor):
        amadeus_np = amadeus_tokens.cpu().numpy()
        use_torch = True
    elif isinstance(amadeus_tokens, list):
        amadeus_np = np.array(amadeus_tokens)
        use_torch = False
    else:
        amadeus_np = np.array(amadeus_tokens)
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
    
    for i in range(num_notes):
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


# Test data
amadeus_array = [
    ['NNN_time_signature_4/4', 'Beat_0', 'Chord_C_Major', 'Tempo_180', 'Instrument_0', 'Note_Pitch_60', 'Note_Duration_4.0', 'Note_Velocity_80'],
    ['SSN', 'Beat_4', 'Chord_C_Major', 'Tempo_180', 'Instrument_0', 'Note_Pitch_64', 'Note_Duration_2.0', 'Note_Velocity_75'],
    ['SSN', 'Beat_12', 'Chord_G_Major', 'Tempo_180', 'Instrument_0', 'Note_Pitch_67', 'Note_Duration_6.0', 'Note_Velocity_85'],
    ['SNN', 'Beat_0', 'Chord_G_Major', 'Tempo_180', 'Instrument_0', 'Note_Pitch_72', 'Note_Duration_4.0', 'Note_Velocity_85'],
]

# Convert
moonbeam_tokens = amadeus_to_moonbeam(amadeus_array)
print("Moonbeam representation shape:", moonbeam_tokens.shape)
print("Moonbeam tokens:\n", moonbeam_tokens)
print("\nFormat: [onset, duration, octave, pitch_class, instrument, velocity]")