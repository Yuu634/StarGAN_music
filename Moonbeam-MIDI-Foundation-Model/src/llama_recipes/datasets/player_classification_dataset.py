import copy
import json

import torch
from torch.utils.data import Dataset
import glob
import mido
from collections import defaultdict
import json
from concurrent.futures import ProcessPoolExecutor
import tqdm
import numpy as np
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
import random

def process_file(args):
    """Helper function to process a single file lazily."""
    label, filename, data_dir, seq_len, non_overlap_ratio, partition = args
    raw_tokens = np.load(os.path.join(data_dir, "processed", filename))
    
    chunked_files, chunked_labels = [], []
    slow, fast = 0, 0

    while fast < len(raw_tokens):
        # Calculate the duration of the current token
        onset_time_slow = raw_tokens[slow][0] / 100  # Convert to seconds
        onset_time_fast = raw_tokens[fast][0] / 100  # Convert to seconds
        current_duration = onset_time_fast - onset_time_slow
        
        if current_duration <= seq_len:
            fast += 1
        else:
            abs_onset = raw_tokens[slow][0]
            raw_tokens_adjust_onset = np.array([[event[0]-abs_onset]+event[1:] for event in raw_tokens[slow:fast].tolist()])
            # chunked_files.append(raw_tokens[slow:fast])
            chunked_files.append(raw_tokens_adjust_onset)
            chunked_labels.append(label)
            
            if partition == "train":  # Create overlap
                slow_new = int(slow + (fast - slow) * non_overlap_ratio)
                if slow_new == slow:  # Ensure progress
                    print(f"breaking the loop to avoid infinite loop: {slow_new} slow old{slow}, len(raw_tokens): {len(raw_tokens)}")
                    slow+=1
                    # break
                else:
                    slow = slow_new
                fast = slow
            elif partition == "test":  # No overlap
                slow = fast
    
    return chunked_files, chunked_labels 
class PlayerClassificationDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        assert partition=="train" or partition=="test"
        self.data_dir = dataset_config.data_dir
        self.seq_len = dataset_config.seq_len #seq duration in seconds
        self.seq_dur = dataset_config.seq_dur
        self.non_overlap_ratio = dataset_config.non_overlap_ratio
        split_data = pd.read_csv(dataset_config.csv_file)
        file_basenames = split_data['file_base_name'].values
        splits = split_data['split'].values     
        labels = split_data['label'].values   
        self.file_basenames = [f for f, s in zip(file_basenames, splits) if s == partition]
        self.labels = [l for l, s in zip(labels, splits) if s == partition]
        self.partition = partition

        # Check seq_dur and seq_len: both can be None, or one can have a value while the other is None, 
        # but they cannot both have values at the same time.
        if dataset_config.individual_eval and partition == "test":
            print("Evaluate each file individually without chunking during evaluation.")
            self.chunked_files, self.chunked_labels = [np.load(os.path.join(self.data_dir, "processed",filename))  for filename in self.file_basenames], self.labels
        else:
            if self.seq_len and not self.seq_dur: #chunk file based on seq_len
                print(f"Chunk file based on seq_len = {self.seq_len} events")
                self.chunked_files, self.chunked_labels = self.get_chunked_files_labels_based_on_seq_len()
            elif self.seq_dur and not self.seq_len: #chunk file based on seq_dur
                print(f"chunk file based on seq_dur = {self.seq_dur} seconds")
                self.chunked_files, self.chunked_labels = self.get_chunked_files_labels_based_on_dur_parallel()
            elif not self.seq_dur and not self.seq_len: #concat all events in midi file
                print("concat all events in midi file")
                self.chunked_files, self.chunked_labels = [np.load(os.path.join(self.data_dir, "processed",filename))  for filename in self.file_basenames], self.labels
            else:
                raise ValueError("seq_len and seq_dur cannot be both not None")
        
        self.tokenizer = tokenizer

    def get_chunked_files_labels_based_on_seq_len(self):
        self.chunked_files, self.chunked_labels = [], []
        for label, filename in zip(self.labels, self.file_basenames):
            raw_tokens = np.load(os.path.join(self.data_dir, "processed",filename))    
            # Split the raw tokens into chunks of length self.seq_len
            hop_length = int(self.seq_len * self.non_overlap_ratio)
            # Process chunks with the defined hop_length
            for start_idx in range(0, len(raw_tokens) - self.seq_len + 1, hop_length):
                chunk = raw_tokens[start_idx:start_idx + self.seq_len]
                self.chunked_files.append(chunk)
                self.chunked_labels.append(label)

            # Handle the remainder, if any
            remainder_start = len(raw_tokens) - (len(raw_tokens) % hop_length)
            if remainder_start<len(raw_tokens):
                remainder = raw_tokens[remainder_start:]

                self.chunked_files.append(remainder)
                self.chunked_labels.append(label)
        return self.chunked_files, self.chunked_labels
    
    def get_chunked_files_labels_based_on_dur_parallel(self):
        import os
        import numpy as np
        from multiprocessing import Pool, cpu_count
    
        self.chunked_files, self.chunked_labels = [], []
        args = [
            (label, filename, self.data_dir, self.seq_dur, self.non_overlap_ratio, self.partition)
            for label, filename in zip(self.labels, self.file_basenames)
        ]
        
        with Pool(cpu_count()) as pool:
            # Use `pool.imap` for lazy iteration
            for chunked_files, chunked_labels in pool.imap(process_file, args):
                self.chunked_files.extend(chunked_files)
                self.chunked_labels.extend(chunked_labels)
        
        return self.chunked_files, self.chunked_labels           
    
    def __len__(self):
        return len(self.chunked_files)

    def __getitem__(self, index):
        # encoded_tokens = self.tokenizer.encode_series(raw_tokens, if_add_sos = True, if_add_eos = True) #SOS and EOS are added in the tokenizer
        encoded_tokens = self.tokenizer.encode_series_player_classification(self.chunked_files[index], if_add_sos = True, if_add_eos = True, if_add_classification_token=True) #SOS and EOS are added in the tokenizer

        classification_labels = [self.chunked_labels[index] for _ in range(len(encoded_tokens))] 

        return {
            "input_ids": encoded_tokens,
            "labels": classification_labels,
            "attention_mask":[0 for _ in range(len(encoded_tokens))] #dummy attention mask
        }
