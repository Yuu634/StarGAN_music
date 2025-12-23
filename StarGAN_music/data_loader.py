from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import json
import numpy as np


class MusicDataset(data.Dataset):
    """Dataset class for the Music dataset."""

    def __init__(self, score_dir, attr_path, selected_attrs, mode):
        """Initialize and preprocess the Music dataset."""
        self.score_dir = score_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)
 
    def preprocess(self):
        """Preprocess the Music attribute file."""
        with open(self.attr_path, 'r') as f:
            for i, line in enumerate(f):
                data = json.loads(line)

                filepath = data.get('location')
                
                label = []
                for attr_name in ['funk', 'celtic', 'instrumentalpop', 'ambient', 'reggae', 'popfolk', 'dance', 'rock', 'classical', 'instrumentalrock', 'folk', 'poprock', 'indie', 'hiphop', 'blues', 'experimental', 'punkrock', 'jazz', 'electronic', 'techno', 'jazzfusion', 'pop', 'alternative', 'electropop', 'soundtrack', 'trance', 'house', 'metal', 'world', 'symphonic', 'lounge', 'easylistening', 'orchestral', 'country', 'newage', 'latin', 'drumnbass', '80s', '90s', 'swing', 'chillout', 'synthpop', 'movie', 'christmas', 'heavy', 'corporate', 'action', 'romantic', 'energetic', 'background', 'children', 'calm', 'adventure', 'motivational', 'summer', 'funny', 'dramatic', 'cool', 'positive', 'emotional', 'holiday', 'deep', 'love', 'dark', 'dream', 'advertising', 'happy', 'soundscape', 'film', 'melodic', 'drama', 'uplifting', 'epic', 'ballad', 'sad', 'relaxing', 'party', 'trailer', 'inspiring', 'soft', 'slow', 'game', 'retro', 'fun', 'meditative', 'sport', 'space', 'commercial', 'documentary', 'upbeat', 'Eb major', 'B major', 'Bb major', 'F# minor', 'F# major', 'G# minor', 'A major', 'B minor', 'E minor', 'D minor', 'F minor', 'G minor', 'F major', 'Eb minor', 'C major', 'A minor', 'G major', 'D major', 'C# major', 'Bb minor', 'Ab major', 'C# minor', 'C minor', 'E major']:
                    if attr_name in data.get('genre', []):
                        label.append(True)
                    elif attr_name in data.get('mood', []):
                        label.append(True)
                    elif attr_name in (data.get('key') or []):
                        label.append(True)
                    else:
                        label.append(False)
                        
                if (i+1) < 2000:
                    self.test_dataset.append([filepath, label])
                else:
                    self.train_dataset.append([filepath, label])

        print('Finished preprocessing the Music dataset...')

    def __getitem__(self, index):
        """Return one score and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filepath, label = dataset[index]
        filepath = self.score_dir + filepath.removesuffix(".mid") + ".npz"
        score = np.load(filepath)['arr_0']
        return score, label

    def __len__(self):
        """Return the number of scores."""
        return self.num_images


def get_loader(score_dir, attr_path, selected_attrs, 
               batch_size=1, dataset='Score', mode='train', num_workers=1):
    """Build and return a data loader."""

    dataset = MusicDataset(score_dir, attr_path, selected_attrs, mode)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader