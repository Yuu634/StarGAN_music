# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from grammar_dataset.grammar_dataset import get_dataset as get_grammar_dataset
from alpaca_dataset import InstructionDataset as get_alpaca_dataset
from samsum_dataset import get_preprocessed_samsum as get_samsum_dataset
from lakh_dataset import LakhDataset as get_lakhmidi_dataset
from merge_dataset import MergeDataset as get_merge_dataset
from player_classification_dataset import PlayerClassificationDataset as get_player_classification_dataset