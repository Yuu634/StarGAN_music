# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]  # Moonbeam-MIDI-Foundation-Model/
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
import fire
from llama_recipes.real_finetuning_player_classification import main

if __name__ == "__main__":
    fire.Fire(main)