# GitHub Copilot Instructions

## Architecture at a Glance
- `StarGAN_music/` adapts the original StarGAN loop for symbolic music: `main.py` wires CLI args → `solver.Solver`, which in turn delegates generation to Amadeus (`Amadeus/generate.py`, `Amadeus/Amadeus/model_zoo.py`) and discrimination to Moonbeam (`Moonbeam-MIDI-Foundation-Model/inference.py`).
- `Amadeus/` holds the symbolic model plus preprocessing scripts under `data_representation/` (step1–4 convert MIDI → events → vocab) and downloadable checkpoints in `models/Amadeus-S`.
- `Moonbeam-MIDI-Foundation-Model/` provides the ScoreArrange domain classifier and all llama_recipes utilities; StarGAN only needs the lightweight inference path plus LoRA adapters in `models/pretrained`.
- `Dataset/` mirrors MidiCaps (captions, splits, raw MIDI) so StarGAN can fetch score tensors and conditioning text without touching disk mid-training.

## Core Workflows
- **Data prep (Amadeus)**: run `python data_representation/step1_midi2corpus.py --dataset <name> --num_features <4|5|7|8>` followed by steps 2–4 to emit `dataset/represented_data/*` and `vocab/*.json`; keep encoding scheme consistent with the vocab referenced by `load_resources`.
- **Model assets**: call `python generate.py -wandb_exp_dir models/Amadeus-S ...` once to download Amadeus checkpoints; Moonbeam checkpoints live under `Moonbeam-MIDI-Foundation-Model/models/pretrained/` and must match the config in `src/llama_recipes/configs/player_classification_config.json`.
- **Training StarGAN-Music**: from `StarGAN_music/StarGAN_music`, run `python main.py --mode train --dataset MidiCaps --g_modelpath ../Amadeus/models/Amadeus-S --d_modelpath ../Moonbeam.../lora_adapter.pt ...`. Samples, logs, and checkpoints go to the `*_dir` flags configured in `main.py`.
- **Evaluation/Inference**: switch `--mode test` and set `--test_iters` to the checkpoint step; the solver will reload generator/discriminator weights and emit translated MIDI to `result_dir`.
- **Discriminator fine-tuning**: when LoRA weights need updating, use `torchrun ... recipes/finetuning/real_finetuning_player_classification.py` with dataset-specific config overrides (see README tables) and reinstall `src/llama_recipes/transformers_minimal` after config edits.

## Coding Patterns & Conventions
- Treat Amadeus tokens as 8-feature rows (`type, beat, chord, tempo, instrument, pitch, duration, velocity`). Helper `solver.Solver.amadeus_to_moonbeam` must be used before feeding Moonbeam classifiers.
- Generator checkpoints are loaded via `Amadeus/generate.load_resources`, which returns `(config, model, vocab)`; always move the model to `self.device` and keep the returned vocab for decoding.
- Conditioning text is encoded with HuggingFace `T5Tokenizer` + `T5EncoderModel`; tensors need `.to(self.device)` before passing into Amadeus contexts.
- Learning-rate schedules follow the original StarGAN pattern: manual decay after `num_iters - num_iters_decay`, gradient penalty via `Solver.gradient_penalty`.
- Logs/samples are sparse on purpose; prefer inserting debug prints over TensorBoard unless `--use_tensorboard` is true (logger defined in `StarGAN/logger.py`).

## External Integration Tips
- Amadeus expects vocab metadata from `vocab/*`; if you add new features, regenerate vocab + retrain before invoking `load_resources`.
- Moonbeam’s `ScoreArrangeDomainClassifier` pulls tokenizer/config paths from `src/llama_recipes/configs`; keep `classification_token` and `num_labels` in sync with the StarGAN attribute list.
- Text/MIDI alignment for MidiCaps lives in `Dataset/MidiCaps/train.json` and related files; loader classes read these JSON files directly, so preserve field names when extending datasets.
- When installing environments, prefer the provided YAMLs: `conda env create -f Amadeus/environment.yml` for generator tooling, and `pip install -e Moonbeam-MIDI-Foundation-Model` followed by `pip install Moonbeam-MIDI-Foundation-Model/src/llama_recipes/transformers_minimal/.` for discriminator utilities.

## Troubleshooting
- CUDA vs CPU mismatches commonly arise because Moonbeam defaults to CUDA; wrap its device argument with `"cuda" if torch.cuda.is_available() else "cpu"` and ensure tensors fed into it are on the same device.
- `amadeus_to_moonbeam` assumes `current_bar` increments whenever a type token begins with `NNN`; malformed tokens lead to negative onsets—add asserts/logs before conversion when debugging datasets.
- If tokenizer/model weights fail to download (e.g., `google/flan-t5-large`), set `HF_HOME` to a writable directory inside the workspace to avoid permission issues on shared systems.
