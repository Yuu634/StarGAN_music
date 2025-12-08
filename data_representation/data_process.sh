#!/usr/bin/env bash
# run_all.sh

BASE_DATASET="lmd_full"
NUM_FEATURES=5
ENCODING="nb"
IN_DIR="../../Dataset/MidiCaps"
OUT_DIR="../dataset/MidiCaps/corpus/"
SUBDIRS=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "a" "b" "c" "d" "e" "f")

for SUBDIR in "${SUBDIRS[@]}"; do
    DATASET="${BASE_DATASET}/${SUBDIR}"

    echo "Running step1 with dataset=$DATASET"
    python3 step1_midi2corpus.py --dataset "$DATASET" --num_features "$NUM_FEATURES" --in_dir "$IN_DIR" --out_dir "$OUT_DIR"

    echo "Running step2 with dataset=$DATASET"
    python3 step2_corpus2event.py --dataset "$DATASET" --num_features "$NUM_FEATURES" --encoding "$ENCODING" --in_dir "$IN_DIR" --out_dir "$OUT_DIR"

    echo "Running step3 with dataset=$DATASET"
    python3 step3_creating_vocab.py --dataset "$DATASET" --num_features "$NUM_FEATURES" --encoding "$ENCODING" --in_dir "$IN_DIR" --out_dir "$OUT_DIR"

    echo "Running step4 with dataset=$DATASET"
    python3 step4_event2tuneidx.py --dataset "$DATASET" --num_features "$NUM_FEATURES" --encoding "$ENCODING" --in_dir "$IN_DIR" --out_dir "$OUT_DIR"
done
