#!/bin/bash
# Quick start script for StarGAN with real Amadeus and Moonbeam models

echo "============================================================"
echo "StarGAN with Real Models - Quick Start"
echo "============================================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Check dependencies
echo -e "\n${YELLOW}Step 1: Checking dependencies...${NC}"
python3 -c "import torch; print('✓ PyTorch:', torch.__version__)" || { echo -e "${RED}✗ PyTorch not found${NC}"; exit 1; }
python3 -c "import transformers; print('✓ Transformers:', transformers.__version__)" || { echo -e "${RED}✗ Transformers not found${NC}"; exit 1; }
python3 -c "import yaml; print('✓ PyYAML installed')" || { echo -e "${RED}✗ PyYAML not found${NC}"; exit 1; }
echo -e "${GREEN}All dependencies installed!${NC}"

# Step 2: Set paths (EDIT THESE!)
echo -e "\n${YELLOW}Step 2: Setting model paths...${NC}"
echo -e "${YELLOW}Please edit this script to set your actual paths!${NC}"

AMADEUS_CONFIG="/mnt/kiso-qnap5/obara/Amadeus/symbolic_yamls/your_config.yaml"
AMADEUS_CHECKPOINT="/mnt/kiso-qnap5/obara/Amadeus/models/checkpoint.pt"
MOONBEAM_CONFIG="/mnt/kiso-qnap5/obara/Moonbeam-MIDI-Foundation-Model/src/llama_recipes/configs/player_classification_config.json"
MOONBEAM_CHECKPOINT="/mnt/kiso-qnap5/obara/Moonbeam-MIDI-Foundation-Model/checkpoints/checkpoint.pt"

# Check if paths exist
if [ ! -f "$AMADEUS_CONFIG" ]; then
    echo -e "${RED}✗ Amadeus config not found: $AMADEUS_CONFIG${NC}"
    echo -e "${YELLOW}Please update AMADEUS_CONFIG in this script${NC}"
    exit 1
fi

if [ ! -f "$MOONBEAM_CONFIG" ]; then
    echo -e "${RED}✗ Moonbeam config not found: $MOONBEAM_CONFIG${NC}"
    echo -e "${YELLOW}Please update MOONBEAM_CONFIG in this script${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Config files found${NC}"

# Step 3: Test model loading
echo -e "\n${YELLOW}Step 3: Testing model loading...${NC}"
python3 test_real_models.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Model loading test passed!${NC}"
else
    echo -e "${RED}✗ Model loading test failed${NC}"
    echo -e "${YELLOW}Please check the error messages above${NC}"
    exit 1
fi

# Step 4: Training (TODO: Add data path)
echo -e "\n${YELLOW}Step 4: Training (TODO)${NC}"
echo -e "${YELLOW}To start training, run:${NC}"
echo ""
echo "python3 train_stargan_real.py \\"
echo "    --amadeus_config $AMADEUS_CONFIG \\"
echo "    --amadeus_checkpoint $AMADEUS_CHECKPOINT \\"
echo "    --moonbeam_config $MOONBEAM_CONFIG \\"
echo "    --moonbeam_checkpoint $MOONBEAM_CHECKPOINT \\"
echo "    --data_dir /path/to/training/data \\"
echo "    --batch_size 16 \\"
echo "    --num_epochs 10 \\"
echo "    --save_dir ./checkpoints"

echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}Setup complete! Ready for training.${NC}"
echo -e "${GREEN}============================================================${NC}"
