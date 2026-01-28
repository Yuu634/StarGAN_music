import json
import torch

vocab_path = "../Amadeus/models/Amadeus-S/files/checkpoints/vocab_LakhALLFined_nb8.json"
ckpt_path = "../Moonbeam-MIDI-Foundation-Model/models/pretrained/moonbeam_309M.pt"

with open(vocab_path, "r", encoding="utf-8") as f:
    vocab = json.load(f)
print(vocab['type'])


ckpt = torch.load(ckpt_path, map_location="cpu")
# 典型的なパターン
state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
#print(state.keys())  # 語彙らしきキーがあるか確認

# もし 'vocab' や 'vocab_list' があれば取り出し
vocab_list = state.get("vocab") or state.get("vocab_list")
#print(vocab_list)