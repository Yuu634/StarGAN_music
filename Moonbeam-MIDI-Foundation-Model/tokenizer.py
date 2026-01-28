from src.llama_recipes.datasets.music_tokenizer import MusicTokenizer
from transformers import LlamaConfig

model_config_path = "src/llama_recipes/configs/player_classification_config.json"
llama_config = LlamaConfig.from_pretrained(model_config_path)
tokenizer = MusicTokenizer(timeshift_vocab_size = llama_config.onset_vocab_size, dur_vocab_size = llama_config.dur_vocab_size, octave_vocab_size = llama_config.octave_vocab_size, pitch_class_vocab_size = llama_config.pitch_class_vocab_size, instrument_vocab_size = llama_config.instrument_vocab_size, velocity_vocab_size = llama_config.velocity_vocab_size, sos_token = llama_config.sos_token, eos_token = llama_config.eos_token, pad_token = llama_config.pad_token) ##TODO:Done,  ADD NEW PLAYER CLASSIFICATION TOKEN 
#print(tokenizer.key)

attrs = [a for a in dir(tokenizer) if not a.startswith("_")]
print(attrs)