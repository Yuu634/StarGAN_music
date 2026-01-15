import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import time
import json

from . import transformer_utils
from . import sub_decoder_zoo
from x_transformers.x_transformers import LayerIntermediates, AbsolutePositionalEmbedding
from data_representation.vocab_utils import LangTokenVocab
import os

class AmadeusModelWrapper(nn.Module):
  def __init__(
    self, 
    *, 
    vocab:LangTokenVocab,                 
    input_length:int,           
    prediction_order:list,       
    input_embedder_name:str,    
    main_decoder_name:str,      
    sub_decoder_name:str,       
    sub_decoder_depth:int,      
    sub_decoder_enricher_use:bool, 
    dim:int,                    
    heads:int,                 
    depth:int,                 
    dropout:float,
    is_regressive:bool
  ):
    '''
    This class wraps the three main components of the AmadeusModel model,
    which are the input embedding layer, the main transformer decoder, and the sub-decoder.
    '''

    super().__init__()
    self.vocab = vocab
    self.vocab_size = vocab.get_vocab_size()
    self.start_token = vocab.sos_token if hasattr(vocab, 'sos_token') else None
    self.end_token = vocab.eos_token if hasattr(vocab, 'eos_token') else None
    self.input_length = input_length
    self.prediction_order = prediction_order
    self._get_input_embedder(input_embedder_name, vocab, dropout, dim)
    self._get_main_decoder(main_decoder_name, input_length, dim, heads, depth, dropout)
    self._get_sub_decoder(sub_decoder_name, prediction_order, vocab, sub_decoder_depth, sub_decoder_enricher_use, dim, heads, dropout)
    self.bos_token_hidden = None
    self.is_regressive = is_regressive
 
  def _get_input_embedder(self, input_embedder_name, vocab, dropout, dim):
    self.emb_dropout = nn.Dropout(dropout)
    self.input_embedder = getattr(transformer_utils, input_embedder_name)(
      vocab=vocab,
      dim_model=dim
    )

  def _get_main_decoder(self, main_decoder_name, input_length, dim, heads, depth, dropout):
    self.pos_enc = AbsolutePositionalEmbedding(dim, input_length)
    self.main_norm = nn.LayerNorm(dim)
    self.main_decoder = getattr(transformer_utils, main_decoder_name)(
      dim=dim,
      depth=depth,
      heads=heads,
      dropout=dropout
    )
  
  def _get_sub_decoder(self, sub_decoder_name, prediction_order, vocab, sub_decoder_depth, sub_decoder_enricher_use, dim, heads, dropout):
    self.sub_decoder = getattr(sub_decoder_zoo, sub_decoder_name)(
      prediction_order=prediction_order,
      vocab=vocab,
      dim=dim,
      sub_decoder_depth=sub_decoder_depth,
      heads=heads,
      dropout=dropout,
      sub_decoder_enricher_use=sub_decoder_enricher_use
    )

  @property
  def device(self):
    return next(self.parameters()).device

  def forward(self, input_seq:torch.Tensor, target:torch.Tensor, context=None):
    if self.is_regressive:
      return self.forward_stepwise(input_seq, target, context=context)
    else:
      embedding = self.input_embedder(input_seq) + self.pos_enc(input_seq)
      embedding = self.emb_dropout(embedding)
      hidden_vec,layer_inter = self.main_decoder(embedding,train=True, context=context)  # B x T x d_model
      hidden_vec = self.main_norm(hidden_vec)
      input_dict = {'hidden_vec':hidden_vec, 'input_seq': input_seq, 'target': target, 'bos_token_hidden': self.bos_token_hidden}
      logits = self.sub_decoder(input_dict)
      # 选择总数中离三分之一最近的层
      num_layers = len(layer_inter.layer_hiddens)
      idx = round(num_layers / 3)
      idx = min(max(idx, 0), num_layers - 1)
      input_dict['hidden_vec'] = layer_inter.layer_hiddens[idx]
      return logits[0], input_dict
      

  def _init_generated_seq(self, batch_size:int, device:torch.device):
    if self.start_token is not None:
      start = torch.tensor(self.start_token, device=device)
      if start.ndim == 1:
        start = start.unsqueeze(0)
      start = start.unsqueeze(0) if start.ndim == 2 else start
      return start.repeat(batch_size, 1, 1)  # B x 1 x num_features
    # fallback: zeros
    return torch.zeros(batch_size, 1, len(self.vocab.feature_list), device=device, dtype=torch.long)

  def _stack_logits(self, logits_steps, target_T:int):
    stacked = {}
    for step_logits in logits_steps:
      for k, v in step_logits.items():
        stacked.setdefault(k, []).append(v)
    for k in stacked.keys():
      stacked[k] = torch.cat(stacked[k], dim=1)  # B x T x vocab
    # pad missing steps if any
    for k, v in stacked.items():
      if v.shape[1] != target_T:
        pad = target_T - v.shape[1]
        stacked[k] = torch.cat([v, v.new_zeros(v.shape[0], pad, v.shape[2])], dim=1)
    return stacked

  def forward_stepwise(self, input_seq:torch.Tensor, target:torch.Tensor, context=None):
    """
    Step-wise conditioned forward for accompaniment - Memory-optimized version:
    - At step n, build sequence = [generated tokens (<= n-1), input score token at n]
    - main_decoder: no_grad for steps < T-1, compute gradients only for final step
    - sub_decoder: compute gradients for all steps
    - Sample argmax token from logits to extend generated sequence.
    """
    B, T = input_seq.shape[0], input_seq.shape[1]
    device = input_seq.device

    generated_seq = self._init_generated_seq(B, device)
    logits_steps = []
    aux_steps = []

    for t in range(1, T):
      score_token = input_seq[:, t:t+1]  # B x 1 x feat
      concat_seq = torch.cat([generated_seq, score_token], dim=1)  # B x L x feat

      # ✅ Memory optimization: Use no_grad for main_decoder in intermediate steps
      if t < T - 1:  # All steps except the last one
        with torch.no_grad():
          embedding = self.input_embedder(concat_seq) + self.pos_enc(concat_seq)
          embedding = self.emb_dropout(embedding)
          hidden_vec, layer_inter = self.main_decoder(embedding, train=True, context=context)
          hidden_vec = self.main_norm(hidden_vec)
          # Detach to prevent gradient computation
          hidden_vec = hidden_vec.detach()
      else:  # Last step: compute gradients for main_decoder
        embedding = self.input_embedder(concat_seq) + self.pos_enc(concat_seq)
        embedding = self.emb_dropout(embedding)
        hidden_vec, layer_inter = self.main_decoder(embedding, train=True, context=context)
        hidden_vec = self.main_norm(hidden_vec)

      # use last hidden only for current prediction
      hidden_last = hidden_vec[:, -1:, :]
      target_slice = target[:, t:t+1] if target is not None else None
      input_dict = {
        'hidden_vec': hidden_last,
        'input_seq': concat_seq,
        'target': target_slice,
        'bos_token_hidden': self.bos_token_hidden
      }

      # ✅ sub_decoder: Always compute gradients
      sub_out = self.sub_decoder(input_dict)
      aux_outputs = None
      if isinstance(sub_out, tuple) and len(sub_out) == 2:
        logits, aux_outputs = sub_out
      else:
        logits = sub_out
      aux_steps.append(aux_outputs)

      # logits is dict(feature -> B x 1 x vocab) for DiffusionDecoder
      logits_steps.append(logits)

      # argmax sample to feed next step (no gradients needed)
      with torch.no_grad():
        sampled_tokens = []
        for feature in self.prediction_order:
          feat_logit = logits[feature]  # B x 1 x V
          sampled = torch.argmax(feat_logit, dim=-1)  # B x 1
          sampled_tokens.append(sampled)
        sampled_token = torch.stack(sampled_tokens, dim=-1)  # B x 1 x num_features
      
      generated_seq = torch.cat([generated_seq, sampled_token], dim=1)
      
      # ✅ Explicit memory cleanup
      del embedding, hidden_vec, input_dict
      torch.cuda.empty_cache() if torch.cuda.is_available() else None

    stacked_logits = self._stack_logits(logits_steps, T)
    return stacked_logits, {'generated_seq': generated_seq, 'aux_outputs': aux_steps}
 

class AmadeusModelAutoregressiveWrapper(nn.Module):
  def __init__(self, net:AmadeusModelWrapper):
    '''
    Initializes an autoregressive wrapper around the AmadeusModelWrapper, 
    which allows sequential token generation.
    '''
    super().__init__()
    self.net = net

  def forward(self, input_seq:torch.Tensor, target:torch.Tensor,context=None): 
    return self.net(input_seq, target, context=context)
  
  def _prepare_inference(self, start_token, manual_seed, condition=None, num_target_measures=4):
    '''
    Prepares the initial tokens for autoregressive inference. If a manual seed is provided, 
    it sets the seed for reproducibility. If a condition is given, it selects a subset of 
    the tokens based on certain criteria related to the encoding scheme.

    Arguments:
    - start_token: The token that represents the start of a sequence.
    - manual_seed: A seed value for reproducibility in inference (if greater than 0).
    - condition: An optional tensor used for conditional generation, which helps select a 
      portion of the input tokens based on the encoding scheme.

    Returns:
    - total_out: A tensor containing the initial tokens for inference, padded to ensure compatibility 
      with the model.
    '''
    if manual_seed > 0:
      torch.manual_seed(manual_seed)

    total_out = []
    if condition is None:
      # Use the start token if no condition is given
      total_out.extend(start_token)
    else:
      # Extract the portion of the sequence depending on encoding scheme (remi, cp, or nb)
      if self.net.vocab.encoding_scheme == 'remi':
        type_boundaries = self.net.vocab.remi_vocab_boundaries_by_key['type']
        # vocab idx -> 0:SOS, 1:EOS, 2:Bar_without_time_signature, ... where_type_ends:Bar_time_signature_end, ...
        measure_bool = (2 <= condition) & (condition < type_boundaries[1]) # between Bar_ts_start and Bar_ts_end 
        conditional_input_len = torch.where(measure_bool)[0][num_target_measures].item()
      elif self.net.vocab.encoding_scheme == 'cp':
        # find the start and end of the measure
        beat_event2idx = self.net.vocab.event2idx['beat']
        for event, idx in beat_event2idx.items():
          if event == 0:
            continue
          if event == 'Bar':
            start_idx = idx
          elif event.startswith('Beat'):
            end_idx = idx
            break
        measure_bool = (condition[:,1] >= start_idx) & (condition[:,1] < end_idx)  # measure tokens
        conditional_input_len = torch.where(measure_bool)[0][num_target_measures].item()
        # measure_bool = (condition[:,1] == 1)  # measure tokens
        conditional_input_len = torch.where(measure_bool)[0][num_target_measures].item()
      elif self.net.vocab.encoding_scheme == 'nb':
        measure_bool = (condition[:,0] == 2) | (condition[:,0] >= 5)  # Empty measure or where new measure starts
        conditional_input_len = torch.where(measure_bool)[0][num_target_measures].item()

      if conditional_input_len == 0:
        conditional_input_len = 50

      selected_tokens = condition[:conditional_input_len].tolist()
      total_out.extend(selected_tokens)

    total_out = torch.LongTensor(total_out).unsqueeze(0).to(self.net.device)
    return total_out

  def _run_one_step(self, input_seq, cache=None, sampling_method=None, threshold=None, temperature=1, bos_hidden_vec=None,context=None):
    '''
    Runs one step of autoregressive generation by taking the input sequence, embedding it,
    passing it through the main decoder, and generating logits and a sampled token.

    Arguments:
    - input_seq: The input sequence tensor to be embedded and processed.
    - cache: Optional cache for attention mechanisms to avoid recomputation.
    - sampling_method: Sampling strategy used to select the next token.
    - threshold: Optional threshold value for sampling methods that require it.
    - temperature: Controls the randomness of predictions (higher temperature increases randomness).

    Returns:
    - logits: The predicted logits for the next token.
    - sampled_token: The token sampled from the logits.
    - intermidiates: Intermediate states from the main decoder, useful for caching.
    '''
    embedding = self.net.input_embedder(input_seq) + self.net.pos_enc(input_seq)
    embedding = self.net.emb_dropout(embedding)

    # Run through the main decoder and normalize
    hidden_vec, intermidiates = self.net.main_decoder(embedding, cache,context_embedding=context)  # B x T x d_model
    hidden_vec = self.net.main_norm(hidden_vec)
    hidden_vec = hidden_vec[:, -1:]  # Keep only the last time step

    input_dict = {'hidden_vec': hidden_vec, 'input_seq': input_seq, 'target': None, 'bos_token_hidden': bos_hidden_vec}

    # Generate the next token
    logits, sampled_token = self.net.sub_decoder(input_dict, sampling_method, threshold, temperature)
    return logits, sampled_token, intermidiates, hidden_vec

  def _update_total_out(self, total_out, sampled_token):
    '''
    Updates the output sequence with the newly sampled token. Depending on the encoding scheme, 
    it either appends the token directly or processes feature-based sampling.

    Arguments:
    - total_out: The tensor containing the previously generated tokens.
    - sampled_token: The newly generated token to be appended.

    Returns:
    - total_out: Updated output tensor with the newly generated token.
    - sampled_token: The processed sampled token.
    '''
    if self.net.vocab.encoding_scheme == 'remi':
      # For remi encoding, directly append the sampled token
      total_out = torch.cat([total_out, sampled_token.unsqueeze(0)], dim=-1)
    else:
      # Handle other encoding schemes by concatenating features
      sampled_token_list = []
      for key in self.net.vocab.feature_list:
        sampled_token_list.append(sampled_token[key])
      sampled_token = torch.cat(sampled_token_list, dim=-1)
      # print(total_out.shape)
      if len(sampled_token.shape) == 2:
        total_out = torch.cat([total_out, sampled_token.unsqueeze(0)], dim=1)
      total_out = torch.cat([total_out, sampled_token.unsqueeze(0).unsqueeze(0)], dim=1)

    return total_out, sampled_token

  @torch.inference_mode()
  def generate(self, manual_seed, max_seq_len, condition=None, num_target_measures=4, sampling_method=None, threshold=None, temperature=1, batch_size=1, context=None, input_note=None):
    '''
    Autoregressively generates a sequence of tokens by repeatedly sampling the next token 
    until the desired maximum sequence length is reached or the end token is encountered.

    Arguments:
    - manual_seed: A seed value for reproducibility in inference.
    - max_seq_len: The maximum length of the generated sequence.
    - condition: An optional conditioning sequence to start generation from.
    - sampling_method: The method used to sample the next token (e.g., greedy, top-k).
    - threshold: Optional threshold for sampling (used in methods like top-p sampling).
    - temperature: Controls the randomness of the token sampling process.
    - batch_size: The number of sequences to generate in parallel.

    Returns:
    - total_out: The generated sequence of tokens as a tensor.
    '''  
    # Prepare the starting sequence for inference
    total_out = self._prepare_inference(self.net.start_token, manual_seed, condition, num_target_measures)

    # === NEW: Process input_note if provided ===
    generation_step = 0

    # If a condition is provided, run one initial step
    if condition is not None:
      if input_note is not None:
        note_vec = input_note[generation_step].unsqueeze(0).unsqueeze(0)
        _, _, cache = self._run_one_step(torch.cat([total_out[:, -self.net.input_length:], note_vec], dim=1), cache=LayerIntermediates(), sampling_method=sampling_method, threshold=threshold, temperature=temperature, context=context)
      else:
        _, _, cache = self._run_one_step(total_out[:, -self.net.input_length:], cache=LayerIntermediates(), sampling_method=sampling_method, threshold=threshold, temperature=temperature, context=context)
    else:
      cache = LayerIntermediates()

    # Continue generating tokens until the maximum sequence length is reached
    pbar = tqdm(total=max_seq_len, desc="Generating tokens", unit="token")
    bos_hidden_vec = None
    hidden_vec_list = []
    token_time_list = []
    while total_out.shape[1] < max_seq_len:
      pbar.update(1)
      generation_step += 1
      input_tensor = total_out[:, -self.net.input_length:] 
      if input_note is not None:
        note_vec = input_note[generation_step].unsqueeze(0).unsqueeze(0)
        input_tensor = torch.cat([input_tensor, note_vec], dim=1)
      # Generate the next token and update the cache
      time_start = time.time()
      _, sampled_token, cache, hidden_vec = self._run_one_step(input_tensor, cache=cache, sampling_method=sampling_method, threshold=threshold, temperature=temperature,bos_hidden_vec=bos_hidden_vec, context=context)
      time_end = time.time()
      token_time_list.append(time_end - time_start)
      if bos_hidden_vec is None:
        bos_hidden_vec = hidden_vec
      hidden_vec_list.append(hidden_vec)
      # Update attention cache to handle autoregressive generation
      for inter in cache.attn_intermediates:
        inter.cached_kv = [t[..., -(self.net.input_length - 1):, :] for t in inter.cached_kv]

      # Update the generated output with the new token
      total_out, sampled_token = self._update_total_out(total_out, sampled_token)

      # Stop if the end token is reached
      if sampled_token.tolist() == self.net.end_token[0]:
        break
    # append hidden_vec to pkl

    # save_path = 'hidden/diffnoaug_hidden_vec.pt'
    # save_time_path = 'hidden/diff_noaug_token_time.json'
    # if os.path.exists(save_path):
    #   # Load existing list and append
    #   hidden_vec_all = torch.load(save_path, map_location="cpu")
    #   hidden_vec_all.extend(hidden_vec_list)
    #   torch.save(hidden_vec_all, save_path)
    # else:
    #   torch.save(hidden_vec_list, save_path)
    
    # if os.path.exists(save_time_path):
    #   # Load existing list and append
    #   token_time_all = json.load(open(save_time_path, 'r'))
    #   token_time_all = token_time_all['token_time_list']
    #   token_time_all.extend(token_time_list)
    #   average_time = sum(token_time_all) / len(token_time_all)
    #   data = {
    #     'average_time': average_time,
    #     'token_time_list': token_time_all
    #   }
    #   json.dump(data, open(save_time_path, 'w'), indent=4)
    # else:
    #   average_time = sum(token_time_list) / len(token_time_list)
    #   data = {
    #     'average_time': average_time,
    #     'token_time_list': token_time_list
    #   }
    #   json.dump(data, open(save_time_path, 'w'), indent=4)

    return total_out
  
  def generate_batch(self, manual_seed, max_seq_len, condition=None, num_target_measures=4, sampling_method=None, threshold=None, temperature=1, batch_size=1, input_note=None):
    '''
    Autoregressively generates a sequence of tokens by repeatedly sampling the next token 
    until the desired maximum sequence length is reached or the end token is encountered.

    Arguments:
    - manual_seed: A seed value for reproducibility in inference.
    - max_seq_len: The maximum length of the generated sequence.
    - condition: An optional conditioning sequence to start generation from.
    - sampling_method: The method used to sample the next token (e.g., greedy, top-k).
    - threshold: Optional threshold for sampling (used in methods like top-p sampling).
    - temperature: Controls the randomness of the token sampling process.
    - batch_size: The number of sequences to generate in parallel.

    Returns:
    - total_out: The generated sequence of tokens as a tensor.
    '''
    
    # Prepare the starting sequence for inference
    total_out = self._prepare_inference(self.net.start_token, manual_seed, condition, num_target_measures)
    # total_out (1,1,num) -> (bs,1,num)
    total_out = total_out.repeat(batch_size, 1, 1)
    
    # === NEW: Process input_note if provided ===
    generation_step = 1
      
    # If a condition is provided, run one initial step
    if condition is not None:
      if input_note is not None:
        note_vec = input_note[generation_step].unsqueeze(0).unsqueeze(0)
        note_vec = note_vec.repeat(batch_size, 1, 1)
        _, _, cache = self._run_one_step(torch.cat([total_out[:, -self.net.input_length:], note_vec], dim=1), cache=LayerIntermediates(), sampling_method=sampling_method, threshold=threshold, temperature=temperature)
      else:
        _, _, cache = self._run_one_step(total_out[:, -self.net.input_length:], cache=LayerIntermediates(), sampling_method=sampling_method, threshold=threshold, temperature=temperature)
    else:
      cache = LayerIntermediates()

    # Continue generating tokens until the maximum sequence length is reached
    pbar = tqdm(total=max_seq_len, desc="Generating tokens", unit="token")
    while total_out.shape[1] < max_seq_len:
      pbar.update(1)
      generation_step += 1
      input_tensor = total_out[:, -self.net.input_length:]

      # === NEW: input_noteのi番目の行を結合 ===
      if input_note is not None and generation_step < input_note.shape[0]:
        # i番目の音符を取得: [num_features]
        note_vec = input_note[generation_step]  # [num_features]
        # バッチ対応: [num_features] -> [batch_size, 1, num_features]
        note_vec = note_vec.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)
        # input_tensorに結合: [batch_size, input_length, num_features] + [batch_size, 1, num_features]
        input_tensor = torch.cat([input_tensor, note_vec], dim=1)
        
      # Generate the next token and update the cache
      _, sampled_token, cache = self._run_one_step(input_tensor, cache=cache, sampling_method=sampling_method, threshold=threshold, temperature=temperature)

      # Update attention cache to handle autoregressive generation
      for inter in cache.attn_intermediates:
        inter.cached_kv = [t[..., -(self.net.input_length - 1):, :] for t in inter.cached_kv]

      # Update the generated output with the new token
      total_out, sampled_token = self._update_total_out(total_out, sampled_token)

      # Stop if the end token is reached
      if sampled_token.tolist() == self.net.end_token[0]:
        break

    return total_out
 
class AmadeusModel(nn.Module):
  def __init__(
    self, 
    vocab:LangTokenVocab,                 
    input_length:int,           
    prediction_order:list,       
    input_embedder_name:str,    
    main_decoder_name:str,      
    sub_decoder_name:str,       
    sub_decoder_depth:int,      
    sub_decoder_enricher_use:bool, 
    dim:int,                    
    heads:int,                 
    depth:int,                 
    dropout:float,
    is_regressive:bool        
  ):
    '''
    This class combines the wrapper classes and initializes the full AmadeusModel model, 
    which can perform autoregressive sequence generation for symbolic music.

    Vocabulary used for tokenization of the symbolic music data.
    Length of the input seqkeuence in tokens.
    Defines the order in which features are predicted in a sequence used for compound shift
    Name of the input embedding model to be used (e.g., one-hot embedding or learned embeddings).
    Name of the main transformer decoder model used for generating the hidden representations for compound tokens.
    Name of the sub-decoder, which processes the hidden states and decodes the sub-tokens inside the compound tokens.
    Depth (number of layers) of the sub-decoder.
    Whether to use an additional enricher module in the sub-decoder to refine representations.
    Dimensionality of the model (hidden size of the transformer layers).
    Number of attention heads in the transformer layers.
    Number of layers in the main decoder.
    Dropout rate for all layers in the model.
    '''

    super().__init__()
    decoder = AmadeusModelWrapper(
      vocab=vocab,
      input_length=input_length,
      prediction_order=prediction_order,
      input_embedder_name=input_embedder_name,
      main_decoder_name=main_decoder_name,
      sub_decoder_name=sub_decoder_name,
      sub_decoder_depth=sub_decoder_depth,
      sub_decoder_enricher_use=sub_decoder_enricher_use,
      dim=dim,
      heads=heads,
      depth=depth,
      dropout=dropout,
      is_regressive=is_regressive
    )
    self.decoder = AmadeusModelAutoregressiveWrapper(
      net=decoder
    )
  
  def forward(self, input_seq:torch.Tensor, target:torch.Tensor, context=None):
    return self.decoder(input_seq, target, context=context)
  
  @torch.inference_mode()
  def generate(self, manual_seed, max_seq_len, condition=None, num_target_measures=4, sampling_method=None, threshold=None, temperature=1,batch_size=1,context=None, input_note=None):
    if batch_size == 1:
      return self.decoder.generate(manual_seed, max_seq_len, condition, num_target_measures, sampling_method, threshold, temperature, context=context, input_note=input_note)
    else:
      return self.decoder.generate_batch(manual_seed, max_seq_len, condition, num_target_measures, sampling_method, threshold, temperature, batch_size, context=context, input_note=input_note)

class AmadeusModel4Encodec(AmadeusModel):
  def __init__(
    self, 
    vocab:LangTokenVocab,                 
    input_length:int,           
    prediction_order:list,       
    input_embedder_name:str,    
    main_decoder_name:str,      
    sub_decoder_name:str,       
    sub_decoder_depth:int,      
    sub_decoder_enricher_use:bool, 
    dim:int,                    
    heads:int,                 
    depth:int,                 
    dropout:float                 
  ):
    super().__init__(
      vocab=vocab, 
      input_length=input_length, 
      prediction_order=prediction_order, 
      input_embedder_name=input_embedder_name, 
      main_decoder_name=main_decoder_name, 
      sub_decoder_name=sub_decoder_name, 
      sub_decoder_depth=sub_decoder_depth, 
      sub_decoder_enricher_use=sub_decoder_enricher_use,
      dim=dim, 
      heads=heads, 
      depth=depth, 
      dropout=dropout
    )

  def _prepare_inference(self, start_token, manual_seed, condition=None):
    if manual_seed > 0:
      torch.manual_seed(manual_seed)
    total_out = []
    if condition is None:
      total_out.extend(start_token)
    else:
      if self.decoder.net.vocab.encoding_scheme == 'remi':
        selected_tokens = condition[:1500].tolist()
      else:
        selected_tokens = condition[:500].tolist()
      total_out.extend(selected_tokens)
    total_out = torch.LongTensor(total_out).unsqueeze(0).to(self.decoder.net.device)
    return total_out

  def _update_total_out(self, total_out, sampled_token):
    if self.decoder.net.vocab.encoding_scheme == 'remi':
      total_out = torch.cat([total_out, sampled_token.unsqueeze(0)], dim=-1)
    else:
      sampled_token_list = []
      for key in self.decoder.net.vocab.feature_list:
        sampled_token_list.append(sampled_token[key])
      sampled_token = torch.cat(sampled_token_list, dim=-1) # B(1) x num_features
      total_out = torch.cat([total_out, sampled_token.unsqueeze(0).unsqueeze(0)], dim=1)
    return total_out, sampled_token

  def _run_one_step(self, input_seq, cache=None, sampling_method=None, threshold=None, temperature=1):
    embedding = self.decoder.net.input_embedder(input_seq) + self.decoder.net.pos_enc(input_seq)
    embedding = self.decoder.net.emb_dropout(embedding)
    hidden_vec, intermidiates = self.decoder.net.main_decoder(embedding, cache) # B x T x d_model
    hidden_vec = self.decoder.net.main_norm(hidden_vec)
    hidden_vec = hidden_vec[:, -1:] # B x 1 x d_model
    input_dict = {'hidden_vec':hidden_vec, 'input_seq': input_seq, 'target': None}
    if self.decoder.net.vocab.encoding_scheme == 'remi':
      feature_class_idx = (input_seq.shape[1] - 1) % 4
      feature_type = self.decoder.net.vocab.feature_list[feature_class_idx]
      logits, sampled_token = self.decoder.net.sub_decoder.run_one_step(input_dict, sampling_method, threshold, temperature, feature_type)
    else:
      logits, sampled_token = self.decoder.net.sub_decoder(input_dict, sampling_method, threshold, temperature)
    return logits, sampled_token, intermidiates

  @torch.inference_mode()
  def generate(self, manual_seed, max_seq_len, condition=None, sampling_method=None, threshold=None, temperature=1):
    total_out = self._prepare_inference(self.decoder.net.start_token, manual_seed, condition)
    if condition is not None:
      _, _, cache = self._run_one_step(total_out[:, -self.decoder.net.input_length:], cache=LayerIntermediates(), sampling_method=sampling_method, threshold=threshold, temperature=temperature)
    else:
      cache = LayerIntermediates()
    while total_out.shape[1] < max_seq_len:
      input_tensor = total_out[:, -self.decoder.net.input_length:]
      _, sampled_token, cache = self._run_one_step(input_tensor, cache=cache, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
      for inter in cache.attn_intermediates:
        inter.cached_kv = [t[..., -(self.decoder.net.input_length - 1):, :] for t in inter.cached_kv] # B x num_heads x T x d_head
      total_out, sampled_token = self._update_total_out(total_out, sampled_token)
      if sampled_token.tolist() == self.decoder.net.end_token[0]:
        break
    return total_out