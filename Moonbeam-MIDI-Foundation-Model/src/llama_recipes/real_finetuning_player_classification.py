# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import json
import dataclasses
import fire
import random
import torch
import torch.optim as optim
from peft import get_peft_model, prepare_model_for_kbit_training
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy
)

from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaConfig,
    LlamaPreTrainedModel,
)
from llama_recipes.datasets.music_tokenizer import MusicTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaModel

from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.configs import ddp_config as DDP_CONFIG
from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.data.concatenator import ConcatDataset_vanilla as ConcatDataset
from llama_recipes.data.concatenator import ConcatDataset_dummy_padding as ConcatDataset_dummy_padding
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing
from llama_recipes.model_checkpointing import load_model_checkpoint_ddp

from llama_recipes.utils import fsdp_auto_wrap_policy
from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
    get_dataloader_kwargs,
)
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset

from llama_recipes.utils.fsdp_utils import hsdp_device_mesh
from llama_recipes.utils.train_utils import (
    train_player_classification,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies,
)
from accelerate.utils import is_xpu_available

"""ソース真偽判定用線形層追加モデル"""
from torch import nn
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.utils import add_start_docstrings_to_model_forward
from transformers.cache_utils import Cache
class LlamaForSequenceDoubleClassification(
    LlamaPreTrainedModel
):
    def __init__(self, config):
        super().__init__(config)
        self.classification_token = config.classification_token
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        # 追加: 真偽分類用線形層
        self.real_fake_score = nn.Linear(config.hidden_size, 2, bias=False)
        self.model.classification_token_embedding = torch.nn.Embedding(1, config.hidden_size)
        self.model.classification_token_embedding.requires_grad = True
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        #outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        position_ids = input_ids
        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids, #if sdpa: position_ids carry information about onset, dur, pitch, instr, vel; elif sdpa_baseline: position_ids are None
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=None,
        )

        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)
        # 追加: 真偽分類スコア
        real_fake_logits = self.real_fake_score(hidden_states)


        where_eos = (input_ids[..., 0] == self.classification_token)  # Shape: (batch_size, seq_len)
        # Use `where_eos` to extract logits
        batch_indices, seq_indices = where_eos.nonzero(as_tuple=True)  # Get batch and sequence indices # get position-1 before the EOS token

        # Gather logits
        pooled_logits = logits[batch_indices, seq_indices]  # Shape: (num_eos_positions, hidden_dim) , because it might learn just to turn off the sequence
        # 追加: 真偽分類用logits抽出
        pooled_real_fake_logits = real_fake_logits[batch_indices, seq_indices]
        
        loss = None
        if labels is not None:
            labels = labels[batch_indices, seq_indices].to(logits.device)
            
            # 追加: 真偽分類用loss計算
            loss_fct = CrossEntropyLoss()
            real_fake_loss = loss_fct(pooled_real_fake_logits.view(-1, 2), labels.view(-1)) #TODO: add weighting param since there might be an arbitrary number of labels here
            number_of_labels = where_eos.sum()
            if number_of_labels ==0:
                real_fake_loss = torch.tensor(0., requires_grad=True).to(pooled_real_fake_logits)
            else:
                # loss = loss/number_of_labels
                real_fake_loss = real_fake_loss
            
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    classification_loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    classification_loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                classification_loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1)) #TODO: add weighting param since there might be an arbitrary number of labels here
                number_of_labels = where_eos.sum()
                if number_of_labels ==0:
                    classification_loss = torch.tensor(0., requires_grad=True).to(pooled_logits)
                else:
                    # classification_loss = classification_loss/number_of_labels
                    classification_loss = classification_loss
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                classification_loss = loss_fct(pooled_logits, labels)
        
        total_loss = classification_loss + real_fake_loss
        
        if not return_dict:
            output = (pooled_logits, pooled_real_fake_logits) + transformer_outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=total_loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )    


def setup_wandb(train_config, fsdp_config, llama_config, **kwargs):
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "You are trying to use wandb which is not currently installed. "
            "Please install it using pip install wandb"
        )
    from llama_recipes.configs import wandb_config as WANDB_CONFIG
    wandb_config = WANDB_CONFIG()
    update_config(wandb_config, **kwargs)
    init_dict = dataclasses.asdict(wandb_config)
    run = wandb.init(**init_dict)
    run.config.update(train_config)
    run.config.update(fsdp_config, allow_val_change=True)

    # Convert the llama_config to a dictionary and then to a JSON string
    config_dict = llama_config.to_dict()
    config_json = json.dumps(config_dict, indent=4)
    
    # Get the wandb run directory
    from pathlib import Path
    # Define the file path within the wandb run directory
    folder_name = (train_config.dist_checkpoint_root_folder+ "/"+ train_config.dist_checkpoint_folder+ "-"+ train_config.model_name)
    save_dir = Path.cwd() / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)
    config_file_path = os.path.join(save_dir, 'llama_config.json')

    # Write the JSON string to the file
    with open(config_file_path, 'w') as f:
        f.write(config_json)
        print(f"config file saved to {config_file_path}!")
    return run


def main(**kwargs):
    # Update the configuration for the training and sharding process
    train_config, fsdp_config, ddp_config = TRAIN_CONFIG(), FSDP_CONFIG(), DDP_CONFIG()
    model_config_path = "src/llama_recipes/configs/player_classification_config.json"
    update_config((train_config, fsdp_config, ddp_config), **kwargs)
    print("updated training config", train_config)
    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    if train_config.enable_fsdp or train_config.enable_ddp:
        setup() #enable nccl / ccl
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        if is_xpu_available():
            torch.xpu.set_device(local_rank)
        elif torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    wandb_run = None

    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp or train_config.enable_ddp else None
    if train_config.enable_fsdp and train_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        if rank == 0:
            model = LlamaForCausalLM.from_pretrained( #TODO: If new model then need to change here! source code: transformers/src/transformers/models/llama/modeling_llama.py
                train_config.model_name,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
                attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            )
        else:
            llama_config = LlamaConfig.from_pretrained(train_config.model_name)
            llama_config.use_cache = use_cache
            with torch.device("meta"):
                model = LlamaForCausalLM(llama_config)

    else: #DDP and non-distributed training
        llama_config = LlamaConfig.from_pretrained(model_config_path)
        llama_config.use_cache = use_cache
        print(f"model_config:{llama_config}")
        model = LlamaForSequenceDoubleClassification(llama_config) #TODO: LOAD FROM CKPT

        model_checkpoint = torch.load(train_config.trained_checkpoint_path) #, , map_location = 'cuda:{}'.format(local_rank)
        
        checkpoint = model_checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith('module.'): # Check if the keys have 'module.' prefix and remove it if necessary
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        # Load the state_dict into the model, ignoring unmatched keys
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False) #some weights are not needed for classification tasks

    if train_config.use_wandb:
        if not train_config.enable_fsdp or rank==0:
            wandb_run = setup_wandb(train_config, fsdp_config, llama_config, **kwargs)


    # Load the tokenizer and add special tokens
    tokenizer = MusicTokenizer(timeshift_vocab_size = llama_config.onset_vocab_size, dur_vocab_size = llama_config.dur_vocab_size, octave_vocab_size = llama_config.octave_vocab_size, pitch_class_vocab_size = llama_config.pitch_class_vocab_size, instrument_vocab_size = llama_config.instrument_vocab_size, velocity_vocab_size = llama_config.velocity_vocab_size, sos_token = llama_config.sos_token, eos_token = llama_config.eos_token, pad_token = llama_config.pad_token) ##TODO:Done,  ADD NEW PLAYER CLASSIFICATION TOKEN 
    tokenizer.add_new_tokens(token_name = "classification_token", token_val = llama_config.classification_token)
    # If there is a mismatch between tokenizer vocab size and embedding matrix, 
    # throw a warning and then expand the embedding matrix
    # if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
    #     print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
    #     model.resize_token_embeddings(len(tokenizer)) #Commented out since there's no tokenizer here

    print_model_size(model, train_config, rank if train_config.enable_fsdp or train_config.enable_ddp else 0)

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_kbit_training(model)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    if train_config.enable_ddp and ddp_config.pure_bf16:
        model.to(torch.bfloat16)

    if train_config.use_peft: #TODO: Switch on PEFT, think how to make the embedding of the extra tokens trainable 
        peft_config = generate_peft_config(train_config, kwargs) 
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        if wandb_run:
            wandb_run.config.update(peft_config)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable parameter during LoRA: {name}")

    hsdp_device_mesh = None 
    if fsdp_config.hsdp and fsdp_config.sharding_strategy == ShardingStrategy.HYBRID_SHARD:
        hsdp_device_mesh = hsdp_device_mesh(replica_group_size=fsdp_config.replica_group_size, sharding_group_size=fsdp_config.sharding_group_size)
        print("HSDP device mesh is ready")

    #setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:

            freeze_transformer_layers(train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer) #TODO: dangerous

        device_id = 0
        if is_xpu_available():
            device_id = torch.xpu.current_device()
        elif torch.cuda.is_available():
            device_id = torch.cuda.current_device()

        model = FSDP(
            model,
            auto_wrap_policy= my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_mesh=hsdp_device_mesh,
            device_id=device_id,
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=(lambda module: module.to_empty(device=torch.device("cuda"), recurse=False))
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model) #TODO: Add DDP
    elif train_config.enable_ddp: #wrap ddp code
        mixed_precision_policy, wrapping_policy = get_policies(ddp_config, rank)
        model.to(local_rank)
        model = DDP(model,
                    mixed_precision=mixed_precision_policy if not ddp_config.pure_bf16 else None, 
                    device_mesh=hsdp_device_mesh,
                    device_ids=[local_rank],
                    find_unused_parameters=False,
                    )
    elif not train_config.quantization and not train_config.enable_fsdp:
        if is_xpu_available():
            model.to("xpu:0")
        elif torch.cuda.is_available():
            model.to("cuda")

    dataset_config = generate_dataset_config(train_config, kwargs)

     # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset( #TODO Done: ADD Player classification datasets
        tokenizer,
        dataset_config,
        split="train",
    )

    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    dataset_val_unconcat = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    ) #TODO Done: ADD Player classification datasets
    if not train_config.enable_fsdp or rank == 0:
            print(f"--> Validation Set Length = {len(dataset_val_unconcat)}")


    if train_config.batching_strategy == "packing":
        dataset_train = ConcatDataset_dummy_padding(dataset_train, chunk_size=train_config.context_length)
    train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, tokenizer, "train")

    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **train_dl_kwargs,
    )

    eval_dataloader = None
    if train_config.run_validation:
        if train_config.batching_strategy == "packing":
            dataset_val = ConcatDataset(dataset_val_unconcat, chunk_size=train_config.context_length) #TODO: add Player classification concat 

        val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, tokenizer, "val")

        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )

    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )

    #if trained_ckpt is provided, continue training
    """if train_config.trained_checkpoint_path: #TODO: think how to load this effectively, whether to do this here or in the beginning
        starting_epoch, starting_step = load_model_checkpoint_ddp(model, optimizer, local_rank, train_config.trained_checkpoint_path)
        print(f"Model loaded with checkpoint: {train_config.trained_checkpoint_path}, {starting_epoch=} {starting_step=}")
    else:
        starting_epoch, starting_step = 0, 0"""
    starting_epoch, starting_step = 0, 0
    
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    # Start the training process
    results = train_player_classification( #TODO: check the training code
        model,
        train_dataloader,
        eval_dataloader,
        dataset_val_unconcat, 
        tokenizer,
        optimizer,
        scheduler,
        starting_epoch,
        starting_step, 
        train_config.gradient_accumulation_steps,
        train_config,
        fsdp_config if train_config.enable_fsdp else None,
        ddp_config if train_config.enable_ddp else None,
        local_rank if (train_config.enable_fsdp or train_config.enable_ddp) else None, #TODO: change this and train_utils
        rank if (train_config.enable_fsdp or train_config.enable_ddp) else None,#TODO: change this and train_utils
        wandb_run,
        train_config.individual_eval
    )
    if not train_config.enable_fsdp or rank==0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]
        if train_config.use_wandb:
            for k,v in results.items():
                wandb_run.summary[k] = v

if __name__ == "__main__":
    fire.Fire(main)
