import os
try:
    api_key = os.environ["WOLFRAM_API_KEY"]
except KeyError:
    os.environ["WOLFRAM_API_KEY"] = 'ALLPX3-LELT3A79XE'

import argparse
import logging
import math
import random
import datasets
from datasets import load_dataset,Features,Value
import torch
from functools import partial
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    SchedulerType,
    DataCollatorForSeq2Seq,
    get_scheduler,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training


import numpy as np
from pathlib import Path
from dataclasses import dataclass
from model.rl_models import CQLModelForCausalLM
from eval.utils import rl_generate_completions, generate_completions

from eval.gsm.run_eval import execute as gsm_execute
from eval.gsm.run_eval import rl_execute as gsm_rl_execute
from eval.strategyqa.run_eval import rl_execute as complex_qa_execute
from pytorch_memlab import MemReporter
from pytorch_memlab.mem_reporter import Optional, LEN, readable_size


logger = get_logger(__name__)
lumos_dir = Path(__file__).parent.parent
model_name_list = [
    'gpt2',
    'lumos',
]


@dataclass
class DataCollatorForSeqRL(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        states = [feature["states"] for feature in features] if "states" in features[0].keys() else None
        actions = [feature["actions"] for feature in features] if "actions" in features[0].keys() else None
        next_states = [feature["next_states"] for feature in features] if "next_states" in features[0].keys() else None
        rewards = [feature["rewards"] for feature in features] if "rewards" in features[0].keys() else None

        assert states is not None and actions is not None and next_states is not None and rewards is not None, "Need to have states, actions, next_states and rewards in features."

        max_state_length = max(len(l) for l in states)
        max_next_state_length = max(len(l) for l in next_states)

        if self.pad_to_multiple_of is not None:
            max_state_length = (
                (max_state_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )
            max_next_state_length = (
                (max_next_state_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        padding_side = self.tokenizer.padding_side
        for feature in features:
            state_remainder = [self.tokenizer.pad_token_id] * (max_state_length - len(feature["states"]))
            next_state_remainder = [self.tokenizer.pad_token_id] * (max_next_state_length - len(feature["next_states"]))
            if isinstance(feature["states"], list):
                feature["states"] = (
                    feature["states"] + state_remainder if padding_side == "right" else state_remainder + feature["states"]
                )
                feature["next_states"] = (
                    feature["next_states"] + next_state_remainder if padding_side == "right" else next_state_remainder + feature["next_states"]
                )
            elif padding_side == "right":
                feature["states"] = np.concatenate([feature["states"], state_remainder]).astype(np.int64)
                feature["next_states"] = np.concatenate([feature["next_states"], next_state_remainder]).astype(np.int64)
            else:
                feature["states"] = np.concatenate([state_remainder, feature["states"]]).astype(np.int64)
                feature["next_states"] = np.concatenate([next_state_remainder, feature["next_states"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        if (
            self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            raise NotImplementedError('LLaMA2 has no attribute "prepare_decoder_input_ids_from_labels"')
        
        # Remove unnecessary items
        features.data.pop('input_ids')
        features.data.pop('attention_mask')

        return features
    

@dataclass
class DataCollatorForSeqRLVF(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        states_user = [feature["states_user"] for feature in features] if "states_user" in features[0].keys() else None
        states_agent = [feature["states_agent"] for feature in features] if "states_agent" in features[0].keys() else None
        actions = [feature["actions"] for feature in features] if "actions" in features[0].keys() else None
        next_states = [feature["next_states"] for feature in features] if "next_states" in features[0].keys() else None
        rewards = [feature["rewards"] for feature in features] if "rewards" in features[0].keys() else None

        assert states_user is not None and states_agent is not None and actions is not None and next_states is not None and rewards is not None, "Need to have states_user, states_agent, actions, next_states and rewards in features."

        max_state_user_length = max(len(l) for l in states_user)
        max_state_agent_length = max(len(l) for l in states_agent)
        max_next_state_length = max(len(l) for l in next_states)

        if self.pad_to_multiple_of is not None:
            max_state_user_length = (
                (max_state_user_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )
            max_state_agent_length = (
                (max_state_agent_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )
            max_next_state_length = (
                (max_next_state_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        padding_side = self.tokenizer.padding_side
        for feature in features:
            state_user_remainder = [self.tokenizer.pad_token_id] * (max_state_user_length - len(feature["states_user"]))
            state_agent_remainder = [self.tokenizer.pad_token_id] * (max_state_agent_length - len(feature["states_agent"]))
            next_state_remainder = [self.tokenizer.pad_token_id] * (max_next_state_length - len(feature["next_states"]))
            if isinstance(feature["states_user"], list):
                feature["states_user"] = (
                    feature["states_user"] + state_user_remainder if padding_side == "right" else state_user_remainder + feature["states_user"]
                )
                feature["states_agent"] = (
                    feature["states_agent"] + state_agent_remainder if padding_side == "right" else state_agent_remainder + feature["states_agent"]
                )
                feature["next_states"] = (
                    feature["next_states"] + next_state_remainder if padding_side == "right" else next_state_remainder + feature["next_states"]
                )
            elif padding_side == "right":
                feature["states_user"] = np.concatenate([feature["states_user"], state_user_remainder]).astype(np.int64)
                feature["states_agent"] = np.concatenate([feature["states_agent"], state_agent_remainder]).astype(np.int64)
                feature["next_states"] = np.concatenate([feature["next_states"], next_state_remainder]).astype(np.int64)
            else:
                feature["states_user"] = np.concatenate([state_user_remainder, feature["states_user"]]).astype(np.int64)
                feature["states_agent"] = np.concatenate([state_agent_remainder, feature["states_agent"]]).astype(np.int64)
                feature["next_states"] = np.concatenate([next_state_remainder, feature["next_states"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        if (
            self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            raise NotImplementedError('LLaMA2 has no attribute "prepare_decoder_input_ids_from_labels"')
        
        # Remove unnecessary items
        features.data.pop('input_ids')
        features.data.pop('attention_mask')

        return features


class CustomMemReporter(MemReporter):
    def print_stats(self, verbose: bool = False, target_device: Optional[torch.device] = None) -> None:
        # header
        show_reuse = verbose
        template_format = '{:<40s}{:>20s}{:>10s}'
        print(template_format.format('Element type', 'Size', 'Used MEM') )
        for device, tensor_stats in self.device_tensor_stat.items():
            # By default, if the target_device is not specified,
            # print tensors on all devices
            if target_device is not None and device != target_device:
                continue
            print('-' * LEN)
            print('Storage on {}'.format(device))
            total_mem = 0
            total_numel = 0

            sorted_tensor_stats = reversed(sorted(tensor_stats, key=lambda x: x[-1]))
            for stat in sorted_tensor_stats:
                name, size, numel, mem = stat
                if not show_reuse:
                    name = name.split('(')[0]
                print(template_format.format(
                    str(name),
                    str(size),
                    readable_size(mem),
                ))
                total_mem += mem
                total_numel += numel

            print('-'*LEN)
            print('Total Tensors: {} \tUsed Memory: {}'.format(
                total_numel, readable_size(total_mem),
            ))

            if device != torch.device('cpu'):
                with torch.cuda.device(device):
                    memory_allocated = torch.cuda.memory_allocated()
                print('The allocated memory on {}: {}'.format(
                    device, readable_size(memory_allocated),
                ))
                if memory_allocated != total_mem:
                    print('Memory differs due to the matrix alignment or'
                          ' invisible gradient buffer tensors')
            print('-'*LEN)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="If passed, will use LORA (low-rank parameter-efficient training) to train the model.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="The rank of lora.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16,
        help="The alpha parameter of lora.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="The dropout rate of lora modules.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help=(
            "Turn on gradient checkpointing. Saves memory but slows training."
        ),
    )
    parser.add_argument(
        "--use_qlora",
        action="store_true",
        help=(
            "Use qLoRA training - main thing is initialising model in quantised form. Not compatible with deepspeed."
        ),
    )
    parser.add_argument(
        '--clip_grad_norm',
        type=float,
        default=-1,
        help='Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead).',
    )
    parser.add_argument(
        '--use_8bit_optimizer',
        action='store_true',
        help='Use 8bit optimizer from bitsandbytes. Not compatible with deepspeed (use deepspeed config instead).',
    )



    # args needed to be updated
    parser.add_argument("--debug", action="store_true", default=False, help="Debug Flag")
    parser.add_argument("--cpu", action="store_true", default=False, help="CPU Flag")
    
    # Dataset utils
    parser.add_argument("--task_name", type=str, default='maths', help="Task name", choices=['maths'])

    # Train utils
    parser.add_argument("--target_update_method", type=str, default='hard', help="Method of updating target net")
    # parser.add_argument("--target_update_method", type=str, default='soft', help="Method of updating target net")

    # parser.add_argument("--target_update_interval", type=int, default=128 * 10, help="Update target net every target_update_interval minibatches")
    parser.add_argument("--target_update_interval", type=int, default=1, help="Update target net every target_update_interval minibatches")

    # parser.add_argument("--checkpointing_steps", type=str, default='1', help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.")  # Same value as target_update_interval
    parser.add_argument("--checkpointing_steps", type=str, default='epoch', help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.")  # Same value as target_update_interval

    # CQL Model
    parser.add_argument("--use_lagrange", action="store_true", default=False, help="Debug Flag")
    parser.add_argument("--num_network", type=int, default=2, help="Multiple Q Trick")
    parser.add_argument("--seed", type=int, default=47, help="A seed for reproducible training.")
    parser.add_argument("--model_name_or_path", type=str, default='/home/ubuntu/yangsihang/llm_ref/.cache/hub/models--ai2lumos--lumos_unified_ground_iterative/snapshots/8adc9ac410a17d2dcf12f8514ec345e59ad86467', help="Path to pretrained model or model identifier from huggingface.co/models.", required=False)
    parser.add_argument("--use_flash_attn", action="store_true", default=False, help="If passed, will use flash attention to train the model.")
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--use_slow_tokenizer", action="store_false", default=True, help="If passed, will use flash attention to train the model.")
    parser.add_argument("--train_file", type=str, default=str(lumos_dir.joinpath(f'data/train/unified/lumos_unified_ground_iterative.jsonl')), help="A csv or a json file containing the training data.")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="The maximum total sequence length (prompt+completion) of each training example.")
    parser.add_argument("--preprocessing_num_workers", type=int, default=16, help="The number of processes to use for the preprocessing.")

    # parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader.")

    # batch_size / num_gpus / batch_size_per_gpu
    parser.add_argument("--gradient_accumulation_steps", type=int, default=int(128 / 1 / 1), help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear", help="The scheduler type to use.", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Ratio of total training steps used for warmup.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=2, help="Total number of training epochs to perform.")
    parser.add_argument("--output_dir", type=str, default=str(lumos_dir.joinpath('results/lumos_unified_rl_iterative')), help="Where to store the final model.")
    parser.add_argument("--with_tracking", action="store_true", default=False, help="Whether to enable experiment trackers for logging.")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--logging_steps", type=int, default=1, help="Log the training loss and learning rate every logging_steps steps.")
    parser.add_argument("--low_cpu_mem_usage", action="store_true", default=False, help=("It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. If passed, LLM loading time and RAM consumption will be benefited."))

    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None:
        raise ValueError("Need either a dataset name or a training file.")
    else:
        if args.train_file is not None:
            extension = str(args.train_file).split(".")[-1]
            assert extension in ["json", "jsonl"], "`train_file` should be a json/jsonl file."
    return args


def encode_with_prompt_completion_format(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated 
    and it doesn't make sense to follow directly with the completion.
    '''
    # if prompt doesn't end with space and completion doesn't start with space, add space
    if not example['prompt'].endswith((' ', '\n', '\t')) and not example['completion'].startswith((' ', '\n', '\t')):
        example_text = example['prompt'] + ' ' + example['completion']
    else:
        example_text = example['prompt'] + example['completion']
    example_text = example_text + tokenizer.eos_token
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(example['prompt'], return_tensors='pt', max_length=max_seq_length, truncation=True)
    # mask the prompt part for avoiding loss
    labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


def encode_with_messages_format(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')
    
    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text
        
    example_text = _concat_messages(messages).strip()
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]), return_tensors='pt', max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx+1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = _concat_messages(messages[:message_idx+1]) + "<|assistant|>\n"
            else:
                messages_so_far = _concat_messages(messages[:message_idx+1])
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt', 
                max_length=max_seq_length, 
                truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100
            
            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


def save_with_accelerate(accelerator, model, tokenizer, output_dir, args):
    unwrapped_model = accelerator.unwrap_model(model)
    # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
    # Otherwise, sometimes the model will be saved with only part of the parameters.
    # Also, accelerator needs to use the wrapped model to get the state_dict.
    state_dict = accelerator.get_state_dict(model)
    if args.use_lora:
        # When using lora, the unwrapped model is a PeftModel, which doesn't support the is_main_process 
        # and has its own save_pretrained function for only saving lora modules.
        # We have to manually specify the is_main_process outside the save_pretrained function.
        if accelerator.is_main_process:
            unwrapped_model.save_pretrained(output_dir, state_dict=state_dict)
    else:
        unwrapped_model.save_pretrained(
            output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=state_dict
        )


def wrap_dataset_vf(args: argparse.Namespace, dataset: datasets.Dataset, task_name: str, task_ids: list, tokenizer: AutoTokenizer, max_seq_length: int) -> dict:
    max_tau_length = 900  # Skip taus which are too long

    total_task_id_list = []
    total_state_user_ids_list = []
    total_state_agent_ids_list = []
    total_action_ids_list = []
    total_next_state_ids_list = []
    total_rewards_list = []
    total_dones_list = []
    total_valid_flags_list = []
    wrap_tqdm = tqdm(range(len(dataset)))

    debug_start_idx = 20000
    if args.debug or 'gpt2' in args.model_name_or_path:
        wrap_tqdm = tqdm(range(debug_start_idx, debug_start_idx + 100))

    wrap_tqdm.set_description("Wrapping dataset to RL format")
    for data_idx in wrap_tqdm:
        task_id = task_ids[data_idx]
        if task_name not in task_id:
            continue
        data_tau = dataset[data_idx]
        # Current setting do not need labels
        tau_str = tokenizer.decode(data_tau['input_ids'])
        tau_attention_mask = data_tau['attention_mask']
        action_prompt = '<|assistant|>\n'
        tau_state_str_list = []
        tau_action_str_list = []
        tau_user_prompt_list = []
        tmp_list = tau_str.split(action_prompt)
        state_str = tmp_list[0] + action_prompt

        multiple_actions_flag = False
        action_split_prompt = ';'
        for block_idx in range(1, len(tmp_list)):
            action_str, next_state_user_prompt = tmp_list[block_idx].split(tokenizer.eos_token)
            action_str = action_str + tokenizer.eos_token  # eos_token is outputed by LLM

            if len(action_str.split(action_split_prompt)) > 1:
                multiple_actions_flag = True

            # Final step
            if block_idx == len(tmp_list) - 1:
                user_prompt = next_state_user_prompt
            else:
                user_prompt = next_state_user_prompt + action_prompt
            next_state_str = state_str + action_str + user_prompt
            tau_state_str_list.append(state_str)
            tau_action_str_list.append(action_str)
            tau_user_prompt_list.append(user_prompt)

            state_str = next_state_str

        tau_state_user_ids_list = []
        tau_state_agent_ids_list = []
        tau_action_ids_list = []
        tau_next_state_ids_list = []
        tau_valid_flags_list = []
        for str_idx in range(len(tau_state_str_list)):
            if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
                # Remove the first bos_token
                macro_step_state_ids = tokenizer(tau_state_str_list[str_idx], max_length=max_seq_length, truncation=True).input_ids[1:]
                maco_step_user_prompt_ids = tokenizer(tau_user_prompt_list[str_idx], max_length=max_seq_length, truncation=True).input_ids[1:]
                macro_step_action_ids = tokenizer(tau_action_str_list[str_idx], max_length=max_seq_length, truncation=True).input_ids[1:]
            elif isinstance(tokenizer, GPT2Tokenizer):
                macro_step_state_ids = tokenizer(tau_state_str_list[str_idx], max_length=max_seq_length, truncation=True).input_ids
                maco_step_user_prompt_ids = tokenizer(tau_user_prompt_list[str_idx], max_length=max_seq_length, truncation=True).input_ids
                macro_step_action_ids = tokenizer(tau_action_str_list[str_idx], max_length=max_seq_length, truncation=True).input_ids
            else:
                raise NotImplementedError
            if str_idx > 0:
                tmp_macro_step_state_ids = tau_next_state_ids_list[-1]
            else:
                tmp_macro_step_state_ids = macro_step_state_ids
            tau_state_user_ids_list.extend([tmp_macro_step_state_ids] * len(macro_step_action_ids))
            state_user_ids_length = len(tmp_macro_step_state_ids)
            for action_id_idx in range(len(macro_step_action_ids)):
                action_id = macro_step_action_ids[action_id_idx]
                tmp_macro_step_state_ids = tmp_macro_step_state_ids + [action_id]
                if action_id_idx < len(macro_step_action_ids) - 1:
                    tmp_macro_step_next_state_ids = tmp_macro_step_state_ids
                    tau_valid_flags_list.append(0)
                else:
                    tmp_macro_step_next_state_ids = tmp_macro_step_state_ids + maco_step_user_prompt_ids
                    tau_valid_flags_list.append(1)
                tau_state_agent_ids_list.append(tmp_macro_step_state_ids[state_user_ids_length:])
                tau_action_ids_list.append(action_id)
                tau_next_state_ids_list.append(tmp_macro_step_next_state_ids)
        
        # state = SU + SA, so use the last step of next_state to obatin the length of tau
        if len(tau_next_state_ids_list[-1]) > max_tau_length:
            continue

        assert len(tau_state_user_ids_list) == len(tau_state_agent_ids_list)
        assert len(tau_state_agent_ids_list) == len(tau_next_state_ids_list)
        assert len(tau_next_state_ids_list) == len(tau_valid_flags_list)
        total_task_id_list.extend([data_idx] * len(tau_next_state_ids_list))
        total_state_user_ids_list.extend(tau_state_user_ids_list)
        total_state_agent_ids_list.extend(tau_state_agent_ids_list)
        total_action_ids_list.extend(tau_action_ids_list)
        total_next_state_ids_list.extend(tau_next_state_ids_list)
        total_rewards_list.extend([0] * (len(tau_next_state_ids_list) - 1) + [1])
        total_dones_list.extend([0] * (len(tau_next_state_ids_list) - 1) + [1])
        total_valid_flags_list.extend(tau_valid_flags_list)

    data_dict = {
        'input_ids': total_action_ids_list,  # Needed by DataCollator

        'states_user': total_state_user_ids_list,
        'states_agent': total_state_agent_ids_list,
        'actions': total_action_ids_list,
        'next_states': total_next_state_ids_list,
        'rewards': total_rewards_list,
        'dones': total_dones_list,
        'valid_flags': total_valid_flags_list,
        'task_ids': total_task_id_list,
    }

    tmp_data_dict = {
        'input_ids': [],
        'states_user': [],
        'states_agent': [],
        'actions': [],
        'next_states': [],
        'rewards': [],
        'dones': [],
        'valid_flags': [],
        'task_ids': [],
    }
    total_steps = len(total_action_ids_list)
    for step in range(total_steps):
        if total_valid_flags_list[step] == 1:
            tmp_data_dict['input_ids'].append(total_action_ids_list[step])
            tmp_data_dict['states_user'].append(total_state_user_ids_list[step])
            tmp_data_dict['states_agent'].append(total_state_agent_ids_list[step])
            tmp_data_dict['actions'].append(total_action_ids_list[step])
            tmp_data_dict['next_states'].append(total_next_state_ids_list[step])
            tmp_data_dict['rewards'].append(total_rewards_list[step])
            tmp_data_dict['dones'].append(total_dones_list[step])
            tmp_data_dict['valid_flags'].append(total_valid_flags_list[step])
            tmp_data_dict['task_ids'].append(total_task_id_list[step])
    
    # data_dict = tmp_data_dict  # test valid flag module

    wrapped_dataset = datasets.Dataset.from_dict(data_dict)
    token_space = torch.unique(torch.as_tensor(total_action_ids_list))

    wrap_result = {
        'wrapped_dataset': wrapped_dataset,
        'token_space': token_space,
    }

    return wrap_result


# Coarse grained
def mind2web_step(batch_dict: dict, tokenizer: AutoTokenizer, actions: str) -> dict:
    step_result = {
        'valid_flag': 1,
    }
    for action in actions.strip().split('; '):
        pos_action = action.find(" = ") + len(" = ")
        pos_parenthesis = action.find('(')
        pos_right_parenthesis = action.rfind(')')

        action_variable = action[: pos_action - len(" = ")].split(", ")[0].strip()
        action_name = action[pos_action: pos_parenthesis]
        action_args = action[pos_parenthesis+1: pos_right_parenthesis]
        arg_variable = action_args[:action_args.find(", ")]

        if action_name == "CLICK":
            query = action_args[action_args.find(", QUERY:") + len(", QUERY:"): ].strip()
        elif action_name == "SELECT" or action_name == "TYPE":
            query = action_args[action_args.find(", QUERY:") + len(", QUERY:"): action_args.rfind(", TEXT:")].strip()
            text = action_args[action_args.find(", TEXT:") + len(", TEXT:"): ].strip()
        else:
            step_result['valid_flag'] = 0
            break
        
    return step_result


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


# Fine grained
def gsm_step(batch_dict: dict, tokenizer: AutoTokenizer, actions: str) -> dict:
    step_result = {
        'valid_flag': 1,
    }

    inter_results = {}
    assistant_block_str = '\n<|assistant|>\n'
    block_list = tokenizer.decode(batch_dict['states_user'][0]).split(assistant_block_str)
    # 0 for task description, -1 for current action
    inter_operation_str_with_user_prompt_list = block_list[1: -1]
    inter_operation_str_list = [operation_str_with_user_prompt[:operation_str_with_user_prompt.find(tokenizer.eos_token)] for operation_str_with_user_prompt in inter_operation_str_with_user_prompt_list]
    for operation_str in inter_operation_str_list:
        for k, op_action in enumerate(operation_str.strip().split('; ')):
            results_variable, execution_results = gsm_rl_execute(op_action.strip(), inter_results)
            if results_variable is None and execution_results is None:
                step_result['valid_flag'] = 0
                return step_result
            elif isinstance(results_variable, str):
                inter_results[results_variable] = execution_results
            elif is_iterable(results_variable):
                for k, variable in enumerate(results_variable):
                    inter_results[variable] = execution_results[k]
            else:
                step_result['valid_flag'] = 0
                return step_result

    for k, action in enumerate(actions.strip().split('; ')):
        try:
            results_variable, execution_results = gsm_execute(action.strip(), inter_results)
            if execution_results:
                if isinstance(results_variable, str):
                    inter_results[results_variable] = execution_results
                else:
                    for k, variable in enumerate(results_variable):
                        inter_results[variable] = execution_results[k]
            else:
                step_result['valid_flag'] = 0
                break
        except:
            step_result['valid_flag'] = 0
            break
    
    return step_result


# Fine grained (Only skip QA related processs at execution-level)
def complex_qa_step(batch_dict: dict, tokenizer: AutoTokenizer, actions: str) -> dict:
    state_user = tokenizer.decode(batch_dict['states_user'][0])
    user_block_str = '<|user|>\n'
    assistant_block_str = '\n<|assistant|>\n'
    curr_subgoals_with_assitant_prompt = state_user.split(user_block_str)[-1]
    curr_subgoals = curr_subgoals_with_assitant_prompt[:curr_subgoals_with_assitant_prompt.find(assistant_block_str)]
    step_result = {
        'valid_flag': 1,
    }
    for k, action in enumerate(actions.strip().split('; ')):
        results_variable, execution_results = complex_qa_execute(action, step_result, subgoals=curr_subgoals.strip().split("; "))
        if results_variable is None and execution_results is None:
            step_result['valid_flag'] = 0
            break
        step_result[results_variable] = execution_results
    
    return step_result


def update_invalid_flags(batch_dict: dict, task_ids: list, model: CQLModelForCausalLM, tokenizer: AutoTokenizer) -> dict:
    model.eval()

    new_line_token = tokenizer.encode("\n", add_special_tokens=False)[-1] # get the last token because the tokenizer may add space tokens at the start.
    eval_batch_size = batch_dict['valid_flags'].shape[0]
    ground_prompts = tokenizer.batch_decode(batch_dict['states_user'])
    actions_list = rl_generate_completions(
                    rl_model=model,
                    tokenizer=tokenizer,
                    prompts=ground_prompts,
                    max_new_tokens=1024 - batch_dict['states_user'].shape[1],
                    batch_size=eval_batch_size,
                    stop_id_sequences=[[new_line_token]],
                    disable_tqdm=True,
                )
    assert len(actions_list) == 1, "Only support batch_size=1 for now."
    actions = actions_list[0]
    pi_valid_flags = []
    for batch_idx in range(eval_batch_size):
        task_type = '_'.join(task_ids[batch_dict['task_ids'][batch_idx]].split('_')[:-1])
        if task_type == 'web_agent':
            step_result = mind2web_step(batch_dict, tokenizer, actions)
        elif task_type == 'maths':
            step_result = gsm_step(batch_dict, tokenizer, actions)
        elif task_type == 'complex_qa':
            step_result = complex_qa_step(batch_dict, tokenizer, actions)
        else:
            raise NotImplementedError
        pi_valid_flags.append(step_result['valid_flag'])
    
    batch_dict['invalid_flags'] = 1 - torch.as_tensor(pi_valid_flags).to(batch_dict['valid_flags'].device)

    # actions = '\n'
    # batch_dict['invalid_flags'] = 1 - batch_dict['valid_flags']

    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
        pi_actions = torch.as_tensor(tokenizer(actions.strip() + tokenizer.eos_token)['input_ids']).to(batch_dict['states_agent'].device)[1:].unsqueeze(dim=0)
    elif isinstance(tokenizer, GPT2Tokenizer):
        pi_actions = torch.as_tensor(tokenizer(actions.strip() + tokenizer.eos_token)['input_ids']).to(batch_dict['states_agent'].device).unsqueeze(dim=0)
    else:
        raise NotImplementedError
    batch_dict['pi_actions'] = pi_actions

    model.train()
    
    return batch_dict


import sys
def redirect_output_to_file(func, filename):
    # Save the current standard output
    original_stdout = sys.stdout
    
    try:
        # Open the file in write mode
        with open(filename, 'w') as file:
            # Redirect the standard output to the file
            sys.stdout = file
            
            # Call the function that prints content
            func()
    finally:
        # Restore the original standard output
        sys.stdout = original_stdout


import deepspeed
from datetime import datetime
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
def main():
    args = parse_args()

    # Test config
    if args.debug:
        args.use_flash_attn = True
        args.use_slow_tokenizer = True
        args.low_cpu_mem_usage = True  # Used by device_map = 'auto'
        args.target_update_interval = 1  # check whether target update works
        args.with_tracking = True  # check whether logger module works
        args.model_name_or_path = '/home/ubuntu/yangsihang/llm_ref/.cache/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10'
        if 'gpt2' in args.model_name_or_path:
            args.use_flash_attn = False  # GPT2 not support it!
        args.output_dir = str(Path(args.output_dir).joinpath('debug'))
    # Train config
    else:
        args.output_dir = str(Path(args.output_dir).joinpath('train'))
        args.use_flash_attn = True
        args.use_slow_tokenizer = True
        args.with_tracking = True

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        # accelerator_log_kwargs["logdir"] = args.output_dir.joinpath('logs')
        accelerator_log_kwargs["project_dir"] = args.output_dir

    # accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, cpu=args.cpu, **accelerator_log_kwargs)
    from accelerate.utils import InitProcessGroupKwargs
    from datetime import timedelta
    accelerate_kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(seconds=1800))]
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, cpu=args.cpu, kwargs_handlers=accelerate_kwargs_handlers, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    accelerator.wait_for_everyone()

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
        )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = str(args.train_file)
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            **dataset_args,
        )

    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        if args.use_qlora:
            raise NotImplementedError
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            device_index = accelerator.local_process_index
            device_map = {"": device_index} # force data-parallel training.
            pretrained_model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                load_in_4bit=True,
                quantization_config=bnb_config,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                use_flash_attention_2=True if args.use_flash_attn else False,
            )
        else:
            use_flash_attn = args.use_flash_attn and not isinstance(tokenizer, GPT2Tokenizer)
            if args.debug and (isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast)):
                pretrained_model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    low_cpu_mem_usage=args.low_cpu_mem_usage,
                    use_flash_attention_2=True if use_flash_attn else False,
                    torch_dtype=config.torch_dtype,
                    device_map='auto',
                )
            else:
                if args.cpu:
                    pretrained_model = AutoModelForCausalLM.from_pretrained(
                        args.model_name_or_path,
                        from_tf=bool(".ckpt" in args.model_name_or_path),
                        config=config,
                        low_cpu_mem_usage=args.low_cpu_mem_usage,
                        use_flash_attention_2=True if use_flash_attn else False,
                        torch_dtype=config.torch_dtype,
                        device_map='cpu',
                    )
                else:
                    pretrained_model = AutoModelForCausalLM.from_pretrained(
                        args.model_name_or_path,
                        from_tf=bool(".ckpt" in args.model_name_or_path),
                        config=config,
                        low_cpu_mem_usage=args.low_cpu_mem_usage,
                        use_flash_attention_2=True if use_flash_attn else False,
                        torch_dtype=config.torch_dtype,
                    )
    else:
        raise NotImplementedError
        logger.info("Training new model from scratch")
        pretrained_model = AutoModelForCausalLM.from_config(config)

    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        })
        assert num_added_tokens in [0, 1], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
    elif isinstance(tokenizer, GPTNeoXTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens({
            "pad_token": "<pad>",
        })
        assert num_added_tokens == 1, "GPTNeoXTokenizer should only add one special token - the pad_token."
    elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(pretrained_model, OPTForCausalLM):
        num_added_tokens = tokenizer.add_special_tokens({'unk_token': '<unk>'})
    elif isinstance(tokenizer, GPT2Tokenizer):
        num_added_tokens = tokenizer.add_special_tokens({
            'unk_token': '<unk>',
            'pad_token': '<pad>',
        })

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = pretrained_model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        pretrained_model.resize_token_embeddings(len(tokenizer))

    if args.seed is not None:
        set_seed(args.seed)

    if args.use_lora:
        raise NotImplementedError
        if args.use_qlora:
            pretrained_model = prepare_model_for_kbit_training(pretrained_model, use_gradient_checkpointing=args.gradient_checkpointing)
        logger.info("Initializing LORA model...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=args.lora_rank, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"]
        )
        pretrained_model = get_peft_model(pretrained_model, peft_config)
        pretrained_model.print_trainable_parameters()

    # Preprocessing the datasets.
    if "prompt" in raw_datasets["train"].column_names and "completion" in raw_datasets["train"].column_names:
        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
        )
    elif "messages" in raw_datasets["train"].column_names:
        encode_function = partial(
            encode_with_messages_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
        )
    else:
        raise ValueError("You need to have either 'prompt'&'completion' or 'messages' in your column names.")
    
    with accelerator.main_process_first():
        lm_datasets = raw_datasets.map(
            encode_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            remove_columns=[name for name in raw_datasets["train"].column_names if name not in ["input_ids", "labels", "attention_mask"]],
            desc="Tokenizing and reformatting instruction data",
        )
        lm_datasets.set_format(type="pt")
        lm_datasets = lm_datasets.filter(lambda example: (example['labels'] != -100).any())

    train_dataset = lm_datasets["train"]
    task_ids = raw_datasets['train']['id']
    wrap_result = wrap_dataset_vf(args=args, dataset=train_dataset, task_name=args.task_name, task_ids=task_ids, tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    rl_train_dataset = wrap_result['wrapped_dataset']
    token_space = wrap_result['token_space']

    # Log a few random samples from the training set:
    # for index in random.sample(range(len(rl_train_dataset)), 3):
        # logger.info(f"Sample {index} of the training set: {rl_train_dataset[index]}.")

    # Wrap LM to RL-style
    q_head_kwargs = {
        'num_network': args.num_network,
    }
    cql_kwargs = {
        'use_lagrange': args.use_lagrange,
    }
    alpha_optim_kwargs = {
        'alpha_multiplier': 1.0,
    }
    policy_optim_kwargs = {
    }
    value_optim_kwargs = {
        'gamma': 0.99,
        'cql_alpha': 5.0,
        'temperature': 1.0,
    }
    ivf_optim_kwargs = {
        'forward_method': 'NN',
        'temperature': 1.0
    }
    model = CQLModelForCausalLM(pretrained_model=pretrained_model,
                                tokenizer=tokenizer,
                                token_space=token_space,
                                q_head_kwargs=q_head_kwargs,
                                cql_kwargs=cql_kwargs,
                                alpha_optim_kwargs=alpha_optim_kwargs,
                                policy_optim_kwargs=policy_optim_kwargs,
                                value_optim_kwargs=value_optim_kwargs,
                                ivf_optim_kwargs=ivf_optim_kwargs,
                                )

    # DataLoaders creation:
    # train_dataloader = DataLoader(
        # rl_train_dataset,
        # shuffle=True, 
        # collate_fn=DataCollatorForSeqRL(tokenizer=tokenizer, model=model.pretrained_model, padding="longest"),
        # batch_size=args.per_device_train_batch_size
    # )
    train_dataloader = DataLoader(
        rl_train_dataset,
        shuffle=True, 
        collate_fn=DataCollatorForSeqRLVF(tokenizer=tokenizer, model=model.pretrained_model, padding="longest"),
        batch_size=args.per_device_train_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    if args.use_qlora:
        from bitsandbytes.optim import AdamW
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            optim_bits=8 if args.use_8bit_optimizer else 32,
            is_paged=True
        )
    else:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler for the `num_processes` times. This is because they assume 
    # the user initialize the scheduler with the entire training set. In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set. So each time the process needs to update the lr multiple times so that the total 
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the entire training set (when epochs is specified) or we need to multiply the 
    # num_training_steps by num_processes so that the total number of updates matches the num_training_steps.
    num_training_steps_for_scheduler = args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )

    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y/%m/%d/%H%M")
        accelerator.init_trackers(f"llm_agent_{formatted_datetime}", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(rl_train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                int(training_difference.replace("step_", ""))
                * args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    target_update_kwargs = {
        'update_method': args.target_update_method,
        'update_interval': args.target_update_interval,
        'tau': 0.005,
    }

    model.train()
    accumulated_steps = 0

    model_name = None
    for tmp_model_name in model_name_list:
        if tmp_model_name in args.model_name_or_path:
            model_name = tmp_model_name
    assert model_name is not None

    for epoch in range(starting_epoch, args.num_train_epochs):
        total_loss = 0
        if (
            args.resume_from_checkpoint
            and epoch == starting_epoch
            and resume_step is not None
        ):
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
        else:
            active_dataloader = train_dataloader
        
        import queue
        log_info_queue = queue.Queue(maxsize=200)
        cuda_usage_dir = lumos_dir.joinpath('cuda_usage')
        cuda_usage_dir.mkdir(exist_ok=True, parents=True)
        is_vf_called_list = []
        for step, batch_dict in enumerate(active_dataloader):
            accumulated_steps += 1
            with accelerator.accumulate(model):
                compute_vf = batch_dict['valid_flags'] == 1
                is_vf_called_list.append(compute_vf)
                reporter = CustomMemReporter(model)
                rl_forward_result = accelerator.unwrap_model(model).rl_forward(batch_dict=batch_dict,
                                                                               compute_vf=compute_vf,
                                                                               task_ids=task_ids,
                                                                               gsm_step=gsm_step,
                                                                               use_distributed=accelerator.use_distributed,
                                                                               )
                loss = rl_forward_result['loss']
                log_info_queue.put(rl_forward_result['log_info'])
                # We keep track of the loss at each logged step
                total_loss += loss.detach().float()

                if not args.debug or isinstance(tokenizer, GPT2Tokenizer):
                    accelerator.backward(loss)
                    # clip gradient norm. don't do this with deepspeed
                    if accelerator.sync_gradients and args.clip_grad_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

            if accumulated_steps % target_update_kwargs['update_interval'] == 0:
                params_to_fetch = [
                    p for p in model.parameters()
                    if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
                ]
                with deepspeed.zero.GatheredParameters(params_to_fetch, modifier_rank=0):
                    accelerator.unwrap_model(model).update_target_value_head(target_update_kwargs)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                # Only record when sync_gradients
                output_filename = cuda_usage_dir.joinpath(f"Desc={model_name.upper()}&Epoch={epoch}&Step={step}.txt")
                redirect_output_to_file(reporter.report, output_filename)

                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = accelerator.gather(total_loss).mean().item() / args.gradient_accumulation_steps / args.logging_steps
                    logger.info(f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}")
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "learning_rate": lr_scheduler.get_last_lr()[0],
                                "train_loss": avg_loss,
                            },
                            step=completed_steps,
                        )
                        loss_info = {
                            'loss/T_loss': np.mean([log_info['T_loss'] for log_info in log_info_queue.queue]),
                            'loss/actor_loss': np.mean([log_info['actor_loss'] for log_info in log_info_queue.queue]),
                            'loss/critic_loss': np.mean([log_info['critic_loss'] for log_info in log_info_queue.queue]),
                            'loss/ivf_loss': np.mean([log_info['ivf_loss'] for log_info in log_info_queue.queue]),
                        }
                        T_loss_info = {
                            f'T_loss/{key}': np.mean([log_info['T_loss_info'][key] for log_info in log_info_queue.queue]) for key in log_info_queue.queue[0]['T_loss_info'].keys()
                        }
                        actor_loss_info = {
                            f'actor_loss/{key}': np.mean([log_info['actor_loss_info'][key] for log_info in log_info_queue.queue]) for key in log_info_queue.queue[0]['actor_loss_info'].keys()
                        }
                        critic_loss_info = {
                            f'critic_loss/{key}': np.mean([log_info['critic_loss_info'][key] for log_info in log_info_queue.queue]) for key in log_info_queue.queue[0]['critic_loss_info'].keys()
                        }
                        ivf_loss_info = {
                            f'ivf_loss/{key}': np.mean([log_info['ivf_loss_info'][key] for log_info in log_info_queue.queue]) for key in log_info_queue.queue[0]['ivf_loss_info'].keys()
                        }

                        accelerator.log(loss_info, step=completed_steps)
                        accelerator.log(T_loss_info, step=completed_steps)
                        accelerator.log(actor_loss_info, step=completed_steps)
                        accelerator.log(critic_loss_info, step=completed_steps)
                        accelerator.log(ivf_loss_info, step=completed_steps)

                    total_loss = 0
                    log_info_queue.queue.clear()
                    
                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        save_with_accelerate(accelerator, model, tokenizer, output_dir, args)

                if completed_steps >= args.max_train_steps:
                    break

            get_accelerator().empty_cache()
        
        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            save_with_accelerate(accelerator, model, tokenizer, output_dir, args)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
        save_with_accelerate(accelerator, model, tokenizer, args.output_dir, args)


if __name__ == "__main__":
    main()
