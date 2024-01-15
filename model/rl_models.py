import sys
import torch
import torch.nn.functional as F

from typing import Tuple
from torch import nn
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
from pathlib import Path


lumos_dir = Path(__file__).parent.parent


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


class InValidFlagHead(nn.Module):
    def __init__(self, config: AutoConfig, head_kwargs: dict):
        super().__init__()
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        else:
            hidden_size = config.hidden_size

        self.tensor_dtype = head_kwargs['tensor_dtype']
        self.invalid_flag_net = nn.Sequential(
            nn.Linear(hidden_size, 1, dtype=self.tensor_dtype),
            nn.Sigmoid(),
        )

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        invalid_flag_tensor = self.invalid_flag_net(hidden_states)

        return invalid_flag_tensor


class EnsembleQHead(nn.Module):
    def __init__(self, config: AutoConfig, head_kwargs: dict):
        super().__init__()
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        else:
            hidden_size = config.hidden_size
        action_space_size = config.vocab_size

        self.num_network = head_kwargs['num_network']
        self.tensor_dtype = head_kwargs['tensor_dtype']
        self.q_net_list = nn.ModuleList()
        for net_idx in range(self.num_network):
            q_module = nn.Sequential(
                nn.Linear(hidden_size, action_space_size, dtype=self.tensor_dtype),
            )
            
            self.q_net_list.append(q_module)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        q_value_list = []
        for net_idx in range(self.num_network):
            q_value = self.q_net_list[net_idx](hidden_states)
            q_value_list.append(q_value.unsqueeze(dim=-1))

        q_value_tensor = torch.cat(q_value_list, dim=-1)

        return q_value_tensor


class Scalar(nn.Module):
    def __init__(self, init_value: float, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=dtype))

    def forward(self) -> nn.Parameter:
        return self.constant


class CQLModelForCausalLM(nn.Module):
    def __init__(self,
                 pretrained_model: AutoModelForCausalLM,
                 tokenizer: AutoTokenizer,
                 token_space: torch.Tensor,
                 q_head_kwargs: dict,
                 cql_kwargs: dict,
                 alpha_optim_kwargs: dict,
                 policy_optim_kwargs: dict,
                 value_optim_kwargs: dict,
                 ivf_optim_kwargs: dict,
                ) -> None:
        super().__init__()
        self.pretrained_model = pretrained_model
        self.tokenizer = tokenizer
        self.tensor_dtype = self.pretrained_model.dtype

        """
        1. Policy Head
            1) = self.pretrained_model.lm_head
            2) ref: https://github.com/tinkoff-ai/CORL/blob/main/algorithms/offline/cql.py#L437
        2. Value Head
            1) = self.q_head
            2) ref: https://github.com/takuseno/d3rlpy/blob/master/d3rlpy/algos/qlearning/torch/cql_impl.py#L219
        """
        self.alpha_optim_kwargs = alpha_optim_kwargs
        self.policy_optim_kwargs = policy_optim_kwargs
        self.value_optim_kwargs = value_optim_kwargs
        self.ivf_optim_kwargs = ivf_optim_kwargs

        self.log_alpha = Scalar(init_value=0.0, dtype=self.tensor_dtype)
        self.alpha_multiplier = self.alpha_optim_kwargs['alpha_multiplier']
        # ref: https://github.com/ray-project/ray/blob/master/rllib/algorithms/sac/sac_torch_model.py#L118
        self.token_space = token_space
        self.action_size = self.token_space.shape[0]
        self.target_entropy = (-torch.log(torch.as_tensor(1.0 / self.action_size)) * 0.98).to(self.tensor_dtype)

        ivf_head_kwargs = {
            'tensor_dtype': self.tensor_dtype,
        }
        q_head_kwargs['tensor_dtype'] = self.tensor_dtype
        self.ivf_head = InValidFlagHead(config=self.pretrained_model.config, head_kwargs=ivf_head_kwargs)
        self.q_head = EnsembleQHead(config=self.pretrained_model.config, head_kwargs=q_head_kwargs)
        self.target_q_head = EnsembleQHead(config=self.pretrained_model.config, head_kwargs=q_head_kwargs)
        self.target_q_head.load_state_dict(self.q_head.state_dict())
        for param in self.target_q_head.parameters():
            param.requires_grad = False

        self.cql_alpha = self.value_optim_kwargs['cql_alpha']

        self.cql_kwargs = cql_kwargs
        if self.cql_kwargs['use_lagrange']:
            raise NotImplementedError
    
    @torch.no_grad()
    def update_target_value_head(self, update_kwargs: dict):
        update_method = update_kwargs['update_method']
        if update_method == 'hard':
            self.target_q_head.load_state_dict(self.q_head.state_dict())
        elif update_method == 'soft':
            tau = update_kwargs['tau']
            for target_param, source_param in zip(self.target_q_head.parameters(), self.q_head.parameters()):
                target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)
        else:
            raise NotImplementedError

    def compute_action_and_log_probs_from_logits(self, logits: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(logits.shape) == 2, "logits should be a 2D tensor with shape = (B, V)!"
        action_dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            actions = action_dist.mode
        else:
            actions = action_dist.sample()

        actions = actions.unsqueeze(dim=-1)  # Used for indexing, so do not need to be converted to self.tensor_dtype
        log_probs = action_dist.logits.to(self.tensor_dtype)

        return actions, log_probs

    def compute_ivf_info(self,
                         ivf_head_input_on_states: torch.Tensor,
                         ) -> dict:
        if self.ivf_optim_kwargs['forward_method'] == 'NN':
            ivf_values = self.ivf_head(ivf_head_input_on_states)  # type = float
            ivf_rewards = -torch.round(ivf_values)  # type = int
        else:
            raise NotImplementedError
        
        ivf_info = {
            'ivf_rewards': ivf_rewards,
            'ivf_values': ivf_values,
        }
        
        return ivf_info

    # ref: d3rlpy.DiscreteSACImpl.update_temp + CORL(https://github.com/tinkoff-ai/CORL/blob/main/algorithms/offline/cql.py#L525)
    def compute_sac_alpha_loss(self, ts_log_probs: torch.Tensor) -> dict:
        with torch.no_grad():
            ts_probs = ts_log_probs.exp()
            expect_ts_log_probs = (ts_probs * ts_log_probs).sum(dim=1, keepdim=True)
            targ_temp = expect_ts_log_probs + self.target_entropy

        alpha_loss = -(
                self.log_alpha().exp() * targ_temp
            ).mean()
        alpha = self.log_alpha().exp() * self.alpha_multiplier

        result = {
            'loss': alpha_loss,
            'alpha': alpha,
        }

        return result
    
    # ref: d3rlpy.DiscreteSACImpl.compute_actor_loss
    def compute_actor_loss(self,
                           alpha: torch.Tensor,
                           ts_log_probs: torch.Tensor,
                           q_table: torch.Tensor,
                           ivf_rewards: torch.Tensor = None,
                           ) -> dict:
        ts_q_table = q_table[:, self.token_space]
        ts_probs = ts_log_probs.exp()

        actor_loss = (ts_probs * (alpha * ts_log_probs - ts_q_table - ivf_rewards)).sum(dim=1).mean(dim=0)
        
        result = {
            'loss': actor_loss,
        }

        return result

    def compute_conservative_loss(self,
                                  q_table: torch.Tensor,
                                  actions: torch.Tensor,
                                  ) -> dict:
        # ref: https://github.com/tinkoff-ai/CORL/blob/main/algorithms/offline/cql.py#L666
        logsumexp = self.value_optim_kwargs['temperature'] * torch.logsumexp(q_table / self.value_optim_kwargs['temperature'], dim=-1, keepdim=True)
        dataset_q_values = torch.gather(q_table, dim=-1, index=actions.unsqueeze(dim=-1))

        conservative_loss = (logsumexp - dataset_q_values).mean()

        result = {
            'loss': conservative_loss,
            'logsumexp': logsumexp,
            'dataset_q_values': dataset_q_values,
        }

        return result

    def compute_critic_loss(self,
                            q_table: torch.Tensor,
                            actions: torch.Tensor,
                            new_next_actions: torch.Tensor,
                            next_q_table: torch.Tensor,
                            rewards: torch.Tensor,
                            dones: torch.Tensor,
                            ) -> dict:
        q_values = torch.gather(q_table, dim=-1, index=actions.unsqueeze(dim=-1))
        next_q_values = torch.gather(next_q_table, dim=-1, index=new_next_actions)
        target_q_values = rewards.unsqueeze(dim=-1).to(self.tensor_dtype) + self.value_optim_kwargs['gamma'] * (1 - dones.unsqueeze(dim=-1).to(self.tensor_dtype)) * next_q_values.detach()
        td_loss = F.mse_loss(q_values, target_q_values).mean(dim=0)
        conservative_loss_result = self.compute_conservative_loss(q_table=q_table, actions=actions)
        conservative_loss = conservative_loss_result['loss']
        critic_loss = td_loss + self.cql_alpha * conservative_loss

        result = {
            'loss': critic_loss,
            'td_loss': td_loss,
            'conservative_loss': conservative_loss,
            'cql_alpha': self.cql_alpha,

            'q_values': q_values,
            'target_q_values': target_q_values,
            'logsumexp': conservative_loss_result['logsumexp'],
            'dataset_q_values': conservative_loss_result['dataset_q_values'],
        }

        return result

    def compute_invalid_flag_loss(self,
                                  ivf_values: torch.Tensor,
                                  invalid_flags: torch.Tensor,
                                ) -> dict:
        ivf_values = ivf_values.flatten()
        invalid_flags = invalid_flags.to(self.tensor_dtype)
        ivf_loss = F.binary_cross_entropy(ivf_values, invalid_flags).mean(dim=0)

        result = {
            'loss': ivf_loss,
            'ivf_values': ivf_values,
        }

        return result

    def rl_forward(self,
                batch_dict: dict,
                **kwargs,
                ) -> dict:
        compute_vf = kwargs['compute_vf']
        use_distributed = kwargs['use_distributed']
        batch_size = batch_dict['actions'].shape[0]

        self.eval()

        max_pi_actions_length = batch_dict['states_agent'].shape[1]
        pi_actions_dict = {}
        invalid_flags = batch_dict['valid_flags'].clone()
        new_line_token = self.tokenizer.encode("\n", add_special_tokens=False)[-1]
        stop_id_sequence = [new_line_token]
        max_new_token_cnt = 256
        # max_new_token_cnt = 16
        task_ids = kwargs['task_ids']
        gsm_step = kwargs['gsm_step']
        for batch_idx in range(batch_size):
            end_idx = max_new_token_cnt
            ivf_action_list = []
            indices = (batch_dict['states_user'][batch_idx] == self.tokenizer.pad_token_id).nonzero()
            # Without padding
            if indices.numel() == 0:
                ivf_input_ids = batch_dict['states_user'][batch_idx].unsqueeze(dim=0)
            # With padding
            else:
                ivf_input_ids = batch_dict['states_user'][batch_idx, :indices[0]].unsqueeze(dim=0)
            for new_token_idx in range(max_new_token_cnt):
                ivf_outputs = self.pretrained_model(input_ids=ivf_input_ids, use_cache=False)
                new_action_logits = ivf_outputs.logits[:, -1, :]
                ivf_action, _ = self.compute_action_and_log_probs_from_logits(logits=new_action_logits, deterministic=True)
                ivf_action_list.append(ivf_action.item())
                ivf_input_ids = torch.cat([ivf_input_ids, ivf_action], dim=-1)

                # Can not break o.t. parallel runing gpus will hang
                if ivf_action.item() == self.tokenizer.eos_token_id or ivf_input_ids[-len(stop_id_sequence):].tolist() == stop_id_sequence:
                    end_idx = min(end_idx, new_token_idx + 1)

            actions_tensor = torch.as_tensor(ivf_action_list)[:end_idx]
            actions = self.tokenizer.decode(actions_tensor)
            task_type = '_'.join(task_ids[batch_dict['task_ids'][batch_idx]].split('_')[:-1])
            if task_type == 'maths':
                step_result = gsm_step(batch_dict, self.tokenizer, actions)
            else:
                raise NotImplementedError

            pi_valid_flag = int(step_result['valid_flag'])
            if compute_vf[batch_idx]:
                max_pi_actions_length = max(max_pi_actions_length, actions_tensor.shape[0])
                pi_actions_dict[batch_idx] = actions_tensor.to(batch_dict['states_agent'].device)
                invalid_flags[batch_idx] = 1 - torch.as_tensor(pi_valid_flag).to(batch_dict['valid_flags'].device)
        
        pi_actions = torch.ones((batch_size, max_pi_actions_length), dtype=torch.long) * self.tokenizer.pad_token_id
        for batch_idx in range(batch_size):
            actions = pi_actions_dict.get(batch_idx, None)
            if actions is None:
                actions = batch_dict['states_agent'][batch_idx]

            pi_actions[batch_idx][:actions.shape[0]] = actions

        self.train()

        states_inputs = dict(
            input_ids = torch.cat([batch_dict['states_user'], batch_dict['states_agent']], dim=1),
        )
        states_outputs = self.pretrained_model(**states_inputs, use_cache=False, output_hidden_states=True)
        states_last_hidden_state = states_outputs.hidden_states[-1]
        next_states_inputs = dict(
            input_ids = batch_dict['next_states'],
        )
        next_states_outputs = self.pretrained_model(**next_states_inputs, use_cache=False, output_hidden_states=True)
        next_states_last_hidden_state = next_states_outputs.hidden_states[-1]
        q_head_input_on_states = states_last_hidden_state[:, -1, :]
        q_head_input_on_next_states = next_states_last_hidden_state[:, -1, :]
        logits_on_states = states_outputs.logits[:, -1, :]
        logits_on_next_states = next_states_outputs.logits[:, -1, :]

        try:
            ivf_inputs = dict(
                input_ids = torch.cat([batch_dict['states_user'], pi_actions.to(batch_dict['states_user'].device)], dim=1),
            )
            ivf_head_input_on_states = self.pretrained_model(**ivf_inputs, use_cache=False, output_hidden_states=True).hidden_states[-1][:, -1, :]
        except RuntimeError:
            error_hint_list = [
                f'=' * 64,
                f'Error states_user dtype: {batch_dict["states_user"].dtype}',
                f'Error states_user: {batch_dict["states_user"]}',
                f'Error pi_actions dtype: {pi_actions.dtype}',
                f'Error pi_actions: {pi_actions}',
                f'=' * 64,
            ]
            error_hint_str = '\n'.join(error_hint_list)
            redirect_output_to_file(print(error_hint_str), lumos_dir.joinpath('results/errors.txt'))
            
            ivf_inputs = dict(
                input_ids = torch.cat([batch_dict['states_user'], pi_actions.to(batch_dict['states_user'].device)], dim=1).to(torch.long),
            )
            ivf_head_input_on_states = self.pretrained_model(**ivf_inputs, use_cache=False, output_hidden_states=True).hidden_states[-1][:, -1, :]

        if not use_distributed:
            self.ivf_head.to(ivf_head_input_on_states.device)

        ivf_info = self.compute_ivf_info(ivf_head_input_on_states=ivf_head_input_on_states)

        # Using token_space to mask out invalid actions
        states_token_space_mask = torch.ones_like(logits_on_states, dtype=torch.bool)
        next_states_token_space_mask = torch.ones_like(logits_on_next_states, dtype=torch.bool)
        states_token_space_mask.scatter_(dim=1, index=self.token_space.unsqueeze(dim=0).repeat(batch_size, 1).to(logits_on_states.device), value=False)
        next_states_token_space_mask.scatter_(dim=1, index=self.token_space.unsqueeze(dim=0).repeat(batch_size, 1).to(logits_on_next_states.device), value=False)
        logits_on_states[states_token_space_mask] = -float('inf')
        logits_on_next_states[next_states_token_space_mask] = -float('inf')

        new_actions, log_probs = self.compute_action_and_log_probs_from_logits(logits=logits_on_states, deterministic=False)
        ts_log_probs = log_probs[:, self.token_space]
        new_next_actions, _ = self.compute_action_and_log_probs_from_logits(logits=logits_on_next_states, deterministic=False)

        if not use_distributed:
            self.q_head.to(q_head_input_on_states.device)
            self.target_q_head.to(q_head_input_on_next_states.device)

        q_table = self.q_head(q_head_input_on_states).min(dim=-1).values
        next_q_table = self.target_q_head(q_head_input_on_next_states).min(dim=-1).values

        sac_alpha_loss_result = self.compute_sac_alpha_loss(ts_log_probs=ts_log_probs)
        sac_alpha = sac_alpha_loss_result['alpha']
        sac_alpha_loss = sac_alpha_loss_result['loss']

        critic_loss_result = self.compute_critic_loss(q_table=q_table, actions=batch_dict['actions'], new_next_actions=new_next_actions, next_q_table=next_q_table, rewards=batch_dict['rewards'], dones=batch_dict['dones'])
        critic_loss = critic_loss_result['loss']
        
        ivf_rewards = ivf_info['ivf_rewards']
        ivf_values = ivf_info['ivf_values']

        actor_loss_result = self.compute_actor_loss(alpha=sac_alpha, ts_log_probs=ts_log_probs, q_table=q_table, ivf_rewards=ivf_rewards)
        actor_loss = actor_loss_result['loss']

        ivf_loss_result = self.compute_invalid_flag_loss(ivf_values=ivf_values, invalid_flags=invalid_flags)
        ivf_loss = ivf_loss_result['loss']
        loss = sac_alpha_loss + actor_loss + critic_loss + ivf_loss

        T_loss_info = {
            'loss': sac_alpha_loss.item(),
            'T': sac_alpha.item(),
        }
        policy_Q = torch.gather(q_table, dim=1, index=new_actions).mean().item()
        dataset_Q = torch.gather(q_table, dim=1, index=batch_dict['actions'].unsqueeze(dim=1)).mean().item()
        actor_loss_info = {
            'loss': actor_loss.item(),
            'T': sac_alpha.item(),
            'ivf_rewards': ivf_rewards.mean().item() if ivf_rewards is not None else 0.0,
            'policy_Q': policy_Q,
            'dataset_Q': dataset_Q,
            'policy_log_probs': torch.gather(log_probs, dim=1, index=new_actions).mean().item(),
            'dataset_log_probs': torch.gather(log_probs, dim=1, index=batch_dict['actions'].unsqueeze(dim=1)).mean().item(),
        }
        critic_loss_info = {
            'loss': critic_loss.item(),
            'td_loss': critic_loss_result['td_loss'].item(),
            'conservative_loss': critic_loss_result['conservative_loss'].item(),
            'cql_alpha': critic_loss_result['cql_alpha'],

            # td_loss
            'q_values': critic_loss_result['q_values'].mean().item(),
            'target_q_values': critic_loss_result['target_q_values'].mean().item(),
            # conservative_loss
            'logsumexp': critic_loss_result['logsumexp'].mean().item(),
            'dataset_q_values': critic_loss_result['dataset_q_values'].mean().item(),
        }
        ivf_loss_info = {
            'loss': ivf_loss.item(),
            'ivf_values': ivf_values.mean().item() if ivf_values is not None else 0.0,
            'compute_ivf': compute_vf.to(self.tensor_dtype).mean().item(),
        }
        log_info = dict(
            T_loss=sac_alpha_loss.item(),
            actor_loss=actor_loss.item(),
            critic_loss=critic_loss.item(),
            ivf_loss=ivf_loss.item(),

            T_loss_info=T_loss_info,
            actor_loss_info=actor_loss_info,
            critic_loss_info=critic_loss_info,
            ivf_loss_info=ivf_loss_info,
        )

        forward_result = dict(
            loss=loss,
            log_info=log_info,
        )

        return forward_result

    def generate(self,
                 input_ids: torch.Tensor,
                 attention_mask: torch.Tensor,
                 stop_id_sequences: list,
                 **generation_kwargs
                ) -> torch.Tensor:
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!"

        outputs = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits[:, -1, :]
        states_token_space_mask = torch.ones_like(logits, dtype=torch.bool)

        token_space = self.token_space.clone()
        if stop_id_sequences[0][0] not in token_space:
            token_space = torch.cat([token_space, torch.as_tensor(stop_id_sequences[0])])

        states_token_space_mask.scatter_(dim=1, index=token_space.unsqueeze(dim=0).to(logits.device), value=False)
        logits[states_token_space_mask] = -float('inf')

        max_new_tokens = generation_kwargs['max_new_tokens']
        assert len(stop_id_sequences) == 1, "Only support one stop_id_sequences for now!"
        stop_id_sequence = stop_id_sequences[0]
        output_ids_tensor = input_ids.clone()

        for idx in range(max_new_tokens):
            new_actions, _ = self.compute_action_and_log_probs_from_logits(logits=logits, deterministic=True)
            output_ids_tensor = torch.cat([output_ids_tensor, new_actions], dim=-1)

            # EOS Token
            if new_actions.item() == self.tokenizer.eos_token_id:
                break
            # Stop Token
            if output_ids_tensor.shape[-1] >= len(stop_id_sequence):
                if output_ids_tensor[0, -len(stop_id_sequence):].tolist() == stop_id_sequence:
                    break

            input_ids = torch.cat([input_ids, new_actions], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(new_actions, dtype=torch.int64)], dim=-1)
            outputs = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits[:, -1, :]
            logits[states_token_space_mask] = -float('inf')

        return output_ids_tensor
    
    def save_pretrained(self, save_directory: str, **save_kwargs):
        self.pretrained_model.save_pretrained(save_directory, **save_kwargs)
        self.tokenizer.save_pretrained(save_directory)

        ivf_head_save_path = Path(save_directory).joinpath('ivf_head.pt')
        torch.save(self.ivf_head.state_dict(), ivf_head_save_path)
        q_head_save_path = Path(save_directory).joinpath('q_head.pt')
        torch.save(self.q_head.state_dict(), q_head_save_path)
        target_q_head_save_path = Path(save_directory).joinpath('target_q_head.pt')
        torch.save(self.target_q_head.state_dict(), target_q_head_save_path)

    def from_pretrained(self, args, save_directory: str):
        config = AutoConfig.from_pretrained(save_directory)
        if args.debug:
            self.pretrained_model = AutoModelForCausalLM.from_pretrained(
                save_directory,
                from_tf=bool(".ckpt" in save_directory),
                config=config,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
                use_flash_attention_2=True if args.use_flash_attn else False,
                torch_dtype=config.torch_dtype,
                device_map='auto',
            )
        else:
            self.pretrained_model = AutoModelForCausalLM.from_pretrained(
                save_directory,
                from_tf=bool(".ckpt" in save_directory),
                config=config,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
                use_flash_attention_2=True if args.use_flash_attn else False,
                torch_dtype=config.torch_dtype,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(save_directory, use_fast=not args.use_slow_tokenizer)

        ivf_head_save_path = Path(save_directory).joinpath('ivf_head.pt')
        self.ivf_head.load_state_dict(torch.load(ivf_head_save_path))
        q_head_save_path = Path(save_directory).joinpath('q_head.pt')
        self.q_head.load_state_dict(torch.load(q_head_save_path))
        target_q_head_save_path = Path(save_directory).joinpath('target_q_head.pt')
        self.target_q_head.load_state_dict(torch.load(target_q_head_save_path))
