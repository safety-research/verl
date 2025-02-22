# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
import requests
import numpy as np

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, get_final_eos_mask, pad_sequence_to_length
#from verl.utils.py_functional import recursive_all_np_to_list
from verl.utils.py_functional import to_1d_np_array
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from vllm import SamplingParams


# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids

def _remove_trailing_pad_tokens(pad_token_id, token_ids: torch.Tensor) -> torch.Tensor:
    # remove the right padding in the token_id
    non_pad_token_locs=torch.nonzero(token_ids != pad_token_id, as_tuple=False)
    assert len(non_pad_token_locs)>0,"No non-pad tokens: "+str(token_ids)
    max_non_pad_token_loc=non_pad_token_locs.max()
    return token_ids[:max_non_pad_token_loc+1]

def _remove_prepending_messages(token_ids: torch.Tensor, message_end_id: int, n_skip_messages: int) -> torch.Tensor:
    # only keep from the n_skip_messages+1 th appearance of message_end_id
    if n_skip_messages==0:
        return token_ids
    message_end_locs=torch.nonzero(token_ids == message_end_id, as_tuple=False)
    if len(message_end_locs)<n_skip_messages:
        assert False,"Not enough messages"
    return token_ids[message_end_locs[n_skip_messages-1]+1:]

def pad_to_max_stack(tensor_list: List[torch.Tensor], pad_token_id: int) -> torch.Tensor:
    assert all([t.ndim==1 for t in tensor_list])
    max_len=max([t.size(0) for t in tensor_list])
    padded_tensor_list=[]
    for t in tensor_list:
        padded_tensor_list.append(torch.cat([t,torch.tensor([pad_token_id]*(max_len-t.size(0)),device=t.device,dtype=t.dtype)],dim=0))
    return torch.stack(padded_tensor_list,dim=0)


class vLLMRollout(BaseRollout):

    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                                  num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"
        self.inference_engine = LLM(
            actor_module,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            tensor_parallel_size=tensor_parallel_size,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            skip_tokenizer_init=False,
            max_model_len=config.prompt_length + config.response_length,
            load_format=config.load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.offload_model_weights()

        kwargs = dict(
            n=1,
            logprobs=1,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # we may detokenize the result all together later
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        #(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            output = self.inference_engine.generate(
                prompts=None,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                prompt_token_ids=idx_list,
                use_tqdm=False)

        # TODO(sgm): disable logprob when recompute_log_prob is enable
        # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
        response = output[0].to(idx.device)
        log_probs = output[1].to(idx.device)

        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            log_probs = pad_sequence_to_length(log_probs, self.config.response_length, self.pad_token_id)

        if self.config.n > 1 and do_sample:
            idx = idx.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n
        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        # free vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)

class vLLMMultiTurnViaChatRollout(BaseRollout):

    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                                  num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"
        self.tokenizer = tokenizer
        self.total_length = config.prompt_length + config.response_length
        self.inference_engine = LLM(
            actor_module,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            tensor_parallel_size=tensor_parallel_size,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            skip_tokenizer_init=False,
            max_model_len=self.total_length,
            load_format=config.load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.offload_model_weights()

        kwargs = dict(
            n=1,
            logprobs=1,  # can be set to 0 and let actor to recompute
            max_tokens=config.environment.per_turn_length
        )

        # we may detokenize the result all together later
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)
        
        #Important:
        #This is multi turn, so we need to set n=1 for sampling params, as we will manually batch n since some samplings might terminate earlier.
        kwargs['n']=1

        #print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    def get_n_tokens(self,prompt,add_generation_prompt=False):
        return len(self.tokenizer.apply_chat_template(prompt,tokenize=True,add_generation_prompt=add_generation_prompt))
    
    def tokenize_with_assistant_mask(self,messages):
        n_messages=len(messages)
        tokenized_messages=self.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=False)
        head=0
        assistant_mask=[]
        for i_last_message in range(n_messages):
            if (i_last_message!=n_messages-1) and (messages[i_last_message+1]["role"]=="assistant"):
                is_next_assistant=True
            else:
                is_next_assistant=False
            last_message_role=messages[i_last_message]["role"]
            n_tokens_with_last_message=self.get_n_tokens(messages[:i_last_message+1],add_generation_prompt=is_next_assistant)
            n_add=n_tokens_with_last_message-head
            if last_message_role=="assistant":
                assistant_mask.append(torch.ones(n_add,dtype=torch.bool))
            else:
                assistant_mask.append(torch.zeros(n_add,dtype=torch.bool))
            head+=n_add
        assistant_mask=torch.cat(assistant_mask,dim=0)
        assert len(assistant_mask)==len(tokenized_messages), "Bug: assistant mask length mismatch"
        return tokenized_messages,assistant_mask

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto,**kwargs)->DataProto:#, environment, **kwargs) -> DataProto: #see verl/single_controller/base/decorator l.54, we can't send these classes as usual.
        #print("START!")
        # rebuild vllm cache engine
        #assert environment is not None, "Environment is required for multi-turn rollout"
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()


        idx=prompts.batch['input_ids']#just for device, size
        batch_size = idx.size(0)

        attention_mask=prompts.batch['attention_mask']
        idx_generation_mask=torch.zeros_like(idx,dtype=attention_mask.dtype,device=attention_mask.device)
        position_ids=prompts.batch['position_ids']
        #eos_token_id = prompts.meta_info['eos_token_id']
        eos_token_id=self.tokenizer.eos_token_id# the meta_info's eos_token_id is is from config and usually includes the pad token, which we don't want when having multiple messages see  get_final_eos_mask to understand this

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

        raw_prompt=prompts.non_tensor_batch['raw_prompt']
        rank=vllm_ps.get_tensor_model_parallel_src_rank()
        #if rank==0:
        #    print("RAW_PROMPT:",raw_prompt)
        #    print("RAW_PROMPT_LEN:",len(raw_prompt))
        #    print("BATCH_SIZE:",batch_size)
        #    print("IDX_TYPE:",type(idx))
        #    print("IDX",idx[:,0])
        #    assert len(raw_prompt)==batch_size, "This bug should be fixed, but didn't try different multi processing settings"
        
        n=1 if prompts.meta_info.get('validate',False) else self.config.n

        todo=list(range(batch_size*n))#manually batch n
        messagess=[]
        prefix_lengths=[]
        for i_batch in range(batch_size):
            raw_prompt_i_batch=raw_prompt[i_batch]
            n_tokens_prefix=self.get_n_tokens(raw_prompt_i_batch,add_generation_prompt=True)
            #if rank==0:
            #    print("PROMPT",raw_prompt_i_batch,"N_TOKENS:",n_tokens)
            for _ in range(n):
                messagess.append(raw_prompt_i_batch.copy())
                prefix_lengths.append(n_tokens_prefix)

        while True:
            #Prepare inputs to generate response from
            messagess_todo=[]
            for i_batch_sample in todo:
                messages_=[{"role":msg["role"],"content":msg["content"]} for msg in messagess[i_batch_sample]]
                messagess_todo.append(messages_)
            #print("START GENERATION")
            with self.update_sampling_params(**kwargs):
                assert self.sampling_params.n==1,"n should be 1 for multi-turn"
                output = self.inference_engine.chat(
                    messages=messagess_todo,
                    sampling_params=self.sampling_params,
                    use_tqdm=False)
            response = output[0].to(idx.device)
            assert response.shape[1] <= self.config.environment.per_turn_length,"Bug: response too long from vllm: "+str(response.shape)

            #trim responses
            assert len(todo)==len(response)
            for i_batch_sample,response_ in zip(todo,response):
                response_text=self.tokenizer.decode(response_,skip_special_tokens=True)
                messagess[i_batch_sample].append({"role":"assistant","content":response_text})
                """
                if torch.all(response_==self.pad_token_id):#some failure case
                    messagess[i_batch_sample].append({"role":"assistant","content":""})
                    todo.remove(i_batch_sample)
                    continue
                response_no_pad=_remove_trailing_pad_tokens(self.pad_token_id,response_)
                response_text=self.tokenizer.decode(response_no_pad.tolist(),skip_special_tokens=True)#remove <|im_end|>
                #if rank==0:
                #    print("MESSAGES:",messagess[i_batch_sample])
                #    print("Gen response text:",response_text)
                messagess[i_batch_sample].append({"role":"assistant","content":response_text})
                """
            #interact with environment
            messagess_todo=[]
            for i_batch_sample in todo:
                messagess_todo.append(messagess[i_batch_sample])

            #We can't do thus yet:
            #env_response_batched=environment.get_response_batched(messages_batched=messagess_todo)
            #So we hardcode it for now:
            payload = {
                "messages_batched": messagess_todo
            }
            url = self.config.environment.url+"/get_env_response_batched"
            #print("GET ENV RESPONSE")
            env_response_batched = requests.post(url, json=payload).json()


            #process environment response
            todo_=todo.copy()#make a copy since we will be modifying todo
            assert len(todo_)==len(env_response_batched)
            for i_batch_sample,env_response in zip(todo_,env_response_batched):
                assert env_response["role"]=="user"
                messagess[i_batch_sample].append(env_response)
                #if rank==0:
                #    print("MESSAGES:",messagess[i_batch_sample])
                n_tokens=self.get_n_tokens(messagess[i_batch_sample],add_generation_prompt=True)
                if n_tokens>=(self.total_length-self.config.environment.per_turn_length):#remove if the total number of tokens goes too long
                    todo.remove(i_batch_sample)
                    continue
                if env_response["done"]:
                    todo.remove(i_batch_sample)
            
            #break if all done
            if len(todo)==0:
                break
            #print("TODO LEFT",len(todo))

        #re-build response in the same format as normal rollout
        #only need idx_list, generation_mask, and prompt_lengths
        response=[]
        response_generation_mask=[]
        for i_batch_sample in range(batch_size*n):
            token_ids,assistant_mask=self.tokenize_with_assistant_mask(messagess[i_batch_sample])
            response.append(torch.tensor(token_ids[prefix_lengths[i_batch_sample]:],device=idx.device,dtype=idx.dtype))
            response_generation_mask.append(assistant_mask[prefix_lengths[i_batch_sample]:].to(device=attention_mask.device,dtype=attention_mask.dtype))
            #eos_token_id,
        response=pad_to_max_stack(response,self.pad_token_id)
        response_generation_mask=pad_to_max_stack(response_generation_mask,0)
        #assert same shape
        assert all([response.size(dim)==response_generation_mask.size(dim) for dim in range(response.ndim)])

        #cut or pad to max length,
        if response.shape[1] > self.config.response_length:
            response=response[:,:self.config.response_length]
            response_generation_mask=response_generation_mask[:,:self.config.response_length]
        elif response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            response_generation_mask = pad_sequence_to_length(response_generation_mask, self.config.response_length, 0)
        #print("RESPONSE.SHAPE_2:",response.shape)
        #print("CONFIG.N:",self.config.n,"DO_SAMPLE:",do_sample)

        if self.config.n > 1 and do_sample:
            idx = idx.repeat_interleave(self.config.n, dim=0)
            idx_generation_mask = idx_generation_mask.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n
        #print("IDX.SHAPE",idx.shape)
        seq = torch.cat([idx, response], dim=-1)
        #print("SEQ.SHAPE",seq.shape)


        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        #print("POSITION_IDS.SHAPE",position_ids.shape)
        #print("POSITION_IDS[0]",position_ids[0])
        #print("POSITION_IDS",position_ids)
        
        generation_mask = torch.cat([idx_generation_mask, response_generation_mask], dim=-1)
        response_attention_mask = get_final_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        #if rank==0:
        #    print("SEQ.0",self.tokenizer.decode(seq[0].tolist(),skip_special_tokens=False))
        #    print("ATTMASK.0",attention_mask[0].tolist())
        #    print("GENMASK.0",generation_mask[0].tolist())
        #    assert False


        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx.contiguous(),
                'responses': response.contiguous(),
                'input_ids': seq.contiguous(),
                'attention_mask': attention_mask.contiguous(),
                'position_ids': position_ids.contiguous(),
                "generation_mask":generation_mask.contiguous()
            },
            batch_size=batch_size)
        non_tensor_batch = {
            'messages': to_1d_np_array(messagess)
        }

        # free vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()
        #print("DONE!")
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

