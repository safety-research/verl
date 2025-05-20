from typing import Dict, Any, List

from safetytooling.data_models.messages import Prompt, ChatMessage
from safetytooling.data_models.inference import LLMResponse

from openai.types.chat.chat_completion import ChatCompletion

from verl.workers.rollout.async_server import ChatCompletionScheduler
from verl.protocol import DataProto

from omegaconf import DictConfig
from tensordict import TensorDict

import os
import sys
import argparse
import torch
import numpy as np
import asyncio
import wandb

sys.path.append(os.environ["COT_DECOMP_ROOT"])

from inference.safety_tooling_inference_api import SafetyToolingMultiTurnInferenceAPI, SafetyToolingInferenceAPI
from data.prompt.multi_turn_factored_decomposition import MultiTurnFactoredDecomposition

sys.path.append(os.path.dirname(__file__))

from naive_chat_scheduler import NaiveChatCompletionScheduler


class VerlInferenceAPIShim:
    def __init__(self, chat_scheduler: ChatCompletionScheduler):
        self.chat_scheduler = chat_scheduler
        self.rate_limit = asyncio.Semaphore(80)

    async def __call__(self, model_id: str, prompt: Prompt, print_prompt_and_response: bool = False, **kwargs):
        assert isinstance(prompt, Prompt)

        oai_format_prompt = prompt.openai_format()

        conversation = None

        # Callback is guaranteed to fire before the submit_chat_completions call returns
        async def callback(completions: ChatCompletion, info: Dict[str, Any], exception: Exception):
            nonlocal conversation
            if exception:
                raise Exception(f"exception: {exception}")
            conversation = completions.choices[0].message.content
            
        new_kwargs = {

        }
        if "temperature" in kwargs:
            new_kwargs["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            new_kwargs["top_p"] = kwargs["top_p"]
        if "max_completion_tokens" in kwargs:
            new_kwargs["max_completion_tokens"] = kwargs["max_completion_tokens"]

        # Hard code to avoid max seq length issues
        new_kwargs["max_completion_tokens"] = 384
            
        async with self.rate_limit:
            try:
                await self.chat_scheduler.submit_chat_completions(
                    callback=callback,
                    callback_additional_info={},
                    model=model_id,
                    messages=oai_format_prompt,
                    **new_kwargs,
                )
            except Exception as e:
                print(f"Error submitting chat completion: {e}")

                return [
                    LLMResponse(
                        model_id=model_id,
                        completion="<subtask_description>Return <answer>The sequence was too long</answer></subtask_description>",
                        stop_reason="stop",
                        duration=0.1,
                        api_duration=0.1,
                        cost=0,
                    )
                ]

        if print_prompt_and_response:
            print(f"Prompt: {oai_format_prompt}")
            print(f"Response: {conversation}")
        return [
            LLMResponse(
                model_id=model_id,
                completion=conversation,
                stop_reason="stop",
                duration=0.1,
                api_duration=0.1,
                cost=0,
            )
        ]
    


class VerlFactoredTaskDecompositionChatScheduler(NaiveChatCompletionScheduler):
    def __init__(
        self,
        config: DictConfig,
        model_path: str,
        server_addresses: List[str],
        max_cache_size: int = 10000,
    ):
        super().__init__(config, model_path, server_addresses, max_cache_size)

        factored_task_decomposition_responder = MultiTurnFactoredDecomposition(
            prompt_generator=None,
            args=argparse.Namespace(
                multi_turn_responder_unanswerable_subtask_response_override="regenerate",
                multi_turn_responder_unanswerable_final_subtask_response_override="regenerate_if_unanswerable_or_directly_answered",
                multi_turn_responder_subtask_prompt_suffix="all_prior_subtask_descriptions_and_results",
                subtask_use_cot=False,
            )
        )

        subtask_model_url = config.get("subtask_model_url", "https://4stqit6suci83a-8888.proxy.runpod.net/v1/")
        if subtask_model_url == "unset":
            raise ValueError("subtask_model_url is not set")
        
        model_name = config.get("subtask_model_name", "Qwen2.5-14B-MMLU-MATH-NUMINATIR-FILTER-UNANSWERABLE-SFT")

        verl_inference_shim = VerlInferenceAPIShim(self)
        
        subtask_inference_api = SafetyToolingMultiTurnInferenceAPI(
            model_name if subtask_model_url != "same_model" else self.model_name,
            multi_turn_responder=factored_task_decomposition_responder,
            print_debug_outputs=False, # TODO(sguo35)
            temperature=0.0,
            use_assistant_message=True,
            cache_dir=None,
            base_url=subtask_model_url,
            force_provider="openai",
            reranker=None
        )
        if subtask_model_url == "same_model":
            subtask_inference_api.API = verl_inference_shim
        factored_task_decomposition_responder.set_subtask_inference_api(subtask_inference_api)

        single_turn_inference_api = SafetyToolingInferenceAPI(
            model_name if subtask_model_url != "same_model" else self.model_name,
            print_debug_outputs=False,
            temperature=0.0,
            use_assistant_message=True,
            cache_dir=None,
            base_url=subtask_model_url,
            force_provider="openai",
            reranker=None,
        )
        if subtask_model_url == "same_model":
            single_turn_inference_api.API = verl_inference_shim
        factored_task_decomposition_responder.set_single_turn_inference_api(single_turn_inference_api)

        self.inference_api = SafetyToolingMultiTurnInferenceAPI(
            model_id=self.model_name,
            multi_turn_responder=factored_task_decomposition_responder,
            cache_dir=None,
            print_debug_outputs=False, # TODO(sguo35)
            use_assistant_message=True,
        )
        self.inference_api.API = verl_inference_shim


        torch.set_printoptions(profile="full")


    def _convert_msg_cache_to_openai_format(self, conversation: List[List[str]]) -> List[dict[str, str]]:
        new_conversation = []
        for msg in conversation:
            if msg[1] == "system":
                new_conversation = []

            new_conversation.append({"role": msg[1], "content": msg[0]})

        if len(new_conversation) > 0 and "final_answer" in new_conversation[-1]["content"]:
            new_conversation[-1]["role"] = "user"

        return new_conversation

    async def generate_sequences(self, batch: DataProto, **sampling_params) -> DataProto:
        kwargs = dict(
            max_completion_tokens=self.config.response_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        do_sample = batch.meta_info.get("do_sample", True)
        is_validate = batch.meta_info.get("validate", False)

        n_samples = self.config.n
        if not do_sample or is_validate:
            n_samples = 1
            kwargs["temperature"] = 0

        kwargs.update(sampling_params)
        print(f"[VerlFactoredTaskDecompositionChatScheduler] generate_sequences sampling params: {kwargs}")

        tasks = []
        batch_conversations = [[] for _ in range(len(batch) * n_samples)]
        for batch_index, conversation in enumerate(batch.non_tensor_batch["raw_prompt"]):

            async def try_catch_failed_error(index, convo):
                try:
                    await self.inference_api(
                            prompt=None,
                            l_msg_cache=batch_conversations[index],
                            use_reranker=False,
                            l_messages=self._convert_conversation_to_safety_tooling(convo),
                            kwargs_override=kwargs
                        )
                except Exception as e:
                    print(f"Error submitting chat completion: {e}")

                    # This example will be masked out in the loss because it has no assistant response
                    batch_conversations[index] = [
                        ["The sequence was too long.", "system"],
                        ["The sequence was too long.", "user"],
                    ]

                    return None

            for i in range(n_samples):
                # raw_prompt: [{"role": "user", "content": ""}, ["role": "assistant", "content"], ...]
                tasks.append(
                    try_catch_failed_error(batch_index * n_samples + i, conversation)
                )

        await asyncio.gather(*tasks)

        for i in range(len(batch_conversations)):
            batch_conversations[i] = self._convert_msg_cache_to_openai_format(batch_conversations[i])

        print("[VerlFactoredTaskDecompositionChatScheduler] generate_sequences done")
        print(len(batch_conversations), len(tasks))

        return self._postprocess(batch, batch_conversations, n_samples)
    
    def _convert_conversation_to_safety_tooling(self, conversation: List[Dict[str, str]]) -> List[ChatMessage]:
        return [ChatMessage(role=message["role"], content=message["content"]) for message in conversation]
    
    def _build_loss_mask(self, input_ids, messages: List[dict[str, str]]) -> torch.Tensor:
        """
        Builds a loss mask for the input ids.
        The loss mask is 1 for the tokens that are part of an assistant message.
        Takes a SINGLE example.
        """

        # Create loss mask by identifying assistant responses
        loss_mask = torch.zeros_like(input_ids, dtype=torch.long)

        # Process each message to find assistant responses
        for i, msg in enumerate(messages):
            # Get tokens for messages up to this point to find the start position
            prefix_messages = messages[: i + 1]
            prefix_tokens = self.tokenizer.apply_chat_template(prefix_messages, tokenize=True, return_tensors="pt", add_generation_prompt=False)

            # Get tokens for messages up to previous point
            prev_tokens = self.tokenizer.apply_chat_template(messages[:i], tokenize=True, return_tensors="pt", add_generation_prompt=False) if i > 0 else None

            # Calculate start and end positions
            start_pos = prev_tokens[0].shape[0] if prev_tokens is not None else 0
            end_pos = prefix_tokens[0].shape[0]

            # If this is an assistant message, set loss mask
            if msg["role"] == "assistant":
                loss_mask[start_pos:end_pos] = 1

        # Padding handled by dp_actor::_forward_micro_batch
        # and the loss mask is already the same shape for the entire batch
        # because the tokenization is batched

        return loss_mask.reshape(1, -1)
        
    def _postprocess(
        self, batch: DataProto, batch_conversations: List[List[List[Dict[str, str]]]], n: int
    ) -> DataProto:
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        # prompts: [prompt] from input dataset
        prompts = [
            self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
            for prompt in batch.non_tensor_batch["raw_prompt"]
        ]

        assert len(batch_conversations) == (len(prompts) * n)

        # sequences: [prompt + response]
        for i, conversation in enumerate(batch_conversations):
            print(f"conversation {i}: {conversation}")
        sequences = [
            self.tokenizer.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
            for conversation in batch_conversations
        ]

        # responses: [response]
        # TODO: mask out tools calling tokens?
        responses = [sequence[len(prompts[i // n]) :] for i, sequence in enumerate(sequences)]

        prompts = self.tokenizer(prompts, return_tensors="pt", padding="longest", padding_side="left")
        responses = self.tokenizer(responses, return_tensors="pt", padding="longest", padding_side="right")
        if n > 1:
            prompts["input_ids"] = prompts["input_ids"].repeat_interleave(n, dim=0)
            prompts["attention_mask"] = prompts["attention_mask"].repeat_interleave(n, dim=0)

        input_ids = torch.cat([prompts["input_ids"], responses["input_ids"]], dim=1)
        attention_mask = torch.cat([prompts["attention_mask"], responses["attention_mask"]], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        loss_mask = torch.cat([
            self._build_loss_mask(input_ids[i], batch_conversations[i])
            for i in range(len(batch_conversations))
        ])

        batch = TensorDict(
            {
                "prompts": prompts["input_ids"],
                "responses": responses["input_ids"],
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            },
            batch_size=len(input_ids),
        )

        non_tensor_batch = {
            "raw_conversations": np.array(batch_conversations, dtype=object)
        }

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
