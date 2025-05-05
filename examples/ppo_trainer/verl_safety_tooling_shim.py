from typing import Dict, Any, List

from safety_tooling.data_models.messages import Prompt, ChatMessage
from safety_tooling.data_models.inference import LLMResponse

from openai.types.chat.chat_completion import ChatCompletion

from verl.workers.rollout.async_server import ChatCompletionScheduler
from verl.protocol import DataProto

from omegaconf import DictConfig
from tensordict import TensorDict

import os
import sys
import argparse
import torch
import asyncio

sys.path.append(os.environ["COT_DECOMP_ROOT"])

from inference.safety_tooling_inference_api import SafetyToolingMultiTurnInferenceAPI
from data.prompt.multi_turn_factored_decomposition import MultiTurnFactoredDecomposition

sys.path.append(os.path.dirname(__file__))

from naive_chat_scheduler import NaiveChatCompletionScheduler


class VerlInferenceAPIShim:
    def __init__(self, chat_scheduler: ChatCompletionScheduler):
        self.chat_scheduler = chat_scheduler

    async def __call__(self, model_id: str, prompt: Prompt, print_prompt_and_response: bool = False, **kwargs):
        assert isinstance(prompt, Prompt)

        oai_format_prompt = prompt.openai_format()

        conversation = None

        # Callback is guaranteed to fire before the submit_chat_completions call returns
        async def callback(completions: ChatCompletion, info: Dict[str, Any], exception: Exception):
            nonlocal conversation
            conversation = completions.choices[0].message.content
            

        await self.chat_scheduler.submit_chat_completions(
            callback=callback,
            model=model_id,
            messages=oai_format_prompt,
            **kwargs,
        )

        if print_prompt_and_response:
            print(f"Prompt: {oai_format_prompt}")
            print(f"Response: {conversation}")

        return [LLMResponse(
            model_id=model_id,
            completion=conversation,
        )]
    


class VerlFactoredTaskDecompositionChatScheduler(NaiveChatCompletionScheduler):
    def __init__(
        self,
        config: DictConfig,
        model_path: str,
        server_addresses: List[str],
        max_cache_size: int = 10000,
    ):
        super().__init__(config, model_path, server_addresses, max_cache_size)

        self.inference_api = SafetyToolingMultiTurnInferenceAPI(
            model_id=self.model_name,
            multi_turn_responder=MultiTurnFactoredDecomposition(
                prompt_generator=None,
                args=argparse.Namespace(
                    multi_turn_responder_unanswerable_subtask_response_override="regenerate",
                    multi_turn_responder_unanswerable_final_subtask_response_override="regenerate_if_unanswerable_or_directly_answered",
                    multi_turn_responder_subtask_prompt_suffix="all_prior_subtask_descriptions_and_results",
                )
            ),
            cache_dir=".cache",
            print_debug_outputs=True, # TODO(sguo35)
            use_assistant_message=True,
        )
        self.inference_api.API = VerlInferenceAPIShim(self)

    async def generate_sequences(self, batch: DataProto, **sampling_params) -> DataProto:
        kwargs = dict(
            n=self.config.n,
            max_completion_tokens=self.config.response_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        do_sample = batch.meta_info.get("do_sample", True)
        is_validate = batch.meta_info.get("validate", False)
        if not do_sample or is_validate:
            kwargs["n"] = 1
            kwargs["temperature"] = 0

        kwargs.update(sampling_params)
        print(f"[VerlFactoredTaskDecompositionChatScheduler] generate_sequences sampling params: {kwargs}")

        tasks = []
        batch_conversations = [[] for _ in range(len(batch))]
        for batch_index, conversation in enumerate(batch.non_tensor_batch["raw_prompt"]):
            # raw_prompt: [{"role": "user", "content": ""}, ["role": "assistant", "content"], ...]
            tasks.append(
                self.inference_api(
                    prompt=None,
                    l_msg_cache=batch_conversations[batch_index],
                    use_reranker=False,
                    l_messages=self._convert_conversation_to_safety_tooling(conversation)
                )
            )

        await asyncio.gather(*tasks)

        tasks = [task.openai_format() for task in tasks]

        print("[VerlFactoredTaskDecompositionChatScheduler] generate_sequences done")

        return self._postprocess(batch, batch_conversations, kwargs["n"])
    
    def _convert_conversation_to_safety_tooling(self, conversation: List[Dict[str, str]]) -> List[ChatMessage]:
        return [ChatMessage(role=message["role"], content=message["content"]) for message in conversation]

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

        # flatten batch_conversations if n > 1
        assert len(batch_conversations) == len(prompts)
        batch_conversations = [conversation for conversations in batch_conversations for conversation in conversations]
        assert len(batch_conversations) == len(prompts) * n

        # sequences: [prompt + response]
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

        batch = TensorDict(
            {
                "prompts": prompts["input_ids"],
                "responses": responses["input_ids"],
                "input_ids": input_ids,
                # TODO(sguo35): Mask out user turns
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=len(input_ids),
        )

        return DataProto(batch=batch)
