from verl import DataProto
import requests
import torch
import numpy as np


class URLEnvironment():

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.url = self.config.actor_rollout_ref.rollout.environment.url

    def get_response_batched(self, messages_batched):
        payload = {"messages_batched": messages_batched}
        url = self.url + "/get_env_response_batched"
        env_response_batched = requests.post(url, json=payload).json()
        return env_response_batched

    def get_reward_batched(self, data: DataProto):  #batched
        messages_batched = []
        reward_locs = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            messages = data_item.non_tensor_batch['messages']
            if isinstance(messages, np.ndarray):
                messages = messages.tolist()
            messages_batched.append(messages)

            attention_mask = data_item.batch['attention_mask']
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = attention_mask[prompt_length:].sum()
            reward_locs.append(valid_response_length - 1)

        url = self.url + "/get_reward_batched"
        payload = {"messages_batched": messages_batched}
        reward_batched = requests.post(url, json=payload).json()

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        for i in range(len(data)):
            reward_tensor[i, reward_locs[i]] = reward_batched[i]
        return reward_tensor
