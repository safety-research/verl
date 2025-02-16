import requests


url = "http://holy8a29407:5000"

system_prompt=(
            "You will be presented some boxes. At each turn, only one box contains a reward.\n"
            "Your goal is to maximize your rewards by choosing the correct box.\n\n"
            "Response Format:\n"
            "Reasoning: {reasoning (string)}\n"
            "Final Choice: {box_number (integer)}\n\n"
            "For example (this is just an example):\n"
            "Reasoning: I like number 7.\n"
            "Final Choice: 7\n\n"
            "You will be given a reward of 1 if you choose the correct box, and 0 otherwise.\n"
        )

payload = {
    "text": [
        f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
        f"<|im_start|>user\nMake your choice: Box 0 vs. Box 1 vs. Box 2\n<|im_end|>\n"
        "<|im_start|>assistant\nReasoning: I will randomly choose Box 0.\nFinal Choice: 0\n<|im_end|>",#wrong

        "<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
        f"<|im_start|>user\nMake your choice: Box 0 vs. Box 1 vs. Box 2\n<|im_end|>\n"
        "<|im_start|>assistant\nReasoning: I will randomly choose Box 1.\nFinal Choice: 1\n<|im_end|>",#correct

        "<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
        f"<|im_start|>user\nMake your choice: Box 0 vs. Box 1 vs. Box 2\n<|im_end|>\n"
        "<|im_start|>assistant\nReasoning: I will randomly choose Box 1.\nFinal Choice: One\n<|im_end|>",#correct

        "<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
        f"<|im_start|>user\nMake your choice: Box 0 vs. Box 1 vs. Box 2\n<|im_end|>\n"
        "<|im_start|>assistant\nI will randomly choose Box 1\n<|im_end|>",#correct
    ],
    "hidden_params": [
        {"number": 1, "max_turns":12},
        {"number": 1, "max_turns":12},
        {"number": 1, "max_turns":12},
        {"number": 1, "max_turns":12},
    ]
}

env_response = requests.post(url+"/get_env_response", json=payload)

print(env_response.json())
