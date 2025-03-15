import sys, os
import json
import numpy as np
from openai import OpenAI
import datasets
from tqdm import tqdm

client = OpenAI(
#   api_key='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' # Sammy's API key
  api_key='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'

)


def load_math_data():
    data_source = '/n/home05/sqin/wall/verl/data/math/MATH_train.json'
    with open(data_source) as f:
        data = json.load(f)

    data_filter = "/n/netscratch/dam_lab/Everyone/wall/cfpark00/grpo_zero_prob_unique_ids.txt"
    with open(data_filter) as f:
        filter_ids = f.read().splitlines()
    filter_ids = list(filter_ids)
    # create a data subset only containing datum with filtered ids
    filter_data = []
    for d in data:
        if d["unique_id"] in filter_ids:
            d['messages'] = d['messages'][0]['content']
            filter_data.append(d)
    print("Filtered data size: ", len(filter_data))
    return filter_data


def ask_openai(formatted_question):
    completion = client.chat.completions.create(
        model="o1-mini",
        messages=[
            {
                "role": "user",
                "content": formatted_question
            }
        ]
    )
    return completion



if __name__ == "__main__":
    # take two argument: start and end
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    # load dataset
    train_data = load_math_data()[start:end] # 1082 entries
    # ask OpenAI to generate detailed reponse
    for i in tqdm(range(len(train_data))):
        openai_responses = ask_openai(train_data[i]["messages"])
        # add OpenAI responses to the dataset
        train_data[i]["detailed_solution"] = openai_responses.choices[0].message.content
    # save the dataset
    with open(f"/n/home05/sqin/wall/verl/data/math/MATH_sft_train_{start}_{end}.json", "w") as f:
        json.dump(train_data, f, indent=4, separators=(',', ': '))
    