import os
import datasets
import argparse
import copy

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

def map_func(datum, idx):
    datum_=copy.deepcopy(datum)
    datum_["data_source"]=data_source
    messages=[
        {
            "role":"system",
            "content":system_prompt
        },
        {
            "role":"system",
            "content":datum["prompt"]
        }
    ]
    datum_["prompt"]=messages
    datum_["reward_model"]={
        "style":"rule",
        "ground_truth":datum["answer"]
    }
    datum_["extra_info"]={
        'split': split,
        'index': idx
    }
    return datum_

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/datasets/boxes')
    args = parser.parse_args()

    data_source = 'cfpark00/boxes'
    dataset = datasets.load_dataset( data_source)

    splits=["always1_train","always1_test","smallest_train","smallest_test","second_train","second_test"]

    for split in splits:
        ds=dataset[split]
        ds=ds.map(function=map_func, with_indices=True)
        local_dir=args.local_dir
        ds.to_parquet(os.path.join(local_dir,f"{split}.parquet"))
        ds.to_json(os.path.join(local_dir,f"{split}.json"))