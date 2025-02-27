import argparse
import os
import glob
import torch
import numpy as np
import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', type=str)
    parser.add_argument('--dest_dir', type=str, default=None)
    parser.add_argument('--regen', action='store_true')
    parser.add_argument('--last_only', action='store_true')
    args = parser.parse_args()
    base_dir = args.base_dir
    assert os.path.exists(base_dir)
    dest_dir = args.dest_dir
    regen = args.regen
    last_only = args.last_only
    ##
    if "/global_step" in base_dir:
        fols = [base_dir]
    else:
        fols = glob.glob(os.path.join(base_dir, 'global_step_*'))
        def get_global_step(fol):
            return int(fol.split('global_step_')[1])
        fols = sorted(fols, key=get_global_step)
    for ifol,fol in enumerate(tqdm.tqdm(fols)):
        if len(glob.glob(os.path.join(fol, 'actor/*.pt')))==0:
            continue
        if not regen and os.path.exists(os.path.join(fol, 'actor/state_dict.pt')):
            continue
        if last_only and ifol<len(fols)-1:
            continue
        weight_files = glob.glob(os.path.join(fol, 'actor/model_*.pt'))
        #print(weight_files)
        #/n/home12/cfpark00/ML/llm-meta-rl/data/verl/verl_grpo_boxes/qwen-2.5_0.5b-instruct-boxes-smallest-v1/global_step_10/actor/model_world_size_2_rank_1.pt
        #get world_size and rank for each file
        world_sizes = []
        ranks = []
        for weight_file in weight_files:
            world_size = int(weight_file.split('world_size_')[1].split('_rank')[0])
            rank = int(weight_file.split('rank_')[1].split('.pt')[0])
            world_sizes.append(world_size)
            ranks.append(rank)
        assert len(set(world_sizes))==1
        world_size = world_sizes[0]
        #print("World size:",world_size)
        #print("n_files:",len(weight_files))
        assert set(ranks)==set(range(world_size))
        #sort by rank
        argsort = np.argsort(ranks)
        weight_files = [weight_files[i] for i in argsort]
        state_dict = torch.load(weight_files[0],weights_only=False)
        for k,v in state_dict.items():
            state_dict[k] = v.to_local()
        for i in range(1,len(weight_files)):
            state_dict_i = torch.load(weight_files[i],weights_only=False)
            for k,v in state_dict_i.items():
                state_dict[k] = torch.cat([state_dict[k],v.to_local()],dim=0)
        #save the merged state_dict
        if dest_dir is None:
            save_path = os.path.join(fol, 'actor/state_dict.pt')
        else:
            save_path = os.path.join(dest_dir, f'state_dict_{ifol}.pt')
        torch.save(state_dict, save_path)
    