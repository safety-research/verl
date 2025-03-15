#!/bin/sh

#SBATCH -p kempner_h100 # Partition to submit to
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks across all nodes
#SBATCH --cpus-per-task=24        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=250G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)
#SBATCH --account=kempner_barak_lab
#SBATCH --exclude=holygpu8a13303
#SBATCH --job-name=grpo_7B    # create a short name for your job

cd /n/home05/sqin/wall/verl
pwd


if [[ -z $SLURM_GPUS_ON_NODE ]]; then
    RAY_NUM_GPUS=0
else
    RAY_NUM_GPUS=$SLURM_GPUS_ON_NODE
fi

# choose available port on the head node
# head_port=`comm -23 <(seq 15000 20000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
head_port=16494
nodes=`scontrol show hostnames $SLURM_JOB_NODELIST`
nodes_array=( $nodes )
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "Head node: $head_node"
echo "Head node ip: $head_node_ip"
echo "Head port: $head_port"
export RAY_HEAD_ADDR="$head_node_ip:$head_port"
echo "Head address: $RAY_HEAD_ADDR"

echo "Starting Ray head on $head_node"
srun -N 1 -n 1 -w "$head_node" ray start --head --node-ip-address="$head_node_ip" --temp-dir /tmp/$USER/$SLURM_JOB_ID/ray \
    --port=$head_port --num-cpus $SLURM_CPUS_PER_TASK --num-gpus $RAY_NUM_GPUS --min-worker-port 20001 --max-worker-port 30000 --block &

# wait for head node to start
sleep 5

# start ray on the rest of the nodes
worker_num=$((SLURM_NNODES - 1))
for (( i = 1; i <= worker_num; i++ )); do
    node=${nodes_array[$i]}
    echo "Starting Ray worker on $node"
    srun -N 1 -n 1 -w "$node" ray start --address="$RAY_HEAD_ADDR" \
        --num-cpus $SLURM_CPUS_PER_TASK --num-gpus $RAY_NUM_GPUS --min-worker-port 20001 --max-worker-port 30000 --block &
    sleep 5
done

export RAY_ADDRESS="$RAY_HEAD_ADDR"


 
python3 -m verl.trainer.main_ppo sunny_scripts/grpo_math_qwen_7B.yaml