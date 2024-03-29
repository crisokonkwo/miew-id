#!/bin/bash
#SBATCH --job-name=miewid-ddp    # create a short name for your job
#SBATCH --partition=npl-2024     # appropriate partition; if not specified, slurm will automatically do it for you
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=2      # set this equals to the number of gpus per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:2             # number of allocated gpus per node
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=charis3@rpi.edu    # change this to your email!

# export your rank 0 information (its address and port)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
echo "MASTER_ADDR="$MASTER_ADDR

source ~/.bashrc
conda activate test_env
python train.py
