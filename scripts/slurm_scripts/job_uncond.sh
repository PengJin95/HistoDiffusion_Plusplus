#!/bin/bash
#SBATCH --account=XXX
#SBATCH --job-name=histo_uncond
#SBATCH --output=/your_home/jobs/output/histo_uncond.log
#SBATCH --error=/your_home/jobs/error/histo_uncond.err
#SBATCH --partition=nextgen
#SBATCH --distribution=cyclic
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=5
#SBATCH --mem-per-gpu=30G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your@email

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $SLURM_JOB_NAME"
echo "Job ID : $SLURM_JOB_ID" 
echo "=========================================================="

# debugging flags (optional)
# export NCCL_DEBUG=INFO

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29502
echo "Head node: $MASTER_ADDR"

# hide duplicated errors using this hack - will be properly fixed in pt-1.12
export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json
export NCCL_IB_DISABLE=0 
export NCCL_P2P_DISABLE=1


GPUS_PER_NODE=1
echo "Num nodes: $SLURM_NNODES"
echo "Num gpus per node: $GPUS_PER_NODE"


cd /your_home

export OMP_NUM_THREADS=5

source activate histo

srun torchrun --nnodes=8 --nproc_per_node=1 --rdzv_backend=c10d \
  --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT --rdzv_id ${SLURM_JOB_ID} \
  generate.py \
  --num-fid-samples 50000 \
  --ckpt /path_to_your_checkpoints/checkpoints/0180000.pt \
  --path-type=linear \
  --encoder-depth=0 \
  --per-proc-batch-size=12 \
  --mode=sde \
  --num-steps=50 \
  --sample-dir /path_to_your_output/repa_baseline_samples/your_experiment_name