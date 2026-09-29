#!/bin/bash
#SBATCH --account=XXX
#SBATCH --job-name=histo
#SBATCH --output=/your_home/jobs/output/histo.log
#SBATCH --error=/your_home/jobs/error/histo.err
#SBATCH --partition=nextgen
#SBATCH --distribution=cyclic
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=3
#SBATCH --mem-per-gpu=30G
#SBATCH --time=5-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your@email

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $SLURM_JOB_NAME"
echo "Job ID : $SLURM_JOB_ID" 
echo "=========================================================="

# debugging flags (optional)
# export NCCL_DEBUG=INFO

cd /your_home

source activate histo

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29505
echo "Head node: $MASTER_ADDR"

# hide duplicated errors using this hack - will be properly fixed in pt-1.12
export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json
export NCCL_IB_DISABLE=0 
export NCCL_P2P_DISABLE=1


GPUS_PER_NODE=1
echo "Num nodes: $SLURM_NNODES"
echo "Num gpus per node: $GPUS_PER_NODE"

LAUNCHER="accelerate launch \
    --multi_gpu \
    --num_processes $(($GPUS_PER_NODE*$SLURM_NNODES)) \
    --num_machines $SLURM_NNODES \
    --rdzv_backend=c10d \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --mixed_precision fp16 \
    --dynamo_backend no \
    "

CMD="train.py --output-dir exp_repa/run1 \
  --allow-tf32 --exp-name=run1 --batch-size 64 \
  --max-train-steps 180000 --proj-coeff 0.5 \
  --gradient-accumulation-steps 4"

srun -K1 bash -c "$LAUNCHER $CMD" 2>&1
