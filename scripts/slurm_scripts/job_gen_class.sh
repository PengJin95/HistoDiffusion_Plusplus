#!/bin/bash
#SBATCH --account=XXX
#SBATCH --job-name=histo_crc_gen
#SBATCH --output=/your_home/jobs/output/histo_crc_gen.log
#SBATCH --error=/your_home/jobs/error/histo_crc_gen.err
#SBATCH --partition=nextgen
#SBATCH --distribution=cyclic
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=30G
#SBATCH --time=03:00:00
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

source activate DiT
export OMP_NUM_THREADS=8
export NCCL_P2P_DISABLE=1

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
head_node_port=29501

srun torchrun --nnodes $SLURM_NNODES --nproc_per_node 2 \
 --rdzv_id $SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$head_node_ip:$head_node_port \
 generate_class_control.py \
 --base-ckpt /your_home/exps/run1/checkpoints/0180000.pt \
 --control-ckpt /your_home/exp_class/crc/run1/checkpoints/0005000.pt \
 --sample-dir /your_output/fake_5k_a --mode sde --batch-size 8 --global-seed 0 \
 --dataset PCam --dist 2>&1
