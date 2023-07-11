#!/bin/bash
#SBATCH --job-name=graph2seq
#SBATCH --output=graph2seq-%j.out
#SBATCH --error=graph2seq-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

# Load the required modules
module load Python/3.8.6-GCCcore-10.2.0
module load root_numpy/4.8.0-foss-2020b-Python-3.8.6
module load PyTorch/1.10.0-fosscuda-2020b

python /home/ucl/ingi/nnepper/TSP-DeepRL-Thesis/experimentation_Graph2Seq_NS.py > /home/ucl/ingi/nnepper/TSP-DeepRL-Thesis/experimentation_Graph2Seq_NS.out