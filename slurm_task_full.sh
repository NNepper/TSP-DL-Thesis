#!/bin/bash
#SBATCH --job-name=FULL
#SBATCH --output=full-%j.out
#SBATCH --error=full-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nathan.nepper@student.ucouvain.be

# Load the required modules
module load Python/3.8.6-GCCcore-10.2.0
module load root_numpy/4.8.0-foss-2020b-Python-3.8.6
module load PyTorch/1.10.0-fosscuda-2020b
module load matplotlib/3.3.3-foss-2020b
module load SciPy-bundle/2020.11-fosscuda-2020b
module load scikit-learn/0.23.2-fosscuda-2020b

mkdir results_full_$SLURM_JOB_ID
python /home/ucl/ingi/nnepper/TSP-DeepRL-Thesis/experimentation_Graph2Seq.py --loss full --directory results_full_$SLURM_JOB_ID --data_train /home/ucl/ingi/nnepper/TSP-DeepRL-Thesis/data/tsp20_train.txt --data_test /home/ucl/ingi/nnepper/TSP-DeepRL-Thesis/data/tsp20_val.txt --n_gpu 2 > results_full_$SLURM_JOB_ID/out.txt