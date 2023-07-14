#!/bin/bash
#SBATCH --job-name=vanill
#SBATCH --output=vanilla-%j.out
#SBATCH --error=vanilla-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nathan.nepper@student.ucouvain.be

# Define a function to run when the script is terminated
function cleanup {
    cp -r "$LOCALSCRATCH/$SLURM_JOB_ID/" "$HOME/$SLURM_JOB_ID" &&\
    rm -rf "$LOCALSCRATCH/$SLURM_JOB_ID"
}
trap cleanup SIGTERM


# Load the required modules
module load releases/2022a
module load Python
module load PyTorch
module load SciPy-bundle
module load scikit-learn 
module load matplotlib

mkdir -p "$LOCALSCRATCH/$SLURM_JOB_ID"
mkdir -p "$HOME/$SLURM_JOB_ID"

python /home/ucl/ingi/nnepper/TSP-DeepRL-Thesis/experimentation_Graph2Seq.py --loss vanilla --directory $LOCALSCRATCH/$SLURM_JOB_ID --data_train  $HOME/TSP-DeepRL-Thesis/data/tsp20_train.txt --data_test $HOME/TSP-DeepRL-Thesis/data/tsp20_val.txt --n_gpu 1
