#SBATCH --job-name=NS
#SBATCH --output=ns-%j.out
#SBATCH --error=ns-%j.err
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
# Load the required modules
module --force purge
module load releases/2022a
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
module load Python
module load matplotlib
module load SciPy-bundle
module load scikit-learn

# Prepare results folder
mkdir -p "$LOCALSCRATCH/$SLURM_JOB_ID"
mkdir -p "$HOME/$SLURM_JOB_ID"
mkdir -p "$LOCALSCRATCH/$SLURM_JOB_ID/result"

# Move code and data to local scratch
cp -r * "$LOCALSCRATCH/$SLURM_JOB_ID/"

python $LOCALSCRATCH/$SLURM_JOB_ID/experimentation_Graph2Seq.py --enc_hid_dim 1024 --enc_num_heads 8  --num_nodes 50  --loss negative_sampling --directory $LOCALSCRATCH/$SLURM_JOB_ID/result --data_train  $LOCALSCRATCH/$SLURM_JOB_ID/data/tsp50_train.txt --data_test $LOCALSCRATCH/$SLURM_JOB_ID/data/tsp50_test.txt --n_gpu 1
