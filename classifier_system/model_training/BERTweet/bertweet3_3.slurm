#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --account=ie-idi
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=80000
#SBATCH --job-name="bertweet3_3"
#SBATCH --output=bertweet3_3.out
#SBATCH --mail-user=eirida@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "We are running from this directory: $SLURM_SUBMIT_DIR"
echo "The name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

module purge
module load Python/3.8.2-GCCcore-9.3.0
source venv/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cluster/apps/eb/software/CUDA/10.1.243-GCC-8.3.0/targets/x86_64-linux/lib/:/cluster/apps/eb/software/cuDNN/8.0.4.30-CUDA-11.1.1/lib64
python model_training/BERTweet/bertweet3_3.py --idun=1
uname -a
