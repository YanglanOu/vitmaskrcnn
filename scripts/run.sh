#!/bin/bash
#SBATCH --job-name=vitg-14-maskrcnn-fb-518      # Name of the job
#SBATCH --output=/dev/null                            # We'll redirect manually
#SBATCH --error=/dev/null                             # We'll redirect manually
#SBATCH --time=0-04:00:00                      # Wall time (D-HH:MM:SS)
#SBATCH --nodes=1                              # Number of nodes
#SBATCH --gres=gpu:1                           # Number of GPUs
#SBATCH --ntasks=1                             # Total number of tasks
#SBATCH --cpus-per-task=4                      # CPUs per task
#SBATCH --mem=64G                              # Memory per node
#SBATCH --partition=gen-h100
#SBATCH --mail-type=END,FAIL                   # Email notifications
#SBATCH --mail-user=ou.yanglan@mayo.edu      # Your email address

# ---------------- Load Conda ----------------
source /home/m341664/miniconda3/etc/profile.d/conda.sh
conda activate vitfasterrcnn   # Replace with your env if different

# ---------------- Create Log Directory ----------------
LOG_DATE=$(date +%Y%m%d)
mkdir -p logs/${LOG_DATE}
echo "Log directory: logs/${LOG_DATE}"

# ---------------- Redirect Output to Date-Based Logs ----------------
exec 1>logs/${LOG_DATE}/vitg-14-fasterrcnn-fb-518_37MRSvitG_${SLURM_JOB_ID}.out
exec 2>logs/${LOG_DATE}/vitg-14-fasterrcnn-fb-518_37MRSvitG_${SLURM_JOB_ID}.err

# ---------------- NCCL Stability Settings ----------------
export NCCL_TIMEOUT=3600
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# ---------------- Job Info ----------------
echo "Running on host: $(hostname)"
echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"

# ---------------- Main Command ----------------
echo "Starting training..."
python train_maskrcnn_improved.py \
--dinov2_checkpoint /dgx1data/skunkworks/pathology/bloodbytes/data2/m328672/dinov2_h200m_results/vitg14_RS_patch37M_5/eval/training_212499/teacher_checkpoint.pth \
--data_root /dgx1data/skunkworks/pathology/bloodbytes/m341664/data/selected_148/selected_72 \
--freeze_backbone \
--output_dir  ./outputs_debug\

