#!/bin/bash
#SBATCH --job-name=vitg-14-maskrcnn-fb-518      # Name of the job
#SBATCH --output=/dev/null                            # We'll redirect manually
#SBATCH --error=/dev/null                             # We'll redirect manually
#SBATCH --time=0-12:00:00                      # Wall time (D-HH:MM:SS) - increased to 12 hours
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

# ---------------- Clear Python Cache ----------------
# Ensure we're using the latest code, not cached bytecode
find /home/m341664/yanglanou/projects/vitmaskrcnn -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find /home/m341664/yanglanou/projects/vitmaskrcnn -name "*.pyc" -delete 2>/dev/null || true

# ---------------- Change to project directory (before creating logs) ----------------
PROJECT_ROOT="/home/m341664/yanglanou/projects/vitmaskrcnn"
cd "$PROJECT_ROOT" || { echo "Error: Could not cd to $PROJECT_ROOT"; exit 1; }

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
echo "Working directory: $(pwd)"
echo "Project root: $PROJECT_ROOT"

# ---------------- Main Command ----------------
echo "Starting training..."
echo "Current working directory: $(pwd)"
echo "Checking if train_maskrcnn_improved.py exists..."
if [ ! -f "$PROJECT_ROOT/train_maskrcnn_improved.py" ]; then
    echo "ERROR: train_maskrcnn_improved.py not found at $PROJECT_ROOT/train_maskrcnn_improved.py"
    exit 1
fi

OUTPUT_DIR="./outputs/panoptils_debug"
echo "Output directory argument: $OUTPUT_DIR"

python "$PROJECT_ROOT/train_maskrcnn_improved.py" \
  --dinov2_checkpoint /dgx1data/skunkworks/pathology/bloodbytes/data2/m328672/dinov2_h200m_results/vitg14_RS_patch37M_5/eval/training_212499/teacher_checkpoint.pth \
  --data_root /dgx1data/skunkworks/pathology/bloodbytes/m341664/data/PanopTILs/bootstrapped_nuclei_labels/fold_1 \
  --freeze_backbone \
  --collapse_categories \
  --output_dir "$OUTPUT_DIR"

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "Training completed successfully. Starting testing..."
    
    # Find the most recent output directory
    OUTPUT_DIR=$(ls -td outputs/panoptils_debug/run_* | head -n1)
    CHECKPOINT_PATH="${OUTPUT_DIR}/best_model.pth"
    
    echo "Using checkpoint: ${CHECKPOINT_PATH}"
    
    # Run testing
    python "$PROJECT_ROOT/test_maskrcnn_improved.py" \
      --checkpoint "${CHECKPOINT_PATH}" \
      --data_root '/dgx1data/skunkworks/pathology/bloodbytes/m341664/data/PanopTILs/bootstrapped_nuclei_labels/fold_1' \
      --batch_size 1
    
    echo "Testing completed."
else
    echo "Training failed. Skipping testing."
    exit 1
fi

echo "Finished at: $(date)"
