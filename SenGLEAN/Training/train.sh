#!/bin/bash
#SBATCH --job-name=senhat-fidelity
#SBATCH --account=def-rkmishra-ab
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/senhat-%j.out
#SBATCH --error=logs/senhat-%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths (edit SCRATCH_BASE if your layout differs)
# ---------------------------------------------------------------------------
SCRATCH_BASE="/scratch/aayushk/superresolution"
SCRATCH_TRAIN="${SCRATCH_BASE}/sentinel_dataset"   # permanent training data
SCRATCH_VALID="${SCRATCH_BASE}/sentinel_test"      # permanent validation data
SCRATCH_CKPT="${SCRATCH_BASE}/weights"             # persistent checkpoints
SCRATCH_SAVE="${SCRATCH_BASE}/samples"             # persistent plots / samples

# Staged copies on node-local NVMe (fast random reads for DataLoader)
STAGE_TRAIN="${SLURM_TMPDIR}/sentinel_dataset"
STAGE_VALID="${SLURM_TMPDIR}/sentinel_test"

# ---------------------------------------------------------------------------
# Environment: point Python at staged data, persistent outputs on /scratch
# ---------------------------------------------------------------------------
export SENHAT_TRAIN_DATA="${STAGE_TRAIN}"
export SENHAT_VALID_DATA="${STAGE_VALID}"
export SENHAT_CHECKPOINT_DIR="${SCRATCH_CKPT}"
export SENHAT_SAVE_DIR="${SCRATCH_SAVE}"

# ---------------------------------------------------------------------------
# Modules + virtualenv (adjust versions to your Narval setup)
# ---------------------------------------------------------------------------
module load python/3.10
module load cuda/12.2
module load gcc/12.3
source "${SCRATCH_BASE}/venv/bin/activate"

# Repo root (directory containing config.py, networks/, datasets/, etc.)
REPO_ROOT="${SCRATCH_BASE}/sr"
cd "${REPO_ROOT}"
mkdir -p logs "${SCRATCH_CKPT}" "${SCRATCH_SAVE}"

echo "============================================================"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          ${SLURMD_NODENAME:-unknown}"
echo "SLURM_TMPDIR:  ${SLURM_TMPDIR}"
echo "Staging FROM:  ${SCRATCH_TRAIN}"
echo "           TO:  ${STAGE_TRAIN}"
echo "Staging FROM:  ${SCRATCH_VALID}"
echo "           TO:  ${STAGE_VALID}"
echo "Checkpoints:   ${SENHAT_CHECKPOINT_DIR}"
echo "Samples:       ${SENHAT_SAVE_DIR}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Stage dataset to node-local SSD (avoids Lustre metadata storm during training)
# ---------------------------------------------------------------------------
stage_start=$(date +%s)
echo "[stage] copying training data..."
rsync -a --info=stats2 "${SCRATCH_TRAIN}/" "${STAGE_TRAIN}/"
echo "[stage] copying validation data..."
rsync -a --info=stats2 "${SCRATCH_VALID}/" "${STAGE_VALID}/"
stage_end=$(date +%s)
echo "[stage] done in $(( stage_end - stage_start ))s"
echo "[stage] train files: $(find "${STAGE_TRAIN}" -name '*.pt' | wc -l)"
echo "[stage] valid files: $(find "${STAGE_VALID}" -name '*.pt' | wc -l)"

# ---------------------------------------------------------------------------
# Optional: quick model sanity check (single forward/backward, no training loop)
# Uncomment for first submission; comment out for production runs.
# ---------------------------------------------------------------------------
# python sanity_check_senhat.py --variant small --rrdbs 23 --batch 2

# ---------------------------------------------------------------------------
# Train (fidelity-only by default; pass --enable_gan later for GAN stage)
# ---------------------------------------------------------------------------
cd "${REPO_ROOT}/SenGLEAN/Training"
python train.py \
    --batch 8 \
    "$@"

echo "Job finished at $(date)"
