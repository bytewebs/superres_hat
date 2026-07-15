#!/bin/bash
#SBATCH --job-name=senhat-phase2
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
SCRATCH_VGG="${SCRATCH_BASE}/models/vgg19-dcbb9e9d.pth"  # pre-staged VGG19 (login node)

# Staged copies on node-local NVMe (fast random reads for DataLoader)
STAGE_TRAIN="${SLURM_TMPDIR}/sentinel_dataset"
STAGE_VALID="${SLURM_TMPDIR}/sentinel_test"

# ---------------------------------------------------------------------------
# Environment: staged data on $SLURM_TMPDIR, persistent outputs on /scratch
# ---------------------------------------------------------------------------
export SENHAT_TRAIN_DATA="${STAGE_TRAIN}"
export SENHAT_VALID_DATA="${STAGE_VALID}"
# Trailing slash is required: train.py concatenates paths as ${dir}gens_7_*.pt
export SENHAT_CHECKPOINT_DIR="${SCRATCH_CKPT}/"
export SENHAT_SAVE_DIR="${SCRATCH_SAVE}/"
export SENHAT_VGG19_PATH="${SCRATCH_VGG}"

# ---------------------------------------------------------------------------
# Modules + virtualenv (STRICT COMBINATION for Narval OpenCV compatibility)
# ---------------------------------------------------------------------------
# Clear out any clashing modules just in case
module purge
# Force the strict combination of modules (order matters!)
module load StdEnv/2023 gcc opencv python/3.11 cuda/12.2

source "${SCRATCH_BASE}/venv/bin/activate"

# Repo root (directory containing config.py, networks/, datasets/, etc.)
REPO_ROOT="${SCRATCH_BASE}/superres_hat"
cd "${REPO_ROOT}"
mkdir -p logs "${SCRATCH_CKPT}" "${SCRATCH_SAVE}"

# VGG19 must be pre-downloaded on a LOGIN node (compute nodes have no internet).
if [[ ! -f "${SCRATCH_VGG}" ]]; then
  echo "ERROR: VGG19 weights not found at ${SCRATCH_VGG}"
  echo ""
  echo "Run ONCE on a Narval LOGIN node (not a compute node):"
  echo "  module purge"
  echo "  module load StdEnv/2023 gcc opencv python/3.11 cuda/12.2"
  echo "  source ${SCRATCH_BASE}/venv/bin/activate"
  echo "  mkdir -p ${SCRATCH_BASE}/models"
  echo "  python -c \"from torchvision.models import vgg19, VGG19_Weights; vgg19(weights=VGG19_Weights.IMAGENET1K_V1)\""
  echo "  cp ~/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth ${SCRATCH_VGG}"
  exit 1
fi
echo "VGG19 weights: ${SENHAT_VGG19_PATH}"

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
# Train. Phase 2 (--enable_gan) needs more VRAM (G + D + MS-SSIM); default batch 4.
# Override with e.g. sbatch train.sh --resume --checkpoint ... --enable_gan --batch 2
cd "${REPO_ROOT}/SenGLEAN/Training"

# If the user already passed --batch in "$@", do not force another value.
BATCH_ARGS=(--batch 4)
for a in "$@"; do
  if [[ "$a" == "--batch" || "$a" == --batch=* ]]; then
    BATCH_ARGS=()
    break
  fi
done

python train.py \
    "${BATCH_ARGS[@]}" \
    "$@"

echo "Job finished at $(date)"
