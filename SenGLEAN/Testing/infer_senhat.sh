#!/bin/bash
#SBATCH --job-name=senhat-infer
#SBATCH --account=def-rkmishra-ab
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/senhat-infer-%j.out
#SBATCH --error=logs/senhat-infer-%j.err

set -euo pipefail

SCRATCH_BASE="/scratch/aayushk/superresolution"
REPO_ROOT="${SCRATCH_BASE}/superres_hat"

# ---- edit these if needed ----
CKPT="${SCRATCH_BASE}/weights/gens_7_659.pt"
S2_TIF="${SCRATCH_BASE}/sentinel2_hatms_test.tif"
OUT_TIF="${SCRATCH_BASE}/samples/senhat_659_enhanced.tif"
# ------------------------------

module purge

module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load python/3.11
module load opencv/4.13.0

source "${SCRATCH_BASE}/venv/bin/activate"

python -c "import cv2; print('OpenCV:', cv2.__version__)"

mkdir -p "${REPO_ROOT}/SenGLEAN/Training/logs" "${SCRATCH_BASE}/samples"
cd "${REPO_ROOT}"

echo "============================================================"
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Node:     ${SLURMD_NODENAME:-unknown}"
echo "CKPT:     ${CKPT}"
echo "INPUT:    ${S2_TIF}"
echo "OUTPUT:   ${OUT_TIF}"
echo "============================================================"

if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT}"
  exit 1
fi
if [[ ! -f "${S2_TIF}" ]]; then
  echo "ERROR: input TIFF not found: ${S2_TIF}"
  exit 1
fi

python - <<PY
from osgeo import gdal
ds = gdal.Open("${S2_TIF}")
print(f"Input bands={ds.RasterCount} size={ds.RasterXSize}x{ds.RasterYSize}")
PY

python SenGLEAN/Testing/infer_senhat.py \
  --checkpoint_g "${CKPT}" \
  --data_path "${S2_TIF}" \
  --out_path "${OUT_TIF}"

ls -lh "${OUT_TIF}"
echo "Done at $(date)"
