#!/usr/bin/env bash
#SBATCH --job-name=euler-train-stream-test
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/euler-train-test}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
EULER_TRAIN_SOURCE="${EULER_TRAIN_SOURCE:-}"
EULER_TRAIN_TEST_OUTPUT_DIR="${EULER_TRAIN_TEST_OUTPUT_DIR:-${SCRATCH:-$PROJECT_DIR}/euler-train-stream-test}"
EULER_TRAIN_TEST_EPOCHS="${EULER_TRAIN_TEST_EPOCHS:-2}"
EULER_TRAIN_TEST_STEPS="${EULER_TRAIN_TEST_STEPS:-8}"

cd "$PROJECT_DIR"
mkdir -p logs

if command -v module >/dev/null 2>&1; then
  module load eth_proxy
fi

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

if [[ -n "$EULER_TRAIN_SOURCE" ]]; then
  python -m pip install -e "$EULER_TRAIN_SOURCE"
fi
python -m pip install -e "$PROJECT_DIR"

export EULER_TRAIN_TEST_OUTPUT_DIR

euler-train-stream-test \
  --check-handshake \
  --epochs "$EULER_TRAIN_TEST_EPOCHS" \
  --steps "$EULER_TRAIN_TEST_STEPS" \
  --local-stream-jsonl "$EULER_TRAIN_TEST_OUTPUT_DIR/stream-events-${SLURM_JOB_ID:-local}.jsonl"
