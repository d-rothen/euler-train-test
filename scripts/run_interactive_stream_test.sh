#!/usr/bin/env bash
# Euler View interactive-shell template for euler-train streaming.
# No #SBATCH headers belong in this file; Euler View should run it inside the
# already allocated interactive shell/session.

set -euo pipefail

# --- 1. Load Modules ---
if command -v module >/dev/null 2>&1; then
  module load stack/2024-06 python/3.11.6
  module load eth_proxy
fi

export PYTHONUNBUFFERED=1

# --- 2. Activate Environment ---
{{__env_activate}}

# --- 3. Euler View Stream Attachment ---
# Euler View injects this reserved value from model_launches.id. euler-train
# treats it as an opaque stream attachment token, not as an auth secret.
export EULER_SESSION_ID="{{__euler_session_token}}"
export EULER_VIEW_STREAM_ATTACH_TOKEN="$EULER_SESSION_ID"
export EULER_VIEW_BASE_URL="{{__euler_view_base_url}}"
export EULER_VIEW_API_TOKEN="{{__euler_view_api_token}}"

if [[ "$EULER_VIEW_BASE_URL" == "{{__euler_view_base_url}}" || -z "$EULER_VIEW_BASE_URL" ]]; then
  echo "EULER_VIEW_BASE_URL was not resolved. Configure __euler_view_base_url on the launch." >&2
  exit 2
fi
if [[ "$EULER_VIEW_API_TOKEN" == "{{__euler_view_api_token}}" || -z "$EULER_VIEW_API_TOKEN" ]]; then
  echo "EULER_VIEW_API_TOKEN was not resolved. Configure __euler_view_api_token on the launch." >&2
  exit 2
fi

# --- 4. Test Package Location ---
PROJECT_DIR="${PROJECT_DIR:-$HOME/euler-train-test}"
EULER_TRAIN_TEST_OUTPUT_DIR="${EULER_TRAIN_TEST_OUTPUT_DIR:-${SCRATCH:-$PROJECT_DIR}/euler-train-stream-test}"
EULER_TRAIN_TEST_EPOCHS="${EULER_TRAIN_TEST_EPOCHS:-2}"
EULER_TRAIN_TEST_STEPS="${EULER_TRAIN_TEST_STEPS:-8}"

cd "$PROJECT_DIR"
mkdir -p "$EULER_TRAIN_TEST_OUTPUT_DIR"

# Set EULER_TRAIN_TEST_INSTALL=1 if the configured environment does not already
# have this package installed.
if [[ "${EULER_TRAIN_TEST_INSTALL:-0}" == "1" ]]; then
  if [[ -n "${EULER_TRAIN_SOURCE:-}" ]]; then
    python -m pip install -e "$EULER_TRAIN_SOURCE"
  fi
  python -m pip install -e "$PROJECT_DIR"
fi

echo "Starting euler-train streaming smoke test"
echo "Host: $(hostname)"
echo "Project: $PROJECT_DIR"
echo "Output: $EULER_TRAIN_TEST_OUTPUT_DIR"
echo "Stream attach token: ${EULER_VIEW_STREAM_ATTACH_TOKEN:0:8}..."

euler-train-stream-test \
  --check-handshake \
  --epochs "$EULER_TRAIN_TEST_EPOCHS" \
  --steps "$EULER_TRAIN_TEST_STEPS" \
  --stream-attach-token "$EULER_VIEW_STREAM_ATTACH_TOKEN" \
  --local-stream-jsonl "$EULER_TRAIN_TEST_OUTPUT_DIR/stream-events-${SLURM_JOB_ID:-interactive}.jsonl"
