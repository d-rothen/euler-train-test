# euler-train-stream-test

Tiny synthetic package for testing `euler-train` streaming from ETH Zurich's
Euler cluster. It does not train a model. It emits the same kinds of records a
real run would produce:

- stream `init` and `finish` lifecycle events
- train and validation metric events
- evaluation metadata updates
- output snapshot metadata
- checkpoint metadata

The command fails when no stream is configured, unless you explicitly use the
local JSONL stream mode for development.

## Local smoke test

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e /Volumes/Volume/git/euler-train-w/train-test
python -m pip install -e ".[dev]"

euler-train-stream-test \
  --local-stream-only \
  --epochs 1 \
  --steps 2 \
  --sleep-sec 0
```

This writes normal `euler-train` run files under `./runs/` and a local stream
copy at `./runs/stream-events.jsonl`.

## Euler View streaming

Set the Euler View connection through environment variables:

```bash
export EULER_VIEW_BASE_URL="https://view.example.com"
export EULER_VIEW_API_TOKEN="..."
export EULER_VIEW_STREAM_ATTACH_TOKEN="..."
```

If you are testing the SLURM fallback path instead of an explicit attachment
token, set `EULER_VIEW_MODEL_ID` instead:

```bash
export EULER_VIEW_MODEL_ID=42
```

Then run:

```bash
euler-train-stream-test --check-handshake
```

Useful options:

- `--epochs` and `--steps`: control how many metric events are emitted.
- `--local-stream-jsonl PATH`: also record emitted stream events locally.
- `--check-handshake`: call `POST /api/model-run-stream/check` before the run.
- `--stream-token`: use a pre-issued ingest token and skip session negotiation.

## Submit on Euler

Copy or clone this package to Euler, install the local `euler-train` package in
the same Python environment, export the variables above, then submit:

```bash
sbatch scripts/run_euler_stream_test.sbatch
```

The job script loads `eth_proxy` before installing or running anything that may
contact Euler View or PyPI.

Optional job environment:

```bash
export PROJECT_DIR="$HOME/euler-train-test"
export EULER_TRAIN_SOURCE="$HOME/euler-train"
export EULER_TRAIN_TEST_OUTPUT_DIR="$SCRATCH/euler-train-stream-test"
export EULER_TRAIN_TEST_EPOCHS=2
export EULER_TRAIN_TEST_STEPS=8
```

`EULER_TRAIN_SOURCE` is only needed when you want the job to install a local
checkout of `euler-train` before installing this package.
