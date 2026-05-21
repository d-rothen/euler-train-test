from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import euler_train
from . import __version__


METRIC_NAMING: dict[str, Any] = {
    "producer_key": "euler_train_stream_test.synthetic",
    "producer_version": __version__,
    "namespaces": {
        "sim.train": {
            "axes": {
                "kind": {
                    "position": 0,
                    "optional": False,
                    "values": ["loss", "stat", "diag"],
                },
                "name": {
                    "position": 1,
                    "optional": False,
                    "values": [
                        "total",
                        "aux",
                        "lr",
                        "grad_norm",
                        "throughput",
                        "progress",
                    ],
                },
            },
        },
        "sim.val": {
            "axes": {
                "kind": {
                    "position": 0,
                    "optional": False,
                    "values": ["loss", "metric"],
                },
                "name": {
                    "position": 1,
                    "optional": False,
                    "values": ["total", "accuracy", "psnr"],
                },
            },
        },
        "sys.train": {
            "axes": {},
        },
    },
}


class JsonlStreamRecorder:
    """Small local stream consumer for debugging the emitted event sequence."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = None

    def bind(self, context: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")
        self._write(
            {
                "type": "bind",
                "run_id": context.run_id,
                "run_dir": str(context.run_dir),
                "project_dir": str(context.project_dir),
            },
        )

    def emit(self, event: dict[str, Any]) -> None:
        self._write(event)

    def flush(self) -> None:
        if self._handle is not None:
            self._handle.flush()

    def close(self) -> None:
        if self._handle is None:
            return
        self._handle.flush()
        self._handle.close()
        self._handle = None

    def _write(self, payload: dict[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("stream recorder is not bound")
        self._handle.write(json.dumps(payload, default=_json_default) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a tiny synthetic euler-train job that emits Euler View "
            "streaming events."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_env_path("EULER_TRAIN_TEST_OUTPUT_DIR"),
        help=(
            "Project directory passed to euler_train.init. Defaults to "
            "$SCRATCH/euler-train-stream-test when SCRATCH is set, otherwise "
            "./runs."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=_env_int("EULER_TRAIN_TEST_EPOCHS", 2),
        help="Number of synthetic epochs to log.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=_env_int("EULER_TRAIN_TEST_STEPS", 8),
        help="Synthetic training steps per epoch.",
    )
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=_env_float("EULER_TRAIN_TEST_SLEEP_SEC", 0.1),
        help="Delay between synthetic train steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_env_int("EULER_TRAIN_TEST_SEED", 7),
        help="Random seed used for synthetic metrics and outputs.",
    )
    parser.add_argument(
        "--run-name",
        default=_env_text("EULER_TRAIN_TEST_RUN_NAME", "stream-smoke-test"),
        help="Human-readable run name stored in meta.json.",
    )
    parser.add_argument(
        "--api-url",
        default=_env_text("EULER_VIEW_BASE_URL"),
        help="Euler View base URL. Also read from EULER_VIEW_BASE_URL.",
    )
    parser.add_argument(
        "--model-id",
        type=int,
        default=_env_int("EULER_VIEW_MODEL_ID"),
        help=(
            "Euler View model ID. Required for SLURM fallback matching when "
            "no stream attach token is provided."
        ),
    )
    parser.add_argument(
        "--stream-attach-token",
        default=_env_text("EULER_VIEW_STREAM_ATTACH_TOKEN"),
        help=(
            "Opaque launch attachment token. Also read from "
            "EULER_VIEW_STREAM_ATTACH_TOKEN."
        ),
    )
    parser.add_argument(
        "--stream-token",
        default=_env_text("EULER_VIEW_STREAM_TOKEN"),
        help=(
            "Pre-issued ingest token. If set, the session handshake is "
            "skipped."
        ),
    )
    parser.add_argument(
        "--datasource-id",
        type=int,
        default=_env_int("EULER_VIEW_DATASOURCE_ID"),
        help="Optional Euler View datasource hint.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=_env_float("EULER_VIEW_STREAM_TIMEOUT_SEC", 10.0),
        help="HTTP timeout for stream session and ingest calls.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_env_int("EULER_VIEW_STREAM_BATCH_SIZE", 1),
        help="Stream event batch size. The default flushes every event.",
    )
    parser.add_argument(
        "--check-handshake",
        action="store_true",
        default=_env_bool("EULER_TRAIN_TEST_CHECK_HANDSHAKE", False),
        help="Call the Euler View dry-run stream handshake before logging.",
    )
    parser.add_argument(
        "--local-stream-jsonl",
        type=Path,
        default=_env_path("EULER_TRAIN_TEST_LOCAL_STREAM_JSONL"),
        help="Also write a JSONL copy of stream events to this path.",
    )
    parser.add_argument(
        "--local-stream-only",
        action="store_true",
        default=_env_bool("EULER_TRAIN_TEST_LOCAL_STREAM_ONLY", False),
        help="Use only the local JSONL stream recorder and skip HTTP calls.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_positive("epochs", args.epochs)
    _validate_positive("steps", args.steps)
    _validate_non_negative("sleep-sec", args.sleep_sec)
    _validate_positive("batch-size", args.batch_size)

    output_dir = args.output_dir or _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        stream = _build_stream(args, output_dir)
        if args.check_handshake and not args.local_stream_only:
            _check_handshake(args)
        run = _run_synthetic(args, output_dir, stream)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"stream test failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "status": "completed",
                "run_id": run.run_id,
                "run_dir": str(run.dir),
            },
            sort_keys=True,
        ),
    )
    return 0


def _run_synthetic(args: argparse.Namespace, output_dir: Path, stream: Any) -> Any:
    rng = np.random.default_rng(args.seed)
    total_steps = args.epochs * args.steps
    config = {
        "purpose": "euler-train streaming smoke test",
        "epochs": args.epochs,
        "steps_per_epoch": args.steps,
        "sleep_sec": args.sleep_sec,
        "seed": args.seed,
        "uses_real_training": False,
        "stream_batch_size": args.batch_size,
    }
    meta = {
        "description": (
            "Synthetic run for validating euler-train streaming from an "
            "Euler compute node."
        ),
        "tags": ["stream-test", "synthetic", "euler-cluster"],
        "test_package": {
            "name": "euler-train-stream-test",
            "version": __version__,
        },
        "runtime": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
        },
    }

    with euler_train.init(
        dir=str(output_dir),
        config=config,
        meta=meta,
        run_name=args.run_name,
        mode="train",
        stream=stream,
        metric_naming=METRIC_NAMING,
    ) as run:
        print(
            json.dumps(
                {
                    "status": "started",
                    "run_id": run.run_id,
                    "run_dir": str(run.dir),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        run.add_evaluation(
            "synthetic_val",
            name="Synthetic validation",
            status="running",
            metadata={"dataset": "synthetic", "split": "val"},
        )

        global_step = 0
        for epoch in range(args.epochs):
            for step_in_epoch in range(args.steps):
                progress = (global_step + 1) / total_steps
                loss = 1.0 / (1.0 + global_step) + 0.01 * rng.random()
                aux_loss = 0.25 * loss
                lr = 1e-3 * (0.5 + 0.5 * math.cos(math.pi * progress))
                grad_norm = 0.4 + 0.05 * rng.random()
                throughput = 128.0 + 8.0 * rng.random()
                run.log(
                    {
                        "sim.train.loss.total": round(loss, 6),
                        "sim.train.loss.aux": round(aux_loss, 6),
                        "sim.train.stat.lr": round(lr, 8),
                        "sim.train.diag.grad_norm": round(grad_norm, 6),
                        "sim.train.stat.throughput": round(throughput, 3),
                        "sim.train.stat.progress": round(progress, 6),
                    },
                    step=global_step,
                    epoch=epoch,
                    mode="train",
                )
                global_step += 1
                if args.sleep_sec:
                    time.sleep(args.sleep_sec)

            val_loss = 0.12 / (epoch + 1) + 0.005 * rng.random()
            accuracy = 0.75 + 0.2 * ((epoch + 1) / args.epochs)
            psnr = 22.0 + 2.0 * epoch + rng.random()
            run.log(
                {
                    "sim.val.loss.total": round(val_loss, 6),
                    "sim.val.metric.accuracy": round(accuracy, 6),
                    "sim.val.metric.psnr": round(psnr, 6),
                },
                step=global_step,
                epoch=epoch,
                mode="val",
            )
            _save_synthetic_outputs(run, rng, epoch=epoch, step=global_step)

        run.finish_evaluation("synthetic_val")
        checkpoint_path = run.dir / "checkpoints" / "synthetic_checkpoint.json"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(
            json.dumps(
                {
                    "note": "No model was trained; this file only tests checkpoint logging.",
                    "epoch": args.epochs - 1,
                    "step": max(global_step - 1, 0),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        run.log_saved_checkpoint(
            checkpoint_path,
            epoch=args.epochs - 1,
            step=max(global_step - 1, 0),
            is_best=True,
        )

    return run


def _save_synthetic_outputs(run: Any, rng: np.random.Generator, *, epoch: int, step: int) -> None:
    pred = rng.random((8, 8)).astype("float32")
    gt = rng.random((8, 8)).astype("float32")
    residual = np.abs(pred - gt).astype("float32")
    run.save_outputs(
        epoch=epoch,
        step=step,
        metadata={"dataset": "synthetic", "split": "val"},
        scalar_field={
            "pred": pred,
            "gt": gt,
            "aux": {
                "residual": residual,
            },
        },
    )


def _build_stream(args: argparse.Namespace, output_dir: Path) -> Any:
    consumers: list[Any] = []
    if not args.local_stream_only:
        consumers.append(_http_stream_config(args))

    local_stream_jsonl = args.local_stream_jsonl
    if args.local_stream_only and local_stream_jsonl is None:
        local_stream_jsonl = output_dir / "stream-events.jsonl"
    if local_stream_jsonl is not None:
        consumers.append(JsonlStreamRecorder(local_stream_jsonl))

    if not consumers:
        raise SystemExit("no stream consumer configured")
    if len(consumers) == 1:
        return consumers[0]
    return consumers


def _http_stream_config(args: argparse.Namespace) -> dict[str, Any]:
    api_token = _env_text("EULER_VIEW_API_TOKEN") or _env_text("EULER_VIEW_ACCESS_TOKEN")
    if not args.api_url:
        raise SystemExit("set EULER_VIEW_BASE_URL or pass --api-url")
    if not args.stream_token and not api_token:
        raise SystemExit(
            "set EULER_VIEW_API_TOKEN, EULER_VIEW_ACCESS_TOKEN, or EULER_VIEW_STREAM_TOKEN"
        )
    if not args.stream_token and args.model_id is None and not args.stream_attach_token:
        raise SystemExit(
            "set EULER_VIEW_STREAM_ATTACH_TOKEN or EULER_VIEW_MODEL_ID for stream attachment"
        )

    return {
        "base_url": args.api_url,
        "api_token": api_token,
        "stream_token": args.stream_token,
        "stream_attach_token": args.stream_attach_token,
        "model_id": args.model_id,
        "datasource_id": args.datasource_id,
        "timeout_sec": args.timeout_sec,
        "batch_size": args.batch_size,
        "flush_interval_sec": 0.0,
    }


def _check_handshake(args: argparse.Namespace) -> None:
    config = _http_stream_config(args)
    if config.get("stream_token"):
        print(
            "skipping dry-run handshake because EULER_VIEW_STREAM_TOKEN bypasses sessions",
            flush=True,
        )
        return

    result = euler_train.check_stream_handshake(config)
    run = result.get("run", {}) if isinstance(result, dict) else {}
    print(
        json.dumps(
            {
                "status": "handshake-ok",
                "resolution": result.get("resolution"),
                "model_id": run.get("modelId"),
                "run_id": run.get("runId"),
                "stream_attach_token_present": bool(run.get("streamAttachToken")),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _default_output_dir() -> Path:
    scratch = _env_text("SCRATCH")
    if scratch:
        return Path(scratch) / "euler-train-stream-test"
    return Path.cwd() / "runs"


def _env_text(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _env_path(name: str) -> Path | None:
    value = _env_text(name)
    return Path(value) if value else None


def _env_int(name: str, default: int | None = None) -> int | None:
    value = _env_text(name)
    if value is None:
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = _env_text(name)
    if value is None:
        return default
    return float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = _env_text(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _validate_positive(name: str, value: int | float | None) -> None:
    if value is None or value <= 0:
        raise SystemExit(f"--{name} must be positive")


def _validate_non_negative(name: str, value: float) -> None:
    if value < 0:
        raise SystemExit(f"--{name} must be non-negative")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
