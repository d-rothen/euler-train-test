from __future__ import annotations

import json
from pathlib import Path

from euler_train_stream_test.cli import main


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_local_stream_only_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "project"

    code = main(
        [
            "--local-stream-only",
            "--output-dir",
            str(output_dir),
            "--epochs",
            "1",
            "--steps",
            "2",
            "--sleep-sec",
            "0",
        ],
    )

    assert code == 0
    run_dirs = list((output_dir / "runs").iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "meta.json").exists()
    assert (run_dirs[0] / "train.jsonl").exists()
    assert (run_dirs[0] / "val.jsonl").exists()

    events = _jsonl(output_dir / "stream-events.jsonl")
    event_types = [event["type"] for event in events]
    assert event_types[0] == "bind"
    assert "init" in event_types
    assert "metric" in event_types
    assert "output_snapshot" in event_types
    assert "checkpoint" in event_types
    assert event_types[-1] == "finish"
