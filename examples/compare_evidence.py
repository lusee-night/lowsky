"""Reproduce evidence comparisons; this does not fit or certify the model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "analysis/constraints-20261013/comparisons"
    )
    parser.add_argument("--only", choices=["all", "bright_sources", "radio_regions"], default="all")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    scripts = {
        "bright_sources": "compare_bright_sources.py",
        "radio_regions": "compare_radio_regions.py",
    }
    jobs = scripts if args.only == "all" else {args.only: scripts[args.only]}
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        JAX_PLATFORMS="cpu",
    )
    receipts = []
    for name, script in jobs.items():
        command = [sys.executable, str(ROOT / "examples" / script), "--output-dir", str(out / name)]
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        receipts.append({"comparison": name, "command": command, "status": "completed"})
    out.mkdir(parents=True, exist_ok=True)
    (out / f"run_{args.only}.json").write_text(
        json.dumps(
            {
                "runs": receipts,
                "interpretation": "Diagnostics with documented observation-operator limits; no refitting or certified region of correctness.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
