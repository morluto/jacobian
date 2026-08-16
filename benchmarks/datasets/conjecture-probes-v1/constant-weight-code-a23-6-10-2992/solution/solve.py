"""Emit the public 2992-word constant-weight-code Oracle construction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/app"))
    args = parser.parse_args()
    source = Path(__file__).with_name("codewords.txt")
    codewords = [
        line.strip().lower()
        for line in source.read_text().splitlines()
        if line.strip() and not line.startswith("$")
    ]
    submission = {"result": {"codewords": codewords}}
    (args.root / "submission.json").write_text(
        json.dumps(submission, sort_keys=True, separators=(",", ":")) + "\n"
    )


if __name__ == "__main__":
    main()
