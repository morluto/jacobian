"""Emit the published R(3,13) lower-bound-61 Oracle graph."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/app"))
    args = parser.parse_args()
    source = Path(__file__).with_name("adjacency.txt").read_text()
    entries = [int(value) for value in re.findall(r"[01]", source)]
    if len(entries) != 60 * 60:
        raise ValueError("published adjacency matrix does not have order 60")
    matrix = [entries[index : index + 60] for index in range(0, len(entries), 60)]
    edges = [[i, j] for i in range(60) for j in range(i + 1, 60) if matrix[i][j]]
    submission = {"result": {"edges": edges}}
    (args.root / "submission.json").write_text(
        json.dumps(submission, sort_keys=True, separators=(",", ":")) + "\n"
    )


if __name__ == "__main__":
    main()
