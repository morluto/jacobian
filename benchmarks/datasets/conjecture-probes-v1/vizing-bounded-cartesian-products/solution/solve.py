"""Emit the frozen Oracle certificate for the bounded Vizing probe."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    root = (
        Path(sys.argv[2])
        if len(sys.argv) == 3 and sys.argv[1] == "--root"
        else Path("/app")
    )
    source = Path(__file__).resolve().parent / "submission.json"
    result = json.loads(source.read_text())["result"]
    (root / "submission.json").write_text(
        json.dumps({"result": result}, sort_keys=True, separators=(",", ":")) + "\n"
    )


if __name__ == "__main__":
    main()
