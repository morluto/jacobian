from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from tools.process_supervisor import run_process_tree

ROOT = Path(__file__).resolve().parents[4]


def test_timeout_kills_a_descendant_that_ignores_sigterm(tmp_path: Path) -> None:
    marker = tmp_path / "child.pid"
    script = tmp_path / "ignore_term.py"
    script.write_text(
        "\n".join(
            [
                "import os",
                "import signal",
                "import time",
                "import sys",
                "child = os.fork()",
                "if child == 0:",
                "    signal.signal(signal.SIGTERM, signal.SIG_IGN)",
                "    with open(sys.argv[1], 'w', encoding='utf-8') as handle:",
                "        handle.write(str(os.getpid()))",
                "        handle.flush()",
                "        os.fsync(handle.fileno())",
                "    time.sleep(30)",
                "    os._exit(0)",
                "time.sleep(30)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_process_tree(
        (sys.executable, str(script), str(marker)),
        timeout=1.0,
        cwd=ROOT,
    )

    assert result.timed_out is True
    assert result.exit_code == 1
    deadline = time.monotonic() + 2
    child_pid: int | None = None
    while time.monotonic() < deadline:
        if marker.exists() and marker.read_text(encoding="utf-8").strip():
            child_pid = int(marker.read_text(encoding="utf-8"))
            break
        time.sleep(0.05)
        if child_pid is None:
            raise AssertionError("descendant did not record its pid before timeout")
        try:
            os.kill(child_pid, 0)
            still_alive = True
        except OSError:
            still_alive = False
        assert still_alive is False
