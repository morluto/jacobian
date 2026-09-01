"""Fixture worker that keeps one nested dialogue child live until cancellation."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from jacobian.process import BoundedWorkerDialogue, run_bounded_worker_dialogue


def main() -> None:
    marker = Path(sys.argv[1])
    child_source = (
        "import os, sys, time\n"
        "from pathlib import Path\n"
        "os.write(1, b'ready!')\n"
        "Path(sys.argv[1]).write_text(str(os.getpid()), encoding='ascii')\n"
        "time.sleep(30)\n"
    )

    def wait_for_child(dialogue: BoundedWorkerDialogue) -> None:
        if dialogue.read_until(b"!", frame_limit=32) != b"ready!":
            raise RuntimeError("nested dialogue child did not become ready")
        time.sleep(30)

    run_bounded_worker_dialogue(
        [sys.executable, "-u", "-c", child_source, str(marker)],
        wait_for_child,
        absolute_deadline=time.monotonic() + 30,
        environment=dict(os.environ),
        stdout_limit=4_096,
        stderr_limit=4_096,
    )


if __name__ == "__main__":
    main()
