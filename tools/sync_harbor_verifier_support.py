"""Update verifier checksum labels for explicitly selected Harbor tasks.

Task support modules are intentionally owned by their task bundles.  This
command only updates the checksum label for each selected task's verifier bundle and
never copies support code, formats benchmark files, or touches unselected
tasks.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.tooling.harbor_suite import (  # noqa: E402
    HarborSuiteError,
    get_suite,
    select_task_refs,
    verifier_bundle_checksum,
)

_CHECKSUM = re.compile(r'jacobian\.checksum="[^"]*"')


def update(dataset: str, tasks: tuple[str, ...]) -> int:
    suite = get_suite(dataset)
    refs = select_task_refs(suite, tasks)
    for ref in refs:
        tests = ref.path / "tests"
        verifier = tests / "verifier.py"
        dockerfile = tests / "Dockerfile"
        if verifier.is_symlink() or not verifier.is_file():
            raise HarborSuiteError(
                f"{verifier.relative_to(ROOT)}: verifier.py must be a regular file"
            )
        if dockerfile.is_symlink() or not dockerfile.is_file():
            raise HarborSuiteError(
                f"{dockerfile.relative_to(ROOT)}: Dockerfile must be a regular file"
            )
        support = tests / "verifier_support.py"
        if support.is_symlink() or not support.is_file():
            raise HarborSuiteError(
                f"{support.relative_to(ROOT)}: verifier_support.py must be a regular file"
            )
        digest = verifier_bundle_checksum(tests)
        text = dockerfile.read_text(encoding="utf-8")
        updated, count = _CHECKSUM.subn(f'jacobian.checksum="{digest}"', text, count=1)
        if not count:
            updated, count = re.subn(
                r"^(FROM [^\n]+\n)",
                f'\\1LABEL jacobian.checksum="{digest}"\n',
                text,
                count=1,
                flags=re.MULTILINE,
            )
        if not count:
            raise HarborSuiteError(
                f"{dockerfile.relative_to(ROOT)}: no FROM line for checksum label"
            )
        if updated != text:
            dockerfile.write_text(updated, encoding="utf-8")
        print(f"Updated verifier checksum: {ref.path.relative_to(ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    args = parser.parse_args()
    try:
        return update(args.dataset, tuple(args.tasks))
    except HarborSuiteError as exc:
        print(f"harbor verifier checksum error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
