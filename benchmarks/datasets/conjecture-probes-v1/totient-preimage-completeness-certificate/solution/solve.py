from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

TASK_ID = "jacobian/totient-preimage-completeness-certificate"
PRIMES = [2, 3, 5, 7, 13, 17]
OPTIONS = [[0, 1, 2, 3, 4, 5], [0, 1, 2], [0, 1], [0, 1], [0, 1], [0, 1]]
LIMITATIONS = [
    "ONE_TARGET_TOTIENT_VALUE_48",
    "EXACT_FINITE_PREIMAGE_CLASSIFICATION",
    "NO_GLOBAL_CARMICHAEL_CONCLUSION",
]


def contribution(p: int, exponent: int) -> int:
    return 1 if exponent == 0 else (p - 1) * p ** (exponent - 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/app"))
    root = parser.parse_args().root
    solutions = []
    for exponents in itertools.product(*OPTIONS):
        if (
            __import__("math").prod(
                contribution(p, a) for p, a in zip(PRIMES, exponents, strict=True)
            )
            != 48
        ):
            continue
        factors = [[p, a] for p, a in zip(PRIMES, exponents, strict=True) if a]
        n = __import__("math").prod(p**a for p, a in factors)
        solutions.append({"n": n, "factorization": factors, "totient": 48})
    result = {
        "candidate_primes": PRIMES,
        "prime_power_options": [
            {"prime": p, "exponents": options}
            for p, options in zip(PRIMES, OPTIONS, strict=True)
        ],
        "enumerated_branch_count": 288,
        "solutions": sorted(solutions, key=lambda row: row["n"]),
        "accepted_count": len(solutions),
    }
    payload = {
        "schema_version": "1",
        "task_id": TASK_ID,
        "result": result,
        "limitations": LIMITATIONS,
    }
    evidence = root / "evidence/answer.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )
    submission = {
        "task_id": TASK_ID,
        "conclusion": "PHI_48_COMPLETE_PREIMAGE_CLASSIFICATION",
        "result": result,
        "claimed_assurance": "CHECKED",
        "scope": "phi-48-complete-preimage-classification-v1",
        "completeness": "COMPLETE",
        "evidence": [
            {
                "path": "evidence/answer.json",
                "sha256": "sha256:" + hashlib.sha256(evidence.read_bytes()).hexdigest(),
            }
        ],
        "limitations": LIMITATIONS,
    }
    (root / "submission.json").write_text(json.dumps(submission, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
