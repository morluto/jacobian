from __future__ import annotations

import argparse
import itertools
import json
from math import prod
from pathlib import Path

PRIMES = [2, 3, 5, 7, 13, 17]
OPTIONS = [[0, 1, 2, 3, 4, 5], [0, 1, 2], [0, 1], [0, 1], [0, 1], [0, 1]]


def contribution(p: int, exponent: int) -> int:
    return 1 if exponent == 0 else (p - 1) * p ** (exponent - 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/app"))
    root = parser.parse_args().root
    solutions = []
    for exponents in itertools.product(*OPTIONS):
        if (
            prod(contribution(p, a) for p, a in zip(PRIMES, exponents, strict=True))
            != 48
        ):
            continue
        factors = [[p, a] for p, a in zip(PRIMES, exponents, strict=True) if a]
        n = prod(p**a for p, a in factors)
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
    submission = {"result": result}
    (root / "submission.json").write_text(json.dumps(submission, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
