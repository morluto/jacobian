import json
import os
from pathlib import Path

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]


def partitions(total, cap=None):
    cap = total if cap is None else min(cap, total)
    if total == 0:
        yield ()
        return
    for first in range(cap, 0, -1):
        for tail in partitions(total - first, first):
            yield (first, *tail)


def value(part):
    result = 1
    for prime, item in zip(PRIMES[: len(part)], part, strict=True):
        result *= prime ** ((1 << item) - 1)
    return result


def table(total):
    return [
        {"partition": list(part), "value": value(part)} for part in partitions(total)
    ]


def factors(number):
    result = []
    for prime in PRIMES:
        exponent = 0
        while number % prime == 0:
            exponent += 1
            number //= prime
        if exponent:
            result.append({"prime": prime, "exponent": exponent})
    return result


current, next_table = table(12), table(13)
a, b = min(item["value"] for item in current), min(item["value"] for item in next_table)
base = Path(os.environ.get("SOLUTION_DIR", "/app"))
submission = {
    "result": {
        "k": 12,
        "current_minimizer": a,
        "current_factors": factors(a),
        "current_divisor_count": 4096,
        "current_candidates": current,
        "next_minimizer": b,
        "next_factors": factors(b),
        "next_divisor_count": 8192,
        "next_candidates": next_table,
        "quotient": b // a,
    },
}
(base / "submission.json").write_text(json.dumps(submission))
