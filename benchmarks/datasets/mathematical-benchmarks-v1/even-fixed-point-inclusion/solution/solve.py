import itertools
import json
import math
import sys
from pathlib import Path


def derive():
    histogram = [0] * 5
    for permutation in itertools.permutations(range(1, 9)):
        fixed = sum(permutation[value - 1] == value for value in (2, 4, 6, 8))
        histogram[fixed] += 1
    terms = [(-1) ** j * math.comb(4, j) * math.factorial(8 - j) for j in range(5)]
    return {
        "signed_inclusion_terms": terms,
        "inclusion_sum": sum(terms),
        "exact_even_fixed_histogram": histogram,
    }


result = derive()
root = (
    Path(sys.argv[2])
    if len(sys.argv) == 3 and sys.argv[1] == "--root"
    else Path("/app")
)
submission = {
    "result": result,
}
(root / "submission.json").write_text(
    json.dumps(submission, sort_keys=True, separators=(",", ":"))
)
