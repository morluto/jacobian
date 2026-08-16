import itertools
import json
import sys
from pathlib import Path

N = 16


def valid(word):
    return all(
        not (word[i] == word[(i + 1) % N] == word[(i + 2) % N]) for i in range(N)
    )


def rotation(word, k):
    return word[k:] + word[:k]


def reflection(word, k):
    return tuple(word[(k - i) % N] for i in range(N))


words = [word for word in itertools.product((0, 1), repeat=N) if valid(word)]
rotations = [sum(rotation(word, k) == word for word in words) for k in range(N)]
reflections = [sum(reflection(word, k) == word for word in words) for k in range(N)]
representatives = sorted(
    {
        "".join(
            map(
                str,
                min(
                    [rotation(word, k) for k in range(N)]
                    + [reflection(word, k) for k in range(N)]
                ),
            )
        )
        for word in words
    }
)
result = {
    "valid_labelled_words": len(words),
    "rotation_fixed_counts": rotations,
    "reflection_fixed_counts": reflections,
    "burnside_numerator": sum(rotations + reflections),
    "orbit_count": len(representatives),
    "canonical_representatives": representatives,
}
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
