import hashlib
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
limitations = ["FINITE_LENGTH_16_INSTANCE", "NO_GENERAL_ENUMERATION_THEOREM"]
evidence = {
    "schema_version": "1",
    "task_id": "jacobian/necklace-burnside-certificate",
    "result": result,
    "limitations": limitations,
}
root = (
    Path(sys.argv[2])
    if len(sys.argv) == 3 and sys.argv[1] == "--root"
    else Path("/app")
)
(root / "evidence").mkdir(parents=True, exist_ok=True)
evidence_path = (
    root / "answer.txt" if root != Path("/app") else root / "evidence/answer.txt"
)
evidence_path.write_text(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
digest = "sha256:" + hashlib.sha256(evidence_path.read_bytes()).hexdigest()
submission = {
    "task_id": "jacobian/necklace-burnside-certificate",
    "conclusion": "DIHEDRAL_ORBITS_ENUMERATED",
    "result": result,
    "claimed_assurance": "COMPUTED",
    "scope": "ALL_LENGTH_16_BINARY_WORDS_AND_ALL_32_DIHEDRAL_ACTIONS",
    "completeness": "COMPLETE",
    "evidence": [{"path": "evidence/answer.txt", "sha256": digest}],
    "limitations": limitations,
}
(root / "submission.json").write_text(
    json.dumps(submission, sort_keys=True, separators=(",", ":"))
)
