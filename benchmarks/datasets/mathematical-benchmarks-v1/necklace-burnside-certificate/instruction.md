# Certify constrained binary necklaces under dihedral symmetry

For cyclic binary words of length 16, forbid every cyclic run of three equal bits. Count equivalence classes under all rotations and reflections.

1. `valid_labelled_words`: the total number of valid labelled words;
2. `rotation_fixed_counts`: a 16-element array where index `k` is the number of valid words fixed by rotation by `k` positions (`word[i] == word[(i+k) mod 16]`);
3. `reflection_fixed_counts`: a 16-element array where index `k` is the number of valid words fixed by the reflection through index `k` (`word[i] == word[(k-i) mod 16]`);
4. `burnside_numerator` and `orbit_count`; and
5. `canonical_representatives`: the sorted list of the lexicographically least representative of every orbit.

The verifier independently enumerates all 65,536 binary words, applies the cyclic constraint, reconstructs every dihedral action, recomputes the fixed-point table, and compares the complete orbit partition. A count without the representatives is incomplete; this is an exact finite replay, not a proof of a general necklace theorem.

<!-- BEGIN PUBLIC CONTRACT SUBMISSION BLOCK -->
## Submission

The verifier replays the task-specific mathematical predicate from the submitted result.

Write `/app/submission.json` to the exact schema in `environment/submission_schema.json`. The submission requires a typed `result`.

<!-- END PUBLIC CONTRACT SUBMISSION BLOCK -->
