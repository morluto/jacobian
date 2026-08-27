"""Contract tests for the Sidon extension-profile operation."""

from __future__ import annotations

from jacobian.math.combinatorics._sidon_extension_models import (
    SidonExtensionProfileRequest,
)
from jacobian.math.combinatorics._sidon_extension_operations import (
    compute_sidon_extension_profile,
)


def _extension(source: list[str], candidates: list[str]):
    request = SidonExtensionProfileRequest(
        source_elements=tuple(source),
        candidate_elements=tuple(candidates),
    )
    return compute_sidon_extension_profile(request)


class TestSidonExtensionProfile:
    def test_basic_fixture(self) -> None:
        """With A={1,2}: candidate 3 fails (diff 1 repeats), candidate 4 succeeds."""
        result = _extension(["1", "2"], ["3", "4"])
        assert result.admissible == ["4"]
        assert len(result.rejected) == 1
        assert result.rejected[0].candidate == "3"
        assert not result.rejected[0].is_admissible

    def test_all_candidates_admissible(self) -> None:
        """If candidates are far enough, all should be admissible."""
        result = _extension(["1", "10"], ["100", "200"])
        assert len(result.admissible) == 2
        assert len(result.rejected) == 0

    def test_all_candidates_rejected(self) -> None:
        """Candidates that repeat any difference from the source are rejected."""
        result = _extension(["1", "2", "5"], ["3"])
        assert len(result.admissible) == 0
        assert len(result.rejected) == 1
        assert result.rejected[0].candidate == "3"

    def test_empty_candidates(self) -> None:
        """No candidates means no admissible and no rejected."""
        result = _extension(["1", "2"], [])
        assert result.admissible == []
        assert result.rejected == []

    def test_obstruction_replays(self) -> None:
        """Each obstruction's pairs must produce the repeated difference."""
        result = _extension(["1", "2"], ["3"])
        obs = result.rejected[0].obstruction
        assert obs is not None
        diff = int(obs.repeated_difference)
        pair_a = int(obs.pair_a[0]) - int(obs.pair_a[1])
        pair_b = int(obs.pair_b[0]) - int(obs.pair_b[1])
        assert pair_a == diff
        assert pair_b == diff
        assert obs.pair_a != obs.pair_b

    def test_admissible_candidates_are_sidon(self) -> None:
        """For every admissible candidate x, A ∪ {x} should be Sidon."""
        result = _extension(["1", "2", "5"], ["3", "4", "10", "20"])
        for x in result.admissible:
            source = [1, 2, 5]
            union = source + [int(x)]
            diffs = []
            for i, a in enumerate(union):
                for j, b in enumerate(union):
                    if i != j:
                        diffs.append(a - b)
            assert len(diffs) == len(set(diffs)), f"A ∪ {{{x}}} is not Sidon"

    def test_partition_is_complete(self) -> None:
        """Admissible + rejected partition the candidates exactly."""
        result = _extension(["1", "2", "5"], ["3", "4", "6", "7", "10"])
        all_candidates = set(result.admissible) | {r.candidate for r in result.rejected}
        assert all_candidates == {"3", "4", "6", "7", "10"}
        assert len(result.admissible) + len(result.rejected) == 5

    def test_translation_preserves_partition(self) -> None:
        """Translating source and candidates by the same amount preserves
        the partition structure (same number of admissible/rejected)."""
        r1 = _extension(["1", "2"], ["3", "4"])
        r2 = _extension(["10", "11"], ["12", "13"])
        assert len(r1.admissible) == len(r2.admissible)
        assert len(r1.rejected) == len(r2.rejected)

    def test_maximality_check(self) -> None:
        """For C = {1,...,N} \ A, an empty admissible class means A is maximal."""
        # A = {1, 2} is not maximal in [1, 4] since 4 is admissible
        result = _extension(["1", "2"], ["3", "4"])
        assert "4" in result.admissible
        # A = {1, 2, 4} should be Sidon and maximal in [1, 5]
        result2 = _extension(["1", "2", "4"], ["3", "5"])
        # Check: 3 should fail (1-2=-1, 4-3=1 but we need a repeated diff)
        # 3: differences of {1,2,4,3} = {1-2,1-4,1-3,2-1,2-4,2-3,4-1,4-2,4-3,3-1,3-2,3-4}
        # 1-2=-1, 4-3=1, no. Actually let's check: 2-1=1, 3-2=1 -> repeated! So 3 is rejected.
        assert "3" not in result2.admissible
