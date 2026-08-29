"""Contract tests for the Sidon extension-profile operation."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from jacobian.canonical import encode_strict_json
from jacobian.math.combinatorics import _sidon_extension_kernel as sidon_kernel
from jacobian.math.combinatorics import (
    _sidon_extension_models as sidon_models,
)
from jacobian.math.combinatorics._sidon_extension import (
    compute_sidon_extension_profile,
)
from jacobian.math.combinatorics._sidon_extension_models import (
    MAX_EXTENSION_RESULT_BYTES,
    SidonExtensionCandidateResult,
    SidonExtensionObstruction,
    SidonExtensionProfileRequest,
    SidonExtensionProfileResult,
    _maximum_result_bytes,
)


def _extension(source: list[str], candidates: list[str]) -> SidonExtensionProfileResult:
    request = SidonExtensionProfileRequest(
        source_elements=tuple(source),
        candidate_elements=tuple(candidates),
    )
    return compute_sidon_extension_profile(request)


class TestSidonExtensionProfile:
    def test_basic_fixture(self) -> None:
        """With A={1,2}: candidate 3 fails (diff 1 repeats), candidate 4 succeeds."""
        result = _extension(["1", "2"], ["3", "4"])
        assert result.admissible == ("4",)
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
        assert result.admissible == ()
        assert result.rejected == ()

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
        """Every admissible candidate should preserve the Sidon property."""
        result = _extension(["1", "2", "5"], ["3", "4", "10", "20"])
        for x in result.admissible:
            source = [1, 2, 5]
            union = [*source, int(x)]
            diffs = []
            for i, a in enumerate(union):
                for j, b in enumerate(union):
                    if i != j:
                        diffs.append(a - b)
            assert len(diffs) == len(set(diffs)), (
                f"adding {x} breaks the Sidon property"
            )

    def test_candidate_admission_is_not_a_fixed_count(self) -> None:
        """An empty source admits 1,001 linear-work candidates."""
        result = _extension([], [str(value) for value in range(1001)])
        assert len(result.admissible) == 1001
        assert result.rejected == ()

    def test_source_admission_is_derived_from_work(self) -> None:
        """A 33-element source is admitted when its actual work is small."""
        source = [str(2**index) for index in range(33)]
        result = _extension(source, [])
        assert result.source_elements == tuple(source)
        assert result.admissible == ()
        assert result.rejected == ()

    def test_large_source_profile_is_rejected_before_materialization(self) -> None:
        """The source-profile dictionary has its own bounded storage budget."""
        # For i > j, the positive difference is
        # (i-j) * (BASE + i+j).  BASE is larger than every possible cross-term,
        # so these differences are distinct; adding the common 120-digit
        # offset keeps the source values at the schema's wide-value boundary.
        base = 10**10
        source = tuple(
            str(10**120 + index * base + index * index) for index in range(2_000)
        )

        with pytest.raises(ValueError) as error:
            compute_sidon_extension_profile(
                SidonExtensionProfileRequest(
                    source_elements=source,
                    candidate_elements=(),
                )
            )

        assert "Sidon source-difference profiling" in str(error.value)

    def test_large_all_admissible_profile_fits_result_budget(self) -> None:
        """An empty source rules out rejected rows in the result bound."""
        candidates = tuple(str(value) for value in range(250_000))
        request = SidonExtensionProfileRequest(
            source_elements=(),
            candidate_elements=candidates,
        )
        result = compute_sidon_extension_profile(request)
        assert result.admissible == candidates
        assert result.rejected == ()
        assert len(encode_strict_json(result.model_dump(mode="json"))) <= (
            MAX_EXTENSION_RESULT_BYTES
        )

    def test_two_element_source_admits_attainable_profile(self) -> None:
        """A source of two elements can still have an all-admissible profile."""
        candidates = tuple(str(value) for value in range(3, 100_003))
        result = _extension(["0", "1"], list(candidates))

        assert result.admissible == candidates
        assert result.rejected == ()
        assert len(encode_strict_json(result.model_dump(mode="json"))) <= (
            MAX_EXTENSION_RESULT_BYTES
        )

    def test_candidate_work_excludes_source_profile_copies(self) -> None:
        """A large source and candidate batch stays within candidate-local work."""
        source = [str(2**index) for index in range(400)]
        candidates = [str(-value) for value in range(1, 2_377)]
        result = _extension(source, candidates)
        assert len(result.admissible) + len(result.rejected) == len(candidates)

    def test_result_bound_covers_the_actual_canonical_result(self) -> None:
        result = _extension(["1", "2", "5"], ["3", "4", "10", "20"])
        actual = len(encode_strict_json(result.model_dump(mode="json")))
        estimated = _maximum_result_bytes(
            result.source_elements,
            result.candidate_elements,
        )
        assert actual <= estimated <= MAX_EXTENSION_RESULT_BYTES

    def test_result_rejects_an_incomplete_partition(self) -> None:
        with pytest.raises(ValidationError):
            SidonExtensionProfileResult(
                source_elements=("1", "2"),
                candidate_elements=("3", "4"),
                admissible=("4",),
                rejected=(),
            )

    def test_result_rejects_an_unbound_obstruction(self) -> None:
        with pytest.raises(ValidationError):
            SidonExtensionProfileResult(
                source_elements=("1", "2"),
                candidate_elements=("3",),
                admissible=(),
                rejected=(
                    SidonExtensionCandidateResult(
                        candidate="3",
                        is_admissible=False,
                        obstruction=SidonExtensionObstruction(
                            candidate="3",
                            repeated_difference="1",
                            pair_a=("2", "1"),
                            pair_b=("4", "3"),
                        ),
                    ),
                ),
            )

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
        """For C = {1,...,N} minus A, empty admissibility means A is maximal."""
        # A = {1, 2} is not maximal in [1, 4] since 4 is admissible
        result = _extension(["1", "2"], ["3", "4"])
        assert "4" in result.admissible
        # A = {1, 2, 4} should be Sidon and maximal in [1, 5]
        result2 = _extension(["1", "2", "4"], ["3", "5"])
        # Check: 3 should fail (1-2=-1, 4-3=1 but we need a repeated diff)
        # 3: differences of {1,2,4,3} = {1-2,1-4,1-3,2-1,2-4,2-3,4-1,4-2,4-3,3-1,3-2,3-4}
        # 1-2=-1, 4-3=1, no. Actually let's check: 2-1=1, 3-2=1 -> repeated! So 3 is rejected.
        assert "3" not in result2.admissible

    def test_kernel_result_construction_does_not_replay_candidate_checks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = 0
        real_obstruction = sidon_models._candidate_obstruction

        def counted_obstruction(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            return real_obstruction(*args, **kwargs)

        monkeypatch.setattr(sidon_kernel, "_candidate_obstruction", counted_obstruction)
        monkeypatch.setattr(sidon_models, "_candidate_obstruction", counted_obstruction)
        _extension(["1", "2"], ["3", "4", "10"])
        assert calls == 3

    def test_kernel_reuses_the_admitted_source_difference_profile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = 0
        real_profile = sidon_models._ordered_difference_pairs

        def counted_profile(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            return real_profile(*args, **kwargs)

        monkeypatch.setattr(sidon_models, "_ordered_difference_pairs", counted_profile)
        _extension(["1", "2", "5"], ["3", "4"])
        assert calls == 1
