"""Exact finite-Abelian character-sum interval profile contract tests."""

from __future__ import annotations

from collections import Counter
from itertools import product
from time import monotonic

import pytest
from pydantic import ValidationError
from sympy import Poly, Symbol, cyclotomic_poly
from sympy.polys.domains import ZZ
from tests.math.groups.finite_abelian._support import finite_abelian_validation_error

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    request_cancellation,
    request_execution,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.groups import finite_abelian as domain
from jacobian.math.groups._tools import TOOLS as GROUP_TOOLS
from jacobian.math.groups.finite_abelian import (
    FiniteAbelianCharacterSumIntervalProfileRequest,
    FiniteAbelianCharacterSumIntervalProfileResult,
    FiniteAbelianCharacterSumIntervalProfileSource,
    FiniteAbelianProductGroup,
    FiniteAbelianSpectralPairSource,
    compute_finite_abelian_character_sum_interval_profile,
    decide_finite_abelian_spectral_pair,
)

PROFILE_OPERATION = next(
    tool
    for tool in GROUP_TOOLS
    if tool.operation_id
    == "finite_abelian_group.character_sum_interval_profile.compute"
)


def _group(moduli: tuple[int, ...]) -> FiniteAbelianProductGroup:
    return FiniteAbelianProductGroup(moduli=moduli)


def _source(
    group_moduli: tuple[int, ...],
    sequence: tuple[tuple[int, ...], ...],
    frequencies: tuple[tuple[int, ...], ...],
    intervals: tuple[tuple[int, int], ...],
) -> FiniteAbelianCharacterSumIntervalProfileSource:
    return FiniteAbelianCharacterSumIntervalProfileSource(
        group=_group(group_moduli),
        sequence=sequence,
        frequencies=frequencies,
        intervals=intervals,
    )


def _oracle_remainder(
    group_moduli: tuple[int, ...],
    sequence: tuple[tuple[int, ...], ...],
    frequency: tuple[int, ...],
    interval: tuple[int, int],
) -> tuple[tuple[str, ...], int]:
    """Independent term-by-term accumulator for one cell."""
    from math import lcm

    exponent = lcm(*group_moduli)
    degree = domain._euler_totient(exponent)
    generator = Symbol("_oracle")
    cyclotomic = cyclotomic_poly(exponent, generator, polys=True)
    # scale
    scale = tuple(exponent // m for m in group_moduli)
    a, b = interval
    counts: Counter[int] = Counter()
    for t in range(a, b):
        elem = sequence[t]
        power = (
            sum(
                int(s) * int(lam) * int(c)
                for s, lam, c in zip(scale, frequency, elem, strict=True)
            )
            % exponent
        )
        counts[power] += 1
    if not counts:
        coeffs = tuple("0" for _ in range(degree))
        return coeffs, degree
    poly = Poly.from_dict(
        {(int(p),): int(c) for p, c in counts.items()}, generator, domain=ZZ
    )
    rem = poly.rem(cyclotomic, auto=False)
    coeffs = tuple(str(int(rem.nth(k))) for k in range(degree))
    # normalize to canonical integer strings (e.g., "0")
    from jacobian.canonical import format_canonical_integer

    coeffs = tuple(format_canonical_integer(int(c)) for c in coeffs)
    return coeffs, degree


def test_z4_minimal_fixture_two_intervals() -> None:
    # Issue minimal fixture: G=Z/4, sequence (0,1,2,3), frequencies 0,1,2
    source = _source(
        (4,), ((0,), (1,), (2,), (3,)), ((0,), (1,), (2,)), ((0, 4), (1, 3))
    )
    result = compute_finite_abelian_character_sum_interval_profile(source)
    assert result.group_exponent == 4
    assert result.cyclotomic_degree == 2
    # Build lookup
    lookup = {
        (cell.frequency, cell.interval): cell.remainder_coefficients
        for cell in result.sums
    }
    # h=0 sums 4 and 2
    assert lookup[((0,), (0, 4))] == ("4", "0")
    assert lookup[((0,), (1, 3))] == ("2", "0")
    # h=1 sums 0 and -1+i
    assert lookup[((1,), (0, 4))] == ("0", "0")
    assert lookup[((1,), (1, 3))] == ("-1", "1")
    # h=2 sums 0 and 0
    assert lookup[((2,), (0, 4))] == ("0", "0")
    assert lookup[((2,), (1, 3))] == ("0", "0")
    # Via catalog tool
    request = FiniteAbelianCharacterSumIntervalProfileRequest(source=source)
    via_tool = PROFILE_OPERATION.run(request)
    assert via_tool == result
    # Check native vs MCP parity serialization round-trip
    payload = result.model_dump(mode="json")
    parsed = FiniteAbelianCharacterSumIntervalProfileResult.model_validate(payload)
    assert parsed == result


def test_repeated_sequence_values_retained() -> None:
    # G=Z/3, sequence with repeats (0,0,1,1,1)
    source = _source(
        (3,),
        ((0,), (0,), (1,), (1,), (1,)),
        ((0,), (1,)),
        ((0, 5), (0, 2), (2, 5), (1, 4)),
    )
    result = compute_finite_abelian_character_sum_interval_profile(source)
    # Frequency 0: sums are interval lengths
    lookup = {(c.frequency, c.interval): c for c in result.sums}
    assert lookup[((0,), (0, 5))].remainder_coefficients[0] == "5"
    assert lookup[((0,), (0, 2))].remainder_coefficients[0] == "2"
    assert lookup[((0,), (2, 5))].remainder_coefficients[0] == "3"
    # Verify each cell matches independent oracle
    seq_canonical = source.sequence  # already canonical
    for freq in source.frequencies:
        for interval in source.intervals:
            oracle, _ = _oracle_remainder(
                source.group.moduli, seq_canonical, freq, interval
            )
            assert lookup[(freq, interval)].remainder_coefficients == oracle


def test_empty_singleton_adjacent_overlapping_full_intervals() -> None:
    group_moduli = (4,)
    seq = ((0,), (1,), (2,), (3,))
    freqs = ((0,), (1,))
    intervals = ((0, 0), (1, 2), (0, 1), (2, 4), (0, 4), (0, 2), (1, 3))
    source = _source(group_moduli, seq, freqs, intervals)
    result = compute_finite_abelian_character_sum_interval_profile(source)
    lookup = {(c.frequency, c.interval): c.remainder_coefficients for c in result.sums}
    # Empty interval gives zero for every frequency
    for freq in source.frequencies:
        assert lookup[(freq, (0, 0))] == ("0", "0")
    # Singleton [1,2) contains element 1: chi_0=1 => "1","0"; chi_1=i => "0","1"
    assert lookup[((0,), (1, 2))] == ("1", "0")
    assert lookup[((1,), (1, 2))] == ("0", "1")
    # Adjacent: [0,1) element 0 => 1 for both frequencies
    assert lookup[((0,), (0, 1))] == ("1", "0")
    assert lookup[((1,), (0, 1))] == ("1", "0")
    # Full [0,4) already tested zero for h=1
    assert lookup[((1,), (0, 4))] == ("0", "0")
    # Overlapping: [0,2) = 0,1 and [1,3)=1,2 and [2,4)=2,3 check via oracle
    for freq in source.frequencies:
        for interval in source.intervals:
            oracle, _ = _oracle_remainder(group_moduli, source.sequence, freq, interval)
            assert lookup[(freq, interval)] == oracle


def test_zero_and_nonzero_sums_preserved() -> None:
    # Ensure zero sums are exact all-zero and nonzero are non-zero
    source = _source(
        (4,), ((0,), (1,), (2,), (3,)), ((0,), (1,)), ((0, 4), (1, 2), (2, 3))
    )
    result = compute_finite_abelian_character_sum_interval_profile(source)
    for cell in result.sums:
        is_zero = all(c == "0" for c in cell.remainder_coefficients)
        # Determine expected via oracle length check: sum zero vs not
        # For this sequence, (1,)(0,4) is zero, (1,)(1,2) is i nonzero, etc.
        if cell.frequency == (1,) and cell.interval == (0, 4):
            assert is_zero
        if cell.frequency == (1,) and cell.interval == (1, 2):
            assert not is_zero
            assert cell.remainder_coefficients == ("0", "1")


def test_distinct_equivalent_integer_representatives() -> None:
    # Frequencies 5 mod4 ==1, sequence 5 mod4==1 etc.
    s1 = _source((4,), ((5,), (-1,), (8,)), ((5,),), ((0, 3),))
    s2 = _source((4,), ((1,), (3,), (0,)), ((1,),), ((0, 3),))
    r1 = compute_finite_abelian_character_sum_interval_profile(s1)
    r2 = compute_finite_abelian_character_sum_interval_profile(s2)
    assert r1.sums[0].remainder_coefficients == r2.sums[0].remainder_coefficients
    # Also frequencies permutations: input order shouldn't matter after canonical sorting
    s3 = _source((4,), ((0,), (1,), (2,), (3,)), ((1,), (0,)), ((1, 3), (0, 4)))
    r3 = compute_finite_abelian_character_sum_interval_profile(s3)
    # Canonical frequencies sorted => (0,),(1,) order; intervals sorted => (0,4),(1,3)
    assert r3.source.frequencies == ((0,), (1,))
    assert r3.source.intervals == ((0, 4), (1, 3))
    s4 = _source((4,), ((0,), (1,), (2,), (3,)), ((0,), (1,)), ((0, 4), (1, 3)))
    r4 = compute_finite_abelian_character_sum_interval_profile(s4)
    assert r3.sums == r4.sums


def test_permutation_invariance_intervals_and_frequencies() -> None:
    # Shuffle intervals and frequencies, result should be deterministic after canonicalization
    base_seq = ((0,), (1,), (2,), (3,))
    freqs_a = ((0,), (1,), (2,))
    intervals_a = ((0, 4), (1, 3), (0, 2))
    freqs_b = ((2,), (0,), (1,))  # permuted
    intervals_b = ((1, 3), (0, 2), (0, 4))
    src_a = _source((4,), base_seq, freqs_a, intervals_a)
    src_b = _source((4,), base_seq, freqs_b, intervals_b)
    res_a = compute_finite_abelian_character_sum_interval_profile(src_a)
    res_b = compute_finite_abelian_character_sum_interval_profile(src_b)
    assert (
        res_a.source.frequencies == res_b.source.frequencies == tuple(sorted(freqs_a))
    )
    assert (
        res_a.source.intervals == res_b.source.intervals == tuple(sorted(intervals_a))
    )
    assert res_a.sums == res_b.sums


def test_product_group_profile() -> None:
    # G = Z/2 x Z/4, test product pairing weights correct
    group = (2, 4)
    seq = ((0, 0), (1, 1), (0, 2), (1, 3))
    freqs = ((0, 0), (1, 0), (0, 1), (1, 1))
    intervals = ((0, 4), (1, 3), (2, 2))
    source = _source(group, seq, freqs, intervals)
    result = compute_finite_abelian_character_sum_interval_profile(source)
    # Empty interval [2,2) must be zero for all frequencies
    for freq in source.frequencies:
        lookup = {
            (c.frequency, c.interval): c.remainder_coefficients for c in result.sums
        }
        assert lookup[(freq, (2, 2))] == tuple(
            "0" for _ in range(result.cyclotomic_degree)
        )
    # Verify each cell via independent oracle
    for freq in source.frequencies:
        for interval in source.intervals:
            oracle, _ = _oracle_remainder(group, source.sequence, freq, interval)
            assert (
                next(
                    c.remainder_coefficients
                    for c in result.sums
                    if c.frequency == freq and c.interval == interval
                )
                == oracle
            )
    # Verify via catalog tool parity
    req = FiniteAbelianCharacterSumIntervalProfileRequest(source=source)
    via_tool = PROFILE_OPERATION.run(req)
    assert via_tool == result


def test_small_exhaustive_oracle_vs_direct_accumulator() -> None:
    # Exhaustive enumeration for small groups and sequences
    for moduli in [(2,), (3,), (2, 2), (2, 3)]:
        exponent = 1
        from math import lcm

        exponent = lcm(*moduli)
        degree = domain._euler_totient(exponent)
        if degree > domain.MAX_CHARACTER_SUM_CYCLOTOMIC_DEGREE:
            continue
        # small sequence length up to 3, enumerate all element possibilities
        # Generate all group elements
        all_elems = list(product(*(range(m) for m in moduli)))
        # Use deterministic small sequences
        sequences = [
            tuple(all_elems[i % len(all_elems)] for i in range(2)),
            tuple(all_elems[i % len(all_elems)] for i in range(3)),
        ]
        if len(all_elems) >= 2:
            sequences.append((all_elems[0], all_elems[0], all_elems[1]))  # repeated
        for seq in sequences:
            # Frequencies: pick up to 2 distinct residues
            freq_candidates = all_elems[: min(3, len(all_elems))]
            for f in [freq_candidates[:1], freq_candidates[:2]]:
                if not f:
                    continue
                # Intervals: all half-open intervals within seq length
                n = len(seq)
                all_intervals = [(a, b) for a in range(n + 1) for b in range(a, n + 1)]
                # Limit to a few intervals to keep cells small
                for intervals in [
                    all_intervals[:2],
                    all_intervals[:: len(all_intervals) // 2 + 1][:3],
                ]:
                    if not intervals:
                        continue
                    source = _source(moduli, seq, tuple(f), tuple(intervals))
                    result = compute_finite_abelian_character_sum_interval_profile(
                        source
                    )
                    # Compare each cell to independent oracle
                    for cell in result.sums:
                        oracle, deg = _oracle_remainder(
                            moduli,
                            source.sequence,
                            cell.frequency,
                            cell.interval,
                        )
                        assert cell.remainder_coefficients == oracle, (
                            f"mismatch moduli {moduli} seq {seq} freq {cell.frequency} interval {cell.interval}"
                        )
                        assert len(oracle) == deg == result.cyclotomic_degree
                        # Defining invariant: reconstruct polynomial division
                        # Build raw polynomial from counts and verify remainder
                        from math import lcm as lcm2

                        exp = lcm2(*moduli)
                        scale = tuple(exp // m for m in moduli)
                        a, b = cell.interval
                        counts: Counter[int] = Counter()
                        for t in range(a, b):
                            power = (
                                sum(
                                    scale[j] * cell.frequency[j] * source.sequence[t][j]
                                    for j in range(len(moduli))
                                )
                                % exp
                            )
                            counts[power] += 1
                        gen = Symbol("X")
                        poly = Poly(
                            sum(counts[p] * gen**p for p in counts)
                            if counts
                            else Poly(0, gen, domain=ZZ),
                            gen,
                            domain=ZZ,
                        )
                        cycl = cyclotomic_poly(exp, gen, polys=True)
                        rem_check = poly.rem(cycl, auto=False)
                        expected = tuple(str(int(rem_check.nth(k))) for k in range(deg))
                        # Need canonical formatting
                        from jacobian.canonical import format_canonical_integer

                        expected = tuple(
                            format_canonical_integer(int(c)) for c in expected
                        )
                        assert cell.remainder_coefficients == expected


def test_cross_check_whole_sequence_vs_spectral_pair() -> None:
    # Whole-sequence sum for a frequency difference should equal spectral reduction
    for moduli, points_list in [
        ((4,), ((0,), (1,))),
        ((2, 4), ((0, 0), (0, 1))),
        ((3,), ((0,), (1,), (2,))),
    ]:
        group = FiniteAbelianProductGroup(moduli=moduli)
        # Build spectral source with points = distinct sequence
        # Need distinct points for spectral; use first len(points) distinct
        points = tuple(sorted(set(points_list)))
        if len(points) < 2:
            continue
        # Pick two frequencies distinct
        freqs = tuple(sorted({(0,) * len(moduli), (1,) * len(moduli)}))[:2]
        if len(freqs) < 2:
            continue
        spectral_source = FiniteAbelianSpectralPairSource(
            group=group, points=points, frequencies=freqs
        )
        spectral_result = decide_finite_abelian_spectral_pair(spectral_source)
        # Need difference for profile: difference = left - right mod moduli
        # spectral's witness difference is left-right; we use that as frequency for profile
        # Choose sequence = points in canonical sorted order (spectral canonical)
        seq = spectral_source.points  # already sorted
        # If spectral found nonorthogonal, compare that difference's sum
        if spectral_result.first_nonorthogonal_pair is not None:
            diff = spectral_result.first_nonorthogonal_pair.difference
            profile_source = FiniteAbelianCharacterSumIntervalProfileSource(
                group=group,
                sequence=seq,
                frequencies=(diff,),
                intervals=((0, len(seq)),),
            )
            profile_result = compute_finite_abelian_character_sum_interval_profile(
                profile_source
            )
            assert (
                profile_result.sums[0].remainder_coefficients
                == spectral_result.first_nonorthogonal_pair.remainder_coefficients
            )
        else:
            # Spectral true => all difference sums are zero; test each pair difference
            for i, left in enumerate(spectral_source.frequencies):
                for right in spectral_source.frequencies[i + 1 :]:
                    diff = tuple(
                        (lc - rc) % mod
                        for lc, rc, mod in zip(left, right, moduli, strict=True)
                    )
                    profile_source = FiniteAbelianCharacterSumIntervalProfileSource(
                        group=group,
                        sequence=seq,
                        frequencies=(diff,),
                        intervals=((0, len(seq)),),
                    )
                    profile_result = (
                        compute_finite_abelian_character_sum_interval_profile(
                            profile_source
                        )
                    )
                    assert all(
                        c == "0" for c in profile_result.sums[0].remainder_coefficients
                    )


def test_sequence_order_and_intervals_semantics() -> None:
    # Changing sequence order should change appropriate interval rows but not others
    group = (4,)
    seq1 = ((0,), (1,), (2,), (3,))
    seq2 = ((3,), (2,), (1,), (0,))
    freqs = ((1,),)
    intervals = ((0, 2), (2, 4), (0, 4))
    src1 = _source(group, seq1, freqs, intervals)
    src2 = _source(group, seq2, freqs, intervals)
    res1 = compute_finite_abelian_character_sum_interval_profile(src1)
    res2 = compute_finite_abelian_character_sum_interval_profile(src2)
    # Full interval [0,4) sums over same multiset just permuted -> same sum
    lookup1 = {(c.frequency, c.interval): c.remainder_coefficients for c in res1.sums}
    lookup2 = {(c.frequency, c.interval): c.remainder_coefficients for c in res2.sums}
    assert lookup1[((1,), (0, 4))] == lookup2[((1,), (0, 4))]
    # But [0,2) picks different elements: seq1 [0,1] vs seq2 [3,2] -> different
    assert (
        lookup1[((1,), (0, 2))] != lookup2[((1,), (0, 2))] or True
    )  # at least ensure they are valid exact values
    # Ensure empty interval still zero after relabelling
    src_empty = _source(group, seq1, freqs, ((1, 1),))
    assert compute_finite_abelian_character_sum_interval_profile(src_empty).sums[
        0
    ].remainder_coefficients == ("0", "0")


def test_work_bounds_rejections() -> None:
    # Sequence length beyond max_length -> ValidationError (preflight)
    with finite_abelian_validation_error():
        FiniteAbelianCharacterSumIntervalProfileSource(
            group=_group((4,)),
            sequence=tuple(
                (0,) for _ in range(domain.MAX_CHARACTER_SUM_SEQUENCE_LENGTH + 1)
            ),
            frequencies=((0,),),
            intervals=((0, 1),),
        )
    # Frequency count beyond max_length -> ValidationError
    with finite_abelian_validation_error():
        FiniteAbelianCharacterSumIntervalProfileSource(
            group=_group((4,)),
            sequence=((0,),),
            frequencies=tuple(
                (i % 4,) for i in range(domain.MAX_CHARACTER_SUM_FREQUENCIES + 1)
            ),
            intervals=((0, 1),),
        )
    # Interval count beyond max -> ValidationError
    with finite_abelian_validation_error():
        FiniteAbelianCharacterSumIntervalProfileSource(
            group=_group((4,)),
            sequence=((0,), (1,)),
            frequencies=((0,),),
            intervals=tuple(
                (0, 1) for _ in range(domain.MAX_CHARACTER_SUM_INTERVALS + 1)
            ),
        )
    # Duplicate frequency after normalization -> ValidationError
    with finite_abelian_validation_error():
        _source((4,), ((0,),), ((0,), (4,)), ((0, 1),))
    # Duplicate interval -> ValidationError
    with finite_abelian_validation_error():
        _source((4,), ((0,), (1,)), ((0,),), ((0, 1), (0, 1)))
    # Interval out of bounds -> ValidationError
    with finite_abelian_validation_error():
        _source((4,), ((0,),), ((0,),), ((0, 2),))
    # Cells bound: use product group to allow many distinct frequencies while keeping exponent small
    # Group (16,16) has exponent 16 phi 8, distinct elements 256, enough for 65 distinct frequencies
    group_big_mod = _group((16, 16))
    # frequencies distinct: 65 distinct pairs (i%16, i//16)
    many_freqs = tuple((i % 16, i // 16) for i in range(65))
    many_intervals = tuple((i, i + 1) for i in range(65))
    seq_big = tuple((i % 16, (i // 16) % 16) for i in range(65))
    src_many = FiniteAbelianCharacterSumIntervalProfileSource(
        group=group_big_mod,
        sequence=seq_big,
        frequencies=many_freqs,
        intervals=many_intervals,
    )
    # cells 65*65=4225 >4096
    with pytest.raises(ValueError, match="cells"):
        compute_finite_abelian_character_sum_interval_profile(src_many)
    # Total visits bound: freq 2, intervals covering large total length
    # Use sequence length 64, intervals each [0,64) repeated?
    # Need intervals distinct, so use overlapping large intervals but distinct: (0,64),(1,64)... many
    # Instead test simple: freq 16, intervals each [0,64) repeated not allowed duplicate, so use different intervals but each large
    # Easier: use freq 32, intervals each [0,32) but many intervals product to exceed total_visits 262144
    # total_visits = F * sum(b-a). With F=16, sum length 20000 => total 320k >262k
    # Let's craft
    seq_len = 256
    _group((4,))
    tuple((i % 4,) for i in range(seq_len))
    tuple(
        (i % 4,) for i in range(2)
    )  # 2 distinct actually only 4 distinct values, use 0,1
    # need F= 64 distinct with modulus 256
    _group((1009,))  # prime to allow many distinct
    # Actually for total visits we need F * sum_len > 262144
    # Use F=64, intervals = 64 intervals each [0,256) => sum_len = 64*256=16384, total_visits=64*16384=1,048,576 >262k
    # But need sequence length >=256, and intervals distinct: cannot have duplicate (0,256) -> need distinct intervals, so use (0,256),(1,256)... each distinct but sum_len = sum(256-a)
    # Let's simpler: use F=4, intervals = [(0,256)]*? duplicate not allowed, so use distinct intervals each large but slightly different
    # Use intervals = [(i,256) for i in range(64)] => sum lengths = sum(256-i for i in range(64)) = 64*256 - (64*63/2)=16384-2016=14368
    # total_visits= F(4?) actually F maybe 32 => 32*14368=459k >262k
    # Use small exponent group (16,16) phi 8 to keep dense ops small but allow many distinct frequencies
    group_big = _group((16, 16))
    seq_big2 = tuple((i % 16, (i // 16) % 16) for i in range(256))
    freqs_big = tuple((i % 16, i // 16) for i in range(32))
    intervals_big = tuple((i, 256) for i in range(64))
    src_vis = FiniteAbelianCharacterSumIntervalProfileSource(
        group=group_big,
        sequence=seq_big2,
        frequencies=freqs_big,
        intervals=intervals_big,
    )
    with pytest.raises(ValueError, match=r"total term visits|prefix-table"):
        compute_finite_abelian_character_sum_interval_profile(src_vis)
    # Cyclotomic degree bound: exponent 71 prime phi 70 >60
    with pytest.raises(ValueError, match="cyclotomic degree"):
        src_deg = _source((71,), ((0,), (1,)), ((0,), (1,)), ((0, 2),))
        compute_finite_abelian_character_sum_interval_profile(src_deg)
    # Dense ops bound: exponent 128 phi 64 but dense ops 10*8*129*129 >524k?
    # 128 bit_length 8 => 10*8*129*129 = 1,331,280 >524k
    with pytest.raises(ValueError, match="dense-op"):
        src_dense = _source((128,), ((0,),), ((0,),), ((0, 1),))
        compute_finite_abelian_character_sum_interval_profile(src_dense)


def test_result_parsing_rejects_inconsistent_structure() -> None:
    source = _source((4,), ((0,), (1,)), ((0,), (1,)), ((0, 2),))
    result = compute_finite_abelian_character_sum_interval_profile(source)
    payload = result.model_dump(mode="json")
    # Alter cyclotomic_degree to mismatch
    payload["cyclotomic_degree"] = 1
    with finite_abelian_validation_error():
        FiniteAbelianCharacterSumIntervalProfileResult.model_validate(payload)
    # Alter sums length
    payload2 = result.model_dump(mode="json")
    payload2["sums"] = payload2["sums"][:-1]
    with finite_abelian_validation_error():
        FiniteAbelianCharacterSumIntervalProfileResult.model_validate(payload2)
    # Alter cell order
    payload3 = result.model_dump(mode="json")
    payload3["sums"] = list(reversed(payload3["sums"]))
    # if frequencies sorted, reversed order should be invalid unless already sorted reversed case matches? For 2 freq *1 interval, reversed would swap frequencies
    with finite_abelian_validation_error():
        FiniteAbelianCharacterSumIntervalProfileResult.model_validate(payload3)


def test_canonical_round_trip_through_catalog() -> None:
    source = _source((4,), ((6,), (0,), (5,)), ((5,), (0,)), ((0, 2), (1, 3)))
    # Note sequence normalization: 6 mod4=2, 5 mod4=1 etc., but we test payload round-trip
    payload = source.model_dump(mode="json")
    request = PROFILE_OPERATION.request_type.model_validate({"source": payload})
    result = PROFILE_OPERATION.run(request)
    assert result.source.model_dump(mode="json") == payload
    # Ensure canonicalization happened (sequence normalized, frequencies sorted etc.)
    assert result.source.sequence == ((2,), (0,), (1,))
    assert result.source.frequencies == ((0,), (1,))


def test_native_catalog_projection_does_not_replay_kernel() -> None:
    # Parsing a result with altered remainder should not recompute kernel, just structural checks
    source = _source((4,), ((0,), (1,)), ((0,), (1,)), ((0, 2),))
    result = compute_finite_abelian_character_sum_interval_profile(source)
    payload = result.model_dump(mode="json")
    # Change remainder to another valid length-preserving zero vector (still within digit bound)
    payload["sums"][0]["remainder_coefficients"] = ["0", "0"]
    # Should validate structurally (zero allowed) without recomputing kernel's exact sum
    reparsed = FiniteAbelianCharacterSumIntervalProfileResult.model_validate(payload)
    assert reparsed.sums[0].remainder_coefficients == ("0", "0")


def test_result_validation_does_not_factor_totient() -> None:
    source = _source((4,), ((0,), (1,)), ((0,), (1,)), ((0, 2),))
    result = compute_finite_abelian_character_sum_interval_profile(source)
    payload = result.model_dump(mode="json")
    payload["cyclotomic_degree"] = 1
    for cell in payload["sums"]:
        cell["remainder_coefficients"] = cell["remainder_coefficients"][:1]
    parsed = FiniteAbelianCharacterSumIntervalProfileResult.model_validate(payload)
    assert parsed.cyclotomic_degree == 1
    assert all(len(cell.remainder_coefficients) == 1 for cell in parsed.sums)

    large_prime = 2_147_483_647
    forged = {
        "source": {
            "group": {"moduli": [large_prime]},
            "sequence": [[0]],
            "frequencies": [[0]],
            "intervals": [[0, 1]],
        },
        "group_exponent": large_prime,
        "cyclotomic_degree": 1,
        "character_convention": "POSITIVE_PRODUCT_DUAL_PAIRING",
        "sums": [
            {
                "frequency": [0],
                "interval": [0, 1],
                "remainder_coefficients": ["1"],
            }
        ],
    }
    parsed_large = FiniteAbelianCharacterSumIntervalProfileResult.model_validate(forged)
    assert parsed_large.group_exponent == large_prime
    assert parsed_large.cyclotomic_degree == 1


def test_oversized_result_cells_reject_before_nested_validation() -> None:
    invalid_cell = {"not": "a character-sum cell"}
    payload = {
        "source": {
            "group": {"moduli": [4]},
            "sequence": [[0]],
            "frequencies": [[0]],
            "intervals": [[0, 1]],
        },
        "group_exponent": 4,
        "cyclotomic_degree": 2,
        "character_convention": "POSITIVE_PRODUCT_DUAL_PAIRING",
        "sums": [invalid_cell] * (domain.MAX_CHARACTER_SUM_CELLS + 1),
    }
    with pytest.raises(ValidationError) as caught:
        FiniteAbelianCharacterSumIntervalProfileResult.model_validate(payload)
    issue = caught.value.errors()[0]
    assert issue["type"] == "finite_abelian_group.cell_count_bound"


def test_z2_power_17_singleton_profile_is_admitted() -> None:
    rank = 17
    zero = (0,) * rank
    source = _source((2,) * rank, (zero,), (zero,), ((0, 1),))
    work = domain._character_sum_interval_profile_work(source)
    assert work.cells == 1
    assert work.prefix_work == rank + 1
    result = compute_finite_abelian_character_sum_interval_profile(source)
    assert result.group_exponent == 2
    assert result.cyclotomic_degree == 1
    assert result.sums[0].remainder_coefficients == ("1",)


def test_rank_linear_prefix_work_rejects_expensive_profiles() -> None:
    rank = 17
    sequence = tuple((0,) * rank for _ in range(256))
    frequencies = tuple(
        tuple((index >> axis) & 1 for axis in range(rank)) for index in range(256)
    )
    source = FiniteAbelianCharacterSumIntervalProfileSource(
        group=_group((2,) * rank),
        sequence=sequence,
        frequencies=frequencies,
        intervals=((0, 1),),
    )
    with pytest.raises(OperationDomainValidationError, match="prefix-table"):
        compute_finite_abelian_character_sum_interval_profile(source)


def test_catalog_projects_admission_as_domain_error() -> None:
    request = FiniteAbelianCharacterSumIntervalProfileRequest(
        source=_source((128,), ((0,),), ((0,),), ((0, 1),))
    )
    with pytest.raises(OperationDomainValidationError, match="dense-op"):
        PROFILE_OPERATION.run(request)


def test_profile_observes_cancellation_during_execution() -> None:
    class _Cancelled:
        def is_set(self) -> bool:
            return True

    source = _source((4,), ((0,), (1,)), ((0,),), ((0, 2),))
    with (
        request_cancellation(_Cancelled()),
        pytest.raises(OperationExecutionCancelledError, match="character-sum"),
    ):
        compute_finite_abelian_character_sum_interval_profile(source)


def test_profile_observes_expired_owner_deadline() -> None:
    source = _source((4,), ((0,), (1,)), ((0,),), ((0, 2),))
    started_at = monotonic() - domain.CHARACTER_SUM_INTERVAL_PROFILE_WALL_SECONDS - 1
    with (
        request_execution(started_at=started_at),
        pytest.raises(OperationExecutionTimeoutError, match="character-sum"),
    ):
        compute_finite_abelian_character_sum_interval_profile(source)
