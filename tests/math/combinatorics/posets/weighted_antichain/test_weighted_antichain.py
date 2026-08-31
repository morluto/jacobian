from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.combinatorics.posets.core._models import (
    PresentationPair,
    ReflexivePairPolicy,
    RelationInterpretation,
)
from jacobian.math.combinatorics.posets.core.operations import materialize_finite_poset
from jacobian.math.combinatorics.posets.weighted_antichain.operations import (
    compute_maximum_weight_antichain,
)


def _cr(num, den=1):
    return CanonicalRational.from_fraction(Fraction(num, den))


def _chain_poset(elements):
    pairs = tuple(
        PresentationPair(lower=elements[i], upper=elements[i + 1])
        for i in range(len(elements) - 1)
    )
    return materialize_finite_poset(
        elements=tuple(elements),
        relation=pairs,
        interpretation=RelationInterpretation.COVER_EDGES,
        reflexive_pairs=ReflexivePairPolicy.FORBIDDEN,
    )


def _antichain_poset(elements):
    return materialize_finite_poset(
        elements=tuple(elements),
        relation=(),
        interpretation=RelationInterpretation.COMPARABLE_PAIRS,
        reflexive_pairs=ReflexivePairPolicy.FORBIDDEN,
    )


def test_chain_single_element() -> None:
    poset = _chain_poset(["a", "b", "c"])
    result = compute_maximum_weight_antichain(poset, (_cr(1), _cr(2), _cr(3)))
    assert result.maximum_weight.as_fraction() == Fraction(3)
    assert result.antichain == ("c",)


def test_antichain_all() -> None:
    poset = _antichain_poset(["a", "b", "c"])
    result = compute_maximum_weight_antichain(poset, (_cr(1), _cr(2), _cr(3)))
    assert result.maximum_weight.as_fraction() == Fraction(6)
    assert set(result.antichain) == {"a", "b", "c"}


def test_empty_poset() -> None:
    poset = _antichain_poset([])
    result = compute_maximum_weight_antichain(poset, ())
    assert result.maximum_weight.as_fraction() == Fraction(0)
    assert result.antichain == ()


def test_zero_weights() -> None:
    poset = _antichain_poset(["a", "b"])
    result = compute_maximum_weight_antichain(poset, (_cr(0), _cr(0)))
    assert result.maximum_weight.as_fraction() == Fraction(0)


def test_witness_is_antichain() -> None:
    poset = _chain_poset(["a", "b", "c", "d"])
    result = compute_maximum_weight_antichain(poset, (_cr(3), _cr(1), _cr(4), _cr(2)))
    assert result.maximum_weight.as_fraction() == Fraction(4)
    assert len(result.antichain) == 1


def test_rational_weights() -> None:
    poset = _chain_poset(["a", "b"])
    result = compute_maximum_weight_antichain(poset, (_cr(1, 2), _cr(3, 4)))
    assert result.maximum_weight.as_fraction() == Fraction(3, 4)


def test_result_preserves_source() -> None:
    poset = _antichain_poset(["a"])
    weights = (_cr(5),)
    result = compute_maximum_weight_antichain(poset, weights)
    assert result.poset == poset
    assert result.weights == weights


def test_rational_arithmetic_work_is_admitted_separately() -> None:
    """Test that the rational growth check rejects large multi-denominator sums.

    With an antichain of width 20 and weights using different denominators
    each near the 32,768-digit limit, the sum's digit growth exceeds the
    canonical envelope.
    """
    import pytest

    poset = _antichain_poset([str(index) for index in range(20)])
    # Use two different denominators near the limit to trigger the
    # multi-denominator growth bound: width * max_digits + len(str(width)).
    weights = tuple(
        CanonicalRational(
            num="1", den="1" + "0" * 16_400 + ("3" if i % 2 == 0 else "7")
        )
        for i in range(20)
    )
    with pytest.raises(ValueError, match="rational growth exceeds"):
        compute_maximum_weight_antichain(poset, weights)


def test_chain_growth_uses_width_not_carrier_size() -> None:
    digits = 16_384
    weights = tuple(
        CanonicalRational(num="1", den="1" + "0" * digits) for _ in range(2)
    )
    result = compute_maximum_weight_antichain(_chain_poset(["a", "b"]), weights)
    assert result.antichain in (("a",), ("b",))


def test_polynomial_chain_21_elements() -> None:
    """A 21-element chain should be handled by the polynomial algorithm."""
    elements = [str(i) for i in range(21)]
    poset = _chain_poset(elements)
    weights = tuple(_cr(1) for _ in range(21))
    result = compute_maximum_weight_antichain(poset, weights)
    # In a chain, the max antichain is a single element with weight 1
    assert result.maximum_weight.as_fraction() == Fraction(1)
    assert len(result.antichain) == 1


def test_polynomial_antichain_30_elements() -> None:
    """A 30-element antichain should sum all weights."""
    elements = [str(i) for i in range(30)]
    poset = _antichain_poset(elements)
    weights = tuple(_cr(i + 1) for i in range(30))
    result = compute_maximum_weight_antichain(poset, weights)
    assert result.maximum_weight.as_fraction() == Fraction(30 * 31, 2)
    assert len(result.antichain) == 30


def test_polynomial_mixed_poset_25_elements() -> None:
    """A 25-element poset with two independent chains should handle polynomially."""
    from jacobian.math.combinatorics.posets.core._models import (
        PresentationPair,
        ReflexivePairPolicy,
        RelationInterpretation,
    )
    from jacobian.math.combinatorics.posets.core.operations import (
        materialize_finite_poset,
    )

    # Use zero-padded names so lexical and numeric order agree.
    elements = [f"e{i:02d}" for i in range(25)]
    # Two chains: e00<e01<...<e12 and e13<e14<...<e24
    pairs = tuple(
        PresentationPair(lower=elements[i], upper=elements[i + 1]) for i in range(12)
    ) + tuple(
        PresentationPair(lower=elements[i], upper=elements[i + 1])
        for i in range(13, 24)
    )
    poset = materialize_finite_poset(
        elements=tuple(elements),
        relation=pairs,
        interpretation=RelationInterpretation.COVER_EDGES,
        reflexive_pairs=ReflexivePairPolicy.FORBIDDEN,
    )
    # Unit weights: max antichain picks one from each chain = 1 + 1 = 2
    weights = tuple(_cr(1) for _ in range(25))
    result = compute_maximum_weight_antichain(poset, weights)
    assert result.maximum_weight.as_fraction() == Fraction(2)
    assert len(result.antichain) == 2


def _allow_big_decimal_strings() -> int:
    """Raise Python's int<->str conversion limit; return previous value."""
    import sys

    previous = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(0)
    return previous


def test_relation_capacity_above_total_finite_vertex_capacity() -> None:
    """Thread 3050: the relation edge must stay uncuttable.

    A fixed relation capacity cheaper than either endpoint lets max-flow cut
    the must-not-cut relation edge and return a bogus weight with an invalid
    antichain. The relation capacity must exceed the total finite vertex
    capacity.
    """
    poset = _chain_poset(["a", "b"])
    weights = (_cr(10**19), _cr(10**19))
    result = compute_maximum_weight_antichain(poset, weights)
    assert result.maximum_weight.as_fraction() == Fraction(10**19)
    assert result.antichain in (("a",), ("b",))


def test_sums_with_one_nontrivial_denominator_are_admitted() -> None:
    """Thread 3050: per-weight numerator/denominator result growth.

    Weights 1/(10^20000+1) and 1 on an edgeless poset have a small reduced
    maximum (10^20000+2)/(10^20000+1); the old uniform width*max denominator
    bound would wrongly reject it.
    """
    import sys

    previous = _allow_big_decimal_strings()
    try:
        a = 10**20000 + 1
        a_str = "1" + "0" * 19999 + "1"
        poset = _antichain_poset(["a", "b"])
        weights = (
            CanonicalRational(num="1", den=a_str),
            CanonicalRational(num="1", den="1"),
        )
        result = compute_maximum_weight_antichain(poset, weights)
        assert result.maximum_weight.as_fraction() == Fraction(a + 1, a)
        assert set(result.antichain) == {"a", "b"}
    finally:
        sys.set_int_max_str_digits(previous)


def test_common_denominator_scale_is_bounded_in_admission() -> None:
    """Thread 3050: an exploding common denominator is rejected up front.

    A 64-element chain of pairwise-coprime 32768-digit denominators would force
    the min-cut to clear to an ~2M-digit common denominator; admission must stop
    it before the kernel builds million-digit scaled integers.
    """
    import random
    import sys

    previous = _allow_big_decimal_strings()
    try:
        random.seed(7)
        base = 10**32767
        denominators = [
            base + random.randrange(2, 10**9, 2) for _ in range(64)
        ]
        denominators = [d if d % 2 else d + 1 for d in denominators]
        poset = _chain_poset([str(i) for i in range(64)])
        weights = tuple(
            CanonicalRational(num="1", den=str(d)) for d in denominators
        )
        import pytest

        with pytest.raises(
            ValueError, match="rational growth exceeds the canonical"
        ):
            compute_maximum_weight_antichain(poset, weights)
    finally:
        sys.set_int_max_str_digits(previous)


def _comparable_map(poset):
    """Transitive comparability reachability from cover edges."""
    elements = list(poset.elements)
    idx = {e: i for i, e in enumerate(elements)}
    n = len(elements)
    down = [set() for _ in range(n)]
    for pair in poset.strict_order_pairs:
        down[idx[pair.lower]].add(idx[pair.upper])
    # Floyd-Warshall closure on the cover graph.
    reach = [set(down[i]) for i in range(n)]
    for k in range(n):
        for i in range(n):
            if k in reach[i]:
                reach[i].update(reach[k])
    return idx, reach


def test_mincut_agrees_with_exhaustive_reference() -> None:
    """Cross-check the polynomial min-cut against brute-force enumeration."""
    from itertools import combinations

    poset = _chain_poset(["a", "b", "c", "d"])
    weights = (_cr(3), _cr(1), _cr(4), _cr(2))
    result = compute_maximum_weight_antichain(poset, weights)
    fracs = [w.as_fraction() for w in weights]
    _, reach = _comparable_map(poset)

    best = Fraction(0)
    for arity in range(1, len(fracs) + 1):
        for combo in combinations(range(len(fracs)), arity):
            invalid = any(
                i != j and (j in reach[i] or i in reach[j])
                for i in combo
                for j in combo
            )
            if invalid:
                continue
            best = max(best, sum(fracs[i] for i in combo))
    assert result.maximum_weight.as_fraction() == best


def test_mincut_matches_networkx_on_scaled_integers() -> None:
    """Cross-check the min-cut reduction against networkx on integer weights."""
    import networkx as nx

    # 0<1<2, 3<4, and 1<4: two interleaved chains, one strict-order pair set
    # built as a valid cover.
    poset = _chain_poset(["0", "1", "2"])
    weights = tuple(_cr(i + 1) for i in range(3))
    result = compute_maximum_weight_antichain(poset, weights)

    n = 3
    fracs = [w.as_fraction() for w in weights]
    total_int_scale = 1000
    int_w = [int(f * total_int_scale) for f in fracs]
    graph = nx.DiGraph()
    graph.add_nodes_from(range(2 + 2 * n))
    idx = {e: i for i, e in enumerate(poset.elements)}
    inf_cap = sum(int_w) + 1
    for i in range(n):
        graph.add_edge(0, 2 + i, weight=int_w[i])
        graph.add_edge(2 + n + i, 1, weight=int_w[i])
    for pair in poset.strict_order_pairs:
        graph.add_edge(
            2 + idx[pair.lower], 2 + n + idx[pair.upper], weight=inf_cap
        )
    cut, _ = nx.minimum_cut(graph, 0, 1, capacity="weight")
    total_weight = sum(w.as_fraction() for w in weights)
    assert result.maximum_weight.as_fraction() == total_weight - Fraction(
        cut, total_int_scale
    )
