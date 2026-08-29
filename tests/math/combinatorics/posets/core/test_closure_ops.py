"""Tests for poset closure, dual, and subposet operations (issue #1746)."""

from __future__ import annotations

from jacobian.math.combinatorics.posets.core._closure_models import (
    DualPosetRequest,
    InducedSubposetRequest,
    LowerClosureRequest,
    UpperClosureRequest,
)
from jacobian.math.combinatorics.posets.core._closure_tools import (
    dual_poset,
    induced_subposet,
    lower_closure,
    upper_closure,
)
from jacobian.math.combinatorics.posets.core._models import (
    FinitePoset,
    FinitePosetRequest,
    PresentationPair,
    RelationInterpretation,
)
from jacobian.math.combinatorics.posets.core.operations import materialize_finite_poset


def make_poset(
    elements: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
) -> FinitePoset:
    """Materialize a poset from cover edges."""
    req = FinitePosetRequest(
        elements=elements,
        relation=tuple(
            PresentationPair(lower=lower, upper=upper) for lower, upper in edges
        ),
        interpretation=RelationInterpretation.COVER_EDGES,
    )
    return materialize_finite_poset(
        req.elements, req.relation, req.interpretation, req.reflexive_pairs
    )


# ---------------------------------------------------------------------------
# Lower closure
# ---------------------------------------------------------------------------


class TestLowerClosure:
    def test_chain_top_element(self) -> None:
        p = make_poset(("a", "b", "c"), (("a", "b"), ("b", "c")))
        r = lower_closure(LowerClosureRequest(poset=p, subset=("c",)))
        assert set(r.closure) == {"a", "b", "c"}

    def test_chain_middle_element(self) -> None:
        p = make_poset(("a", "b", "c"), (("a", "b"), ("b", "c")))
        r = lower_closure(LowerClosureRequest(poset=p, subset=("b",)))
        assert set(r.closure) == {"a", "b"}

    def test_chain_bottom_element(self) -> None:
        p = make_poset(("a", "b", "c"), (("a", "b"), ("b", "c")))
        r = lower_closure(LowerClosureRequest(poset=p, subset=("a",)))
        assert set(r.closure) == {"a"}

    def test_antichain(self) -> None:
        p = make_poset(("a", "b", "c"), ())
        r = lower_closure(LowerClosureRequest(poset=p, subset=("a",)))
        assert set(r.closure) == {"a"}

    def test_v_shape(self) -> None:
        p = make_poset(("a", "b", "c"), (("a", "b"), ("a", "c")))
        r = lower_closure(LowerClosureRequest(poset=p, subset=("b",)))
        assert set(r.closure) == {"a", "b"}
        r = lower_closure(LowerClosureRequest(poset=p, subset=("c",)))
        assert set(r.closure) == {"a", "c"}

    def test_multiple_elements(self) -> None:
        p = make_poset(("a", "b", "c"), (("a", "b"), ("a", "c")))
        r = lower_closure(LowerClosureRequest(poset=p, subset=("b", "c")))
        assert set(r.closure) == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# Upper closure
# ---------------------------------------------------------------------------


class TestUpperClosure:
    def test_chain_bottom_element(self) -> None:
        p = make_poset(("a", "b", "c"), (("a", "b"), ("b", "c")))
        r = upper_closure(UpperClosureRequest(poset=p, subset=("a",)))
        assert set(r.closure) == {"a", "b", "c"}

    def test_chain_top_element(self) -> None:
        p = make_poset(("a", "b", "c"), (("a", "b"), ("b", "c")))
        r = upper_closure(UpperClosureRequest(poset=p, subset=("c",)))
        assert set(r.closure) == {"c"}

    def test_antichain(self) -> None:
        p = make_poset(("a", "b", "c"), ())
        r = upper_closure(UpperClosureRequest(poset=p, subset=("a",)))
        assert set(r.closure) == {"a"}


# ---------------------------------------------------------------------------
# Dual
# ---------------------------------------------------------------------------


class TestDual:
    def test_chain_reverses_order(self) -> None:
        p = make_poset(("a", "b", "c"), (("a", "b"), ("b", "c")))
        d = dual_poset(DualPosetRequest(poset=p))
        dual_pairs = {(x.lower, x.upper) for x in d.poset.strict_order_pairs}
        assert ("b", "a") in dual_pairs
        assert ("c", "b") in dual_pairs
        assert ("c", "a") in dual_pairs

    def test_antichain_unchanged(self) -> None:
        p = make_poset(("a", "b"), ())
        d = dual_poset(DualPosetRequest(poset=p))
        assert len(d.poset.strict_order_pairs) == 0

    def test_dual_swaps_minimal_maximal(self) -> None:
        p = make_poset(("a", "b", "c"), (("a", "b"), ("b", "c")))
        d = dual_poset(DualPosetRequest(poset=p))
        assert set(d.poset.minimal_elements) == set(p.maximal_elements)
        assert set(d.poset.maximal_elements) == set(p.minimal_elements)


# ---------------------------------------------------------------------------
# Induced subposet
# ---------------------------------------------------------------------------


class TestInducedSubposet:
    def test_chain_subposet(self) -> None:
        p = make_poset(("a", "b", "c"), (("a", "b"), ("b", "c")))
        r = induced_subposet(InducedSubposetRequest(poset=p, subset=("a", "b")))
        assert set(r.subposet.elements) == {"a", "b"}
        assert len(r.subposet.strict_order_pairs) == 1

    def test_single_element(self) -> None:
        p = make_poset(("a", "b", "c"), (("a", "b"), ("b", "c")))
        r = induced_subposet(InducedSubposetRequest(poset=p, subset=("b",)))
        assert set(r.subposet.elements) == {"b"}
        assert len(r.subposet.strict_order_pairs) == 0

    def test_antichain_subposet(self) -> None:
        p = make_poset(("a", "b", "c"), ())
        r = induced_subposet(InducedSubposetRequest(poset=p, subset=("a", "b")))
        assert set(r.subposet.elements) == {"a", "b"}
        assert len(r.subposet.strict_order_pairs) == 0

    def test_v_shape_preserves_structure(self) -> None:
        p = make_poset(
            ("a", "b", "c", "d"), (("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"))
        )
        r = induced_subposet(InducedSubposetRequest(poset=p, subset=("a", "b", "c")))
        assert set(r.subposet.elements) == {"a", "b", "c"}
        pairs = {(x.lower, x.upper) for x in r.subposet.strict_order_pairs}
        assert ("a", "b") in pairs
        assert ("a", "c") in pairs
        assert ("b", "c") not in pairs

    def test_non_convex_subset(self) -> None:
        """Subposet skipping an intermediate element must recompute the cover."""
        # Chain a < b < c; restrict to {a, c}, skipping b.
        p = make_poset(("a", "b", "c"), (("a", "b"), ("b", "c")))
        r = induced_subposet(InducedSubposetRequest(poset=p, subset=("a", "c")))
        assert set(r.subposet.elements) == {"a", "c"}
        pairs = {(x.lower, x.upper) for x in r.subposet.strict_order_pairs}
        assert ("a", "c") in pairs
        # The cover should be {a, c}, not empty
        covers = {(x.lower, x.upper) for x in r.subposet.cover_relations}
        assert ("a", "c") in covers
