"""Exact native kernels over finite greedoids and antimatroids.

All functions are deterministic and complete for accepted values. They need
no ``UNKNOWN``, timeout-as-mathematics, search budget, or solver outcome.
Every function accepting a ``FiniteFeasibleSetSystem`` enforces the same
bounded carrier envelope as its MCP request contract before any kernel
expansion.
"""

from __future__ import annotations

from jacobian.math.greedoids._models import require_bounded_carrier
from jacobian.math.greedoids.values import FiniteFeasibleSetSystem

__all__ = [
    "antimatroid_to_convex_geometry",
    "bases",
    "basic_word_profile",
    "convex_geometry_to_antimatroid",
    "feasible_continuations",
    "rank",
    "recognize",
    "union_closed",
]


def _is_feasible(system: FiniteFeasibleSetSystem, subset: frozenset[int]) -> bool:
    return tuple(sorted(subset)) in system.feasible_index()


def _feasible_sets(system: FiniteFeasibleSetSystem) -> list[frozenset[int]]:
    return [frozenset(row) for row in system.feasible]


def _accessibility_obstruction(
    feasible_rows: list[tuple[int, ...]],
    index: dict[tuple[int, ...], int],
) -> tuple[int, ...] | None:
    """Return the first nonempty feasible set with no removable element."""
    for row in sorted(feasible_rows, key=lambda r: (len(r), r)):
        if not row:
            continue
        fs = frozenset(row)
        removable = any(tuple(sorted(fs - {elem})) in index for elem in sorted(fs))
        if not removable:
            return tuple(sorted(fs))
    return None


def _exchange_obstruction(
    feasible_sets: list[frozenset[int]],
    index: dict[tuple[int, ...], int],
) -> dict[str, object] | None:
    """Return the first exchange violation as a result dict, or None."""
    by_size: dict[int, list[frozenset[int]]] = {}
    for fs in feasible_sets:
        by_size.setdefault(len(fs), []).append(fs)
    sizes = sorted(by_size)
    for large_size in reversed(sizes):
        for small_size in sizes:
            if large_size <= small_size:
                continue
            for x_set in by_size[large_size]:
                for y_set in by_size[small_size]:
                    augmenting = [
                        elem
                        for elem in sorted(x_set - y_set)
                        if tuple(sorted(y_set | {elem})) in index
                    ]
                    if not augmenting:
                        return {
                            "status": "NOT_A_GREEDOID",
                            "obstruction": "exchange_violation",
                            "larger_set": tuple(sorted(x_set)),
                            "smaller_set": tuple(sorted(y_set)),
                        }
    return None


def recognize(
    system: FiniteFeasibleSetSystem,
) -> dict[str, object]:
    """Return ``GREEDOID`` with rank and bases, or ``NOT_A_GREEDOID`` with the
    first exact obstruction under deterministic order.

    For accepted finite sizes, exhaust every licensed feasible set/pair. A
    sample of exchange pairs cannot return ``GREEDOID``.
    """
    require_bounded_carrier(system)
    index = system.feasible_index()
    n = len(system.ground)
    if () not in index:
        return {"status": "NOT_A_GREEDOID", "obstruction": "missing_empty_set"}
    access = _accessibility_obstruction(list(system.feasible), index)
    if access is not None:
        return {
            "status": "NOT_A_GREEDOID",
            "obstruction": "inaccessible_feasible_set",
            "feasible_set": access,
        }
    feasible_sets = _feasible_sets(system)
    exch = _exchange_obstruction(feasible_sets, index)
    if exch is not None:
        return exch
    r, basis_list = _bases_unchecked(system, None)
    return {
        "status": "GREEDOID",
        "rank": r,
        "bases": [tuple(sorted(b)) for b in basis_list],
        "ground_size": n,
    }


def _require_greedoid(system: FiniteFeasibleSetSystem) -> None:
    """Reject a structural family that has not established the greedoid axioms."""
    recognized = recognize(system)
    if recognized["status"] != "GREEDOID":
        raise ValueError(
            "a greedoid operation requires a recognized greedoid "
            f"(first obstruction: {recognized['obstruction']})"
        )


def _require_antimatroid(system: FiniteFeasibleSetSystem) -> None:
    """Reject a greedoid that has not established the antimatroid axioms."""
    _require_greedoid(system)
    full_ground = tuple(range(len(system.ground)))
    if full_ground not in system.feasible_index() or not union_closed(system):
        raise ValueError(
            "an antimatroid operation requires full support and union closure"
        )


def union_closed(system: FiniteFeasibleSetSystem) -> bool:
    """Return whether the feasible family is closed under pairwise union."""
    require_bounded_carrier(system)
    feasible_sets = _feasible_sets(system)
    index = system.feasible_index()
    for i, a in enumerate(feasible_sets):
        for b in feasible_sets[i + 1 :]:
            if tuple(sorted(a | b)) not in index:
                return False
    return True


def rank(system: FiniteFeasibleSetSystem, subset: frozenset[int] | None = None) -> int:
    """Return ``r(X) = max{|F| : F feasible and F ⊆ X}``.

    If ``subset`` is ``None``, the rank of the whole greedoid (the common size
    of its bases) is returned.
    """
    _require_greedoid(system)
    return _rank_unchecked(system, subset)


def _rank_unchecked(
    system: FiniteFeasibleSetSystem, subset: frozenset[int] | None
) -> int:
    """Compute rank after the greedoid axioms have been established."""
    if subset is None:
        candidates = _feasible_sets(system)
    else:
        candidates = [fs for fs in _feasible_sets(system) if fs <= subset]
    return max((len(fs) for fs in candidates), default=0)


def bases(
    system: FiniteFeasibleSetSystem, subset: frozenset[int] | None = None
) -> tuple[int, list[frozenset[int]]]:
    """Return ``(rank, basis_list)`` for the supplied ground subset.

    A basis of ``X`` is a maximal feasible subset of ``X``. All bases have the
    same cardinality under the greedoid theorem convention.
    """
    _require_greedoid(system)
    return _bases_unchecked(system, subset)


def _bases_unchecked(
    system: FiniteFeasibleSetSystem, subset: frozenset[int] | None
) -> tuple[int, list[frozenset[int]]]:
    """Compute bases after the greedoid axioms have been established."""
    if subset is None:
        subset = frozenset(range(len(system.ground)))
    feasible_in_subset = [fs for fs in _feasible_sets(system) if fs <= subset]
    if not feasible_in_subset:
        return 0, []
    # Bases are inclusion-maximal feasible sets (not just max-cardinality).
    basis_list = [
        fs
        for fs in feasible_in_subset
        if not any(other > fs for other in feasible_in_subset)
    ]
    max_size = max(len(fs) for fs in basis_list) if basis_list else 0
    return max_size, basis_list


def feasible_continuations(
    system: FiniteFeasibleSetSystem, feasible_set: frozenset[int]
) -> list[int]:
    """Return ``Gamma(X) = {e in E\\X : X union {e} in F}`` for a feasible ``X``."""
    _require_greedoid(system)
    index = system.feasible_index()
    if tuple(sorted(feasible_set)) not in index:
        raise ValueError("input set must be feasible")
    n = len(system.ground)
    return [
        e
        for e in range(n)
        if e not in feasible_set and tuple(sorted(feasible_set | {e})) in index
    ]


def basic_word_profile(
    system: FiniteFeasibleSetSystem, word: tuple[int, ...]
) -> dict[str, object]:
    """Return whether ``word`` is a basic word.

    A basic word is a finite sequence of distinct ground elements such that
    every prefix set is feasible. A full basic word has length ``r(E)`` and its
    underlying set is a greedoid basis.
    """
    _require_greedoid(system)
    return _basic_word_profile_unchecked(system, word)


def _basic_word_profile_unchecked(
    system: FiniteFeasibleSetSystem, word: tuple[int, ...]
) -> dict[str, object]:
    """Profile a basic word after the greedoid axioms have been established."""
    n = len(system.ground)
    seen: set[int] = set()
    for elem in word:
        if not 0 <= elem < n:
            return {
                "status": "NOT_A_BASIC_WORD",
                "obstruction": "foreign_element",
                "prefix_index": len(seen),
            }
        if elem in seen:
            return {
                "status": "NOT_A_BASIC_WORD",
                "obstruction": "repeated_element",
                "prefix_index": len(seen),
            }
        seen.add(elem)
    index = system.feasible_index()
    prefix: frozenset[int] = frozenset()
    for i, elem in enumerate(word):
        prefix = prefix | {elem}
        if tuple(sorted(prefix)) not in index:
            return {
                "status": "NOT_A_BASIC_WORD",
                "obstruction": "infeasible_prefix",
                "prefix_index": i,
                "prefix_set": tuple(sorted(prefix)),
            }
    r, _ = _bases_unchecked(system, None)
    is_full = len(word) == r
    return {
        "status": "BASIC_WORD",
        "prefix_length": len(word),
        "is_full": is_full,
        "rank": r,
    }


def antimatroid_to_convex_geometry(
    system: FiniteFeasibleSetSystem,
) -> tuple[list[tuple[int, ...]], dict[tuple[int, ...], tuple[int, ...]]]:
    """Return the complementary closed-set family of a full-support antimatroid.

    The complementary family ``C = {E\\F : F in F}`` is an intersection-closed
    finite closure system satisfying anti-exchange. Returns the closed-set
    family (sorted tuples of ground indices) and the feasible->closed
    complement map.
    """
    _require_antimatroid(system)
    return _antimatroid_to_convex_geometry_unchecked(system)


def _antimatroid_to_convex_geometry_unchecked(
    system: FiniteFeasibleSetSystem,
) -> tuple[list[tuple[int, ...]], dict[tuple[int, ...], tuple[int, ...]]]:
    """Project an already-established antimatroid to its closed family."""
    n = len(system.ground)
    ground_set = frozenset(range(n))
    complement_map: dict[tuple[int, ...], tuple[int, ...]] = {}
    closed_family: list[tuple[int, ...]] = []
    for feasible in system.feasible:
        fs = frozenset(feasible)
        closed = ground_set - fs
        closed_tuple = tuple(sorted(closed))
        complement_map[feasible] = closed_tuple
        closed_family.append(closed_tuple)
    return closed_family, complement_map


def convex_geometry_to_antimatroid(
    closed_family: list[tuple[int, ...]], n: int
) -> list[tuple[int, ...]]:
    """Return the complementary feasible family of a convex geometry.

    ``n`` is the ground-set size; each closed set in ``closed_family`` is a
    sorted tuple of ground indices. The feasible family is ``{E\\C : C in F}``.
    """
    ground_set = frozenset(range(n))
    return [tuple(sorted(ground_set - frozenset(c))) for c in closed_family]
