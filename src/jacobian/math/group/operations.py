"""Exact finite group operations backed by SymPy combinatorics."""

from __future__ import annotations

from typing import Any

from jacobian.math.group._models import PermutationGroupRequest

__all__ = [
    "element_order",
    "group_conjugacy_classes",
    "group_orbit",
    "group_order",
    "group_stabilizer",
    "subgroup_lattice",
]


def _backend_group(group: PermutationGroupRequest) -> Any:
    from sympy.combinatorics import Permutation, PermutationGroup

    return PermutationGroup(
        [Permutation(list(generator)) for generator in group.generators]
    )


def group_order(group: PermutationGroupRequest) -> int:
    """Return the exact order of a permutation group via Schreier-Sims."""
    return int(_backend_group(group).order())


def element_order(degree: int, generator: list[int]) -> int:
    """Return the exact order of one permutation."""
    from sympy.combinatorics import Permutation

    if len(generator) != degree or sorted(generator) != list(range(degree)):
        raise ValueError("generator must be a permutation of 0..n-1")
    return int(Permutation(list(generator)).order())


def group_orbit(group: PermutationGroupRequest, point: int) -> list[int]:
    """Return the sorted orbit of a point under a permutation group."""
    if not 0 <= point < group.degree:
        raise ValueError("point must be in 0..n-1")
    return sorted(_backend_group(group).orbit(point))


def _full_generator(perm: Any, degree: int) -> tuple[int, ...]:
    form = list(perm.array_form)
    # array_form truncates trailing fixed points; restore them exactly.
    form.extend(range(len(form), degree))
    return tuple(form)


def group_conjugacy_classes(
    degree: int, generators: list[list[int]]
) -> list[list[list[int]]]:
    """Return conjugacy classes as lists of permutation array forms.

    Two elements are conjugate iff they lie in the same class.  The returned
    classes partition the group; each class is a list of permutations (as
    array forms over ``0..n-1``). The result is canonically ordered: members
    of each class are sorted lexicographically and classes are sorted by
    their smallest member, so equal groups serialize identically. The
    generated group must have order at most 5000 (degree up to 64 alone
    does not bound enumeration; e.g., S8 has order 40320); larger groups
    are rejected before enumeration.
    """
    from sympy.combinatorics import Permutation, PermutationGroup

    if not 1 <= degree <= 64:
        raise ValueError("group degree must be between 1 and 64")
    if not generators:
        raise ValueError("at least one generator is required")
    perms = []
    for perm in generators:
        if len(perm) != degree or sorted(perm) != list(range(degree)):
            raise ValueError("each generator must be a permutation of 0..n-1")
        perms.append(Permutation(list(perm)))
    group = PermutationGroup(perms)
    # Bound enumeration by group order before materializing all |G| elements.
    # S12 has order 479M and would exhaust memory; reject conservatively.
    from jacobian.math.group._models import MAX_CONJUGACY_CLASSES_GROUP_ORDER

    order = int(group.order())
    if order > MAX_CONJUGACY_CLASSES_GROUP_ORDER:
        raise ValueError(
            f"group order {order} exceeds the bounded maximum "
            f"{MAX_CONJUGACY_CLASSES_GROUP_ORDER} for conjugacy classes "
            f"(would materialize |G|={order} elements)"
        )
    classes = group.conjugacy_classes()
    canonical = [sorted(list(p.array_form) for p in cls) for cls in classes]
    canonical.sort(key=lambda cls: tuple(cls[0]))
    return canonical


def group_stabilizer(
    group: PermutationGroupRequest, point: int
) -> PermutationGroupRequest:
    """Return the point stabilizer as the canonical permutation-group value.

    The stabilizer of ``point`` is the subgroup of elements fixing ``point``.
    By the orbit-stabilizer theorem,
    ``|G| = |orbit(point)| * |stabilizer(point)|``. The trivial stabilizer is
    represented by the identity generator ``[0,...,degree-1]`` so the value is
    consumable unchanged by every permutation-group consumer, including
    ``group_order`` and a chained stabilizer request.
    """
    if not 0 <= point < group.degree:
        raise ValueError("point must be in 0..n-1")
    stabilizer = _backend_group(group).stabilizer(point)
    generators = tuple(
        _full_generator(generator, group.degree) for generator in stabilizer.generators
    )
    # The canonical group value requires at least one generator; represent the
    # trivial stabilizer by the identity permutation.
    if not generators:
        generators = (tuple(range(group.degree)),)
    return PermutationGroupRequest(degree=group.degree, generators=generators)


def subgroup_lattice(
    degree: int, generators: list[list[int]]
) -> list[tuple[list[list[int]], int]]:
    """Return all subgroups of a permutation group.

    Returns a list of (generators_of_subgroup, subgroup_order) tuples.
    The traversal is exponential in the generated group's order, so it is
    bounded to groups of order at most 64.
    """
    from sympy.combinatorics import Permutation, PermutationGroup

    from jacobian.math.group._models import MAX_SUBGROUP_LATTICE_GROUP_ORDER

    if not 1 <= degree <= 64:
        raise ValueError("group degree must be between 1 and 64")
    if not generators:
        raise ValueError("at least one generator is required")
    perms = []
    for perm in generators:
        if len(perm) != degree or sorted(perm) != list(range(degree)):
            raise ValueError("each generator must be a permutation of 0..n-1")
        perms.append(Permutation(list(perm)))
    group = PermutationGroup(perms)
    if int(group.order()) > MAX_SUBGROUP_LATTICE_GROUP_ORDER:
        raise ValueError(
            "subgroup lattice computation is bounded to groups of order "
            f"at most {MAX_SUBGROUP_LATTICE_GROUP_ORDER}"
        )
    # Traverse distinct subgroups instead of the 2^|G|-element power set:
    # every subgroup is the closure of an existing subgroup extended by one
    # element, so a fixpoint frontier enumerates each exactly once with
    # |subgroups| x |G| closure constructions - feasible under the admitted
    # order bound.
    identity = Permutation(list(range(degree)))
    trivial_key = frozenset((tuple(identity.array_form),))
    subgroup_keys: set[frozenset[tuple[int, ...]]] = {trivial_key}
    result: list[tuple[list[list[int]], int]] = [([list(identity.array_form)], 1)]
    frontier = [trivial_key]
    elements = list(group.elements)
    while frontier:
        current = frontier.pop()
        [Permutation(list(member)) for member in current]
        for element in elements:
            element_form = tuple(element.array_form)
            if element_form in current:
                continue
            generated = group.subgroup(
                [Permutation(list(member)) for member in current] + [element]
            )
            key = frozenset(tuple(p.array_form) for p in generated.elements)
            if key not in subgroup_keys:
                subgroup_keys.add(key)
                result.append(
                    (
                        sorted(
                            (list(p.array_form) for p in generated.generators),
                        ),
                        int(generated.order()),
                    )
                )
                frontier.append(key)
    # Canonical entry order keeps serialized lattices hash-seed independent.
    result.sort(key=lambda entry: (entry[1], entry[0]))
    return result
