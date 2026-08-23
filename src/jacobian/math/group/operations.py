"""Exact finite group operations backed by SymPy combinatorics."""

from __future__ import annotations

from typing import Any

from jacobian.math.group._models import PermutationGroupRequest, SubgroupEntry

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


class SubgroupLatticeBudgetExceededError(ValueError):
    """The lattice traversal exhausted its declared closure-construction budget."""


# The traversal performs one closure construction per (discovered subgroup,
# outside element) pair. Over an admitted source this is at most
# |subgroups| x |G| <= 2825 x 63 ~= 178k pairs; the extremal admitted group
# C2^6 needs 154,413. The budget covers every admitted request and fails
# closed rather than letting a frontier run without a mathematical bound.
MAX_SUBGROUP_LATTICE_CLOSURES = 200_000


def _require_admitted_lattice_source(group: PermutationGroupRequest) -> Any:
    """Return the bounded backend permutation group for an admitted source."""
    from jacobian.math.group._models import MAX_SUBGROUP_LATTICE_GROUP_ORDER

    backend_group = _backend_group(group)
    if int(backend_group.order()) > MAX_SUBGROUP_LATTICE_GROUP_ORDER:
        raise ValueError(
            "subgroup lattice computation is bounded to groups of order "
            f"at most {MAX_SUBGROUP_LATTICE_GROUP_ORDER}"
        )
    return backend_group


def _canonical_form(permutation: Any, degree: int) -> tuple[int, ...]:
    form = list(permutation.array_form)
    # array_form truncates trailing fixed points; restore them exactly.
    form.extend(range(len(form), degree))
    return tuple(form)


def _element_table(
    group: Any, degree: int
) -> tuple[list[tuple[int, ...]], list[list[int]]]:
    """Index every element once with its full multiplication table.

    At most 64x64 exact backend products; every later closure step is then a
    table lookup instead of a backend construction.
    """
    from sympy.combinatorics import Permutation

    elements = sorted({_canonical_form(element, degree) for element in group.elements})
    element_index = {form: position for position, form in enumerate(elements)}
    index_perms = [Permutation(list(form)) for form in elements]
    mul = [
        [
            element_index[_canonical_form(index_perms[i] * index_perms[j], degree)]
            for j in range(len(elements))
        ]
        for i in range(len(elements))
    ]
    return elements, mul


def _extend_by_element(
    members: list[int], candidate: int, mask: int, mul: list[list[int]]
) -> tuple[int, list[int]]:
    """Return the element set generated by ``members`` plus ``candidate``."""
    seen = mask | (1 << candidate)
    generated = [*members, candidate]
    i = 0
    while i < len(generated):
        row = mul[generated[i]]
        j = 0
        while j < len(generated):
            product = row[generated[j]]
            if not seen >> product & 1:
                seen |= 1 << product
                generated.append(product)
            j += 1
        i += 1
    return seen, generated


def subgroup_lattice(group: PermutationGroupRequest) -> list[SubgroupEntry]:
    """Return every subgroup of a permutation group as canonical entries.

    Each entry retains its subgroup as the canonical permutation-group value
    plus its exact order, so an enumerated subgroup passes unchanged to
    ``group_order``, ``group_orbit``, a chained stabilizer request, or any
    other permutation-group consumer. The trivial subgroup is represented by
    the identity generator ``[0,...,degree-1]`` so every entry value stays
    consumable without synthesizing a generator. The lattice is complete and
    canonically ordered: entries are sorted by order and then by generators,
    so equal groups serialize identically. The traversal is bounded to groups
    of order at most 64 and counts its closure constructions against
    ``MAX_SUBGROUP_LATTICE_CLOSURES``, so the search work is derived from the
    subgroup/search-node count instead of only the admitted group order;
    exhausting the budget raises :class:`SubgroupLatticeBudgetExceededError`,
    which the wire operation reports as a typed ``LIMIT_EXCEEDED`` outcome.
    """
    degree = group.degree
    backend_group = _require_admitted_lattice_source(group)
    elements, mul = _element_table(backend_group, degree)
    element_count = len(elements)

    identity_position = elements.index(tuple(range(degree)))
    trivial_mask = 1 << identity_position
    known_masks = {trivial_mask}
    # Every discovered subgroup extends its discoverer's chain by one
    # element, and each accepted extension at least doubles the generated
    # subgroup (Lagrange), so chains stay within log2(order) <= 6 links.
    identity_generators = (tuple(elements[identity_position]),)
    lattice: list[tuple[tuple[tuple[int, ...], ...], int]] = [(identity_generators, 1)]
    frontier: list[tuple[int, tuple[int, ...]]] = [(trivial_mask, ())]
    closures = 0
    while frontier:
        mask, chain = frontier.pop()
        members = [
            position for position in range(element_count) if mask >> position & 1
        ]
        for candidate in range(element_count):
            if mask >> candidate & 1:
                continue
            closures += 1
            if closures > MAX_SUBGROUP_LATTICE_CLOSURES:
                raise SubgroupLatticeBudgetExceededError(
                    f"lattice traversal exceeded {MAX_SUBGROUP_LATTICE_CLOSURES} "
                    "closure constructions"
                )
            seen, generated = _extend_by_element(members, candidate, mask, mul)
            if seen not in known_masks:
                known_masks.add(seen)
                child_chain = (*chain, candidate)
                if child_chain:
                    generators_out = tuple(
                        sorted(elements[position] for position in child_chain)
                    )
                else:
                    generators_out = identity_generators
                lattice.append((generators_out, len(generated)))
                frontier.append((seen, child_chain))
    # Canonical entry order keeps serialized lattices hash-seed independent.
    lattice.sort(key=lambda entry: (entry[1], entry[0]))
    return [
        SubgroupEntry(
            group=PermutationGroupRequest(degree=degree, generators=generators),
            order=order,
        )
        for generators, order in lattice
    ]
