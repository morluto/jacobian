"""Exact bounded finite semigroup operations."""

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.finite_semigroups._models import (
    AdjoinIdentityResult,
    AdjoinZeroResult,
    ElementPowerResult,
    FiniteSemigroup,
    GeneratedSubsemigroupResult,
    GreenRelationsResult,
    IdealEnumerationResult,
    IdempotentsResult,
    KaroubiProjectionResult,
    LocalStructureResult,
    NilpotentElementsResult,
    OppositeResult,
    PowerProfileResult,
    PrincipalIdealsResult,
    ProductResult,
    ReesQuotientResult,
    RegularElementsResult,
)


def _require_associative(semigroup: FiniteSemigroup) -> None:
    """Establish the semigroup law once for a native operation."""

    from pydantic_core import PydanticCustomError

    try:
        semigroup._check_associativity()
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("semigroup",), code=exc.type, message=exc.message()
        ) from exc


def _require_declared(
    declared: tuple[str, ...],
    requested: tuple[str, ...],
    *,
    field: str,
    code: str,
    message: str,
) -> None:
    if any(element not in declared for element in requested):
        raise OperationDomainValidationError(
            location=(field,),
            code=code,
            message=message,
        )


def _element_power(
    elements: tuple[str, ...],
    multiplication: tuple[tuple[str, ...], ...],
    element: str,
    exponent: int,
) -> str:
    """Compute ``element^exponent`` using its bounded eventual period.

    ``exponent`` must be at least 1; the semigroup may have no identity, so
    ``a^0`` is undefined and rejected at the request boundary.
    """

    powers, index, period, _, _ = _power_profile_data(elements, multiplication, element)
    if exponent < index:
        return powers[exponent - 1]
    return powers[index - 1 + (exponent - index) % period]


def _idempotents(
    elements: tuple[str, ...],
    multiplication: tuple[tuple[str, ...], ...],
) -> tuple[str, ...]:
    """Return every element ``e`` with ``e*e = e`` in declared order."""

    idx = {label: i for i, label in enumerate(elements)}
    return tuple(e for e in elements if multiplication[idx[e]][idx[e]] == e)


def _principal_ideals(
    elements: tuple[str, ...],
    multiplication: tuple[tuple[str, ...], ...],
    requested: tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    """Return the principal ideal of each requested element.

    The principal two-sided ideal of ``a`` is ``S^1 a S^1``. Ideals are
    returned in declared element order with each ideal's elements in declared
    semigroup order.
    """

    idx = {label: i for i, label in enumerate(elements)}
    n = len(elements)
    ideals: list[tuple[str, ...]] = []
    for element in requested:
        ideal = {element}
        i = idx[element]
        for left in range(n):
            left_product = multiplication[left][i]
            ideal.add(left_product)
            ideal.add(multiplication[i][left])
            for right in range(n):
                ideal.add(multiplication[idx[left_product]][right])
        ideals.append(tuple(e for e in elements if e in ideal))
    return tuple(ideals)


def _power_profile_data(
    elements: tuple[str, ...],
    multiplication: tuple[tuple[str, ...], ...],
    element: str,
) -> tuple[tuple[str, ...], int, int, str, tuple[str, ...]]:
    """Compute ``(powers, index, period, idempotent, cyclic)`` for one element.

    ``powers[i]`` is ``a^(i+1)`` (1-based exponents).  ``index`` is the
    smallest positive exponent whose power first repeats, ``period`` the
    cycle length, and ``idempotent`` the unique idempotent ``a^k`` with
    ``k >= index`` and ``k ≡ 0 (mod period)``.
    """

    idx = {label: i for i, label in enumerate(elements)}
    a = element
    powers: list[str] = [a]
    seen: dict[str, int] = {a: 0}

    current = a
    zero_based_index = 0
    period = 1
    while True:
        next_power = multiplication[idx[current]][idx[a]]
        if next_power in seen:
            zero_based_index = seen[next_power]
            period = len(powers) - zero_based_index
            break
        seen[next_power] = len(powers)
        powers.append(next_power)
        current = next_power

    index = zero_based_index + 1
    k = index + ((-index) % period)
    idempotent = powers[k - 1]
    cyclic = tuple(powers)
    return tuple(powers), index, period, idempotent, cyclic


def power_profile(semigroup: "FiniteSemigroup", element: str) -> PowerProfileResult:
    """Compute the power profile of one element in a finite semigroup."""

    _require_associative(semigroup)
    _require_declared(
        semigroup.elements,
        (element,),
        field="element",
        code="finite_semigroup.element_not_in_semigroup",
        message="element must be in the semigroup",
    )
    powers, index, period, idempotent, cyclic = _power_profile_data(
        semigroup.elements,
        semigroup.multiplication,
        element,
    )
    return PowerProfileResult._from_kernel(
        semigroup, element, powers, index, period, idempotent, cyclic
    )


def generated_subsemigroup(
    semigroup: "FiniteSemigroup", generators: tuple[str, ...]
) -> GeneratedSubsemigroupResult:
    """Compute the subsemigroup generated by a set of elements."""

    _require_associative(semigroup)
    if not generators:
        raise OperationDomainValidationError(
            location=("generators",),
            code="finite_semigroup.generators_empty",
            message="at least one generator is required",
        )
    elements = semigroup.elements
    _require_declared(
        elements,
        generators,
        field="generators",
        code="finite_semigroup.generator_not_in_semigroup",
        message="every generator must be in the semigroup",
    )
    mult = semigroup.multiplication
    idx = {label: i for i, label in enumerate(elements)}

    generated = set(generators)
    changed = True
    while changed:
        changed = False
        current = list(generated)
        for a in current:
            for b in current:
                product = mult[idx[a]][idx[b]]
                if product not in generated:
                    generated.add(product)
                    changed = True

    # Sort by original element ordering
    result_elements = tuple(e for e in elements if e in generated)
    return GeneratedSubsemigroupResult._from_kernel(
        semigroup=semigroup,
        generators=generators,
        elements=result_elements,
    )


def verify_generated_subsemigroup(claim: GeneratedSubsemigroupResult) -> bool:
    """Verify closure and generation against the retained semigroup source."""

    try:
        return generated_subsemigroup(claim.semigroup, claim.generators) == claim
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


def element_power(
    semigroup: "FiniteSemigroup", element: str, exponent: int
) -> ElementPowerResult:
    """Compute ``element^exponent`` in a finite semigroup."""

    _require_associative(semigroup)
    _require_declared(
        semigroup.elements,
        (element,),
        field="element",
        code="finite_semigroup.element_not_in_semigroup",
        message="element must be in the semigroup",
    )
    if type(exponent) is not int or exponent < 1:
        raise OperationDomainValidationError(
            location=("exponent",),
            code="finite_semigroup.exponent_positive",
            message="exponent must be a positive integer",
        )
    power = _element_power(
        semigroup.elements, semigroup.multiplication, element, exponent
    )
    return ElementPowerResult._from_kernel(semigroup, element, exponent, power)


def idempotents(semigroup: "FiniteSemigroup") -> IdempotentsResult:
    """Find every idempotent element ``e`` with ``e*e = e``."""

    _require_associative(semigroup)
    values = _idempotents(semigroup.elements, semigroup.multiplication)
    return IdempotentsResult._from_kernel(semigroup, values)


def regular_elements(semigroup: "FiniteSemigroup") -> RegularElementsResult:
    """Return all ``a`` for which some ``x`` satisfies ``a*x*a = a``."""

    _require_associative(semigroup)
    index = {label: position for position, label in enumerate(semigroup.elements)}
    regular = []
    for a in semigroup.elements:
        ai = index[a]
        for x in semigroup.elements:
            if (
                semigroup.multiplication[index[semigroup.multiplication[ai][index[x]]]][
                    ai
                ]
                == a
            ):
                regular.append((a, x))
                break
    return RegularElementsResult._from_kernel(semigroup, tuple(regular))


def nilpotent_elements(
    semigroup: "FiniteSemigroup", zero: str
) -> NilpotentElementsResult:
    """Return all elements with a positive power equal to an absorbing zero."""

    _require_associative(semigroup)
    _require_declared(
        semigroup.elements,
        (zero,),
        field="zero",
        code="finite_semigroup.zero_not_in_semigroup",
        message="zero must be an element of the semigroup",
    )
    index = {label: position for position, label in enumerate(semigroup.elements)}
    if any(
        semigroup.multiplication[index[zero]][position] != zero
        or semigroup.multiplication[position][index[zero]] != zero
        for position in range(len(semigroup.elements))
    ):
        raise OperationDomainValidationError(
            location=("zero",),
            code="finite_semigroup.not_absorbing_zero",
            message="zero must absorb multiplication on both sides",
        )
    values = []
    for element in semigroup.elements:
        powers, _, _, _, _ = _power_profile_data(
            semigroup.elements, semigroup.multiplication, element
        )
        for exponent, power in enumerate(powers, start=1):
            if power == zero:
                values.append((element, exponent))
                break
    return NilpotentElementsResult._from_kernel(semigroup, zero, tuple(values))


def principal_ideals(
    semigroup: "FiniteSemigroup", elements: tuple[str, ...]
) -> PrincipalIdealsResult:
    """Compute the principal ideal of each requested element."""

    _require_associative(semigroup)
    declared = semigroup.elements
    _require_declared(
        declared,
        elements,
        field="elements",
        code="finite_semigroup.element_not_in_semigroup",
        message="every element must be in the semigroup",
    )
    if len(set(elements)) != len(elements):
        raise OperationDomainValidationError(
            location=("elements",),
            code="finite_semigroup.requested_elements_not_distinct",
            message="requested elements must be distinct",
        )
    if elements != tuple(element for element in declared if element in elements):
        raise OperationDomainValidationError(
            location=("elements",),
            code="finite_semigroup.requested_elements_wrong_order",
            message="requested elements must use declared semigroup order",
        )
    ideals = _principal_ideals(
        semigroup.elements,
        semigroup.multiplication,
        elements,
    )
    return PrincipalIdealsResult._from_kernel(semigroup, elements, ideals)


def _left_ideals(
    elements: tuple[str, ...],
    multiplication: tuple[tuple[str, ...], ...],
) -> list[frozenset[str]]:
    """Compute the principal left ideal S^1 a of each element."""

    n = len(elements)
    ideals: list[frozenset[str]] = []
    for i in range(n):
        ideal = {elements[i]}
        for j in range(n):
            ideal.add(multiplication[j][i])
        ideals.append(frozenset(ideal))
    return ideals


def _right_ideals(
    elements: tuple[str, ...],
    multiplication: tuple[tuple[str, ...], ...],
) -> list[frozenset[str]]:
    """Compute the principal right ideal a S^1 of each element."""

    n = len(elements)
    ideals: list[frozenset[str]] = []
    for i in range(n):
        ideal = {elements[i]}
        for j in range(n):
            ideal.add(multiplication[i][j])
        ideals.append(frozenset(ideal))
    return ideals


def _two_sided_ideals(
    elements: tuple[str, ...],
    multiplication: tuple[tuple[str, ...], ...],
) -> list[frozenset[str]]:
    """Compute the principal two-sided ideal S^1 a S^1 of each element."""

    idx = {label: i for i, label in enumerate(elements)}
    n = len(elements)
    ideals: list[frozenset[str]] = []
    for i in range(n):
        ideal = {elements[i]}
        for j in range(n):
            ideal.add(multiplication[j][i])
            ideal.add(multiplication[i][j])
            for k in range(n):
                ideal.add(multiplication[j][idx[multiplication[i][k]]])
        ideals.append(frozenset(ideal))
    return ideals


def _partition_from_ideals(
    elements: tuple[str, ...],
    ideals: list[frozenset[str]],
) -> tuple[tuple[str, ...], ...]:
    """Group elements by equality of their principal ideals.

    Returns a tuple of equivalence-class tuples in declared element order.
    """

    groups: list[list[str]] = []
    assigned: list[bool] = [False] * len(elements)
    for i in range(len(elements)):
        if assigned[i]:
            continue
        group = [elements[i]]
        assigned[i] = True
        for j in range(i + 1, len(elements)):
            if not assigned[j] and ideals[i] == ideals[j]:
                group.append(elements[j])
                assigned[j] = True
        groups.append(group)
    return tuple(tuple(g) for g in groups)


def _green_relations(
    elements: tuple[str, ...],
    multiplication: tuple[tuple[str, ...], ...],
) -> tuple[
    tuple[tuple[str, ...], ...],
    tuple[tuple[str, ...], ...],
    tuple[tuple[str, ...], ...],
    tuple[tuple[str, ...], ...],
    tuple[tuple[str, ...], ...],
]:
    """Compute the Green relations L, R, H, D, J.

    For a finite semigroup S:
    - a L b iff S^1 a = S^1 b (principal left ideals agree)
    - a R b iff a S^1 = b S^1 (principal right ideals agree)
    - H = L ∩ R
    - J is defined by principal two-sided ideals: a J b iff S^1 a S^1 = S^1 b S^1
    - D = L ∨ R (the join), equivalently the relation whose blocks are the
      connected components of the L-R intersection graph

    Returns each as a tuple of equivalence-class tuples in declared element order.
    """  # noqa: RUF002

    left = _left_ideals(elements, multiplication)
    right = _right_ideals(elements, multiplication)
    two_sided = _two_sided_ideals(elements, multiplication)

    L_classes = _partition_from_ideals(elements, left)  # noqa: N806
    R_classes = _partition_from_ideals(elements, right)  # noqa: N806
    J_classes = _partition_from_ideals(elements, two_sided)  # noqa: N806

    # H = L ∩ R: two elements are H-related iff they are both L-related and R-related
    H_classes = _intersection_partition(elements, L_classes, R_classes)  # noqa: N806

    # D = L ∨ R: build a graph where two elements are connected if L-related or R-related  # noqa: RUF003
    # then D-classes are the connected components
    D_classes = _join_partition(elements, L_classes, R_classes)  # noqa: N806

    return L_classes, R_classes, H_classes, D_classes, J_classes


def _intersection_partition(
    elements: tuple[str, ...],
    partition_a: tuple[tuple[str, ...], ...],
    partition_b: tuple[tuple[str, ...], ...],
) -> tuple[tuple[str, ...], ...]:
    """Compute the partition that is the intersection of two partitions."""

    a_map: dict[str, int] = {}
    b_map: dict[str, int] = {}
    for i, cls in enumerate(partition_a):
        for e in cls:
            a_map[e] = i
    for i, cls in enumerate(partition_b):
        for e in cls:
            b_map[e] = i
    groups: dict[tuple[int, int], list[str]] = {}
    for e in elements:
        key = (a_map[e], b_map[e])
        groups.setdefault(key, []).append(e)
    # Return in declared element order
    seen: set[str] = set()
    result: list[tuple[str, ...]] = []
    for e in elements:
        if e in seen:
            continue
        key = (a_map[e], b_map[e])
        result.append(tuple(groups[key]))
        seen.update(groups[key])
    return tuple(result)


def _join_partition(
    elements: tuple[str, ...],
    partition_a: tuple[tuple[str, ...], ...],
    partition_b: tuple[tuple[str, ...], ...],
) -> tuple[tuple[str, ...], ...]:
    """Compute the join (least upper bound) of two partitions via union-find."""

    n = len(elements)
    idx = {e: i for i, e in enumerate(elements)}
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for cls in partition_a:
        for i in range(1, len(cls)):
            union(idx[cls[0]], idx[cls[i]])
    for cls in partition_b:
        for i in range(1, len(cls)):
            union(idx[cls[0]], idx[cls[i]])

    groups: dict[int, list[str]] = {}
    for e in elements:
        groups.setdefault(find(idx[e]), []).append(e)
    # Return in declared element order
    seen: set[str] = set()
    result: list[tuple[str, ...]] = []
    for e in elements:
        if e in seen:
            continue
        root = find(idx[e])
        result.append(tuple(groups[root]))
        seen.update(groups[root])
    return tuple(result)


def green_relations(semigroup: FiniteSemigroup) -> GreenRelationsResult:
    """Compute the Green relations of a finite semigroup."""

    _require_associative(semigroup)
    L, R, H, D, J = _green_relations(  # noqa: N806
        semigroup.elements, semigroup.multiplication
    )
    return GreenRelationsResult._from_kernel(semigroup, L, R, H, D, J)


def _find_identity(
    elements: tuple[str, ...], multiplication: tuple[tuple[str, ...], ...]
) -> str | None:
    idx = {label: i for i, label in enumerate(elements)}
    for candidate in elements:
        ci = idx[candidate]
        if all(
            multiplication[ci][j] == elements[j]
            and multiplication[j][ci] == elements[j]
            for j in range(len(elements))
        ):
            return candidate
    return None


def local_structure(semigroup: FiniteSemigroup) -> LocalStructureResult:
    """Return units, local monoids, maximal subgroups, and the minimal ideal."""

    from jacobian.math.finite_semigroups._models import LocalMonoidValue

    _require_associative(semigroup)
    elements = semigroup.elements
    multiplication = semigroup.multiplication
    idx = {label: i for i, label in enumerate(elements)}
    identity = _find_identity(elements, multiplication)
    if identity is None:
        units: tuple[str, ...] = ()
    else:
        ei = idx[identity]
        units = tuple(
            a
            for a in elements
            if any(
                multiplication[idx[a]][idx[b]] == identity
                and multiplication[idx[b]][idx[a]] == identity
                for b in elements
            )
        )
    idempotent_list = _idempotents(elements, multiplication)
    monoids: list[LocalMonoidValue] = []
    subgroups: list[tuple[str, ...]] = []
    for e in idempotent_list:
        ei = idx[e]
        # eSe carrier: {e*s*e} in declared order.
        seen_carrier: set[str] = set()
        for s in elements:
            value = multiplication[ei][idx[multiplication[idx[s]][ei]]]
            seen_carrier.add(value)
        carrier = tuple(a for a in elements if a in seen_carrier)
        # Units of the local monoid with identity e.
        subgroup = tuple(
            a
            for a in carrier
            if any(
                multiplication[idx[a]][idx[b]] == e
                and multiplication[idx[b]][idx[a]] == e
                for b in carrier
            )
        )
        monoids.append(
            LocalMonoidValue(idempotent=e, carrier=carrier, maximal_subgroup=subgroup)
        )
        subgroups.append(subgroup)
    # Minimal (kernel) ideal: smallest principal two-sided ideal.
    principals = _principal_ideals(elements, multiplication, elements)
    kernel = min(principals, key=len)
    for ideal in principals:
        if set(ideal) < set(kernel):
            kernel = ideal
    return LocalStructureResult._from_kernel(
        semigroup, identity, units, tuple(monoids), tuple(subgroups), kernel
    )


def ideal_enumeration(semigroup: FiniteSemigroup) -> IdealEnumerationResult:
    """Enumerate all two-sided ideals and subsemigroups under an honest bound."""

    _require_associative(semigroup)
    n = len(semigroup.elements)
    if n > 12:
        raise OperationResourceAdmissionError(
            location=("semigroup",),
            code="finite_semigroup.enumeration_bound",
            message="ideal/subsemigroup enumeration requires at most 12 elements",
        )
    elements = semigroup.elements
    index = {label: position for position, label in enumerate(elements)}
    table = tuple(
        tuple(index[value] for value in row) for row in semigroup.multiplication
    )
    # Work entirely in the subset lattice.  The result is still necessarily
    # exponential, but accepted subsets no longer allocate labels and sets or
    # perform dictionary lookups inside their quadratic closure tests.
    product_bits = tuple(
        tuple(1 << table[left][right] for right in range(n)) for left in range(n)
    )
    left_ideal_bits = tuple(
        sum(1 << table[left][member] for left in range(n)) for member in range(n)
    )
    right_ideal_bits = tuple(
        sum(1 << table[member][right] for right in range(n)) for member in range(n)
    )
    ideals: list[tuple[str, ...]] = []
    subsemigroups: list[tuple[str, ...]] = []
    for mask in range(1, 1 << n):
        members = tuple(index for index in range(n) if mask & (1 << index))
        closed = True
        for left in members:
            products = 0
            for right in members:
                products |= product_bits[left][right]
            if products & ~mask:
                closed = False
                break
        if closed:
            subsemigroups.append(tuple(elements[index] for index in members))
        if all(
            not ((left_ideal_bits[member] | right_ideal_bits[member]) & ~mask)
            for member in members
        ):
            ideals.append(tuple(elements[index] for index in members))
    return IdealEnumerationResult._from_kernel(
        semigroup, tuple(ideals), tuple(subsemigroups)
    )


def opposite_semigroup(semigroup: FiniteSemigroup) -> OppositeResult:
    """Return the opposite semigroup with transposed multiplication."""

    from jacobian.math.finite_semigroups._models import OppositeResult

    _require_associative(semigroup)
    n = len(semigroup.elements)
    transposed = tuple(
        tuple(semigroup.multiplication[j][i] for j in range(n)) for i in range(n)
    )
    opposite = FiniteSemigroup(elements=semigroup.elements, multiplication=transposed)
    return OppositeResult._from_kernel(semigroup, opposite)


def product_semigroup(left: FiniteSemigroup, right: FiniteSemigroup) -> ProductResult:
    """Return the direct product with componentwise multiplication."""

    from jacobian.math.finite_semigroups._models import ProductResult

    _require_associative(left)
    _require_associative(right)
    if len(left.elements) * len(right.elements) > 50:
        raise OperationResourceAdmissionError(
            location=("left", "right"),
            code="finite_semigroup.product_bound",
            message="direct product exceeds the 50-element bound",
        )
    elements: list[str] = []
    left_proj: list[tuple[str, str]] = []
    right_proj: list[tuple[str, str]] = []
    for a in left.elements:
        for b in right.elements:
            label = f"{a}\u00d7{b}"
            elements.append(label)
            left_proj.append((label, a))
            right_proj.append((label, b))
    left_idx = {label: i for i, label in enumerate(left.elements)}
    right_idx = {label: i for i, label in enumerate(right.elements)}
    # Build the table directly over the product axis.
    pairs = [(a, b) for a in left.elements for b in right.elements]
    table = tuple(
        tuple(
            (
                f"{left.multiplication[left_idx[a1]][left_idx[a2]]}"
                f"\u00d7{right.multiplication[right_idx[b1]][right_idx[b2]]}"
            )
            for (a2, b2) in pairs
        )
        for (a1, b1) in pairs
    )
    product = FiniteSemigroup(elements=tuple(elements), multiplication=table)
    return ProductResult._from_kernel(
        left, right, product, tuple(left_proj), tuple(right_proj)
    )


def _fresh_label(elements: tuple[str, ...], base: str) -> str:
    if base not in elements:
        return base
    index = 1
    while f"{base}#{index}" in elements:
        index += 1
    return f"{base}#{index}"


def adjoin_identity(semigroup: FiniteSemigroup) -> AdjoinIdentityResult:
    """Adjoin a fresh two-sided identity with the inclusion embedding."""

    from jacobian.math.finite_semigroups._models import AdjoinIdentityResult

    _require_associative(semigroup)
    one = _fresh_label(semigroup.elements, "1")
    elements = (*semigroup.elements, one)
    idx = {label: i for i, label in enumerate(semigroup.elements)}
    table = tuple(
        tuple(
            semigroup.multiplication[idx[a]][idx[b]]
            if a in idx and b in idx
            else (a if b == one else b)
            for b in elements
        )
        for a in elements
    )
    result = FiniteSemigroup(elements=elements, multiplication=table)
    return AdjoinIdentityResult._from_kernel(
        semigroup, result, tuple((a, a) for a in semigroup.elements)
    )


def adjoin_zero(semigroup: FiniteSemigroup) -> AdjoinZeroResult:
    """Adjoin a fresh absorbing zero with the inclusion embedding."""

    from jacobian.math.finite_semigroups._models import AdjoinZeroResult

    _require_associative(semigroup)
    zero = _fresh_label(semigroup.elements, "0")
    elements = (*semigroup.elements, zero)
    idx = {label: i for i, label in enumerate(semigroup.elements)}
    table = tuple(
        tuple(
            semigroup.multiplication[idx[a]][idx[b]] if a in idx and b in idx else zero
            for b in elements
        )
        for a in elements
    )
    result = FiniteSemigroup(elements=elements, multiplication=table)
    return AdjoinZeroResult._from_kernel(
        semigroup, result, tuple((a, a) for a in semigroup.elements)
    )


def rees_quotient(
    semigroup: FiniteSemigroup, ideal: tuple[str, ...]
) -> ReesQuotientResult:
    """Collapse a two-sided ideal to a single zero class."""

    from jacobian.math.finite_semigroups._models import ReesQuotientResult

    _require_associative(semigroup)
    declared = set(semigroup.elements)
    if any(element not in declared for element in ideal):
        raise OperationDomainValidationError(
            location=("ideal",),
            code="finite_semigroup.ideal_not_declared",
            message="ideal elements must belong to the semigroup",
        )
    ideal_set = set(ideal)
    if tuple(a for a in semigroup.elements if a in ideal_set) != ideal:
        raise OperationDomainValidationError(
            location=("ideal",),
            code="finite_semigroup.ideal_order",
            message="ideal must use declared semigroup order without repeats",
        )
    idx = {label: i for i, label in enumerate(semigroup.elements)}
    for s in semigroup.elements:
        for a in ideal:
            if (
                semigroup.multiplication[idx[s]][idx[a]] not in ideal_set
                or semigroup.multiplication[idx[a]][idx[s]] not in ideal_set
            ):
                raise OperationDomainValidationError(
                    location=("ideal",),
                    code="finite_semigroup.not_two_sided_ideal",
                    message="rees quotient requires a two-sided ideal",
                )
    if not ideal:
        raise OperationDomainValidationError(
            location=("ideal",),
            code="finite_semigroup.empty_ideal",
            message="rees quotient requires a nonempty ideal",
        )
    zero = (
        "0"
        if "0" not in [a for a in semigroup.elements if a not in ideal_set]
        else "0#1"
    )
    survivors = tuple(a for a in semigroup.elements if a not in ideal_set)
    elements = (*survivors, zero)

    def _class(value: str) -> str:
        return zero if value in ideal_set else value

    # Rebuild over the quotient axis: survivors keep products, ideal maps to zero.
    rows: list[tuple[str, ...]] = []
    source_of = {a: a for a in survivors}
    source_of[zero] = ideal[0]
    for a in elements:
        row: list[str] = []
        for b in elements:
            sa, sb = source_of[a], source_of[b]
            row.append(_class(semigroup.multiplication[idx[sa]][idx[sb]]))
        rows.append(tuple(row))
    quotient = FiniteSemigroup(elements=elements, multiplication=tuple(rows))
    projection = tuple((a, _class(a)) for a in semigroup.elements)
    return ReesQuotientResult._from_kernel(semigroup, ideal, quotient, projection)


def karoubi_projection(semigroup: FiniteSemigroup) -> KaroubiProjectionResult:
    """Return the Karoubi envelope over the semigroup idempotents."""

    from jacobian.math.finite_categories.values import (
        CategoryIdentifier,
        FiniteCategory,
        MorphismSpec,
    )
    from jacobian.math.finite_semigroups._models import KaroubiProjectionResult

    _require_associative(semigroup)
    elements = semigroup.elements
    multiplication = semigroup.multiplication
    idx = {label: i for i, label in enumerate(elements)}
    idempotents_list = _idempotents(elements, multiplication)
    # One entry per Hom(e, f) generator: the spec plus its e, s, f data so the
    # composition step never has to destructure a nested identifier.
    generators: list[tuple[MorphismSpec, str, str, str]] = []
    for e in idempotents_list:
        for target in idempotents_list:
            for s in elements:
                # Hom(e, f) = {s | f*s*e == s}.
                if (
                    multiplication[idx[target]][idx[multiplication[idx[s]][idx[e]]]]
                    == s
                ):
                    generators.append(
                        (
                            MorphismSpec(
                                morphism_id=((e, s), target),
                                source=e,
                                target=target,
                            ),
                            e,
                            s,
                            target,
                        )
                    )
    if len(generators) > 4_096:
        raise OperationResourceAdmissionError(
            location=("semigroup",),
            code="finite_semigroup.karoubi_morphism_bound",
            message="karoubi envelope exceeds the morphism bound",
        )
    by_id: dict[CategoryIdentifier, MorphismSpec] = {
        spec.morphism_id: spec for spec, _e, _s, _t in generators
    }
    composition: list[
        tuple[CategoryIdentifier, CategoryIdentifier, CategoryIdentifier]
    ] = []
    for g_spec, _g_e, g_s, g_target in generators:
        for f_spec, f_source, _f_s, f_target in generators:
            if f_target != g_spec.source:
                continue
            product_label = multiplication[idx[g_s]][idx[_f_s]]
            result_id = ((f_source, product_label), g_target)
            if result_id in by_id:
                composition.append((g_spec.morphism_id, f_spec.morphism_id, result_id))
    category = FiniteCategory(
        objects=tuple(idempotents_list),
        morphisms=tuple(spec for spec, _e, _s, _t in generators),
        identities=tuple((e, ((e, e), e)) for e in idempotents_list),
        composition=tuple(composition),
    )
    return KaroubiProjectionResult._from_kernel(
        semigroup, tuple(idempotents_list), category
    )


__all__ = [
    "adjoin_identity",
    "adjoin_zero",
    "element_power",
    "generated_subsemigroup",
    "green_relations",
    "ideal_enumeration",
    "idempotents",
    "karoubi_projection",
    "local_structure",
    "nilpotent_elements",
    "opposite_semigroup",
    "power_profile",
    "principal_ideals",
    "product_semigroup",
    "rees_quotient",
    "regular_elements",
    "verify_generated_subsemigroup",
]
