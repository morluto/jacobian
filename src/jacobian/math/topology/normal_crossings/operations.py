"""Exact dual-complex and nearby-cycle lattice operations for SNC presentations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import pairwise
from math import comb
from types import MappingProxyType

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.matrices.certified_snf.operations import (
    smith_normal_form_certificate,
)
from jacobian.math.matrices.values import IntegerMatrix
from jacobian.math.topology._models import (
    MAX_TOPOLOGY_FACETS,
    BoundarySquareLedgerEntry,
    FiniteSimplicialComplex,
    canonical_complex,
)
from jacobian.math.topology.chain_complexes.operations import (
    construct_chain_complex,
)
from jacobian.math.topology.chain_complexes.values import (
    MAX_BASIS_SIZE,
    MAX_OPERATION_MATRIX_CELLS,
    CoefficientRing,
)
from jacobian.math.topology.normal_crossings._models import (
    MAX_NC_SPECIALIZATION_CELLS,
    MAX_NC_SPECIALIZATION_MAPS,
    DualComplexResult,
    NearbyCycleLattice,
    NearbyCycleLatticesResult,
    NormalCrossingsPresentation,
    NormalCrossingsStratum,
    NormalCrossingsStratumRecord,
    SpecializationMap,
    StrataInCardinality,
    maximal_strata,
    stratum_subsets,
)

MAX_NC_SUBSET_INCIDENCES = 65_536
"""Bound on the total stratum-subset incidence pairs enumerated at admission."""

StratumKey = tuple[str, ...]


def _canonical_order(keys: Iterable[StratumKey]) -> list[StratumKey]:
    """Sort stratum keys by cardinality first, then lexicographically."""

    return sorted(keys, key=lambda key: (len(key), key))


def _domain_error(
    location: tuple[str | int, ...],
    code: str,
    message: str,
) -> OperationDomainValidationError:
    return OperationDomainValidationError(
        location=location,
        code=f"topology.normal_crossings.{code}",
        message=message,
    )


def _resource_error(
    location: tuple[str | int, ...],
    code: str,
    message: str,
) -> OperationResourceAdmissionError:
    return OperationResourceAdmissionError(
        location=location,
        code=f"topology.normal_crossings.{code}",
        message=message,
    )


@dataclass(frozen=True, slots=True)
class AdmittedPresentation:
    """One admitted strict-SNC presentation with its canonical derived axes."""

    value: NormalCrossingsPresentation
    index: Mapping[StratumKey, NormalCrossingsStratum]
    groups: tuple[tuple[StratumKey, ...], ...]
    """Stratum keys grouped by cardinality; ``groups[c - 1]`` has size ``c``."""

    @property
    def max_cardinality(self) -> int:
        return len(self.groups)

    def require(self, key: StratumKey) -> NormalCrossingsStratum:
        stratum = self.index.get(key)
        if stratum is None:
            raise _domain_error(
                ("strata",),
                "undeclared_stratum",
                f"component set {key} is not a declared stratum",
            )
        return stratum


def admit_presentation(
    components: tuple[str, ...],
    strata: tuple[NormalCrossingsStratum, ...],
) -> AdmittedPresentation:
    """Validate and canonicalize one finite strict-SNC incidence presentation.

    This is the single shared admission for the native operations and the
    catalog projections.  It establishes the semantic incidence identities
    (distinct strata keyed by exact component sets, singleton coverage,
    downward closure, and the transverse dimension identity) once, before
    any dual-complex, Cech, lattice, or specialization value is materialized.
    """

    ordered_components = tuple(sorted(components))
    if len(set(ordered_components)) != len(ordered_components):
        raise _domain_error(
            ("components",),
            "duplicate_component",
            "presentation components must be distinct labels",
        )
    declared = set(ordered_components)

    keys: list[StratumKey] = []
    for position, stratum in enumerate(strata):
        key = tuple(stratum.components)
        if not set(key) <= declared:
            raise _domain_error(
                ("strata", position, "components"),
                "undeclared_component",
                "every stratum component must be a declared component",
            )
        keys.append(key)
    if len(set(keys)) != len(keys):
        raise _domain_error(
            ("strata",),
            "duplicate_stratum",
            "a component set identifies at most one stratum; duplicate "
            "component sets are rejected by the canonical contract",
        )

    total_incidences = sum(len(stratum_subsets(key)) for key in keys)
    if total_incidences > MAX_NC_SUBSET_INCIDENCES:
        raise _resource_error(
            ("strata",),
            "subset_incidence_budget",
            "stratum subset incidence enumeration exceeds the "
            f"{MAX_NC_SUBSET_INCIDENCES}-pair admission bound",
        )

    index = dict(zip(keys, strata, strict=True))
    for component in ordered_components:
        if (component,) not in index:
            raise _domain_error(
                ("strata",),
                "missing_component_stratum",
                f"component {component} requires its singleton stratum",
            )
    for key in keys:
        for subset in stratum_subsets(key):
            if subset not in index:
                raise _domain_error(
                    ("strata",),
                    "not_downward_closed",
                    f"subset {subset} of stratum {key} is not a declared stratum",
                )
    for key in keys:
        stratum = index[key]
        for component in key:
            singleton = index[(component,)]
            if stratum.dimension != singleton.dimension - (len(key) - 1):
                raise _domain_error(
                    ("strata",),
                    "dimension_identity_violated",
                    f"stratum {key} must satisfy dim S = dim {{i}} - (r - 1) "
                    f"for each component i of S",
                )

    ordered_keys = _canonical_order(index)
    max_cardinality = max(len(key) for key in keys)
    groups: list[list[StratumKey]] = [[] for _ in range(max_cardinality)]
    for key in ordered_keys:
        groups[len(key) - 1].append(key)
    value = NormalCrossingsPresentation(
        components=ordered_components,
        strata=tuple(index[key] for key in ordered_keys),
    )
    return AdmittedPresentation(
        value=value,
        index=MappingProxyType(index),
        groups=tuple(tuple(group) for group in groups),
    )


def _stratum_records(
    admitted: AdmittedPresentation,
) -> tuple[NormalCrossingsStratumRecord, ...]:
    return tuple(
        NormalCrossingsStratumRecord(
            components=stratum.components,
            dimension=stratum.dimension,
            branch_multiplicity=stratum.branch_multiplicity,
        )
        for stratum in admitted.value.strata
    )


def _cardinality_groups(
    admitted: AdmittedPresentation,
) -> tuple[StrataInCardinality, ...]:
    return tuple(
        StrataInCardinality(cardinality=cardinality, strata=group)
        for cardinality, group in enumerate(admitted.groups, start=1)
    )


def _cech_matrices(
    admitted: AdmittedPresentation,
) -> tuple[tuple[tuple[str, ...], ...], ...]:
    """Assemble the signed Cech incidence matrices in canonical stratum order.

    For a stratum ``S`` with sorted components ``(i_0, ..., i_k)`` the
    alternating differential is ``d(S) = sum_j (-1)^j (S \\ {i_j})``; the
    target rows use each cardinality group's canonical sorted order.
    """

    matrices: list[tuple[tuple[str, ...], ...]] = []
    for degree in range(1, len(admitted.groups)):
        source = admitted.groups[degree]
        target = admitted.groups[degree - 1]
        row_for = {key: row for row, key in enumerate(target)}
        dense = [[0] * len(source) for _ in target]
        for column, key in enumerate(source):
            for position in range(len(key)):
                face = key[:position] + key[position + 1 :]
                sign = 1 if position % 2 == 0 else -1
                dense[row_for[face]][column] += sign
        matrices.append(tuple(tuple(str(value) for value in row) for row in dense))
    return tuple(matrices)


def _require_dual_complex_admission(
    admitted: AdmittedPresentation,
    facets: tuple[StratumKey, ...],
) -> None:
    if len(facets) > MAX_TOPOLOGY_FACETS:
        raise _resource_error(
            ("strata",),
            "dual_complex_facet_budget",
            "the dual complex exceeds the "
            f"{MAX_TOPOLOGY_FACETS}-facet canonical complex bound",
        )
    for group in admitted.groups:
        if len(group) > MAX_BASIS_SIZE:
            raise _resource_error(
                ("strata",),
                "cech_group_budget",
                "a Cech incidence group exceeds the "
                f"{MAX_BASIS_SIZE}-stratum per-degree bound",
            )
    sizes = tuple(len(group) for group in admitted.groups)
    total_cells = sum(rows * columns for rows, columns in pairwise(sizes))
    if total_cells > MAX_OPERATION_MATRIX_CELLS:
        raise _resource_error(
            ("strata",),
            "cech_cell_budget",
            "the Cech incidence matrices exceed the "
            f"{MAX_OPERATION_MATRIX_CELLS}-cell aggregate bound",
        )


def dual_complex(
    components: tuple[str, ...],
    strata: tuple[NormalCrossingsStratum, ...],
) -> DualComplexResult:
    """Compute the dual complex and Cech incidence complex of an SNC presentation.

    The dual complex has one vertex per irreducible component and one simplex
    per inclusion-maximal stratum component set; downward closure makes every
    stratum a face.  The Cech / normalization incidence complex places the
    strata of cardinality ``k + 1`` in degree ``k`` with the alternating
    signed-incidence differential ``d(S) = sum_j (-1)^j (S \\ {i_j})`` over
    ``ZZ``; the shared chain-complex kernel replays ``d^2 = 0`` before the
    result is returned.
    """

    admitted = admit_presentation(components, strata)
    keys = tuple(stratum.components for stratum in admitted.value.strata)
    facets = maximal_strata(keys)
    _require_dual_complex_admission(admitted, facets)
    complex_value: FiniteSimplicialComplex = canonical_complex(
        admitted.value.components, facets
    )
    sizes = tuple(len(group) for group in admitted.groups)
    cech_value = construct_chain_complex(
        sizes,
        _cech_matrices(admitted),
        coefficient_ring=CoefficientRing.INTEGER,
    )
    ledger = tuple(
        BoundarySquareLedgerEntry(
            upper_dimension=degree,
            product_rows=sizes[degree - 2] if degree >= 2 else 0,
            product_columns=sizes[degree],
        )
        for degree in range(1, len(sizes))
    )
    return DualComplexResult._from_kernel(
        presentation=admitted.value,
        strata=_stratum_records(admitted),
        dual_complex=complex_value,
        strata_by_cardinality=_cardinality_groups(admitted),
        cech_value=cech_value,
        differential_squared_zero=ledger,
    )


_CERTIFIED_KERNEL_RANKS: dict[int, None] = {}


def _successive_difference_basis(r: int) -> IntegerMatrix:
    """Return the canonical saturated basis ``b_k = e_k - e_{k+1}`` of ``K_r``."""

    if r == 1:
        return IntegerMatrix(row_count=0, column_count=1, entries=())
    rows = tuple(
        tuple(1 if column == k else -1 if column == k + 1 else 0 for column in range(r))
        for k in range(r - 1)
    )
    return IntegerMatrix(row_count=r - 1, column_count=r, entries=rows)


def _certified_saturated_kernel(r: int) -> IntegerMatrix:
    """Return the Smith-certified saturated basis of ``ker(sum: Z^r -> Z)``.

    The all-ones row has certified rank one, so the kernel has rank
    ``r - 1``; the successive-difference basis has certified rank ``r - 1``
    with every invariant factor equal to one, so its row span is saturated
    and therefore equals the full kernel.
    """

    basis = _successive_difference_basis(r)
    if r in _CERTIFIED_KERNEL_RANKS:
        return basis
    ones = smith_normal_form_certificate(
        IntegerMatrix(row_count=1, column_count=r, entries=((1,) * r,))
    )
    certificate = ones if r == 1 else smith_normal_form_certificate(basis)
    expected_rank = 1 if r == 1 else r - 1
    if (
        ones.rank != 1
        or certificate.rank != expected_rank
        or (
            r > 1
            and tuple(int(factor) for factor in certificate.invariant_factors)
            != (1,) * (r - 1)
        )
    ):
        raise _domain_error(
            ("strata",),
            "kernel_certificate_invalid",
            f"the sum-zero kernel basis of Z^{r} failed its Smith certificate",
        )
    _CERTIFIED_KERNEL_RANKS[r] = None
    return basis


def _fold_matrix(source: StratumKey, target: StratumKey) -> IntegerMatrix:
    """Express the fiber-sum fold ``K_J -> K_I`` in successive-difference bases.

    The ambient fold ``Z^J -> Z^I`` is the identity on ``I`` and sends every
    branch of ``J \\ I`` to the largest component of ``I`` in sorted order.
    Row ``k`` expands the image of the ``k``-th source basis vector in the
    target basis, so ``K_J`` coordinates map to ``K_I`` coordinates as
    ``v -> v^T M`` and cover compositions multiply in path order.
    """

    positions = {label: position for position, label in enumerate(target)}
    fold_target = positions[target[-1]]

    def fold(label: str) -> int:
        return positions.get(label, fold_target)

    rows: list[tuple[int, ...]] = []
    for k in range(len(source) - 1):
        row = [0] * (len(target) - 1)
        start = fold(source[k])
        end = fold(source[k + 1])
        if start < end:
            for a in range(start, end):
                row[a] += 1
        elif end < start:
            for a in range(end, start):
                row[a] -= 1
        rows.append(tuple(row))
    if not rows:
        return IntegerMatrix(row_count=0, column_count=len(target) - 1, entries=())
    return IntegerMatrix(
        row_count=len(rows),
        column_count=len(target) - 1,
        entries=tuple(rows),
    )


def _cover_pairs(
    admitted: AdmittedPresentation,
) -> tuple[tuple[StratumKey, StratumKey], ...]:
    """Return every cover inclusion ``(target, source)`` in canonical order."""

    pairs: list[tuple[StratumKey, StratumKey]] = []
    for cardinality, group in enumerate(admitted.groups, start=1):
        if cardinality < 2:
            continue
        for source in group:
            for position in range(len(source)):
                target = source[:position] + source[position + 1 :]
                pairs.append((target, source))
    return tuple(sorted(pairs))


def _require_lattice_admission(
    pairs: tuple[tuple[StratumKey, StratumKey], ...],
) -> None:
    if len(pairs) > MAX_NC_SPECIALIZATION_MAPS:
        raise _resource_error(
            ("strata",),
            "specialization_map_budget",
            "the cover specialization maps exceed the "
            f"{MAX_NC_SPECIALIZATION_MAPS}-map publication bound",
        )
    total_cells = sum((len(source) - 1) * (len(target) - 1) for target, source in pairs)
    if total_cells > MAX_NC_SPECIALIZATION_CELLS:
        raise _resource_error(
            ("strata",),
            "specialization_cell_budget",
            "the specialization matrices exceed the "
            f"{MAX_NC_SPECIALIZATION_CELLS}-cell aggregate bound",
        )


def nearby_cycle_lattices(
    components: tuple[str, ...],
    strata: tuple[NormalCrossingsStratum, ...],
) -> NearbyCycleLatticesResult:
    """Compute the integral nearby-cycle stalk lattices of an SNC presentation.

    Each stratum ``S`` with ``r`` branches carries the Milnor-fibre phase
    lattice ``K_S = ker(sum: Z^r -> Z)`` with its canonical saturated
    successive-difference basis, lattice rank ``r - 1``, and exact
    Milnor-fibre cohomology ranks ``rank H^q(MF_S; Z) = C(r - 1, q)`` for
    ``q`` in ``0..r-1``.  Each cover inclusion ``I ⊂ J`` of component sets
    publishes the specialization ``K_J -> K_I`` as the fiber-sum fold that is
    the identity on ``I`` and folds the added branch onto the largest
    component of ``I`` in sorted order, expressed in the saturated bases.
    """

    admitted = admit_presentation(components, strata)
    pairs = _cover_pairs(admitted)
    _require_lattice_admission(pairs)
    lattices = tuple(
        NearbyCycleLattice(
            components=stratum.components,
            branch_multiplicity=stratum.branch_multiplicity,
            lattice_rank=stratum.branch_multiplicity - 1,
            saturated_basis=_certified_saturated_kernel(stratum.branch_multiplicity),
            milnor_fiber_cohomology_ranks=tuple(
                comb(stratum.branch_multiplicity - 1, q)
                for q in range(stratum.branch_multiplicity)
            ),
        )
        for stratum in admitted.value.strata
    )
    specializations = tuple(
        SpecializationMap(
            source_components=source,
            target_components=target,
            matrix=_fold_matrix(source, target),
        )
        for target, source in pairs
    )
    return NearbyCycleLatticesResult._from_kernel(
        presentation=admitted.value,
        lattices=lattices,
        specializations=specializations,
    )


def specialization_matrix(
    components: tuple[str, ...],
    strata: tuple[NormalCrossingsStratum, ...],
    source_components: StratumKey,
    target_components: StratumKey,
) -> SpecializationMap:
    """Return the direct specialization ``K_J -> K_I`` for any inclusion ``I ⊂ J``.

    The direct map uses the same fiber-sum fold convention as the published
    cover maps: identity on ``I`` and every branch of ``J \\ I`` folded onto
    the largest component of ``I``.  Cover compositions equal this map.
    """

    admitted = admit_presentation(components, strata)
    source = tuple(sorted(source_components))
    target = tuple(sorted(target_components))
    admitted.require(source)
    admitted.require(target)
    if not set(target) < set(source):
        raise _domain_error(
            ("strata",),
            "specialization_inclusion_invalid",
            "specialization requires the target component set to be a proper "
            "subset of the source",
        )
    return SpecializationMap(
        source_components=source,
        target_components=target,
        matrix=_fold_matrix(source, target),
    )


def verify_dual_complex_claim(claim: DualComplexResult) -> bool:
    """Verify a dual-complex claim by replaying the shared admitted kernel."""

    try:
        return (
            dual_complex(claim.presentation.components, claim.presentation.strata)
            == claim
        )
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


def verify_nearby_cycle_lattices_claim(claim: NearbyCycleLatticesResult) -> bool:
    """Verify a nearby-cycle lattice claim by replaying the admitted kernel."""

    try:
        return (
            nearby_cycle_lattices(
                claim.presentation.components, claim.presentation.strata
            )
            == claim
        )
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


__all__ = [
    "MAX_NC_SUBSET_INCIDENCES",
    "AdmittedPresentation",
    "admit_presentation",
    "dual_complex",
    "nearby_cycle_lattices",
    "specialization_matrix",
    "verify_dual_complex_claim",
    "verify_nearby_cycle_lattices_claim",
]
