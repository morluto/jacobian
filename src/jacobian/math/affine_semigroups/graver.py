"""Complete Graver-basis enumeration for small one-row configurations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import product
from math import comb, gcd

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.affine_semigroups.graver_models import (
    IntegerConfigurationGraverBasis,
    IntegerConfigurationMarkovBasis,
)
from jacobian.math.affine_semigroups.operations import relation_lattice
from jacobian.math.affine_semigroups.semigroup import (
    AffineConfiguration,
    _admit_configuration,
)
from jacobian.math.matrices.values import IntegerMatrix
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    RationalPolynomial,
    RationalPolynomialIdeal,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

MAX_GRAVER_WORK = 100_000_000
MAX_GRAVER_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_TORIC_IDEAL_GENERATORS = 64
MAX_TORIC_IDEAL_OUTPUT_BYTES = 8 * 1024 * 1024
_POLYNOMIAL_VARIABLE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,31}\Z")


def _normalized_graver_weights(configuration: IntegerMatrix) -> tuple[int, ...]:
    rows, columns = configuration.row_count, configuration.column_count
    entries = tuple(int(value) for value in configuration.entries[0]) if rows else ()
    if rows != 1 or not 1 <= columns <= 5:
        raise OperationResourceAdmissionError(
            location=("configuration",),
            code="affine_semigroup.graver_shape",
            message="Graver enumeration currently admits one-row configurations with 1..5 columns",
        )
    if any(abs(value) >= 10**8 for value in entries):
        raise OperationResourceAdmissionError(
            location=("configuration",),
            code="affine_semigroup.graver_digits",
            message="Graver configuration entries are limited to 8 decimal digits",
        )
    divisor = 0
    for value in entries:
        divisor = gcd(divisor, abs(value))
    if divisor:
        entries = tuple(value // divisor for value in entries)
    return entries


def _graver_search_radius(entries: tuple[int, ...]) -> int:
    return max(1, 2 * max((abs(value) for value in entries), default=0))


def _graver_box_states(columns: int, radius: int) -> int:
    side = 2 * radius + 1
    states = 1
    for _ in range(columns):
        states *= side
    return states


def _admit_graver_search(entries: tuple[int, ...]) -> tuple[int, int]:
    columns = len(entries)
    bound = _graver_search_radius(entries)
    box_states = _graver_box_states(columns, bound)
    candidate_states = box_states
    if candidate_states * candidate_states > MAX_GRAVER_WORK:
        raise OperationResourceAdmissionError(
            location=("configuration",),
            code="affine_semigroup.graver_work",
            message=(
                "complete Graver enumeration exceeds the 100,000,000 candidate-pair work envelope"
            ),
        )
    max_digits = max(1, len(str(bound)))
    output_bound = ((candidate_states - 1) // 2) * (columns * (max_digits + 2) + 2)
    if output_bound > MAX_GRAVER_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("configuration",),
            code="affine_semigroup.graver_output",
            message="worst-case Graver result exceeds the 8 MiB output envelope",
        )

    return bound, candidate_states


def _enumerate_graver_vectors(
    entries: tuple[int, ...], bound: int
) -> tuple[tuple[int, ...], ...]:
    """Enumerate a previously admitted one-row Graver coordinate box."""
    columns = len(entries)
    candidates = []
    for vector in product(range(-bound, bound + 1), repeat=columns):
        if not any(vector):
            continue
        if sum(a * z for a, z in zip(entries, vector, strict=True)) == 0:
            candidates.append(vector)
    candidates.sort(key=lambda vector: (sum(map(abs, vector)), vector))
    minima: list[tuple[int, ...]] = []
    result = []
    for vector in candidates:
        if any(
            all(
                left == 0 or (left > 0) == (right > 0)
                for left, right in zip(minimal, vector, strict=True)
            )
            and all(
                abs(left) <= abs(right)
                for left, right in zip(minimal, vector, strict=True)
            )
            for minimal in minima
        ):
            continue
        divisor = 0
        for value in vector:
            divisor = gcd(divisor, abs(value))
        # A nonprimitive relation is necessarily conformally decomposable.
        if divisor != 1:
            continue
        minima.append(vector)
        first = next(value for value in vector if value)
        result.append(vector if first > 0 else tuple(-value for value in vector))
    return tuple(sorted(set(result)))


def graver_basis(configuration: IntegerMatrix) -> IntegerConfigurationGraverBasis:
    """Return all conformally indecomposable vectors in an admitted kernel.

    For a one-row integer matrix, expansion of a relation into unit signed
    summands gives a minimal zero-sum sequence in [-M, M]. Such a sequence has
    length at most 2M, where M is the largest absolute entry. Therefore every
    Graver vector lies in the admitted coordinate box. We enumerate that full
    box and retain exactly the componentwise minima in each sign orthant.
    """
    if configuration.row_count == 1 and configuration.column_count <= 5:
        entries = _normalized_graver_weights(configuration)
        bound, _candidate_states = _admit_graver_search(entries)
        vectors = _enumerate_graver_vectors(entries, bound)
    else:
        # In nullity at most one, the primitive generator of ker_Z(A) is the
        # entire Graver basis (or the basis is empty). Reuse the admitted exact
        # Smith/Hermite kernel instead of searching an arbitrary coordinate box.
        lattice = relation_lattice(configuration)
        if lattice.nullity > 1:
            raise OperationResourceAdmissionError(
                location=("configuration",),
                code="affine_semigroup.graver_nullity",
                message=(
                    "outside the one-row, at-most-five-column enumeration and "
                    "nullity-at-most-one exact Graver slices"
                ),
            )
        if lattice.nullity == 0:
            vectors = ()
        else:
            vector = tuple(int(value) for value in lattice.relation_basis.entries[0])
            first = next(value for value in vector if value)
            vectors = (vector if first > 0 else tuple(-value for value in vector),)
    return IntegerConfigurationGraverBasis(configuration=configuration, vectors=vectors)


def markov_basis(configuration: IntegerMatrix) -> IntegerConfigurationMarkovBasis:
    """Return a global Markov basis for a bounded one-row configuration.

    The complete Graver basis generates the toric ideal for an integer
    configuration. The fundamental theorem of Markov bases then gives
    connectivity of every nonnegative fiber, rather than only tested fibers.
    """
    graver = graver_basis(configuration)
    return IntegerConfigurationMarkovBasis(
        configuration=configuration,
        moves=graver.vectors,
    )


def _toric_candidate_count_bound(columns: int, entries: tuple[int, ...]) -> int:
    """Bound sign-normalized Graver outputs by the complete l1 candidate ball."""
    if not any(entries):
        # The zero row has exactly the signed unit vectors as its Graver basis.
        return columns
    maximum = max(entries)
    radius = 2 * maximum
    lattice_points = sum(
        (1 << support) * comb(columns, support) * comb(radius, support)
        for support in range(min(columns, radius) + 1)
    )
    # The one-row Graver proof bounds each vector's l1 norm by 2*maximum.
    # The ball is symmetric, and the ideal keeps one of every opposite pair.
    return (lattice_points - 1) // 2


@dataclass(frozen=True, slots=True)
class _ToricIdealPlan:
    configuration: AffineConfiguration
    reduced_weights: tuple[int, ...]
    search_radius: int
    precomputed_vectors: tuple[tuple[int, ...], ...] | None


def _canonical_toric_configuration(value: object) -> AffineConfiguration:
    if type(value) is not AffineConfiguration:
        raise OperationDomainValidationError(
            location=("configuration",),
            code="affine_semigroup.toric_configuration",
            message="configuration must be a canonical labelled affine configuration",
        )
    configuration = _admit_configuration(value)
    columns = len(configuration.generator_labels)
    if configuration.rows != 1 or not 1 <= columns <= 5:
        raise OperationDomainValidationError(
            location=("configuration",),
            code="affine_semigroup.toric_shape",
            message="toric ideals currently admit one-row configurations with 1..5 generators",
        )
    if any(type(weight) is not int for weight in configuration.entries[0]):
        raise OperationDomainValidationError(
            location=("configuration", "entries"),
            code="affine_semigroup.toric_entries",
            message="configuration weights must be exact integers",
        )
    if any(weight < 0 for weight in configuration.entries[0]):
        raise OperationDomainValidationError(
            location=("configuration", "entries"),
            code="affine_semigroup.toric_nonnegative",
            message="toric monomial-map weights must be nonnegative",
        )
    if any(
        type(label) is not str
        or not 1 <= len(label) <= 32
        or not _POLYNOMIAL_VARIABLE.fullmatch(label)
        for label in configuration.generator_labels
    ):
        raise OperationDomainValidationError(
            location=("configuration", "generator_labels"),
            code="affine_semigroup.toric_variable_axis",
            message=(
                "generator labels must be valid ordered polynomial variable names "
                "so the toric ideal retains the exact generator axis"
            ),
        )

    return configuration


def _small_toric_plan(
    configuration: AffineConfiguration,
) -> _ToricIdealPlan:
    entries = configuration.entries[0]
    columns = len(entries)
    # In one and two variables the complete integer kernel is generated by at
    # most two explicit primitive vectors. Use that exact closed form instead
    # of expanding a coordinate box whose size depends on the input weights.
    vectors: tuple[tuple[int, ...], ...]
    if columns == 1:
        vectors = () if entries[0] > 0 else ((1,),)
    elif not any(entries):
        vectors = ((0, 1), (1, 0))
    else:
        divisor = gcd(*entries)
        vector: tuple[int, ...] = (
            entries[1] // divisor,
            -(entries[0] // divisor),
        )
        if next(value for value in vector if value) < 0:
            vector = tuple(-value for value in vector)
        vectors = (vector,)
    exponent_bound = max(
        (abs(value) for vector in vectors for value in vector), default=0
    )
    if exponent_bound > MAX_POLYNOMIAL_EXPONENT:
        raise OperationResourceAdmissionError(
            location=("configuration",),
            code="affine_semigroup.toric_exponent_bound",
            message=(
                "the exact primitive kernel move exceeds the shared polynomial "
                f"exponent limit {MAX_POLYNOMIAL_EXPONENT}"
            ),
        )
    output_bound = 4096 + len(vectors) * (2048 + 5 * 256)
    if output_bound > MAX_TORIC_IDEAL_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("configuration",),
            code="affine_semigroup.toric_output_bound",
            message="the complete toric ideal presentation exceeds its 8 MiB output envelope",
        )
    return _ToricIdealPlan(configuration, entries, 0, vectors)


def _graver_toric_plan(
    configuration: AffineConfiguration,
) -> _ToricIdealPlan:
    entries = configuration.entries[0]
    columns = len(entries)

    divisor = 0
    for weight in entries:
        divisor = gcd(divisor, weight)
    reduced = tuple(weight // divisor for weight in entries) if divisor else entries
    generator_bound = _toric_candidate_count_bound(columns, reduced)
    if generator_bound > MAX_TORIC_IDEAL_GENERATORS:
        raise OperationResourceAdmissionError(
            location=("configuration",),
            code="affine_semigroup.toric_generator_bound",
            message=(
                "the l1 Graver candidate envelope permits more than "
                f"{MAX_TORIC_IDEAL_GENERATORS} ideal generators"
            ),
        )

    maximum = max(reduced, default=0)
    # Match the Graver kernel's exact complete-box and candidate-pair estimate.
    radius = _graver_search_radius(reduced)
    box_states = _graver_box_states(columns, radius)
    if box_states * box_states > MAX_GRAVER_WORK:
        raise OperationResourceAdmissionError(
            location=("configuration",),
            code="affine_semigroup.toric_work",
            message=(
                f"complete Graver candidate-pair work exceeds {MAX_GRAVER_WORK} states"
            ),
        )

    exponent_bound = 1 if maximum == 0 else 2 * maximum
    if exponent_bound > MAX_POLYNOMIAL_EXPONENT:
        raise OperationResourceAdmissionError(
            location=("configuration",),
            code="affine_semigroup.toric_exponent_bound",
            message=(
                "the complete Graver coordinate envelope exceeds the shared "
                f"polynomial exponent limit {MAX_POLYNOMIAL_EXPONENT}"
            ),
        )
    # At most 64 generators, two unit-coefficient terms each, five bounded
    # exponents per term, and five variable labels (repeated in each value).
    output_bound = 4096 + generator_bound * (2048 + 5 * 256)
    if output_bound > MAX_TORIC_IDEAL_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("configuration",),
            code="affine_semigroup.toric_output_bound",
            message="the complete toric ideal presentation exceeds its 8 MiB output envelope",
        )
    return _ToricIdealPlan(configuration, reduced, radius, None)


def _admit_toric_configuration(value: object) -> _ToricIdealPlan:
    configuration = _canonical_toric_configuration(value)
    if len(configuration.generator_labels) <= 2:
        return _small_toric_plan(configuration)
    return _graver_toric_plan(configuration)


def toric_ideal(configuration: AffineConfiguration) -> RationalPolynomialIdeal:
    """Return a Graver-generated presentation of a one-row QQ toric ideal.

    For the admitted nonnegative weights ``a_j``, this is the ideal
    ``ker(Q[x_1,...,x_n] -> Q[t], x_j -> t**a_j)``. The returned Graver
    binomials generate that ideal; they are not claimed to be a minimal
    generating set.
    """
    plan = _admit_toric_configuration(configuration)
    configuration = plan.configuration
    variables = configuration.generator_labels
    if len(variables) == 1 and configuration.entries[0][0] != 0:
        zero = RationalPolynomial(
            variables=variables,
            polynomial=SparseRationalPolynomial(terms=()),
        )
        return RationalPolynomialIdeal(variables=variables, generators=(zero,))

    vectors = (
        plan.precomputed_vectors
        if plan.precomputed_vectors is not None
        else _enumerate_graver_vectors(plan.reduced_weights, plan.search_radius)
    )
    one = CanonicalRational(num=1, den=1)
    minus_one = CanonicalRational(num=-1, den=1)
    generators = []
    for vector in vectors:
        positive = tuple(max(value, 0) for value in vector)
        negative = tuple(max(-value, 0) for value in vector)
        terms = tuple(
            RationalPolynomialTerm(coefficient=coefficient, exponents=exponents)
            for exponents, coefficient in sorted(
                ((positive, one), (negative, minus_one)),
                key=lambda item: item[0],
                reverse=True,
            )
        )
        generators.append(
            RationalPolynomial(
                variables=variables,
                polynomial=SparseRationalPolynomial(terms=terms),
            )
        )
    if not generators:
        # RationalPolynomialIdeal represents the zero ideal by one zero input.
        generators.append(
            RationalPolynomial(
                variables=variables,
                polynomial=SparseRationalPolynomial(terms=()),
            )
        )
    return RationalPolynomialIdeal(
        variables=variables,
        generators=tuple(generators),
    )
