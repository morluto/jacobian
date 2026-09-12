"""Exact frame and vector-family contract tests."""

import json
from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.matrices.values import IntegerMatrix
from jacobian.math.number_theory.number_fields import GaussianRational
from jacobian.math.topology.frames._models import (
    ComplexFrameProfileRequest,
    GramResult,
    MutuallyUnbiasedBasesRequest,
    SicProfileRequest,
)
from jacobian.math.topology.frames._tools import (
    _coherence,
    _complex_frame_profile,
    _frame_potential,
    _gram,
    _mutually_unbiased_bases,
    _sic_profile,
    _tight_equiangular_profile,
)
from jacobian.math.topology.frames.operations import gram, verify_gram
from jacobian.math.topology.frames.values import (
    MAX_VECTOR_CELLS,
    ComplexFrame,
    VectorFamily,
)


def _z(real: int, imaginary: int = 0) -> GaussianRational:
    return GaussianRational.from_fractions(Fraction(real), Fraction(imaginary))


def _repeated_standard_basis(
    *, dimension: int, repeats: int
) -> tuple[tuple[int, ...], ...]:
    basis = tuple(
        tuple(int(row == column) for column in range(dimension))
        for row in range(dimension)
    )
    return basis * repeats


def test_gram_accepts_nonspanning_vector_family() -> None:
    assert _gram(
        VectorFamily.model_validate(
            {"dimension": len(([[1, 0], [2, 0]])[0]), "vectors": [[1, 0], [2, 0]]}
        )
    ).gram == (
        (1, 2),
        (2, 4),
    )


def test_decoded_gram_rejects_shape_forgery() -> None:
    result = gram(VectorFamily(dimension=2, vectors=((1, 0), (0, 1))))
    payload = result.model_dump()
    payload["gram"]["row_count"] = 1
    with pytest.raises(ValueError, match="shape"):
        GramResult.model_validate(payload)
    malformed = GramResult.model_construct(
        vectors=((1, 0), (0, 1)),
        dimension=2,
        gram=IntegerMatrix(entries=((1,),)),
    )
    assert not verify_gram(malformed)
    assert not verify_gram(result.model_copy(update={"gram": None}))
    assert not verify_gram(result.model_copy(update={"vectors": ()}))


def test_vector_family_schema_advertises_cell_budget() -> None:
    description = VectorFamily.model_json_schema()["properties"]["vectors"][
        "description"
    ]

    assert f"len(vectors) * dimension <= {MAX_VECTOR_CELLS}" in description


def test_gram_accepts_a_single_vector_beyond_the_old_side_cap() -> None:
    vector = (1,) * 513
    result = gram(VectorFamily(dimension=len(vector), vectors=(vector,)))

    assert result.dimension == 513
    assert result.gram == ((513,),)


def test_frame_operations_admit_shape_sensitive_vector_count() -> None:
    vectors = ((1,),) * 1_025

    result = _frame_potential(VectorFamily(dimension=len(vectors[0]), vectors=vectors))

    assert result.potential == 1_025**2


def test_frame_operations_admit_coefficient_beyond_the_old_value_cap() -> None:
    result = _frame_potential(VectorFamily(dimension=1, vectors=((1_001,),)))

    assert result.potential == 1004006004001


def test_frame_requires_full_ambient_span() -> None:
    request = VectorFamily.model_validate(
        {"dimension": len(([[1, 0], [2, 0]])[0]), "vectors": [[1, 0], [2, 0]]}
    )
    with pytest.raises(OperationDomainValidationError) as error:
        _frame_potential(request)
    assert error.value.errors()[0]["type"] == "frames.frame_does_not_span"


def test_coherence_rejects_zero_vector() -> None:
    request = VectorFamily.model_validate(
        {
            "dimension": len(([[0, 0], [1, 0], [0, 1]])[0]),
            "vectors": [[0, 0], [1, 0], [0, 1]],
        }
    )
    with pytest.raises(OperationDomainValidationError) as error:
        _coherence(request)
    assert error.value.errors()[0]["type"] == "frames.zero_vector"


def test_tight_equiangular_profile_is_exact_and_serializable() -> None:
    result = _tight_equiangular_profile(
        VectorFamily(dimension=2, vectors=((1, 0), (0, 1)))
    )
    assert result.tight is True
    assert result.tight_constant == 1
    assert result.equiangular is True
    assert result.common_squared_inner_product is not None
    assert result.common_squared_inner_product.as_integer_ratio() == (0, 1)
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_exact_complex_mub_profile_and_forged_shape_rejection() -> None:
    standard = ComplexFrame(dimension=2, vectors=((_z(1), _z(0)), (_z(0), _z(1))))
    hadamard = ComplexFrame(dimension=2, vectors=((_z(1), _z(1)), (_z(1), _z(-1))))
    request = MutuallyUnbiasedBasesRequest(dimension=2, bases=(standard, hadamard))
    result = _mutually_unbiased_bases(request)
    assert result.is_mutually_unbiased is True
    assert result.basis_pair_count == 1
    assert result.cross_gram_squared[0][0][0].as_integer_ratio() == (1, 2)
    assert result.basis_grams[0][0][1].as_fractions() == (Fraction(0), Fraction(0))
    assert type(result).model_validate_json(result.model_dump_json()) == result
    forged = json.loads(result.model_dump_json())
    forged["basis_pair_count"] = 0
    with pytest.raises(ValueError, match="pair count"):
        type(result).model_validate_json(json.dumps(forged))
    non_mub = _mutually_unbiased_bases(
        MutuallyUnbiasedBasesRequest(dimension=2, bases=(standard, standard))
    )
    assert non_mub.is_mutually_unbiased is False
    with pytest.raises(ValueError, match="at most 16"):
        MutuallyUnbiasedBasesRequest(dimension=2, bases=(standard,) * 17)


def test_exact_complex_sic_and_design_profiles_are_decisions() -> None:
    basis = ComplexFrame(dimension=2, vectors=((_z(1), _z(0)), (_z(0), _z(1))))
    design = _complex_frame_profile(ComplexFrameProfileRequest(frame=basis))
    assert design.tight is True
    assert design.equiangular is True
    assert design.common_squared_overlap is not None
    assert design.common_squared_overlap.as_integer_ratio() == (0, 1)

    sic = _sic_profile(
        SicProfileRequest(frame=ComplexFrame(dimension=1, vectors=((_z(1),),)))
    )
    assert sic.is_sic is True
    assert sic.cardinality_residual == 0
    assert sic.equiangular is True
    assert sic.common_squared_overlap is not None
    assert sic.common_squared_overlap.as_integer_ratio() == (1, 2)
    assert sic.common_squared_overlap_residual is not None
    assert sic.common_squared_overlap_residual.as_integer_ratio() == (0, 1)
    assert sic.squared_overlaps[0][0].as_integer_ratio() == (1, 1)
    assert sic.tight_residual[0][0].as_fractions() == (Fraction(0), Fraction(0))
    assert type(sic).model_validate_json(sic.model_dump_json()) == sic
    forged = json.loads(sic.model_dump_json())
    forged["squared_overlaps"] = []
    with pytest.raises(ValueError, match="SIC ledgers"):
        type(sic).model_validate_json(json.dumps(forged))

    phase_scaled = _sic_profile(
        SicProfileRequest(
            frame=ComplexFrame(
                dimension=1,
                vectors=((GaussianRational.from_fractions(Fraction(0), Fraction(3)),),),
            )
        )
    )
    assert phase_scaled.is_sic is True
    assert phase_scaled.frame_operator[0][0].as_fractions() == (
        Fraction(1),
        Fraction(0),
    )
    non_sic = _sic_profile(SicProfileRequest(frame=basis))
    assert non_sic.is_sic is False
    assert non_sic.cardinality_residual == -2
    assert non_sic.equiangular is True
    assert non_sic.common_squared_overlap is not None
    assert non_sic.common_squared_overlap.as_integer_ratio() == (0, 1)
    assert non_sic.common_squared_overlap_residual is not None
    assert non_sic.common_squared_overlap_residual.as_integer_ratio() == (-1, 3)
    assert type(non_sic).model_validate_json(non_sic.model_dump_json()) == non_sic

    equal_norm_wrong_overlap = _sic_profile(
        SicProfileRequest(
            frame=ComplexFrame(
                dimension=2,
                vectors=((_z(1), _z(0)),) * 4,
            )
        )
    )
    assert equal_norm_wrong_overlap.cardinality_residual == 0
    assert equal_norm_wrong_overlap.equiangular is True
    assert equal_norm_wrong_overlap.common_squared_overlap is not None
    assert equal_norm_wrong_overlap.common_squared_overlap.as_integer_ratio() == (1, 1)
    assert equal_norm_wrong_overlap.common_squared_overlap_residual is not None
    assert (
        equal_norm_wrong_overlap.common_squared_overlap_residual.as_integer_ratio()
        == (
            2,
            3,
        )
    )
    assert equal_norm_wrong_overlap.is_sic is False


def test_complex_operator_uses_vector_times_conjugate_vector_and_trace_average() -> (
    None
):
    frame = ComplexFrame(
        dimension=2,
        vectors=(
            (_z(1), _z(0, 1)),
            (_z(0), _z(2)),
        ),
    )
    result = _complex_frame_profile(ComplexFrameProfileRequest(frame=frame))

    assert result.frame_operator[0][1].as_fractions() == (Fraction(0), Fraction(-1))
    assert result.frame_operator[1][0].as_fractions() == (Fraction(0), Fraction(1))
    assert result.tight is False
    assert result.tight_residual[0][0].as_fractions() == (
        Fraction(-2),
        Fraction(0),
    )
    assert result.tight_residual[1][1].as_fractions() == (
        Fraction(2),
        Fraction(0),
    )


def test_mub_accepts_scaled_nonunit_representatives() -> None:
    standard = ComplexFrame(
        dimension=2,
        vectors=((_z(2), _z(0)), (_z(0), _z(3))),
    )
    hadamard = ComplexFrame(
        dimension=2,
        vectors=((_z(5), _z(5)), (_z(7), _z(-7))),
    )

    result = _mutually_unbiased_bases(
        MutuallyUnbiasedBasesRequest(dimension=2, bases=(standard, hadamard))
    )

    assert result.is_mutually_unbiased is True
    assert result.cross_gram_squared[0][0][0].as_integer_ratio() == (1, 2)


def test_sic_ledgers_are_invariant_under_independent_representative_scaling() -> None:
    base = ComplexFrame(
        dimension=2,
        vectors=(
            (_z(1), _z(0)),
            (_z(0), _z(1)),
            (_z(1), _z(1)),
            (_z(1), _z(2)),
        ),
    )
    scaled = ComplexFrame(
        dimension=2,
        vectors=(
            (_z(2), _z(0)),
            (_z(0), _z(3)),
            (_z(4), _z(4)),
            (_z(5), _z(10)),
        ),
    )

    base_result = _sic_profile(SicProfileRequest(frame=base))
    scaled_result = _sic_profile(SicProfileRequest(frame=scaled))

    assert scaled_result.is_sic == base_result.is_sic
    assert scaled_result.cardinality_residual == base_result.cardinality_residual
    assert scaled_result.equiangular == base_result.equiangular
    assert scaled_result.common_squared_overlap == base_result.common_squared_overlap
    assert (
        scaled_result.common_squared_overlap_residual
        == base_result.common_squared_overlap_residual
    )
    assert scaled_result.squared_overlaps == base_result.squared_overlaps
    assert scaled_result.frame_operator == base_result.frame_operator
    assert scaled_result.tight_residual == base_result.tight_residual
    assert base_result.equiangular is False
    assert base_result.common_squared_overlap is None
    assert base_result.common_squared_overlap_residual is None
    assert (
        type(base_result).model_validate_json(base_result.model_dump_json())
        == base_result
    )


def test_complex_accumulation_height_is_admitted_before_arithmetic() -> None:
    huge = GaussianRational.from_fractions(Fraction(10**127), Fraction(0))
    frame = ComplexFrame(dimension=1, vectors=((huge,),) * 64)
    with pytest.raises(OperationResourceAdmissionError, match="accumulation"):
        _sic_profile(SicProfileRequest(frame=frame))


def test_complex_derived_denominator_growth_is_admitted_before_basis_grams() -> None:
    primes = (
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
        73,
        79,
        83,
        89,
        97,
        101,
        103,
        107,
        109,
        113,
        127,
        131,
        137,
    )
    denominators = []
    for prime in primes:
        denominator = prime
        while len(str(denominator)) < 70:
            denominator *= prime
        denominators.append(denominator)
    vector = tuple(
        GaussianRational.from_fractions(Fraction(1, denominator), Fraction(0))
        for denominator in denominators
    )
    frame = ComplexFrame(dimension=33, vectors=(vector,) * 33)

    with pytest.raises(OperationResourceAdmissionError, match="height") as error:
        _mutually_unbiased_bases(
            MutuallyUnbiasedBasesRequest(dimension=33, bases=(frame,))
        )
    assert error.value.errors()[0]["type"] == "frames.complex_inner_product_height"


@pytest.mark.parametrize("operation", (_complex_frame_profile, _sic_profile))
def test_complex_profiles_reject_empty_frames_before_tightness(
    operation: object,
) -> None:
    empty = ComplexFrame(dimension=2, vectors=())
    request_type = (
        ComplexFrameProfileRequest
        if operation is _complex_frame_profile
        else SicProfileRequest
    )
    with pytest.raises(OperationDomainValidationError) as error:
        operation(request_type(frame=empty))  # type: ignore[operator]
    assert error.value.errors()[0]["type"] == "frames.empty_complex_frame"


def test_complex_profile_rejects_forged_vector_axes_at_native_boundary() -> None:
    malformed = ComplexFrame.model_construct(dimension=2, vectors=((_z(1),),))
    request = ComplexFrameProfileRequest.model_construct(frame=malformed)

    with pytest.raises(OperationDomainValidationError) as error:
        _complex_frame_profile(request)
    assert error.value.errors()[0]["type"] == (
        "frames.complex_vector_dimension_mismatch"
    )


def test_complex_profile_rejects_forged_noncanonical_or_oversized_scalars() -> None:
    noncanonical = CanonicalRational.model_construct(num=2, den=4)
    scalar = GaussianRational.model_construct(real=noncanonical, imaginary=_z(0).real)
    frame = ComplexFrame.model_construct(dimension=1, vectors=((scalar,),))
    request = ComplexFrameProfileRequest.model_construct(frame=frame)
    with pytest.raises(OperationDomainValidationError) as error:
        _complex_frame_profile(request)
    assert error.value.errors()[0]["type"] == "frames.complex_scalar_component"

    oversized = CanonicalRational.model_construct(num=10**5000, den=1)
    scalar = GaussianRational.model_construct(real=oversized, imaginary=_z(0).imaginary)
    frame = ComplexFrame.model_construct(dimension=1, vectors=((scalar,),))
    request = ComplexFrameProfileRequest.model_construct(frame=frame)
    with pytest.raises(OperationResourceAdmissionError) as error:
        _complex_frame_profile(request)
    assert error.value.errors()[0]["type"] == "frames.complex_scalar_height"


def test_complex_profile_rejects_scalar_height_before_expansion() -> None:
    huge = GaussianRational.from_fractions(Fraction(10**129), Fraction(0))
    frame = ComplexFrame(dimension=1, vectors=((huge,),))
    with pytest.raises(OperationResourceAdmissionError, match="scalar components"):
        _sic_profile(SicProfileRequest(frame=frame))


def test_sic_profile_rejects_forged_structural_residuals() -> None:
    result = _sic_profile(
        SicProfileRequest(frame=ComplexFrame(dimension=1, vectors=((_z(1),),)))
    )
    forged = json.loads(result.model_dump_json())
    forged["cardinality_residual"] = "1"
    with pytest.raises(ValueError, match="cardinality residual"):
        type(result).model_validate_json(json.dumps(forged))

    forged = json.loads(result.model_dump_json())
    forged["common_squared_overlap_residual"] = None
    with pytest.raises(ValueError, match="present together"):
        type(result).model_validate_json(json.dumps(forged))

    forged = json.loads(result.model_dump_json())
    forged["equiangular"] = False
    with pytest.raises(ValueError, match="equiangular status"):
        type(result).model_validate_json(json.dumps(forged))

    forged = json.loads(result.model_dump_json())
    forged["common_squared_overlap"] = {"num": "0", "den": "1"}
    forged["common_squared_overlap_residual"] = {"num": "-1", "den": "2"}
    with pytest.raises(ValueError, match="common overlap"):
        type(result).model_validate_json(json.dumps(forged))

    forged = json.loads(result.model_dump_json())
    forged["common_squared_overlap_residual"] = {"num": "1", "den": "1"}
    with pytest.raises(ValueError, match="residual"):
        type(result).model_validate_json(json.dumps(forged))

    forged = json.loads(result.model_dump_json())
    forged["is_sic"] = False
    with pytest.raises(ValueError, match="SIC status"):
        type(result).model_validate_json(json.dumps(forged))


@pytest.mark.parametrize(
    ("operation", "request_type"),
    (
        (_tight_equiangular_profile, VectorFamily),
        (_complex_frame_profile, ComplexFrameProfileRequest),
        (_mutually_unbiased_bases, MutuallyUnbiasedBasesRequest),
        (_sic_profile, SicProfileRequest),
    ),
)
def test_new_frame_operations_reject_untyped_native_requests(
    operation: object, request_type: type[object]
) -> None:
    expected_message = (
        "VectorFamily" if operation is _tight_equiangular_profile else "request"
    )
    with pytest.raises(OperationDomainValidationError, match=expected_message) as error:
        operation({})  # type: ignore[operator]
    expected_code = (
        "frames.vector_family_type"
        if operation is _tight_equiangular_profile
        else "frames.request_type"
    )
    assert error.value.errors()[0]["type"] == expected_code
    assert request_type.__name__ in str(error.value)


@pytest.mark.parametrize("operation", (_gram, _coherence, _frame_potential))
def test_existing_frame_operations_reject_untyped_native_requests(
    operation: object,
) -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        operation({})  # type: ignore[operator]
    assert error.value.errors()[0]["type"] == "frames.vector_family_type"


def test_existing_frame_operations_reject_forged_vector_axes_at_native_boundary() -> (
    None
):
    malformed = VectorFamily.model_construct(dimension=2, vectors=((1,),))
    with pytest.raises(OperationDomainValidationError) as error:
        _gram(malformed)
    assert error.value.errors()[0]["type"] == "frames.vector_dimension_mismatch"


def test_coherence_is_exact_and_carries_canonical_maximizer() -> None:
    result = _coherence(
        VectorFamily.model_validate(
            {
                "dimension": len(([[1, 1], [1, 0], [0, 1]])[0]),
                "vectors": [[1, 1], [1, 0], [0, 1]],
            }
        )
    )
    assert result.coherence_squared.as_integer_ratio() == (1, 2)
    assert result.maximizing_pair == (0, 2)


def test_potential_remains_exact_above_json_safe_integer() -> None:
    repeated = [1000] * 16
    final = [1000] * 15 + [999]
    vectors = (
        [repeated] * 5 + [final] + [[int(i == j) for j in range(16)] for i in range(16)]
    )
    result = _frame_potential(
        VectorFamily.model_validate(
            {"dimension": len((vectors)[0]), "vectors": vectors}
        )
    )
    expected = sum(
        sum(a * b for a, b in zip(left, right, strict=True)) ** 2
        for left in result.vectors
        for right in result.vectors
    )
    assert result.potential == expected


def test_flint_gram_reconstructs_dot_products_and_quadratic_form() -> None:
    vectors = tuple(
        tuple(((row * 17 + column * 31) % 11) - 5 for column in range(128))
        for row in range(256)
    )
    result = gram(VectorFamily(dimension=len(vectors[0]), vectors=vectors))

    assert result.gram[17][203] == sum(
        left * right for left, right in zip(vectors[17], vectors[203], strict=True)
    )
    coefficients = (2, -3, 1)
    indices = (4, 91, 177)
    quadratic = sum(
        coefficients[i] * result.gram[indices[i]][indices[j]] * coefficients[j]
        for i in range(len(indices))
        for j in range(len(indices))
    )
    combined = tuple(
        sum(coefficients[i] * vectors[indices[i]][column] for i in range(len(indices)))
        for column in range(len(vectors[0]))
    )
    assert quadratic == sum(entry * entry for entry in combined) >= 0


def test_repeated_basis_retains_exact_frame_results_and_wire_values() -> None:
    vectors = _repeated_standard_basis(dimension=2, repeats=2)
    result = gram(VectorFamily(dimension=len(vectors[0]), vectors=vectors))
    assert result.gram == ((1, 0, 1, 0), (0, 1, 0, 1), (1, 0, 1, 0), (0, 1, 0, 1))
    potential = _frame_potential(
        VectorFamily(dimension=len(vectors[0]), vectors=vectors)
    )
    assert potential.potential == 8
    coherence = _coherence(VectorFamily(dimension=len(vectors[0]), vectors=vectors))
    assert coherence.coherence_squared.as_integer_ratio() == (1, 1)
    assert coherence.maximizing_pair == (1, 3)
    assert (
        GramResult.model_validate_json(
            encode_strict_json(result.model_dump(mode="json"))
        )
        == result
    )
    assert type(potential).model_validate_json(potential.model_dump_json()) == potential


def test_sparse_high_height_gram_retains_zero_and_repeated_dot_products() -> None:
    vectors = ((1_000, 0), (0, 1_000)) * 2
    result = gram(VectorFamily(dimension=len(vectors[0]), vectors=vectors))
    assert result.gram == (
        (1_000_000, 0, 1_000_000, 0),
        (0, 1_000_000, 0, 1_000_000),
        (1_000_000, 0, 1_000_000, 0),
        (0, 1_000_000, 0, 1_000_000),
    )
    assert GramResult.model_validate_json(result.model_dump_json()) == result


def test_high_coefficients_retain_exact_gram_entries() -> None:
    vectors = ((70_000_000, 70_000_000),) * 2
    result = _gram(VectorFamily(dimension=len(vectors[0]), vectors=vectors))
    expected = 2 * 70_000_000**2
    assert expected > 2**53
    assert result.gram == ((expected, expected), (expected, expected))


@pytest.mark.scale
def test_result_sensitive_operations_diverge_at_full_carrier_boundary() -> None:
    dimension = 512
    vectors = _repeated_standard_basis(dimension=dimension, repeats=2)
    family = VectorFamily(dimension=len(vectors[0]), vectors=vectors)

    assert len(vectors) == MAX_VECTOR_CELLS // dimension
    gram_result = gram(family)
    potential = _frame_potential(
        VectorFamily(dimension=len(vectors[0]), vectors=vectors)
    )
    coherence = _coherence(VectorFamily(dimension=len(vectors[0]), vectors=vectors))
    assert len(gram_result.gram) == MAX_VECTOR_CELLS // dimension
    assert gram_result.gram[0][dimension] == 1
    assert potential.potential == 2 * (MAX_VECTOR_CELLS // dimension)
    assert coherence.coherence_squared.as_integer_ratio() == (1, 1)
    assert coherence.maximizing_pair == (
        dimension - 1,
        MAX_VECTOR_CELLS // dimension - 1,
    )
    assert encode_strict_json(gram_result.model_dump(mode="json"))
    assert encode_strict_json(potential.model_dump(mode="json"))


@pytest.mark.scale
def test_sparse_high_height_gram_is_admitted_by_occupancy() -> None:
    dimension = 512
    vectors = tuple(
        tuple(1_000 * entry for entry in vector)
        for vector in _repeated_standard_basis(dimension=dimension, repeats=2)
    )
    family = VectorFamily(dimension=len(vectors[0]), vectors=vectors)
    naive_entry_bound = 512 * 1_000**2
    naive_chars = len(str(naive_entry_bound)) + int(naive_entry_bound > 0)
    assert naive_chars > 0
    result = gram(family)
    encoded = encode_strict_json(result.model_dump(mode="json"))
    assert result.gram[0][0] == 1_000_000
    assert result.gram[0][1] == 0
    assert result.gram[0][512] == 1_000_000
    assert encoded


def test_sparse_row_norm_controls_gram_entry_admission() -> None:
    family = VectorFamily(dimension=2, vectors=((70_000_000, 0),))

    result = _gram(VectorFamily(dimension=family.dimension, vectors=family.vectors))

    assert result.gram == ((4_900_000_000_000_000,),)


@pytest.mark.scale
def test_dense_high_height_gram_uses_the_structural_work_bound() -> None:
    dimension = 512
    basis = tuple(
        tuple(1_000 if row == column else 999 for column in range(dimension))
        for row in range(dimension)
    )
    vectors = basis * 2
    result = _gram(VectorFamily(dimension=len(vectors[0]), vectors=vectors))

    potential = _frame_potential(
        VectorFamily(dimension=len(vectors[0]), vectors=vectors)
    )
    diagonal = 1_000**2 + (dimension - 1) * 999**2
    off_diagonal = 2 * 1_000 * 999 + (dimension - 2) * 999**2
    expected = 4 * dimension * (diagonal**2 + (dimension - 1) * off_diagonal**2)
    assert result.gram[0][0] == diagonal
    assert potential.potential == expected


@pytest.mark.scale
def test_high_coefficients_remain_exact_within_the_cell_bound() -> None:
    dimension = 512
    vectors = ((4_000_000,) * dimension,) * (MAX_VECTOR_CELLS // dimension)

    result = _gram(VectorFamily(dimension=len(vectors[0]), vectors=vectors))

    assert result.gram[0][0] == dimension * 4_000_000**2


def test_flint_rank_rejects_nonspanning_family_above_previous_boundary() -> None:
    vector = (1,) * 32
    request = VectorFamily(dimension=len(vector), vectors=(vector,) * 64)

    with pytest.raises(OperationDomainValidationError) as error:
        _frame_potential(request)
    assert error.value.errors()[0]["type"] == "frames.frame_does_not_span"


def test_coherence_maximizer_matches_complete_exact_profile() -> None:
    vectors = _repeated_standard_basis(dimension=32, repeats=2)
    result = _coherence(VectorFamily(dimension=len(vectors[0]), vectors=vectors))
    gram_result = _gram(VectorFamily(dimension=len(vectors[0]), vectors=vectors)).gram
    candidates = (
        (
            Fraction(
                gram_result[left][right] ** 2,
                gram_result[left][left] * gram_result[right][right],
            ),
            (left, right),
        )
        for left in range(len(vectors))
        for right in range(left + 1, len(vectors))
    )

    maximum, pair = max(candidates)
    assert result.coherence_squared.as_integer_ratio() == (
        maximum.numerator,
        maximum.denominator,
    )
    assert result.maximizing_pair == pair


def test_gram_verifier_propagates_unexpected_kernel_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.topology.frames import operations

    claim = gram(VectorFamily(dimension=1, vectors=((1,),)))

    def unavailable(
        vectors: tuple[tuple[int, ...], ...],
    ) -> tuple[tuple[int, ...], ...]:
        raise ValueError("backend arithmetic unavailable")

    monkeypatch.setattr(operations, "integer_gram", unavailable)
    with pytest.raises(ValueError, match="backend arithmetic unavailable"):
        verify_gram(claim)


def _sylvester_hadamard(order: int) -> tuple[tuple[int, ...], ...]:
    rows: tuple[tuple[int, ...], ...] = ((1,),)
    while len(rows) < order:
        rows = tuple(row + row for row in rows) + tuple(
            row + tuple(-entry for entry in row) for row in rows
        )
    return rows


def test_dimension_32_standard_hadamard_mub_skips_operator_height() -> None:
    dimension = 32
    standard = ComplexFrame(
        dimension=dimension,
        vectors=tuple(
            tuple(_z(int(row == column)) for column in range(dimension))
            for row in range(dimension)
        ),
    )
    hadamard = ComplexFrame(
        dimension=dimension,
        vectors=tuple(
            tuple(_z(entry) for entry in row) for row in _sylvester_hadamard(dimension)
        ),
    )

    result = _mutually_unbiased_bases(
        MutuallyUnbiasedBasesRequest(dimension=dimension, bases=(standard, hadamard))
    )

    assert result.is_mutually_unbiased is True
    assert result.basis_pair_count == 1


def test_sic_profile_rejects_asymmetric_squared_overlaps() -> None:
    result = _sic_profile(
        SicProfileRequest(frame=ComplexFrame(dimension=1, vectors=((_z(1),), (_z(1),))))
    )
    forged = json.loads(result.model_dump_json())
    forged["squared_overlaps"][1][0] = {"num": "0", "den": "1"}

    with pytest.raises(ValueError, match="symmetric"):
        type(result).model_validate_json(json.dumps(forged))
