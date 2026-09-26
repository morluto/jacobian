from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory._kempner_models import KempnerDigitSet
from jacobian.math.number_theory.kempner.operations import enclose_kempner_series
from jacobian.math.number_theory.modular_forms._models import (
    ModularFormCoordinatesHeckeRequest,
)
from jacobian.math.number_theory.modular_forms._tools import TOOLS
from jacobian.math.number_theory.modular_forms.basis import (
    BASIS_ID,
    modular_form_coordinates_hecke,
)
from jacobian.math.number_theory.modular_forms.operations import (
    named_q_expansion,
    sturm_bound,
)
from jacobian.math.number_theory.modular_forms.transforms import (
    formal_q_series_u_operator,
    formal_q_series_v_operator,
)
from jacobian.math.number_theory.modular_forms.values import (
    MAX_Q_TRANSFORM_OUTPUT_PRECISION,
    MAX_Q_TRANSFORM_SOURCE_ORDER,
    ModularFormCoordinates,
    ModularFormSpace,
)
from jacobian.math.polynomials.series._models import TruncatedSeries


def test_dense_kempner_prefix_recurrence_is_exact() -> None:
    family = KempnerDigitSet(base=10, allowed_digits=tuple(range(9)))
    result = enclose_kempner_series(family, 4)
    expected = sum(
        (Fraction(1, n) for n in range(1, 10_000) if "9" not in str(n)), Fraction(0)
    )
    assert result.partial_sum.as_fraction() == expected


def _coordinate_hecke(
    form: ModularFormCoordinates, index: int
) -> ModularFormCoordinates:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "modular_form.coordinates.hecke.apply"
    )
    return tool.run(ModularFormCoordinatesHeckeRequest(form=form, index=index))


def test_gamma0_sturm_and_coordinate_hecke_metadata() -> None:
    space = ModularFormSpace(level=1, weight=4, kind="M")
    assert sturm_bound(space).bound == 0
    form = ModularFormCoordinates(
        space=space,
        basis_id=BASIS_ID,
        coordinates=(CanonicalRational(num=1, den=1),),
    )
    transformed = _coordinate_hecke(form, 2)
    assert transformed.space == form.space
    assert transformed.basis_id == form.basis_id
    expected_eigenvalue = sum(divisor**3 for divisor in (1, 2))
    assert transformed.coordinates == (
        CanonicalRational(num=expected_eigenvalue, den=1),
    )


def test_q_transforms_accept_the_exact_last_source_index_boundary() -> None:
    space = ModularFormSpace(level=1, weight=4, kind="M")
    short = named_q_expansion(space, "E4", 3)
    longer = named_q_expansion(space, "E4", 4)

    assert formal_q_series_u_operator(
        short.q_expansion, 2, 2
    ) == formal_q_series_u_operator(longer.q_expansion, 2, 2)
    assert formal_q_series_v_operator(
        short.q_expansion, 2, 2
    ) == formal_q_series_v_operator(longer.q_expansion, 2, 2)


def test_formal_u_and_v_return_only_truncated_series_values() -> None:
    source_space = ModularFormSpace(level=1, weight=4, kind="M")
    source = named_q_expansion(source_space, "E4", 8)
    for transform in (formal_q_series_u_operator, formal_q_series_v_operator):
        result = transform(source.q_expansion, 2, 2)
        assert result.truncation_order == 2
        assert not hasattr(result, "space")
        restored = type(result).model_validate(result.model_dump())
        assert restored == result


def test_named_form_native_admission_is_stable() -> None:
    with pytest.raises(OperationDomainValidationError):
        named_q_expansion("bad", "E4", 2)  # type: ignore[arg-type]
    with pytest.raises(OperationDomainValidationError):
        named_q_expansion(ModularFormSpace(level=1, weight=6, kind="M"), "E4", 2)


def test_sturm_level_one_boundary_remains_explicit() -> None:
    assert sturm_bound(ModularFormSpace(level=1, weight=12, kind="M")).bound == 1


def test_q_transforms_reject_forged_scalar_carriers_before_arithmetic() -> None:
    forged_scalar = CanonicalRational.model_construct(num=1, den=0)
    forged_series = TruncatedSeries.model_construct(
        variable="q", truncation_order=2, coefficients=(forged_scalar, forged_scalar)
    )
    for transform in (formal_q_series_u_operator, formal_q_series_v_operator):
        with pytest.raises(OperationDomainValidationError):
            transform(forged_series, 2, 1)


def test_q_transforms_reject_unadmitted_prefix_and_output_before_indexing() -> None:
    source = named_q_expansion(ModularFormSpace(level=1, weight=4, kind="M"), "E4", 2)
    with pytest.raises(OperationResourceAdmissionError):
        formal_q_series_u_operator(
            source.q_expansion, 2, MAX_Q_TRANSFORM_OUTPUT_PRECISION + 1
        )
    forged_series = TruncatedSeries.model_construct(
        variable="q",
        truncation_order=MAX_Q_TRANSFORM_SOURCE_ORDER + 1,
        coefficients=(),
    )
    with pytest.raises(OperationResourceAdmissionError):
        formal_q_series_v_operator(forged_series, 2, 1)


def test_formal_operator_prime_bound_precedes_primality() -> None:
    source = named_q_expansion(ModularFormSpace(level=1, weight=4, kind="M"), "E4", 2)
    with pytest.raises(OperationResourceAdmissionError):
        formal_q_series_v_operator(source.q_expansion, 10**100, 1)


def test_coordinate_hecke_rejects_index_outside_its_envelope() -> None:
    form = ModularFormCoordinates(
        space=ModularFormSpace(level=1, weight=4, kind="M"),
        basis_id=BASIS_ID,
        coordinates=(CanonicalRational(num=1, den=1),),
    )

    with pytest.raises(OperationResourceAdmissionError) as error:
        modular_form_coordinates_hecke(form, MAX_Q_TRANSFORM_SOURCE_ORDER + 1)
    assert error.value.errors()[0]["type"] == (
        "modular_form.coordinates_hecke_index_bound"
    )


def test_coordinate_hecke_weight_zero_uses_inverse_divisor_factor() -> None:
    space = ModularFormSpace(level=1, weight=0, kind="M")
    source = ModularFormCoordinates(
        space=space,
        basis_id=BASIS_ID,
        coordinates=(CanonicalRational(num=1, den=1),),
    )
    result = _coordinate_hecke(source, 3)
    assert result.space == space
    expected_eigenvalue = sum((Fraction(1, divisor) for divisor in (1, 3)), Fraction(0))
    assert result.coordinates == (CanonicalRational.from_fraction(expected_eigenvalue),)
