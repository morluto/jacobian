"""Public contract, admission, and error semantics for exact complex tori."""

import copy
from fractions import Fraction

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError
from tests.math.geometry.complex_tori._support import (
    nonmonic_quadratic_torus,
    rational,
)

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_cancellation,
    request_execution,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.complex_tori import (
    LatticeComplexStructure,
    compute_neron_severi_lattice,
    compute_riemann_form_profile,
)
from jacobian.math.geometry.complex_tori import operations as complex_torus_operations
from jacobian.math.geometry.complex_tori._models import (
    NeronSeveriLatticeRequest,
    RiemannFormProfileRequest,
)
from jacobian.math.geometry.complex_tori._tools import TOOLS
from jacobian.math.lattices.invariant_forms import FormKind, IntegralBilinearForm
from jacobian.math.matrices.values import IntegerMatrix, RationalMatrix


def _elliptic_torus() -> LatticeComplexStructure:
    return LatticeComplexStructure(
        coordinate_axis=("e1", "e2"),
        complex_structure=RationalMatrix(
            entries=(
                (rational(0), rational(1)),
                (rational(-1), rational(0)),
            )
        ),
    )


def _deep_rational_mapping() -> object:
    nested: object = "0"
    for _ in range(1_500):
        nested = {"num": nested, "den": "1"}
    return nested


def test_exact_complex_structure_identity_is_required() -> None:
    torus = LatticeComplexStructure(
        coordinate_axis=("e1", "e2"),
        complex_structure=RationalMatrix(
            entries=(
                (rational(1), rational(0)),
                (rational(0), rational(1)),
            )
        ),
    )

    with pytest.raises(OperationDomainValidationError) as exc_info:
        compute_neron_severi_lattice(torus)

    assert exc_info.value.errors()[0]["type"] == ("complex_torus.not_complex_structure")


def test_complex_torus_axis_must_have_even_rank() -> None:
    with pytest.raises(ValidationError) as exc_info:
        LatticeComplexStructure(
            coordinate_axis=("e1", "e2", "e3"),
            complex_structure=RationalMatrix(
                entries=tuple(
                    tuple(rational(row == column) for column in range(3))
                    for row in range(3)
                )
            ),
        )

    assert exc_info.value.errors()[0]["type"] == "complex_torus.odd_lattice_rank"


def test_neron_severi_schema_requires_the_exact_real_domain_discriminator() -> None:
    payload = {"torus": _elliptic_torus().model_dump(mode="json")}
    missing_discriminator = copy.deepcopy(payload)
    del missing_discriminator["torus"]["complex_structure"]["domain"]

    validator = Draft202012Validator(NeronSeveriLatticeRequest.model_json_schema())
    assert not list(validator.iter_errors(payload))
    assert list(validator.iter_errors(missing_discriminator))
    with pytest.raises(ValidationError):
        NeronSeveriLatticeRequest.model_validate(missing_discriminator)


def test_riemann_profile_request_schema_requires_the_exact_real_discriminator() -> None:
    torus = _elliptic_torus()
    form = IntegralBilinearForm(
        coordinate_axis=torus.coordinate_axis,
        kind="ALTERNATING",
        matrix=IntegerMatrix(entries=(("0", "1"), ("-1", "0"))),
    )
    payload = {
        "torus": torus.model_dump(mode="json"),
        "form": form.model_dump(mode="json"),
    }
    missing_discriminator = copy.deepcopy(payload)
    del missing_discriminator["torus"]["complex_structure"]["domain"]

    validator = Draft202012Validator(RiemannFormProfileRequest.model_json_schema())
    assert not list(validator.iter_errors(payload))
    assert list(validator.iter_errors(missing_discriminator))
    with pytest.raises(ValidationError):
        RiemannFormProfileRequest.model_validate(missing_discriminator)


@pytest.mark.parametrize(
    ("axis", "kind", "expected_code"),
    (
        (("x", "y"), "ALTERNATING", "complex_torus.form_axis"),
        (("e1", "e2"), "BILINEAR", "complex_torus.form_kind"),
    ),
)
def test_profile_rejects_wrong_axis_or_form_kind(
    axis: tuple[str, str],
    kind: FormKind,
    expected_code: str,
) -> None:
    torus = _elliptic_torus()
    form = IntegralBilinearForm(
        coordinate_axis=axis,
        kind=kind,
        matrix=IntegerMatrix(entries=(("0", "1"), ("-1", "0"))),
    )

    with pytest.raises(OperationDomainValidationError) as exc_info:
        compute_riemann_form_profile(torus, form)

    assert exc_info.value.errors()[0]["type"] == expected_code


def test_profile_observes_cancellation_after_complex_structure_products() -> None:
    class CancelAfterChecks:
        def __init__(self, limit: int) -> None:
            self.checks = 0
            self.limit = limit

        def is_set(self) -> bool:
            self.checks += 1
            return self.checks >= self.limit

    torus = _elliptic_torus()
    form = IntegralBilinearForm(
        coordinate_axis=torus.coordinate_axis,
        kind="ALTERNATING",
        matrix=IntegerMatrix(entries=(("0", "1"), ("-1", "0"))),
    )
    cancellation = CancelAfterChecks(4)
    with (
        request_cancellation(cancellation),
        pytest.raises(
            OperationExecutionCancelledError,
            match="after exact complex-structure recognition",
        ),
    ):
        compute_riemann_form_profile(torus, form)
    assert cancellation.checks == 4


def test_neron_severi_owner_deadline_reaches_invariant_form_phases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checks = 0

    def monotonic() -> float:
        nonlocal checks
        checks += 1
        # started_at=10 fixes the owner deadline at 610. The first expired
        # observation occurs inside the nested invariant-form kernel.
        return 700.0 if checks >= 7 else 500.0

    monkeypatch.setattr(complex_torus_operations, "monotonic", monotonic)
    with (
        request_execution(started_at=10.0),
        pytest.raises(
            OperationExecutionTimeoutError,
            match="during exact invariant-form constraint expansion",
        ),
    ):
        compute_neron_severi_lattice(_elliptic_torus())


def test_riemann_profile_preserves_a_stricter_deadline_through_inertia(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torus = _elliptic_torus()
    form = IntegralBilinearForm(
        coordinate_axis=torus.coordinate_axis,
        kind="ALTERNATING",
        matrix=IntegerMatrix(entries=(("0", "1"), ("-1", "0"))),
    )
    checks = 0

    def monotonic() -> float:
        nonlocal checks
        checks += 1
        # The owner deadline would be 610, but the inherited 550 deadline must
        # remain authoritative when the nested inertia phase observes 560.
        return 560.0 if checks >= 8 else 500.0

    monkeypatch.setattr(complex_torus_operations, "monotonic", monotonic)
    with request_execution(started_at=10.0):
        bind_request_deadline(550.0)
        with pytest.raises(
            OperationExecutionTimeoutError,
            match="during exact rational congruence elimination",
        ):
            compute_riemann_form_profile(torus, form)
        execution = current_request_execution()
        assert execution is not None
        assert execution.deadline == 550.0


def test_raw_torus_validation_does_not_recurse_through_a_malformed_matrix() -> None:
    nested: object = None
    for _ in range(1_500):
        nested = {"next": nested}

    with pytest.raises(ValidationError):
        LatticeComplexStructure.model_validate(
            {
                "coordinate_axis": ["e1", "e2"],
                "complex_structure": {
                    "domain": "QQ",
                    "entries": [[nested]],
                },
            }
        )


@pytest.mark.parametrize("component", ("num", "den"))
def test_raw_neron_severi_request_rejects_deep_rational_components(
    component: str,
) -> None:
    payload = {"torus": _elliptic_torus().model_dump(mode="json")}
    payload["torus"]["complex_structure"]["entries"][0][0][component] = (
        _deep_rational_mapping()
    )

    with pytest.raises(ValidationError) as exc_info:
        NeronSeveriLatticeRequest.model_validate(payload)

    assert exc_info.value.errors(include_input=False)[0]["type"] == (
        "matrix.shape_mismatch"
    )


def test_raw_riemann_request_rejects_deep_rational_shaped_integer() -> None:
    torus = _elliptic_torus()
    form = IntegralBilinearForm(
        coordinate_axis=torus.coordinate_axis,
        kind="ALTERNATING",
        matrix=IntegerMatrix(entries=(("0", "1"), ("-1", "0"))),
    )
    payload = {
        "torus": torus.model_dump(mode="json"),
        "form": form.model_dump(mode="json"),
    }
    payload["form"]["matrix"]["entries"][0][0] = {
        "num": _deep_rational_mapping(),
        "den": "1",
    }

    with pytest.raises(ValidationError) as exc_info:
        RiemannFormProfileRequest.model_validate(payload)

    assert exc_info.value.errors(include_input=False)[0]["type"] == (
        "matrix.shape_mismatch"
    )


@pytest.mark.parametrize("malformed_part", ("row_axis", "coefficient_num"))
def test_raw_neron_severi_request_rejects_allowed_key_depth_traps(
    malformed_part: str,
) -> None:
    payload = {
        "torus": nonmonic_quadratic_torus().model_dump(mode="json"),
    }
    matrix = payload["torus"]["complex_structure"]
    if malformed_part == "row_axis":
        matrix["entries"] = [42]
        expected_type = "matrix.shape_mismatch"
    else:
        nested: object = "0"
        for _ in range(1_500):
            nested = {"next": nested}
        matrix["entries"][0][0]["coefficients_ascending"][0]["num"] = nested
        expected_type = "string_type"

    with pytest.raises(ValidationError) as exc_info:
        NeronSeveriLatticeRequest.model_validate(payload)

    assert exc_info.value.errors(include_input=False)[0]["type"] == expected_type


def test_neron_severi_admits_nested_work_before_testing_j_squared() -> None:
    dimension = 18
    height = 10**10
    zero = rational(0)
    torus = LatticeComplexStructure(
        coordinate_axis=tuple(f"e{index}" for index in range(dimension)),
        complex_structure=RationalMatrix(
            entries=tuple(
                tuple(
                    rational(height)
                    if row % 2 == 0 and column == row + 1
                    else rational(Fraction(-1, height))
                    if row % 2 == 1 and column == row - 1
                    else zero
                    for column in range(dimension)
                )
                for row in range(dimension)
            )
        ),
    )

    with pytest.raises(OperationDomainValidationError) as exc_info:
        compute_neron_severi_lattice(torus)

    assert exc_info.value.errors()[0]["type"] == "complex_torus.budget_exceeded"


def test_profile_admits_inertia_before_hodge_and_associated_form_products() -> None:
    dimension = 64
    zero = rational(0)
    block = (
        (rational(Fraction(1, 10)), rational(1)),
        (rational(Fraction(-101, 100)), rational(Fraction(-1, 10))),
    )
    torus = LatticeComplexStructure(
        coordinate_axis=tuple(f"e{index}" for index in range(dimension)),
        complex_structure=RationalMatrix(
            entries=tuple(
                tuple(
                    block[row % 2][column % 2] if row // 2 == column // 2 else zero
                    for column in range(dimension)
                )
                for row in range(dimension)
            )
        ),
    )
    form = IntegralBilinearForm(
        coordinate_axis=torus.coordinate_axis,
        kind="ALTERNATING",
        matrix=IntegerMatrix(
            entries=tuple(
                tuple(
                    "1"
                    if row % 2 == 0 and column == row + 1
                    else "-1"
                    if row % 2 == 1 and column == row - 1
                    else "0"
                    for column in range(dimension)
                )
                for row in range(dimension)
            )
        ),
    )

    with pytest.raises(OperationDomainValidationError) as exc_info:
        compute_riemann_form_profile(torus, form)

    assert exc_info.value.errors()[0]["type"] == "complex_torus.budget_exceeded"


def test_catalog_examples_round_trip_through_each_result_contract() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "complex_torus.neron_severi_lattice.compute",
        "complex_torus.riemann_form.profile.compute",
    }
    for tool in TOOLS:
        request = tool.request_type.model_validate(tool.examples[0].input)
        result = tool.run(request)
        round_tripped = tool.result_type.model_validate(result.model_dump(mode="json"))
        assert round_tripped == result
