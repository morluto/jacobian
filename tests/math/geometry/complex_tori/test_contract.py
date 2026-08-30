"""Public contract, admission, and error semantics for exact complex tori."""

import time

import pytest
from pydantic import ValidationError
from tests.math.geometry.complex_tori._support import rational

from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    request_execution,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.complex_tori import (
    LatticeComplexStructure,
    compute_neron_severi_lattice,
    compute_riemann_form_profile,
)
from jacobian.math.geometry.complex_tori._tools import TOOLS
from jacobian.math.lattices.invariant_forms import IntegralBilinearForm
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


@pytest.mark.parametrize(
    ("axis", "kind", "expected_code"),
    (
        (("x", "y"), "ALTERNATING", "complex_torus.form_axis"),
        (("e1", "e2"), "BILINEAR", "complex_torus.form_kind"),
    ),
)
def test_profile_rejects_wrong_axis_or_form_kind(
    axis: tuple[str, str],
    kind: str,
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


def test_outer_request_deadline_is_reused_before_exact_backend_work() -> None:
    started_at = time.monotonic()
    with request_execution(started_at):
        bind_request_deadline(started_at - 1)
        with pytest.raises(OperationExecutionTimeoutError):
            compute_neron_severi_lattice(_elliptic_torus())


def test_catalog_publishes_both_operations_with_replayable_examples() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "complex_torus.neron_severi_lattice.compute",
        "complex_torus.riemann_form.profile.compute",
    }
    for tool in TOOLS:
        request = tool.request_type.model_validate(tool.examples[0].input)
        result = tool.run(request)
        replayed = tool.result_type.model_validate(result.model_dump(mode="json"))
        assert replayed == result
