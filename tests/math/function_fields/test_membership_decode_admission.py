"""Shared decoded-profile admission preserves rational prime-place semantics."""

from time import monotonic

import pytest
from pydantic import ValidationError

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    request_execution,
)
from jacobian.math.function_fields._models import (
    FiniteFunctionField,
    FiniteFunctionFieldElement,
    FunctionFieldDivisor,
    FunctionFieldDivisorTerm,
    FunctionFieldPlace,
    FunctionFieldRiemannRochMembership,
    FunctionFieldRiemannRochMembershipRow,
    PrimeFieldPolynomial,
    PrimeFieldRationalFunction,
)
from jacobian.math.function_fields.valuation import membership_profile_valuations


def one(p: int) -> PrimeFieldRationalFunction:
    poly = PrimeFieldPolynomial(characteristic=p, coefficients=(1,))
    return PrimeFieldRationalFunction(numerator=poly, denominator=poly)


def profile(
    p: int = 5, *, extension: bool = False, reducible: bool = False
) -> dict[str, object]:
    unit = one(p)
    field = FiniteFunctionField(
        characteristic=p,
        defining_polynomial=(unit, unit, unit) if extension else (unit,),
    )
    element = FiniteFunctionFieldElement(
        field=field, coordinates=(unit, unit) if extension else (unit,)
    )
    poly = PrimeFieldPolynomial(
        characteristic=p, coefficients=(0, 0, 1) if reducible else (0, 1)
    )
    place = FunctionFieldPlace(
        field=field, kind="FINITE", prime_polynomial=poly, degree=poly.degree
    )
    row = FunctionFieldRiemannRochMembershipRow(
        place=place, element_valuation=0, divisor_multiplicity=1, sum=1
    )
    return {
        "element": element,
        "divisor": FunctionFieldDivisor(
            field=field, terms=(FunctionFieldDivisorTerm(place=place, multiplicity=1),)
        ),
        "status": "IN_SPACE",
        "profile": (row,),
    }


@pytest.mark.parametrize(
    "options", [{"p": 4}, {"extension": True}, {"reducible": True}]
)
def test_decoding_rejects_invalid_place_semantics(options: dict[str, object]) -> None:
    payload = profile(**options)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="rational-field prime place"):
        FunctionFieldRiemannRochMembership.model_validate(payload)


def largest_places() -> tuple[
    FiniteFunctionFieldElement, tuple[FunctionFieldPlace, ...]
]:
    p = 257
    unit = one(p)
    field = FiniteFunctionField(characteristic=p, defining_polynomial=(unit,))
    element = FiniteFunctionFieldElement(field=field, coordinates=(unit,))
    coefficients: list[tuple[int, ...]] = [(a, 1) for a in range(p)]
    coefficients += [
        (a, 0, 1) for a in range(1, p) if pow((-a) % p, (p - 1) // 2, p) == p - 1
    ][:24]
    places = tuple(
        FunctionFieldPlace(
            field=field,
            kind="FINITE",
            prime_polynomial=PrimeFieldPolynomial(characteristic=p, coefficients=c),
            degree=len(c) - 1,
        )
        for c in coefficients
    )
    return element, places


def test_maximum_row_axis_accepts_valid_prime_places() -> None:
    element, places = largest_places()
    assert len(places) == 281
    assert membership_profile_valuations(element, places) == (0,) * 281


def test_profile_replay_obeys_shared_deadline() -> None:
    element, places = largest_places()
    started = monotonic()
    with (
        request_execution(started, outer_deadline=started - 1),
        pytest.raises(OperationExecutionTimeoutError),
    ):
        membership_profile_valuations(element, places)


def test_profile_replay_checks_cancellation_during_rows() -> None:
    class CancelAfterChecks:
        checks = 0

        def is_set(self) -> bool:
            self.checks += 1
            return self.checks >= 290

    element, places = largest_places()
    signal = CancelAfterChecks()
    with (
        request_execution(monotonic(), cancellation_signal=signal),
        pytest.raises(OperationExecutionCancelledError),
    ):
        membership_profile_valuations(element, places)
    assert signal.checks == 290


def test_aggregate_decode_work_is_refused_before_irreducibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.catalog.models import OperationResourceAdmissionError
    from jacobian.math.function_fields import valuation

    p = 257
    unit = one(p)
    field = FiniteFunctionField(characteristic=p, defining_polynomial=(unit,))
    numerator = PrimeFieldPolynomial(
        characteristic=p, coefficients=(1,) + (0,) * 11 + (1,)
    )
    denominator = PrimeFieldPolynomial(
        characteristic=p, coefficients=(2,) + (0,) * 11 + (1,)
    )
    element = FiniteFunctionFieldElement(
        field=field,
        coordinates=(
            PrimeFieldRationalFunction(numerator=numerator, denominator=denominator),
        ),
    )
    places = tuple(
        FunctionFieldPlace(
            field=field,
            kind="FINITE",
            degree=12,
            prime_polynomial=PrimeFieldPolynomial(
                characteristic=p, coefficients=(i % p, i // p) + (0,) * 10 + (1,)
            ),
        )
        for i in range(281)
    )

    def forbidden_test(*args: object, **kwargs: object) -> bool:
        pytest.fail("irreducibility ran before aggregate admission")

    monkeypatch.setattr(valuation, "is_irreducible_over_gf", forbidden_test)
    with pytest.raises(OperationResourceAdmissionError) as caught:
        membership_profile_valuations(element, places)
    assert (
        caught.value.errors()[0]["type"]
        == "function_field.riemann_roch_membership_work_exceeds_envelope"
    )


@pytest.mark.parametrize(
    "mutation", ["empty", "omit_infinity", "omit_finite", "divisor_coefficient"]
)
def test_decode_binds_complete_support_and_divisor(mutation: str) -> None:
    from jacobian.math.function_fields.operations import (
        function_field_riemann_roch_membership,
    )

    unit = one(5)
    field = FiniteFunctionField(characteristic=5, defining_polynomial=(unit,))
    element = FiniteFunctionFieldElement(
        field=field,
        coordinates=(
            PrimeFieldRationalFunction(
                numerator=PrimeFieldPolynomial(characteristic=5, coefficients=(0, 1)),
                denominator=unit.denominator,
            ),
        ),
    )
    divisor = FunctionFieldDivisor(field=field, terms=())
    result = function_field_riemann_roch_membership(element, divisor)
    payload = result.model_dump()
    if mutation == "empty":
        payload["profile"] = ()
        payload["status"] = "IN_SPACE"
    elif mutation == "omit_infinity":
        payload["profile"] = tuple(
            row.model_dump() for row in result.profile if row.place.kind == "FINITE"
        )
        payload["status"] = "IN_SPACE"
    elif mutation == "omit_finite":
        payload["profile"] = tuple(
            row.model_dump() for row in result.profile if row.place.kind == "INFINITE"
        )
    else:
        payload["profile"] = tuple(
            row.model_copy(
                update={"divisor_multiplicity": 2, "sum": row.element_valuation + 2}
            ).model_dump()
            for row in result.profile
        )
        payload["status"] = "IN_SPACE"
    with pytest.raises(ValidationError, match="complete support"):
        FunctionFieldRiemannRochMembership.model_validate(payload)


def test_semantically_duplicate_places_are_rejected() -> None:
    payload = profile()
    row = payload["profile"][0]  # type: ignore[index]
    duplicate = row.model_copy(
        update={
            "place": row.place.model_copy(
                update={
                    "prime_polynomial": PrimeFieldPolynomial(
                        characteristic=5, coefficients=(0, 2)
                    )
                }
            )
        }
    )
    payload["profile"] = tuple(
        sorted((row, duplicate), key=lambda r: r.place.model_dump_json())
    )
    with pytest.raises(ValidationError, match="complete support"):
        FunctionFieldRiemannRochMembership.model_validate(payload)


def test_model_decode_preserves_the_aggregate_resource_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.function_fields import valuation
    from jacobian.math.function_fields.operations import (
        function_field_riemann_roch_membership,
    )

    unit = one(5)
    field = FiniteFunctionField(characteristic=5, defining_polynomial=(unit,))
    element = FiniteFunctionFieldElement(
        field=field,
        coordinates=(
            PrimeFieldRationalFunction(
                numerator=PrimeFieldPolynomial(characteristic=5, coefficients=(0, 1)),
                denominator=unit.denominator,
            ),
        ),
    )
    result = function_field_riemann_roch_membership(
        element, FunctionFieldDivisor(field=field, terms=())
    )
    monkeypatch.setattr(valuation, "MAX_RIEMANN_ROCH_MEMBERSHIP_FACTOR_WORK", 0)
    with pytest.raises(ValidationError) as error:
        FunctionFieldRiemannRochMembership.model_validate_json(result.model_dump_json())
    assert (
        error.value.errors()[0]["type"]
        == "function_field.riemann_roch_membership_work_exceeds_envelope"
    )
