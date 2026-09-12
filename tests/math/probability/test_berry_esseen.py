"""Behavioral tests for the pinned finite i.i.d. Berry--Esseen operation."""

import json
import time
from collections.abc import Callable
from fractions import Fraction
from importlib import import_module
from math import isqrt
from typing import cast

import pytest

from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    request_execution,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.probability import _berry_esseen as berry_module
from jacobian.math.probability._berry_esseen import (
    BERRY_ESSEEN_CONSTANT,
    BERRY_ESSEEN_THEOREM_VARIANT,
    MAX_BERRY_ESSEEN_ATOMS,
    MAX_BERRY_ESSEEN_SAMPLE_COUNT,
    BerryEsseenRequest,
    BerryEsseenResult,
    berry_esseen_bound,
)
from jacobian.math.probability._berry_esseen_tools import BERRY_ESSEEN_OPERATION


def _distribution(*atoms: tuple[int | Fraction, Fraction]) -> dict[str, object]:
    return {
        "atoms": [
            {
                "value": {
                    "num": Fraction(value).numerator,
                    "den": Fraction(value).denominator,
                },
                "probability": {
                    "num": probability.numerator,
                    "den": probability.denominator,
                },
            }
            for value, probability in atoms
        ]
    }


def _request(
    distribution: dict[str, object], sample_count: int = 1
) -> BerryEsseenRequest:
    return BerryEsseenRequest.model_validate(
        {"distribution": distribution, "sample_count": sample_count}
    )


def _five_atom_oracle_distribution() -> dict[str, object]:
    # Issue #2503's independent exact oracle: mean=33, variance=4224/5,
    # rho=161792/5 and rho^2/sigma^6=62410/35937.
    return _distribution(
        (1, Fraction(1, 5)),
        (9, Fraction(1, 5)),
        (25, Fraction(1, 5)),
        (49, Fraction(1, 5)),
        (81, Fraction(1, 5)),
    )


def test_five_atom_oracle_returns_exact_source_moments() -> None:
    result = berry_esseen_bound(_request(_five_atom_oracle_distribution(), 5))

    assert result.mean.as_fraction() == Fraction(33)
    assert result.variance.as_fraction() == Fraction(4224, 5)
    assert result.third_absolute_central_moment.as_fraction() == Fraction(161792, 5)
    assert (
        result.third_absolute_central_moment.as_fraction() ** 2
        / result.variance.as_fraction() ** 3
        == Fraction(62410, 35937)
    )


def _dyadic_sqrt_interval(value: Fraction) -> tuple[Fraction, Fraction]:
    scale = 1 << 64
    scaled_numerator = value.numerator * scale * scale
    if scaled_numerator % value.denominator == 0:
        exact_scaled = scaled_numerator // value.denominator
        root = isqrt(exact_scaled)
        if root * root == exact_scaled:
            exact = Fraction(root, scale)
            return exact, exact
    scaled_floor = scaled_numerator // value.denominator
    lower_numerator = isqrt(scaled_floor)
    return Fraction(lower_numerator, scale), Fraction(lower_numerator + 1, scale)


def test_symmetric_bernoulli_has_exact_single_sample_bound() -> None:
    result = berry_esseen_bound(
        _request(_distribution((0, Fraction(1, 2)), (1, Fraction(1, 2))))
    )
    lower, upper = _dyadic_sqrt_interval(Fraction(196, 625))

    assert result.theorem_variant == BERRY_ESSEEN_THEOREM_VARIANT
    assert result.universal_constant.as_fraction() == BERRY_ESSEEN_CONSTANT
    assert result.mean.as_fraction() == Fraction(1, 2)
    assert result.variance.as_fraction() == Fraction(1, 4)
    assert result.third_absolute_central_moment.as_fraction() == Fraction(1, 8)
    assert result.bound_squared.as_fraction() == Fraction(196, 625)
    assert result.bound_lower.as_fraction() == lower
    assert result.bound_upper.as_fraction() == upper
    assert lower * lower <= Fraction(196, 625) <= upper * upper
    assert (lower * (1 << 64)).denominator == 1
    assert upper - lower == Fraction(1, 1 << 64)


def test_asymmetric_bernoulli_and_n_monotonicity() -> None:
    distribution = _distribution((0, Fraction(1, 3)), (2, Fraction(2, 3)))
    one = berry_esseen_bound(_request(distribution, 1))
    four = berry_esseen_bound(_request(distribution, 4))

    assert one.mean.as_fraction() == Fraction(4, 3)
    assert one.variance.as_fraction() == Fraction(8, 9)
    assert one.third_absolute_central_moment.as_fraction() == Fraction(80, 81)
    assert four.bound_squared.as_fraction() == one.bound_squared.as_fraction() / 4
    assert four.bound_upper.as_fraction() <= one.bound_upper.as_fraction()


def test_large_positive_count_is_admitted_without_sample_expansion() -> None:
    result = berry_esseen_bound(
        _request(_distribution((0, Fraction(1, 2)), (1, Fraction(1, 2))), 257)
    )

    assert result.source.sample_count == 257
    assert result.bound_squared.as_fraction() == Fraction(196, 625 * 257)


def test_native_trillion_plus_count_is_admitted_from_growth() -> None:
    result = berry_esseen_bound(
        _request(
            _distribution((0, Fraction(1, 2)), (1, Fraction(1, 2))),
            1_000_000_000_001,
        )
    )

    assert result.source.sample_count == 1_000_000_000_001
    assert result.bound_squared.as_fraction() == Fraction(196, 625 * 1_000_000_000_001)


def test_affine_rescaling_cancels_before_intermediate_height() -> None:
    result = berry_esseen_bound(
        _request(_distribution((0, Fraction(1, 2)), (10**100, Fraction(1, 2))))
    )

    assert result.mean.as_fraction() == Fraction(10**100, 2)
    assert result.variance.as_fraction() == Fraction(10**200, 4)
    assert result.bound_squared.as_fraction() == Fraction(196, 625)
    lower, upper = _dyadic_sqrt_interval(Fraction(196, 625))
    assert result.bound_lower.as_fraction() == lower
    assert result.bound_upper.as_fraction() == upper


def test_successful_compute_does_not_replay_distribution_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kernel result construction does not repeat source normalization."""

    berry_module = import_module("jacobian.math.probability._berry_esseen")
    call_count = 0
    original = cast(Callable[..., object], berry_module.require_input_distribution)

    def counted(*args: object, **kwargs: object) -> object:
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(berry_module, "require_input_distribution", counted)
    result = berry_esseen_bound(_request(_five_atom_oracle_distribution(), 7))

    assert result.mean.as_fraction() == Fraction(33)
    assert call_count == 1


def test_sample_count_boundary_is_bounded_and_preflighted() -> None:
    distribution = _request(
        _distribution((0, Fraction(1, 2)), (1, Fraction(1, 2))),
        MAX_BERRY_ESSEEN_SAMPLE_COUNT,
    )
    assert distribution.sample_count == MAX_BERRY_ESSEEN_SAMPLE_COUNT

    with pytest.raises(OperationResourceAdmissionError, match="squared bound"):
        berry_esseen_bound(distribution)

    over_digits = "1" + "0" * 512
    with pytest.raises(ValueError, match="digit bound"):
        BerryEsseenRequest.model_validate(
            {
                "distribution": _distribution((0, Fraction(1, 2)), (1, Fraction(1, 2))),
                "sample_count": int(over_digits),
            }
        )
    with pytest.raises(ValueError):
        BerryEsseenRequest.model_validate_json(
            json.dumps(
                {
                    "distribution": json.loads(
                        _request(
                            _distribution((0, Fraction(1, 2)), (1, Fraction(1, 2)))
                        ).distribution.model_dump_json()
                    ),
                    "sample_count": over_digits,
                }
            )
        )
    with pytest.raises(
        OperationDomainValidationError, match="sample_count must be positive"
    ):
        berry_esseen_bound(
            BerryEsseenRequest.model_construct(
                distribution=distribution.distribution,
                sample_count=0,
            )
        )


def test_sample_count_uses_the_exact_integer_wire_contract() -> None:
    schema = BerryEsseenRequest.model_json_schema()["properties"]["sample_count"]

    assert schema["type"] == "string"
    assert schema["pattern"].startswith("^[1-9]") or "0," in schema["pattern"]
    assert "511" in schema["pattern"] or schema.get("maxLength") == 513

    distribution = json.loads(
        _request(
            _distribution((0, Fraction(1, 2)), (1, Fraction(1, 2)))
        ).distribution.model_dump_json()
    )
    request = BerryEsseenRequest.model_validate_json(
        json.dumps({"distribution": distribution, "sample_count": "4"})
    )
    assert request.sample_count == 4

    for sample_count in ("0", "-1"):
        with pytest.raises(ValueError, match="must be positive"):
            BerryEsseenRequest.model_validate_json(
                json.dumps({"distribution": distribution, "sample_count": sample_count})
            )


def test_atom_count_boundary_is_admitted_and_overflow_is_preflighted() -> None:
    count = MAX_BERRY_ESSEEN_ATOMS
    distribution = _distribution(
        *((value, Fraction(1, count)) for value in range(count))
    )
    assert (
        len(berry_esseen_bound(_request(distribution)).source.distribution.atoms)
        == count
    )

    over_bound = _distribution(
        *((value, Fraction(1, count + 1)) for value in range(count + 1))
    )
    overflow_request = BerryEsseenRequest.model_validate(
        {"distribution": over_bound, "sample_count": 1}
    )
    with pytest.raises(OperationResourceAdmissionError, match="16384"):
        berry_esseen_bound(overflow_request)
    schema = BerryEsseenRequest.model_json_schema()
    atoms_schema = schema["$defs"]["FiniteRationalDistribution"]["properties"]["atoms"]
    assert atoms_schema["maxItems"] == MAX_BERRY_ESSEEN_ATOMS
    assert "BerryEsseenDistribution" not in schema.get("$defs", {})


def test_input_rational_height_boundary_is_enforced() -> None:
    denominator = 10**127 + 19
    first_value = Fraction(1, denominator)
    second_value = Fraction(denominator + 1, denominator)
    accepted = berry_esseen_bound(
        _request(
            _distribution(
                (first_value, Fraction(1, 2)),
                (second_value, Fraction(1, 2)),
            )
        )
    )
    assert accepted.source.distribution.atoms[-1].value.as_fraction() == second_value

    with pytest.raises(OperationResourceAdmissionError, match="128-digit bound"):
        berry_esseen_bound(
            _request(_distribution((0, Fraction(1, 2)), (10**128, Fraction(1, 2))))
        )


def test_input_height_is_checked_before_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("normalization must not precede input-height admission")

    monkeypatch.setattr(berry_module, "require_input_distribution", fail)
    with pytest.raises(OperationResourceAdmissionError, match="128-digit bound"):
        berry_esseen_bound(
            _request(_distribution((0, Fraction(1, 2)), (10**128, Fraction(1, 2))))
        )


def test_native_sample_count_rejects_non_integers() -> None:
    request = _request(_distribution((0, Fraction(1, 2)), (1, Fraction(1, 2))))
    for sample_count in ("4", True):
        with pytest.raises(
            OperationDomainValidationError, match="sample_count must be an integer"
        ):
            berry_esseen_bound(
                BerryEsseenRequest.model_construct(
                    distribution=request.distribution,
                    sample_count=sample_count,  # type: ignore[arg-type]
                )
            )


def test_berry_esseen_moment_scan_honors_an_expired_deadline() -> None:
    request = _request(_distribution((0, Fraction(1, 2)), (1, Fraction(1, 2))))
    with request_execution(time.monotonic()):
        bind_request_deadline(time.monotonic() - 1)
        with pytest.raises(OperationExecutionTimeoutError, match="deadline expired"):
            berry_esseen_bound(request)


def test_serialized_result_preserves_source_and_interval_invariants() -> None:
    result = berry_esseen_bound(_request(_five_atom_oracle_distribution(), 7))
    restored = BerryEsseenResult.model_validate_json(result.model_dump_json())
    lower = restored.bound_lower.as_fraction()
    upper = restored.bound_upper.as_fraction()
    squared = restored.bound_squared.as_fraction()

    assert restored == result
    assert restored.source == result.source
    assert lower <= upper
    assert lower * lower <= squared <= upper * upper


def test_result_rejects_forged_source_normalization() -> None:
    genuine = berry_esseen_bound(_request(_five_atom_oracle_distribution(), 7))
    payload = json.loads(genuine.model_dump_json())
    payload["source"]["distribution"]["atoms"][0]["probability"] = {
        "num": "1",
        "den": "1",
    }

    with pytest.raises(ValueError, match="sum exactly to 1"):
        BerryEsseenResult.model_validate_json(json.dumps(payload))


def test_result_rejects_forged_source_input_height() -> None:
    genuine = berry_esseen_bound(_request(_five_atom_oracle_distribution(), 7))
    payload = genuine.model_dump()
    payload["source"]["distribution"]["atoms"][-1]["value"] = {
        "num": 10**128,
        "den": 1,
    }

    with pytest.raises(ValueError, match="128-digit bound"):
        BerryEsseenResult.model_validate(payload)


def test_result_requires_consecutive_dyadic_grid_endpoints() -> None:
    genuine = berry_esseen_bound(_request(_five_atom_oracle_distribution(), 7))
    payload = genuine.model_dump()
    payload["bound_squared"] = {"num": 1, "den": 4}
    payload["bound_lower"] = {"num": 0, "den": 1}
    payload["bound_upper"] = {"num": 1, "den": 1}

    with pytest.raises(ValueError, match="consecutive"):
        BerryEsseenResult.model_validate(payload)


def test_result_requires_dyadic_singleton_or_consecutive_grid() -> None:
    genuine = berry_esseen_bound(
        _request(_distribution((0, Fraction(1, 2)), (1, Fraction(1, 2))))
    )
    payload = genuine.model_dump()
    payload["bound_squared"] = {"num": 196, "den": 625}
    payload["bound_lower"] = {"num": 14, "den": 25}
    payload["bound_upper"] = {"num": 14, "den": 25}

    with pytest.raises(ValueError, match="dyadic"):
        BerryEsseenResult.model_validate(payload)


def test_result_enforces_owner_rational_height_bound() -> None:
    genuine = berry_esseen_bound(_request(_five_atom_oracle_distribution(), 7))
    payload = genuine.model_dump()
    payload["mean"] = {"num": 1, "den": 10**512}

    with pytest.raises(ValueError, match="512-digit bound"):
        BerryEsseenResult.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("universal_constant", {"num": 1, "den": 1}),
        ("variance", {"num": -1, "den": 1}),
        ("third_absolute_central_moment", {"num": 0, "den": 1}),
        ("bound_squared", {"num": 0, "den": 1}),
        ("bound_lower", {"num": 2, "den": 1}),
    ),
)
def test_result_rejects_structurally_invalid_claims(
    field: str, value: dict[str, int]
) -> None:
    genuine = berry_esseen_bound(_request(_five_atom_oracle_distribution(), 7))
    payload = genuine.model_dump()
    payload[field] = value

    with pytest.raises(ValueError):
        BerryEsseenResult.model_validate(payload)


def test_zero_variance_and_bad_normalization_are_rejected_at_operation_boundary() -> (
    None
):
    with pytest.raises(OperationDomainValidationError, match="positive variance"):
        berry_esseen_bound(_request(_distribution((7, Fraction(1, 1)))))
    with pytest.raises(OperationDomainValidationError, match="sum exactly to 1"):
        berry_esseen_bound(
            _request(_distribution((0, Fraction(1, 1)), (1, Fraction(1, 1))))
        )


def test_normalization_intermediate_height_is_a_resource_refusal() -> None:
    left = 10**260 + 57
    right = 10**260 + 99
    with pytest.raises(OperationResourceAdmissionError, match="intermediate bound"):
        berry_esseen_bound(
            _request(
                _distribution(
                    (0, Fraction(1, left)),
                    (1, Fraction(1, right)),
                )
            )
        )


def test_operation_declaration_pins_iid_constant_and_contract() -> None:
    assert BERRY_ESSEEN_OPERATION.operation_id.endswith(
        "berry_esseen_iid_05600.compute"
    )
    assert "general-independent constant" in BERRY_ESSEEN_OPERATION.description
    assert "0.5600" in BERRY_ESSEEN_OPERATION.description
    assert "16,384 atoms" in BERRY_ESSEEN_OPERATION.description
    assert "128 decimal digits" in BERRY_ESSEEN_OPERATION.description
    assert "512 decimal digits" in BERRY_ESSEEN_OPERATION.description
    assert BERRY_ESSEEN_OPERATION.request_type is BerryEsseenRequest
    assert BERRY_ESSEEN_OPERATION.result_type is BerryEsseenResult
