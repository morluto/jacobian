from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.math.polynomials.local_series._tools import TOOLS
from jacobian.math.polynomials.local_series.contact_profile import (
    PuiseuxContactProfile,
    PuiseuxContactRequest,
    puiseux_contact_profile,
)
from jacobian.math.polynomials.local_series.puiseux_values import (
    PuiseuxTerm,
    TruncatedPuiseuxWindow,
)


def _rational(value: int | Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _prefix(*terms: tuple[Fraction, Fraction]) -> TruncatedPuiseuxWindow:
    return TruncatedPuiseuxWindow(
        variable="t",
        center=_rational(0),
        valuation_lower=_rational(0),
        precision=_rational(2),
        ramification_index=2,
        terms=tuple(
            PuiseuxTerm(
                exponent=_rational(exponent), coefficient=_rational(coefficient)
            )
            for exponent, coefficient in terms
        ),
    )


def test_contact_profile_matches_exact_coefficient_oracle_and_roundtrips() -> None:
    # For finite sparse prefixes with a shared exponent window, the valuation
    # of the difference is the least exponent where their coefficient maps vary.
    prefixes = (
        _prefix((Fraction(1, 2), Fraction(3)), (Fraction(3, 2), Fraction(7))),
        _prefix((Fraction(1, 2), Fraction(3)), (Fraction(1), Fraction(5))),
        _prefix((Fraction(1, 2), Fraction(4))),
    )
    result = puiseux_contact_profile(PuiseuxContactRequest(prefixes=prefixes))

    assert tuple(
        (row.left_index, row.right_index, row.status, row.contact_order)
        for row in result.pairs
    ) == (
        (0, 1, "DETERMINED", _rational(1)),
        (0, 2, "DETERMINED", _rational(Fraction(1, 2))),
        (1, 2, "DETERMINED", _rational(Fraction(1, 2))),
    )
    assert result.prefixes == prefixes
    assert PuiseuxContactProfile.model_validate_json(result.model_dump_json()) == result


def test_equal_prefixes_are_unresolved_through_finite_precision() -> None:
    prefix = _prefix((Fraction(1, 2), Fraction(3)))
    result = puiseux_contact_profile(PuiseuxContactRequest(prefixes=(prefix, prefix)))

    assert result.pairs[0].status == "UNRESOLVED"
    assert result.pairs[0].contact_order is None
    assert result.pairs[0].common_precision == _rational(2)


def test_contact_profile_requires_common_known_window() -> None:
    prefix = _prefix((Fraction(1, 2), Fraction(3)))
    different_precision = TruncatedPuiseuxWindow(
        variable="t",
        center=_rational(0),
        valuation_lower=_rational(0),
        precision=_rational(3),
        ramification_index=2,
        terms=(
            PuiseuxTerm(exponent=_rational(Fraction(1, 2)), coefficient=_rational(3)),
        ),
    )
    with pytest.raises(ValueError, match="known exponent window"):
        PuiseuxContactRequest(prefixes=(prefix, different_precision))


def test_contact_profile_is_published_with_formal_prefix_contract() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "local_series.puiseux.contact_profile.compute"
    )
    assert tool.request_type is PuiseuxContactRequest
    assert tool.result_type is PuiseuxContactProfile
    result = tool.run(
        PuiseuxContactRequest(
            prefixes=(
                _prefix((Fraction(1, 2), Fraction(1))),
                _prefix((Fraction(1, 2), Fraction(2))),
            )
        )
    )
    assert result.pairs[0].contact_order == _rational(Fraction(1, 2))


def test_native_boundary_revalidates_reconstructed_request() -> None:
    from jacobian.catalog.models import OperationResourceAdmissionError

    with pytest.raises(OperationResourceAdmissionError) as error:
        puiseux_contact_profile(PuiseuxContactRequest.model_construct(prefixes=()))
    assert "invalid Puiseux contact request" in str(error.value)


def test_profile_rejects_unbounded_pairwise_prefix_carrier() -> None:
    prefix = _prefix((Fraction(1, 2), Fraction(1)))
    with pytest.raises(ValueError):
        PuiseuxContactProfile.model_validate({"prefixes": (prefix,) * 33, "pairs": ()})
