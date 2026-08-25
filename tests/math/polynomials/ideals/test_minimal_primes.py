"""Tests for exact source-bound rational ideal minimal primes."""

from __future__ import annotations

import shutil

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResult
from jacobian.math.polynomials.ideals import _operations as operations_module
from jacobian.math.polynomials.ideals._models import (
    MAX_OUTPUT_GENERATORS,
    MAX_OUTPUT_TERMS,
    IdealMinimalPrimesRequest,
    IdealMinimalPrimesResult,
    computed_minimal_primes_result,
)
from jacobian.math.polynomials.ideals._operations import (
    compute_ideal_minimal_primes,
)
from jacobian.math.polynomials.ideals._singular import SingularMinimalPrimesResult
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialIdeal,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

_VERIFIER_TARGET = "jacobian.math.polynomials.ideals._singular"
_PRODUCER_TARGET = (
    "jacobian.math.polynomials.ideals._operations.run_singular_minimal_primes"
)
_VERIFICATION_TARGET = (
    "jacobian.math.polynomials.ideals._operations"
    ".run_singular_minimal_primes_verification"
)


def _poly(
    variables: tuple[str, ...],
    *terms: tuple[int, int, tuple[int, ...]],
) -> RationalPolynomial:
    return RationalPolynomial.model_validate(
        {
            "domain": "QQ",
            "variables": list(variables),
            "polynomial": {
                "terms": [
                    {
                        "coefficient": {"num": str(numerator), "den": str(denominator)},
                        "exponents": list(exponents),
                    }
                    for numerator, denominator, exponents in terms
                ]
            },
        }
    )


def _ideal(
    variables: tuple[str, ...], *generators: RationalPolynomial
) -> RationalPolynomialIdeal:
    return RationalPolynomialIdeal(variables=variables, generators=generators)


def _axes_request() -> IdealMinimalPrimesRequest:
    variables = ("x", "y")
    return IdealMinimalPrimesRequest(
        ideal=_ideal(variables, _poly(variables, (1, 1, (1, 1))))
    )


def _axes_components() -> tuple[RationalPolynomialIdeal, ...]:
    variables = ("x", "y")
    return tuple(
        sorted(
            (
                _ideal(variables, _poly(variables, (1, 1, (1, 0)))),
                _ideal(variables, _poly(variables, (1, 1, (0, 1)))),
            ),
            key=lambda ideal: ideal.model_dump_json(),
        )
    )


def _transport_oversized_components() -> tuple[RationalPolynomialIdeal, ...]:
    variables = ("x",)
    coefficient = CanonicalRational(
        num="9" * 32_768,
        den="9" * 32_767 + "8",
    )
    polynomial = RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=coefficient,
                    exponents=(exponent,),
                )
                for exponent in range(160, -1, -1)
            )
        ),
    )
    return (_ideal(variables, polynomial),)


def _product_ideal(
    variables: tuple[str, ...], degrees: tuple[int, ...]
) -> RationalPolynomialIdeal:
    """The ideal <x_i^degree - x_i> whose family has degree-product size."""

    generators = []
    for index, degree in enumerate(degrees):
        high = tuple(degree if slot == index else 0 for slot in range(len(variables)))
        low = tuple(1 if slot == index else 0 for slot in range(len(variables)))
        generators.append(_poly(variables, (1, 1, high), (-1, 1, low)))
    return _ideal(variables, *generators)


def _forbid_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(source: object, components: object, budget: object) -> str:
        raise AssertionError("the independent verifier must not be launched")

    monkeypatch.setattr(
        f"{_VERIFIER_TARGET}.run_singular_minimal_primes_verification", forbidden
    )


def test_computed_family_passes_independent_defining_invariant_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _axes_request()
    components = _axes_components()
    seen: list[tuple[object, object]] = []

    def verify(
        source: RationalPolynomialIdeal,
        claimed: tuple[RationalPolynomialIdeal, ...],
        budget: object,
    ) -> str:
        seen.append((source, claimed))
        return "VERIFIED"

    monkeypatch.setattr(
        f"{_VERIFIER_TARGET}.run_singular_minimal_primes_verification", verify
    )
    monkeypatch.setattr(
        _PRODUCER_TARGET,
        lambda *args: pytest.fail("the model validator must not re-run the kernel"),
    )

    result = IdealMinimalPrimesResult(
        request=request,
        outcome="COMPUTED",
        components=components,
        backend_version="4.4.0",
    )

    assert result.components == components
    assert seen == [(request.ideal, components)]


@pytest.mark.parametrize("verdict", ["REFUTED", "TIMEOUT", "UNAVAILABLE", "ERROR"])
def test_non_verified_verdicts_cannot_authorize_a_result(
    monkeypatch: pytest.MonkeyPatch, verdict: str
) -> None:
    request = _axes_request()
    components = _axes_components()

    monkeypatch.setattr(
        f"{_VERIFIER_TARGET}.run_singular_minimal_primes_verification",
        lambda source, claimed, budget: verdict,
    )

    with pytest.raises(ValidationError):
        IdealMinimalPrimesResult(
            request=request,
            outcome="COMPUTED",
            components=components,
            backend_version="4.4.0",
        )


def test_computed_family_rejects_an_incomplete_family_by_radical_intersection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _axes_request()
    components = _axes_components()

    monkeypatch.setattr(
        f"{_VERIFIER_TARGET}.run_singular_minimal_primes_verification",
        lambda source, claimed, budget: "REFUTED",
    )

    with pytest.raises(ValidationError):
        IdealMinimalPrimesResult(
            request=request,
            outcome="COMPUTED",
            components=components[:1],
            backend_version="4.4.0",
        )


def test_missing_backend_is_typed_and_makes_no_component_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _PRODUCER_TARGET,
        lambda source, budget: SingularMinimalPrimesResult(
            outcome="UNAVAILABLE", detail="backend is unavailable"
        ),
    )
    _forbid_verifier(monkeypatch)

    result = compute_ideal_minimal_primes(_axes_request())

    assert result.outcome == "UNAVAILABLE"
    assert result.components is None


def test_producer_verifies_once_without_a_third_backend_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _axes_request()
    components = _axes_components()
    calls = {"produce": 0, "verify": 0}

    def backend(
        source: RationalPolynomialIdeal,
        budget: object,
        *,
        wall_seconds: float | None = None,
    ) -> SingularMinimalPrimesResult:
        calls["produce"] += 1
        assert source == request.ideal
        return SingularMinimalPrimesResult(
            outcome="COMPUTED", components=components, backend_version="4.4.0"
        )

    def verify(
        source: RationalPolynomialIdeal,
        claimed: tuple[RationalPolynomialIdeal, ...],
        budget: object,
        *,
        wall_seconds: float | None = None,
    ) -> str:
        calls["verify"] += 1
        return "VERIFIED"

    monkeypatch.setattr(_PRODUCER_TARGET, backend)
    monkeypatch.setattr(_VERIFICATION_TARGET, verify)

    result = compute_ideal_minimal_primes(request)

    assert result.outcome == "COMPUTED"
    assert result.components == components
    assert calls == {"produce": 1, "verify": 1}


def test_stable_defective_kernel_family_is_refuted_by_independent_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deterministic kernel defect that repeats across passes is caught."""

    variables = ("x", "y")
    request = IdealMinimalPrimesRequest(
        ideal=_ideal(variables, _poly(variables, (1, 1, (1, 1))))
    )
    defective = (_ideal(variables, _poly(variables, (1, 1, (1, 1)))),)

    def backend(
        source: RationalPolynomialIdeal,
        budget: object,
        *,
        wall_seconds: float | None = None,
    ) -> SingularMinimalPrimesResult:
        return SingularMinimalPrimesResult(
            outcome="COMPUTED", components=defective, backend_version="4.4.0"
        )

    monkeypatch.setattr(_PRODUCER_TARGET, backend)
    monkeypatch.setattr(
        _VERIFICATION_TARGET,
        lambda source, claimed, budget, *, wall_seconds=None: "REFUTED",
    )

    result = compute_ideal_minimal_primes(request)

    assert result.outcome == "ERROR"
    assert result.components is None
    assert result.detail is not None


def test_duplicate_producer_family_is_a_typed_error_without_a_third_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variables = ("x",)
    prime = _ideal(variables, _poly(variables, (1, 1, (1,))))
    duplicated = (prime, prime)
    request = IdealMinimalPrimesRequest(
        ideal=_ideal(variables, _poly(variables, (1, 2, (2,))))
    )
    calls = {"produce": 0, "verify": 0}

    def backend(
        source: RationalPolynomialIdeal,
        budget: object,
        *,
        wall_seconds: float | None = None,
    ) -> SingularMinimalPrimesResult:
        calls["produce"] += 1
        return SingularMinimalPrimesResult(
            outcome="COMPUTED",
            components=duplicated,
            backend_version="4.4.0",
        )

    def verify(
        source: RationalPolynomialIdeal,
        claimed: tuple[RationalPolynomialIdeal, ...],
        budget: object,
        *,
        wall_seconds: float | None = None,
    ) -> str:
        calls["verify"] += 1
        return "VERIFIED"

    monkeypatch.setattr(_PRODUCER_TARGET, backend)
    monkeypatch.setattr(_VERIFICATION_TARGET, verify)

    result = compute_ideal_minimal_primes(request)

    assert calls == {"produce": 1, "verify": 1}
    assert result.outcome == "ERROR"
    assert result.components is None
    assert result.detail is not None


def test_trusted_factory_enforces_shape_ring_ordering_and_uniqueness() -> None:
    request = _axes_request()
    components = _axes_components()
    foreign_ring = (_ideal(("y", "x"), _poly(("y", "x"), (1, 1, (0, 1)))),)

    computed = computed_minimal_primes_result(request, components, "4.4.0")
    assert computed.outcome == "COMPUTED"
    assert computed.components == components
    assert computed.backend_version == "4.4.0"
    assert computed.detail is None

    with pytest.raises(ValueError, match="ordered ring"):
        computed_minimal_primes_result(request, foreign_ring, "4.4.0")
    with pytest.raises(ValueError, match="unique and canonically ordered"):
        computed_minimal_primes_result(request, tuple(reversed(components)), "4.4.0")
    with pytest.raises(ValueError, match="unique and canonically ordered"):
        computed_minimal_primes_result(request, (components[0], components[0]), "4.4.0")
    with pytest.raises(ValueError, match="requires components and backend version"):
        computed_minimal_primes_result(request, None, None)


def test_external_family_must_respect_the_generator_and_term_envelopes() -> None:
    variables = tuple(f"v{index}" for index in range(8))
    request = IdealMinimalPrimesRequest(
        ideal=_ideal(variables, _poly(variables, (1, 1, (1,) * 8)))
    )
    wide = (
        _ideal(
            variables,
            *(
                _poly(variables, (1, 1, (index, 0, 0, 0, 0, 0, 0, 0)))
                for index in range(9)
            ),
        ),
    )
    computed = computed_minimal_primes_result(request, wide, "4.4.0")
    assert computed.outcome == "COMPUTED"
    assert computed.components == wide

    first_component_generators = MAX_OUTPUT_GENERATORS // 2
    over_generators = (
        _ideal(
            variables,
            *(
                _poly(variables, (1, 1, (0,) * 8))
                for _ in range(first_component_generators)
            ),
        ),
        _ideal(
            variables,
            *(
                _poly(variables, (-1, 1, (0,) * 8))
                for _ in range(MAX_OUTPUT_GENERATORS - first_component_generators + 1)
            ),
        ),
    )
    with pytest.raises(
        ValueError, match=rf"{MAX_OUTPUT_GENERATORS}-generator exact-result envelope"
    ):
        computed_minimal_primes_result(request, over_generators, "4.4.0")

    heavy_terms = tuple(
        sorted(
            (
                (1, 1, (exponent // 33, exponent % 33, 0, 0, 0, 0, 0, 0))
                for exponent in range(MAX_OUTPUT_TERMS + 1)
            ),
            key=lambda term: term[2],
            reverse=True,
        )
    )
    oversized = (_ideal(variables, _poly(variables, *heavy_terms)),)
    with pytest.raises(
        ValueError, match=rf"{MAX_OUTPUT_TERMS}-term exact-result envelope"
    ):
        computed_minimal_primes_result(request, oversized, "4.4.0")


def test_validator_rejects_duplicate_components_before_any_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _axes_request()
    duplicated = (_axes_components()[0], _axes_components()[0])

    def forbidden(
        source: RationalPolynomialIdeal,
        claimed: tuple[RationalPolynomialIdeal, ...],
        budget: object,
    ) -> str:
        raise AssertionError("structural rejection must precede the backend replay")

    monkeypatch.setattr(
        f"{_VERIFIER_TARGET}.run_singular_minimal_primes_verification", forbidden
    )

    with pytest.raises(ValidationError):
        IdealMinimalPrimesResult(
            request=request,
            outcome="COMPUTED",
            components=duplicated,
            backend_version="4.4.0",
        )


class _Clock:
    """Deterministic ``time.monotonic`` stand-in with preprogrammed readings."""

    def __init__(self, *readings: float) -> None:
        self._readings = list(readings)

    def monotonic(self) -> float:
        return self._readings.pop(0)


def test_verification_is_charged_only_the_remaining_wall_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _axes_request()
    components = _axes_components()
    charged: list[float | None] = []

    def backend(
        source: RationalPolynomialIdeal,
        budget: object,
        *,
        wall_seconds: float | None = None,
    ) -> SingularMinimalPrimesResult:
        assert source == request.ideal
        return SingularMinimalPrimesResult(
            outcome="COMPUTED", components=components, backend_version="4.4.0"
        )

    def verify(
        source: RationalPolynomialIdeal,
        claimed: tuple[RationalPolynomialIdeal, ...],
        budget: object,
        *,
        wall_seconds: float | None = None,
    ) -> str:
        charged.append(wall_seconds)
        return "VERIFIED"

    monkeypatch.setattr(_PRODUCER_TARGET, backend)
    monkeypatch.setattr(_VERIFICATION_TARGET, verify)
    monkeypatch.setattr(
        "jacobian.math.polynomials.ideals._operations.time",
        _Clock(100.0, 103.25),
    )

    result = compute_ideal_minimal_primes(request)

    assert result.outcome == "COMPUTED"
    assert charged == [pytest.approx(10.0 - 3.25)]


def test_exhausted_deadline_is_a_typed_timeout_without_a_verification_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _axes_request()
    launches = 0

    def backend(
        source: RationalPolynomialIdeal,
        budget: object,
        *,
        wall_seconds: float | None = None,
    ) -> SingularMinimalPrimesResult:
        nonlocal launches
        launches += 1
        return SingularMinimalPrimesResult(
            outcome="COMPUTED",
            components=_axes_components(),
            backend_version="4.4.0",
        )

    monkeypatch.setattr(_PRODUCER_TARGET, backend)
    _forbid_verifier(monkeypatch)
    monkeypatch.setattr(
        "jacobian.math.polynomials.ideals._operations.time",
        _Clock(100.0, 110.5),
    )

    result = compute_ideal_minimal_primes(request)

    assert launches == 1
    assert result.outcome == "TIMEOUT"
    assert result.components is None
    assert result.detail is not None


@pytest.mark.parametrize(
    ("verdict", "outcome"),
    [
        ("REFUTED", "ERROR"),
        ("TIMEOUT", "TIMEOUT"),
        ("UNAVAILABLE", "UNAVAILABLE"),
        ("CANCELLED", "CANCELLED"),
        ("ERROR", "ERROR"),
    ],
)
def test_verification_verdicts_map_to_typed_outcomes(
    monkeypatch: pytest.MonkeyPatch, verdict: str, outcome: str
) -> None:
    request = _axes_request()
    components = _axes_components()

    monkeypatch.setattr(
        _PRODUCER_TARGET,
        lambda source, budget: SingularMinimalPrimesResult(
            outcome="COMPUTED", components=components, backend_version="4.4.0"
        ),
    )
    monkeypatch.setattr(
        _VERIFICATION_TARGET,
        lambda source, claimed, budget, *, wall_seconds=None: verdict,
    )

    result = compute_ideal_minimal_primes(request)

    assert result.outcome == outcome
    assert result.components is None
    assert result.detail is not None


def test_producer_cancellation_is_a_typed_outcome_without_a_verification_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled producing pass keeps its CANCELLED verdict end to end."""

    monkeypatch.setattr(
        _PRODUCER_TARGET,
        lambda source, budget: SingularMinimalPrimesResult(
            outcome="CANCELLED",
            detail="Singular execution was cancelled before producing a result.",
        ),
    )
    _forbid_verifier(monkeypatch)

    result = compute_ideal_minimal_primes(_axes_request())

    assert result.outcome == "CANCELLED"
    assert result.components is None
    assert result.backend_version is None
    assert result.detail is not None


def test_certified_families_above_the_envelope_are_rejected_before_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rational-root certificate rejects guaranteed overflow preflight.

    For <x1^2 - x1, ..., xk^2 - xk> each generator is single-variable and
    vanishes at two distinct certified rationals (0 and 1), so the source
    provably has at least 2^k minimal primes with at least k generators
    apiece. At k = 5 that is 160 aggregate generators against a fixed
    64-generator exact-result envelope: every execution could only end in
    typed LIMIT_EXCEEDED, so admission rejects the source before Singular
    launches instead of spending backend work on a guaranteed overflow.
    """

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("the backend must not launch for a rejected source")

    monkeypatch.setattr(_PRODUCER_TARGET, forbidden)
    _forbid_verifier(monkeypatch)

    for variable_count in (5, 6, 7):
        variables = tuple(f"x{index}" for index in range(1, variable_count + 1))
        with pytest.raises(ValidationError):
            compute_ideal_minimal_primes(
                IdealMinimalPrimesRequest(
                    ideal=_product_ideal(variables, (2,) * variable_count)
                )
            )


def test_bezout_boundary_family_is_admitted() -> None:
    variables = tuple(f"x{index}" for index in range(1, 5))
    # Four certified generators give 4 * 2^4 = 64 aggregate generators:
    # exactly the envelope boundary, so the family may still fit and the
    # source stays admitted.
    request = IdealMinimalPrimesRequest(ideal=_product_ideal(variables, (2,) * 4))

    assert len(request.ideal.generators) == 4


def test_negative_root_certificate_rejects_and_admits_at_the_boundary() -> None:
    """Certification is by exact evaluation, not one fixed root pattern.

    Each generator x_i^2 - 1 vanishes at the distinct probes -1 and 1 and
    never at 0: five certified generators force 5 * 2^5 = 160 aggregate
    generators and are rejected, while four sit exactly at the envelope
    boundary and stay admitted.
    """

    def unit_offset_ideal(count: int) -> RationalPolynomialIdeal:
        variables = tuple(f"x{index}" for index in range(1, count + 1))
        return _ideal(
            variables,
            *(
                _poly(
                    variables,
                    (
                        1,
                        1,
                        tuple(2 if slot == index else 0 for slot in range(count)),
                    ),
                    (-1, 1, (0,) * count),
                )
                for index in range(count)
            ),
        )

    with pytest.raises(ValidationError):
        IdealMinimalPrimesRequest(ideal=unit_offset_ideal(5))

    request = IdealMinimalPrimesRequest(ideal=unit_offset_ideal(4))

    assert len(request.ideal.generators) == 4


def test_irreducible_constraint_stays_admitted_below_the_envelope() -> None:
    """The forced-height bonus respects the same exact envelope boundary.

    Three certified generators give 2^3 = 8 feasible root choices and,
    with x4^2 - 2 solely occupying its own uncertified variable, every
    counted minimal prime needs at least 4 generators: 8 * 4 = 32
    aggregate generators stay provably inside the 64-generator envelope,
    so the source is admitted for the backend to answer exactly.
    """

    variables = tuple(f"x{index}" for index in range(1, 5))
    request = IdealMinimalPrimesRequest(
        ideal=_ideal(
            variables,
            *_product_ideal(variables, (2,) * 3).generators,
            _poly(
                variables,
                (1, 1, (0, 0, 0, 2)),
                (-2, 1, (0, 0, 0, 0)),
            ),
        )
    )

    assert len(request.ideal.generators) == 4


def test_shared_extra_slot_is_not_counted_as_a_forced_constraint() -> None:
    """A slot shared with another generator certifies no extra height.

    <x1(x1-1), ..., x4(x4-1), x5, x5 - 1> contains comaximal generators
    x5 and x5 - 1, so it is the unit ideal with an empty family and fits
    trivially; naively counting either generator's slot as one forced
    constraint would claim 16 * 5 = 80 aggregate generators and reject a
    fitting source. The shared slot is neither exempted from the point
    feasibility test nor counted toward any component's forced height.
    """

    variables = tuple(f"x{index}" for index in range(1, 6))
    request = IdealMinimalPrimesRequest(
        ideal=_ideal(
            variables,
            *_product_ideal(variables, (2,) * 4).generators,
            _poly(variables, (1, 1, (0, 0, 0, 0, 1))),
            _poly(variables, (1, 1, (0, 0, 0, 0, 1)), (-1, 1, (0,) * 5)),
        )
    )

    assert len(request.ideal.generators) == 6


def test_coupled_irreducible_constraint_falls_back_to_plain_certification() -> None:
    """An irreducible constraint sharing its slot loses the height bonus.

    Adding the coupling x5 * x1 to the rejected family makes x5's slot
    shared, so specialization could combine the constraints into a unit;
    the certificate then counts neither extra height nor exempt choices.
    The coupling does not vanish at any pinned probe point, so no choice
    stays feasible and admission retains the conservative four-certified
    behavior without rejecting the source.
    """

    variables = tuple(f"x{index}" for index in range(1, 6))
    request = IdealMinimalPrimesRequest(
        ideal=_ideal(
            variables,
            *_product_ideal(variables, (2,) * 4).generators,
            _poly(
                variables,
                (1, 1, (0, 0, 0, 0, 2)),
                (-2, 1, (0, 0, 0, 0, 0)),
            ),
            _poly(variables, (1, 1, (1, 0, 0, 0, 1))),
        )
    )

    assert len(request.ideal.generators) == 6


def test_coupling_generators_remove_infeasible_root_choices() -> None:
    """Root choices are certified only when every generator stays feasible.

    For <x_i^2 - x_i (1<=i<=5), x_2 - x_1, ..., x_5 - x_1> the coupling
    equations vanish only on the two diagonal choices (all zero or all
    one), so the source provably holds just two minimal primes carrying at
    least 10 aggregate generators; rejecting from the univariate generators
    alone would falsely claim 160. The source stays admitted.
    """

    def coupled_ideal(
        variable_count: int, coupled_count: int
    ) -> RationalPolynomialIdeal:
        variables = tuple(f"x{index}" for index in range(1, variable_count + 1))
        idempotents = _product_ideal(variables, (2,) * variable_count).generators
        couplings = tuple(
            _poly(
                variables,
                (-1, 1, tuple(1 if slot == 0 else 0 for slot in range(variable_count))),
                (
                    1,
                    1,
                    tuple(1 if slot == index else 0 for slot in range(variable_count)),
                ),
            )
            for index in range(1, coupled_count)
        )
        return _ideal(variables, *idempotents, *couplings)

    request = IdealMinimalPrimesRequest(ideal=coupled_ideal(5, 5))

    assert len(request.ideal.generators) == 9

    partially_coupled = coupled_ideal(7, 2)

    with pytest.raises(ValidationError):
        IdealMinimalPrimesRequest(ideal=partially_coupled)


def test_incompatible_extra_generators_block_certification_entirely() -> None:
    """An extra generator vanishing on no root choice certifies nothing.

    <x^2 - x, y^2 - y, x + y - 2> admits exactly one feasible choice
    (1, 1), so the certificate bounds the family by one two-generator
    component and the source stays admitted.
    """

    variables = ("x", "y")
    request = IdealMinimalPrimesRequest(
        ideal=_ideal(
            variables,
            *_product_ideal(variables, (2, 2)).generators,
            _poly(variables, (1, 1, (1, 0)), (1, 1, (0, 1)), (-2, 1, (0, 0))),
        )
    )

    assert len(request.ideal.generators) == 3


def test_pure_power_sources_admit_large_degree_products() -> None:
    """<x^20, y^20> has Bezout product 400 yet exactly one minimal prime.

    A pure power of every ring variable forces every prime over the source
    to contain <x,...,y>, so the complete family is that single component
    regardless of how large the degree product is.
    """

    variables = ("x", "y")
    pure_powers = _ideal(
        variables,
        _poly(variables, (1, 1, (20, 0))),
        _poly(variables, (1, 1, (0, 20))),
    )
    request = IdealMinimalPrimesRequest(ideal=pure_powers)
    assert len(request.ideal.generators) == 2

    mixed = IdealMinimalPrimesRequest(
        ideal=_ideal(
            variables,
            *pure_powers.generators,
            _poly(variables, (1, 1, (3, 3)), (-1, 1, (2, 4))),
        )
    )
    assert len(mixed.ideal.generators) == 3


def test_monomial_sources_on_few_active_variables_admit_their_family_bound() -> None:
    variables = ("w", "x", "y", "z")
    # One monomial on three active variables: at most 2^3 components with
    # at most 3 single-term generators each fits the aggregate envelope.
    request = IdealMinimalPrimesRequest(
        ideal=_ideal(variables, _poly(variables, (1, 1, (0, 3, 2, 14))))
    )

    assert len(request.ideal.generators) == 1


def test_wide_monomial_sources_without_full_pure_powers_are_admitted() -> None:
    """Seven active monomial variables are admitted without certification.

    The former worst-case rejection relied on the Bezout bound's value
    across ideals; admission now retains every source within the input
    envelopes and lets the decoded family answer against the aggregate
    exact-result envelopes.
    """

    variables = tuple(f"x{index}" for index in range(1, 9))
    generators = [
        _poly(
            variables,
            (1, 1, tuple(7 if slot == index else 0 for slot in range(8))),
        )
        for index in range(7)
    ]

    request = IdealMinimalPrimesRequest(ideal=_ideal(variables, *generators))

    assert len(request.ideal.generators) == 7


def test_unit_and_zero_degenerate_sources_admit_their_exact_families() -> None:
    variables = tuple(f"x{index}" for index in range(1, 7))
    constant = IdealMinimalPrimesRequest(
        ideal=_ideal(
            variables,
            _poly(variables, (1, 1, (0,) * 6)),
            *_product_ideal(variables, (2,) * 6).generators,
        )
    )
    assert len(constant.ideal.generators) == 7

    dropped = IdealMinimalPrimesRequest(
        ideal=_ideal(("x", "y"), _poly(("x", "y")), _poly(("x", "y"), (1, 1, (1, 1))))
    )
    assert len(dropped.ideal.generators) == 2


def test_request_description_advertises_the_enforced_budgets() -> None:
    description = IdealMinimalPrimesRequest.model_fields["ideal"].description

    assert description is not None
    assert "at most 8 variables" in description
    assert "at most 32 generators" in description
    assert "total degree is at most 20" in description
    assert "LIMIT_EXCEEDED" in description


@pytest.mark.skipif(
    shutil.which("Singular") is None,
    reason="Singular 4.4 backend is not installed",
)
def test_coordinate_axes_are_the_two_qq_minimal_primes() -> None:
    result = compute_ideal_minimal_primes(_axes_request())

    assert result.outcome == "COMPUTED"
    assert result.components == _axes_components()


def test_transport_oversize_is_typed_before_operation_result_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = IdealMinimalPrimesRequest(
        ideal=_ideal(("x",), _poly(("x",), (1, 1, (1,))))
    )
    components = _transport_oversized_components()
    monkeypatch.setattr(
        operations_module,
        "run_singular_minimal_primes",
        lambda ideal, budget: SingularMinimalPrimesResult(
            outcome="COMPUTED",
            components=components,
            backend_version="4.4.0",
        ),
    )
    monkeypatch.setattr(
        operations_module,
        "run_singular_minimal_primes_verification",
        lambda *args, **kwargs: "VERIFIED",
    )

    result = compute_ideal_minimal_primes(request)

    assert result.outcome == "LIMIT_EXCEEDED"
    assert result.components is None
    public = OperationResult(
        operation_id="polynomial.ideal.minimal_primes.compute",
        runtime_ms=0,
        output=result.model_dump(mode="json"),
    )
    assert public.output["outcome"] == "LIMIT_EXCEEDED"


@pytest.mark.skipif(
    shutil.which("Singular") is None,
    reason="Singular 4.4 backend is not installed",
)
def test_high_degree_pure_power_source_has_the_single_axis_component() -> None:
    """<x^20, y^20> computes its exact one-component family end to end."""

    variables = ("x", "y")
    request = IdealMinimalPrimesRequest(
        ideal=_ideal(
            variables,
            _poly(variables, (1, 1, (20, 0))),
            _poly(variables, (1, 1, (0, 20))),
        )
    )
    axes = {
        _poly(variables, (1, 1, (1, 0))).model_dump_json(),
        _poly(variables, (1, 1, (0, 1))).model_dump_json(),
    }

    result = compute_ideal_minimal_primes(request)

    assert result.outcome == "COMPUTED"
    assert result.components is not None
    assert len(result.components) == 1
    assert {
        generator.model_dump_json() for generator in result.components[0].generators
    } == axes


@pytest.mark.skipif(
    shutil.which("Singular") is None,
    reason="Singular 4.4 backend is not installed",
)
def test_forged_well_shaped_family_fails_independent_verification() -> None:
    variables = ("x", "y")
    request = IdealMinimalPrimesRequest(
        ideal=_ideal(variables, _poly(variables, (1, 1, (1, 1))))
    )
    forged = (_ideal(variables, _poly(variables, (1, 1, (1, 1)))),)

    with pytest.raises(ValidationError):
        IdealMinimalPrimesResult(
            request=request,
            outcome="COMPUTED",
            components=forged,
            backend_version="4.4.0",
        )


@pytest.mark.skipif(
    shutil.which("Singular") is None,
    reason="Singular 4.4 backend is not installed",
)
def test_unit_zero_and_embedded_sources_have_their_exact_family_shapes() -> None:
    variables = ("x", "y")
    zero = compute_ideal_minimal_primes(
        IdealMinimalPrimesRequest(ideal=_ideal(variables, _poly(variables)))
    )
    unit = compute_ideal_minimal_primes(
        IdealMinimalPrimesRequest(
            ideal=_ideal(variables, _poly(variables, (1, 1, (0, 0))))
        )
    )
    embedded = compute_ideal_minimal_primes(
        IdealMinimalPrimesRequest(
            ideal=_ideal(
                variables,
                _poly(variables, (1, 1, (2, 0))),
                _poly(variables, (1, 1, (1, 1))),
            )
        )
    )

    assert zero.outcome == unit.outcome == embedded.outcome == "COMPUTED"
    assert zero.components == (_ideal(variables, _poly(variables)),)
    assert unit.components == ()
    assert embedded.components == (_ideal(variables, _poly(variables, (1, 1, (1, 0)))),)


@pytest.mark.skipif(
    shutil.which("Singular") is None,
    reason="Singular 4.4 backend is not installed",
)
def test_prime_wider_than_the_ring_dimension_is_computed() -> None:
    """A minimal prime whose standard basis exceeds the variable count.

    The affine cone over the rational normal quartic curve is one prime in
    five variables whose reduced standard basis has six elements; the
    Eisenbud-Evans set-theoretic bound does not bound the emitted
    presentation.
    """

    variables = ("x0", "x1", "x2", "x3", "x4")
    request = IdealMinimalPrimesRequest(
        ideal=_ideal(
            variables,
            _poly(variables, (1, 1, (1, 0, 1, 0, 0)), (-1, 1, (0, 2, 0, 0, 0))),
            _poly(variables, (1, 1, (1, 0, 0, 1, 0)), (-1, 1, (0, 1, 1, 0, 0))),
            _poly(variables, (1, 1, (1, 0, 0, 0, 1)), (-1, 1, (0, 0, 2, 0, 0))),
            _poly(variables, (1, 1, (0, 1, 0, 1, 0)), (-1, 1, (0, 0, 2, 0, 0))),
            _poly(variables, (1, 1, (0, 1, 0, 0, 1)), (-1, 1, (0, 0, 1, 1, 0))),
            _poly(variables, (1, 1, (0, 0, 1, 0, 1)), (-1, 1, (0, 0, 0, 2, 0))),
        )
    )

    result = compute_ideal_minimal_primes(request)

    assert result.outcome == "COMPUTED"
    assert result.components is not None
    assert len(result.components) == 1
    assert all(
        len(component.generators) > len(variables) for component in result.components
    )


def test_irreducible_constraint_overflow_is_rejected_before_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sole irreducible constraint raises every component's forced height.

    For <x1(x1-1), ..., x4(x4-1), x5^2 - 2> the four certified generators
    give 16 feasible root choices, and x5^2 - 2 solely occupies its own
    variable with no rational root among the fixed probes: over QQ every
    counted minimal prime contains the four linear forms plus x5^2 - 2, so
    each needs at least five generators and the family carries at least 80
    aggregate generators against a fixed 64-generator envelope. Admission
    rejects the source before Singular launches instead of spending
    backend work on a guaranteed overflow.
    """

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("the backend must not launch for a rejected source")

    monkeypatch.setattr(_PRODUCER_TARGET, forbidden)
    _forbid_verifier(monkeypatch)

    variables = tuple(f"x{index}" for index in range(1, 6))
    with pytest.raises(ValidationError):
        compute_ideal_minimal_primes(
            IdealMinimalPrimesRequest(
                ideal=_ideal(
                    variables,
                    *_product_ideal(variables, (2,) * 4).generators,
                    _poly(
                        variables,
                        (1, 1, (0, 0, 0, 0, 2)),
                        (-2, 1, (0, 0, 0, 0, 0)),
                    ),
                )
            )
        )


@pytest.mark.skipif(
    shutil.which("Singular") is None,
    reason="Singular 4.4 backend is not installed",
)
def test_coupled_source_computes_its_two_diagonal_components() -> None:
    """The coupled idempotent source computes exactly two maximal components.

    The couplings x_j - x_1 collapse the five-dimensional {0,1} grid to the
    two diagonal points, so the complete family is <x1,...,x5> together
    with <x1-1,...,x5-1>: admission must retain the source and the backend
    must return both components.
    """

    variables = tuple(f"x{index}" for index in range(1, 6))
    request = IdealMinimalPrimesRequest(
        ideal=_ideal(
            variables,
            *_product_ideal(variables, (2,) * 5).generators,
            *(
                _poly(
                    variables,
                    (-1, 1, (1, 0, 0, 0, 0)),
                    (1, 1, tuple(1 if slot == index else 0 for slot in range(5))),
                )
                for index in range(1, 5)
            ),
        )
    )

    result = compute_ideal_minimal_primes(request)

    expected = (
        _ideal(
            variables,
            *(
                _poly(variables, (1, 1, tuple(1 if v == i else 0 for v in range(5))))
                for i in range(5)
            ),
        ),
        _ideal(
            variables,
            *(
                _poly(
                    variables,
                    (1, 1, tuple(1 if v == i else 0 for v in range(5))),
                    (-1, 1, (0, 0, 0, 0, 0)),
                )
                for i in range(5)
            ),
        ),
    )

    assert result.outcome == "COMPUTED"
    assert result.components is not None
    assert {
        frozenset(generator.model_dump_json() for generator in component.generators)
        for component in result.components
    } == {
        frozenset(generator.model_dump_json() for generator in component.generators)
        for component in expected
    }
