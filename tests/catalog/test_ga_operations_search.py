"""Catalog search checks for the Ga fixed-subspace and stable-subrepresentation
operations.

These open the catalog, so they live in the catalog lane rather than under
``tests/math``.
"""

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import MathTool, OperationMatchRequest
from jacobian.math.polynomials.derivations._models import (
    PolynomialDerivation,
)
from jacobian.math.polynomials.derivations._stable_kernels import (
    ga_stable_subrepresentation,
)
from jacobian.math.polynomials.derivations._stable_models import (
    PolynomialGaFixedSubspaceRequest,
    PolynomialGaStableSubrepresentationRequest,
)
from jacobian.math.polynomials.derivations._tools import TOOLS
from jacobian.math.polynomials.derivations._weight_models import (
    RationalPolynomial,
)
from jacobian.math.polynomials.derivations.operations import (
    ga_action_from_derivation,
)


def _poly(
    variables: tuple[str, ...], terms: tuple[tuple[int, ...], ...]
) -> RationalPolynomial:
    return RationalPolynomial.model_validate(
        {
            "variables": list(variables),
            "polynomial": {
                "terms": [
                    {"coefficient": {"num": 1, "den": 1}, "exponents": list(exponents)}
                    for exponents in terms
                ]
            },
        }
    )


def _monomial(variable: str, exponent: int | None) -> RationalPolynomial:
    terms = () if exponent is None else ((1, exponent),)
    return RationalPolynomial.model_validate(
        {
            "variables": [variable],
            "polynomial": {
                "terms": [
                    {"coefficient": {"num": c, "den": 1}, "exponents": [e]}
                    for c, e in terms
                ]
            },
        }
    )


def _translation_action():
    x = _monomial("x", 1)
    return ga_action_from_derivation(
        PolynomialDerivation(variables=("x",), images=(_monomial("x", 0),)),
        ((x, _monomial("x", 0), _monomial("x", None)),),
    )


def test_catalog_contract_and_request_roundtrip() -> None:
    stable = ga_stable_subrepresentation(
        _translation_action(), (_poly(("x",), ((0,),)), _poly(("x",), ((1,),)))
    )
    request = PolynomialGaFixedSubspaceRequest(subrepresentation=stable)
    assert request.model_validate_json(request.model_dump_json()) == request
    assert any(
        tool.operation_id == "algebraic_group.ga.fixed_subspace.compute"
        for tool in TOOLS
    )
    matches = Catalog.open().match(OperationMatchRequest(need="Ga fixed polynomials"))
    assert any(
        match.operation_id == "algebraic_group.ga.fixed_subspace.compute"
        for match in matches.matches
    )


def test_request_preserves_explicit_axis_and_catalog_example() -> None:
    request = PolynomialGaStableSubrepresentationRequest(
        action=_translation_action(), basis=(_monomial("x", 0), _monomial("x", 1))
    )
    assert request.basis[1].variables == ("x",)
    assert any(
        tool.operation_id == "algebraic_group.ga.stable_subrepresentation.compute"
        for tool in TOOLS
    )
    assert isinstance(TOOLS[-1], MathTool)
    matches = Catalog.open().match(
        OperationMatchRequest(need="Ga-stable finite polynomial span")
    )
    assert any(
        result.operation_id == "algebraic_group.ga.stable_subrepresentation.compute"
        for result in matches.matches
    )
