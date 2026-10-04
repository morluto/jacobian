"""Operational admission never becomes a false specialized resultant claim."""

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials._models import PolynomialValue
from jacobian.math.polynomials.multivariate import operations
from jacobian.math.polynomials.multivariate._resultant import (
    MultivariateResultantResult,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _polynomial(
    axes: tuple[str, ...], terms: dict[tuple[int, ...], int]
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=axes,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    exponents=e, coefficient=CanonicalRational.from_integer_ratio(c, 1)
                )
                for e, c in sorted(terms.items(), reverse=True)
                if c
            )
        ),
    )


def _claim(source: RationalPolynomial, answer: int = 0) -> MultivariateResultantResult:
    axes = tuple(v for v in source.variables if v != "x")
    value = _polynomial(axes, {(0,) * len(axes): answer})
    claim = MultivariateResultantResult(
        left=source,
        right=source,
        elimination_variable="x",
        resultant=PolynomialValue(value=value),
    )
    return MultivariateResultantResult.model_validate_json(
        claim.model_dump_json(), strict=True
    )


@pytest.mark.parametrize("axes", [("x", "y"), ("y", "x")])
def test_admitted_true_and_false_relations_survive_strict_json(
    axes: tuple[str, ...],
) -> None:
    source = _polynomial(axes, {tuple(32 if v == "x" else 0 for v in axes): 1})
    assert operations.verify_multivariate_resultant(_claim(source))
    assert not operations.verify_multivariate_resultant(_claim(source, 1))


@pytest.mark.parametrize(
    "kind", ["degree", "terms", "exponent", "coefficient", "support"]
)
def test_resource_refusal_is_typed_and_precedes_backend(
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    axes: tuple[str, ...] = ("x", "y")
    terms: dict[tuple[int, ...], int] = {(33, 0): 1}
    suffix = "degree_budget"
    if kind == "terms":
        terms = dict.fromkeys([(i, j) for i in range(8) for j in range(65)][:513], 1)
        suffix = "source_budget"
    elif kind == "exponent":
        terms, suffix = {(65, 0): 1}, "source_budget"
    elif kind == "coefficient":
        terms, suffix = {(1, 0): 10**256}, "source_budget"
    elif kind == "support":
        axes = ("x", "y", "z")
        terms, suffix = {(1, 0, 0): 1, (0, 32, 0): 1, (0, 0, 32): 1}, "support_budget"
    source = _polynomial(axes, terms)
    claim = _claim(source)

    def forbidden(*args: object) -> None:
        pytest.fail("resultant kernel reached after resource admission should refuse")

    monkeypatch.setattr(operations, "_sylvester_resultant_value", forbidden)
    for invoke in (
        lambda: operations.multivariate_resultant(source, source, "x"),
        lambda: operations.verify_multivariate_resultant(claim),
    ):
        with pytest.raises(OperationResourceAdmissionError) as error:
            invoke()
        assert (
            error.value.errors()[0]["type"]
            == f"polynomial.multivariate_resultant.{suffix}"
        )


@pytest.mark.parametrize("wrong_axis", [False, True])
def test_domain_invalidity_stays_separate_from_work_refusal(wrong_axis: bool) -> None:
    source = _polynomial(("x", "y"), {(1, 0): 1})
    claim = _claim(source)
    if wrong_axis:
        claim = claim.model_copy(update={"elimination_variable": "z"})
    else:
        claim = claim.model_copy(update={"right": _polynomial(("x", "z"), {(1, 0): 1})})
    assert not operations.verify_multivariate_resultant(claim)
    with pytest.raises(OperationDomainValidationError) as error:
        operations.multivariate_resultant(
            claim.left, claim.right, claim.elimination_variable
        )
    assert not isinstance(error.value, OperationResourceAdmissionError)


def test_resource_diagnostic_names_actual_degree_and_bound() -> None:
    source = _polynomial(("x", "y"), {(33, 0): 1})
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        operations.verify_multivariate_resultant(_claim(source))
    assert (
        exc_info.value.errors()[0]["type"]
        == "polynomial.multivariate_resultant.degree_budget"
    )


def test_right_source_budget_has_right_location() -> None:
    left = _polynomial(("x", "y"), {(1, 0): 1})
    right = _polynomial(("x", "y"), {(65, 0): 1})
    claim = _claim(left).model_copy(update={"right": right})
    for invoke in (
        lambda: operations.multivariate_resultant(left, right, "x"),
        lambda: operations.verify_multivariate_resultant(claim),
    ):
        with pytest.raises(OperationResourceAdmissionError) as error:
            invoke()
        assert error.value.errors()[0]["loc"] == ("right",)


@pytest.mark.parametrize("cancelled", [False, True])
def test_backend_noncompletion_is_not_a_false_claim(
    cancelled: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian._execution import (
        OperationExecutionCancelledError,
        OperationExecutionTimeoutError,
    )

    failure = (
        OperationExecutionCancelledError("stopped")
        if cancelled
        else OperationExecutionTimeoutError("deadline")
    )

    def stopped(*args: object) -> None:
        raise failure

    monkeypatch.setattr(operations, "_sylvester_resultant_value", stopped)
    source = _polynomial(("x", "y"), {(1, 0): 1})
    with pytest.raises(type(failure)):
        operations.verify_multivariate_resultant(_claim(source))
