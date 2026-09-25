from __future__ import annotations

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.free_algebras._models import (
    FreeAlgebraIdeal,
    FreeAlgebraPolynomial,
    FreeAlgebraTerm,
)
from jacobian.math.free_algebras.operations import (
    groebner_shirshov_through_degree,
    ideal_generated_prefix,
)
from jacobian.math.koszul.module_models import (
    BasedFiniteModule,
    FiniteCommutativeAlgebra,
    ModuleDifferential,
    ModuleKoszulComplex,
    ModuleKoszulRequest,
)
from jacobian.math.koszul.module_operations import (
    module_koszul_complex,
    module_koszul_homology,
)
from jacobian.math.ore_algebras._models import (
    DifferentialOreOperator,
    DifferentialOreTerm,
)
from jacobian.math.ore_algebras.operations import (
    differential_operator_apply,
    differential_operator_multiply,
)
from jacobian.math.polynomials.derivations._models import PolynomialDerivation
from jacobian.math.polynomials.derivations.operations import (
    apply_derivation,
    derivation_iterates,
)
from jacobian.math.polynomials.values import (
    RationalFunction,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _q(value: int) -> CanonicalRational:
    return CanonicalRational(num=value, den=1)


def _nc(*terms: tuple[int, tuple[str, ...]]) -> FreeAlgebraPolynomial:
    return FreeAlgebraPolynomial(
        alphabet=("a", "b", "c"),
        terms=tuple(
            FreeAlgebraTerm(coefficient=_q(coefficient), word=word)
            for coefficient, word in sorted(
                terms, key=lambda item: item[1], reverse=True
            )
        ),
    )


def test_groebner_shirshov_checks_reverse_ordered_overlap_and_reaches_fixed_point() -> (
    None
):
    # The first generator has leading word bc and the second ab.  Only the
    # reverse ordered pair (ab, bc) has the b overlap.
    ideal = FreeAlgebraIdeal(
        alphabet=("a", "b", "c"),
        generators=(
            _nc((1, ("b", "c")), (-1, ("a", "c"))),
            _nc((1, ("a", "b")), (-1, ("a", "c"))),
        ),
        side="two-sided",
    )
    result = groebner_shirshov_through_degree(ideal, 3)

    assert result.status == "COMPLETE_THROUGH_DEGREE"
    assert any(
        {term.word for term in composition.terms} == {("a",), ("c",)}
        for composition in result.compositions
    )
    # The returned basis has the reverse overlap remainder and its final
    # ordered-pair checks have reached a zero-composition fixed point.
    assert any(
        {term.word for term in value.terms} == {("a",), ("c",)}
        for value in result.basis
    )


def test_groebner_shirshov_preflights_composition_coefficient_growth() -> None:
    large = 10**63
    first = FreeAlgebraPolynomial(
        alphabet=("a", "b", "c"),
        terms=(
            FreeAlgebraTerm(coefficient=_q(large), word=("a", "b")),
            FreeAlgebraTerm(coefficient=_q(1), word=("c", "c")),
        ),
    )
    second = FreeAlgebraPolynomial(
        alphabet=("a", "b", "c"),
        terms=(
            FreeAlgebraTerm(
                coefficient=CanonicalRational(num=1, den=large),
                word=("b", "c"),
            ),
            FreeAlgebraTerm(coefficient=_q(1), word=("b", "b")),
        ),
    )
    ideal = FreeAlgebraIdeal(
        alphabet=("a", "b", "c"),
        generators=(first, second),
        side="two-sided",
    )

    with pytest.raises(
        OperationResourceAdmissionError,
        match="coefficient multiplication exceeds",
    ):
        groebner_shirshov_through_degree(ideal, 3)


def test_koszul_cycle_and_boundary_dimensions_follow_chain_degrees() -> None:
    algebra = FiniteCommutativeAlgebra(basis=("1",), multiplication=(((_q(1),),),))
    module = BasedFiniteModule(algebra=algebra, basis=("m",), action=(((_q(1),),),))
    complex_value = module_koszul_complex(
        ModuleKoszulRequest(algebra=algebra, module=module, sequence=((_q(1),),))
    )

    homology = module_koszul_homology(complex_value)

    assert homology.cycle_dimensions == (1, 0)
    assert homology.boundary_dimensions == (1, 0)
    assert homology.dimensions == (0, 0)


def test_koszul_homology_rejects_forged_differential_axes() -> None:
    algebra = FiniteCommutativeAlgebra(basis=("1",), multiplication=(((_q(1),),),))
    module = BasedFiniteModule(algebra=algebra, basis=("m",), action=(((_q(1),),),))
    request = ModuleKoszulRequest(algebra=algebra, module=module, sequence=((_q(1),),))
    complex_value = module_koszul_complex(request)
    forged = ModuleKoszulComplex.model_construct(
        algebra=algebra,
        module=module,
        sequence=request.sequence,
        basis_sizes=complex_value.basis_sizes,
        differentials=(
            ModuleDifferential.model_construct(
                row_count=1, column_count=1, entries=((2, 0, _q(1)),)
            ),
        ),
        square_zero=True,
    )
    with pytest.raises(OperationDomainValidationError):
        module_koszul_homology(forged)


def test_derivation_re_admits_forged_polynomial_shape() -> None:
    zero = RationalPolynomial(
        domain="QQ",
        variables=("x",),
        polynomial=SparseRationalPolynomial(terms=()),
    )
    derivation = PolynomialDerivation(variables=("x",), images=(zero,))
    forged = RationalPolynomial.model_construct(
        domain="QQ",
        variables=("x",),
        polynomial={
            "terms": ({"coefficient": _q(1), "exponents": ()},),
        },
    )
    with pytest.raises(OperationDomainValidationError):
        apply_derivation(derivation, forged)


def test_derivation_rejects_result_coefficient_growth_before_multiplication() -> None:
    large = 10**127
    image = RationalPolynomial(
        domain="QQ",
        variables=("x",),
        polynomial=SparseRationalPolynomial(
            terms=(RationalPolynomialTerm(coefficient=_q(large), exponents=(1,)),)
        ),
    )
    derivation = PolynomialDerivation(variables=("x",), images=(image,))
    source = RationalPolynomial(
        domain="QQ",
        variables=("x",),
        polynomial=SparseRationalPolynomial(
            terms=(RationalPolynomialTerm(coefficient=_q(large), exponents=(1,)),)
        ),
    )

    with pytest.raises(OperationResourceAdmissionError) as error:
        apply_derivation(derivation, source)
    assert error.value.errors()[0]["type"] == (
        "polynomial_derivation.result_coefficient_budget"
    )


def test_derivation_iterates_re_admit_generated_intermediates() -> None:
    exponents = tuple(
        (left, total - left) for total in range(65) for left in range(total + 1)
    )

    def polynomial(axis: tuple[tuple[int, int], ...]) -> RationalPolynomial:
        return RationalPolynomial(
            domain="QQ",
            variables=("x", "y"),
            polynomial=SparseRationalPolynomial(
                terms=tuple(
                    RationalPolynomialTerm(coefficient=_q(1), exponents=value)
                    for value in sorted(axis, reverse=True)
                )
            ),
        )

    derivation = PolynomialDerivation(
        variables=("x", "y"),
        images=(polynomial(exponents[:256]), polynomial(exponents[256:261])),
    )
    source = polynomial(((1, 0), (0, 1)))

    with pytest.raises(OperationResourceAdmissionError) as error:
        derivation_iterates(derivation, source, 2)
    assert error.value.errors()[0]["type"] == (
        "polynomial_derivation.source_term_budget"
    )


def test_differential_ore_re_admits_rational_function_parent() -> None:
    forged_function = RationalFunction.model_construct(
        domain="QQ",
        variables=("y",),
        numerator=SparseRationalPolynomial.model_construct(terms=()),
        denominator=SparseRationalPolynomial.model_construct(terms=()),
    )
    with pytest.raises(OperationDomainValidationError):
        differential_operator_apply(DifferentialOreOperator(terms=()), forged_function)


def _one_rational_function() -> RationalFunction:
    return RationalFunction(
        domain="QQ",
        variables=("x",),
        numerator=SparseRationalPolynomial(
            terms=(RationalPolynomialTerm(coefficient=_q(1), exponents=(0,)),)
        ),
        denominator=SparseRationalPolynomial(
            terms=(RationalPolynomialTerm(coefficient=_q(1), exponents=(0,)),)
        ),
    )


def _large_rational_function() -> RationalFunction:
    return RationalFunction(
        domain="QQ",
        variables=("x",),
        numerator=SparseRationalPolynomial(
            terms=(RationalPolynomialTerm(coefficient=_q(1), exponents=(0,)),)
        ),
        denominator=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(coefficient=_q(1), exponents=(64,)),
                RationalPolynomialTerm(coefficient=_q(1), exponents=(0,)),
            )
        ),
    )


def test_differential_ore_growth_is_admitted_before_expansion() -> None:
    function = _large_rational_function()
    derivative = DifferentialOreOperator(
        terms=(
            DifferentialOreTerm(
                order=16,
                coefficient=_one_rational_function(),
            ),
        )
    )
    with pytest.raises(OperationResourceAdmissionError):
        differential_operator_apply(derivative, function)


def test_differential_ore_product_growth_is_admitted_before_expansion() -> None:
    function = _large_rational_function()
    identity = DifferentialOreOperator(
        terms=(DifferentialOreTerm(order=16, coefficient=_one_rational_function()),)
    )
    coefficient = DifferentialOreOperator(
        terms=(DifferentialOreTerm(order=0, coefficient=function),)
    )
    with pytest.raises(OperationResourceAdmissionError):
        differential_operator_multiply(identity, coefficient)


def test_free_ideal_prefix_admits_aggregate_terms_before_expansion() -> None:
    # Repeat-free degree-six words: 64 terms, hence 64 contexts times 64
    # terms at degree twelve on the left-sided prefix.
    words = tuple(
        tuple("x" if mask & (1 << index) else "y" for index in range(6))
        for mask in range(64)
    )
    generator = FreeAlgebraPolynomial(
        alphabet=("x", "y"),
        terms=tuple(
            FreeAlgebraTerm(coefficient=_q(1), word=word)
            for word in sorted(words, reverse=True)
        ),
    )
    ideal = FreeAlgebraIdeal(alphabet=("x", "y"), generators=(generator,), side="left")
    with pytest.raises(OperationResourceAdmissionError):
        ideal_generated_prefix(ideal, 12)
