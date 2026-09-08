"""Operational failure at a native claim consumer never refutes its claim."""

import importlib
import json
from typing import Any

import pytest

from jacobian._execution import OperationExecutionTimeoutError
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import OperationResourceAdmissionError

CASES = (
    (
        "matrix.symbolic.eigenvalues.compute",
        "matrices.symbolic.operations",
        "verify_symbolic_eigenvalues",
        "_symbolic_characteristic_polynomial_kernel",
    ),
    (
        "polynomial.ideal.normal_form.compute",
        "polynomials.ideals.operations",
        "verify_ideal_normal_form",
        "ideal_normal_form",
    ),
    (
        "polynomial.ideal.containment.decide",
        "polynomials.ideals.operations",
        "verify_ideal_containment",
        "ideal_containment",
    ),
    (
        "polynomial.ideal.equality.decide",
        "polynomials.ideals.operations",
        "verify_ideal_equality",
        "ideal_equality",
    ),
    (
        "polynomial.system.real_root_box.certify",
        "polynomials.root_boxes.operations",
        "verify_real_root_box",
        "certify_real_root_box",
    ),
    (
        "polynomial.box.enclosure.compute",
        "polynomials.intervals._tools",
        "verify_polynomial_box_enclosure",
        "polynomial_box_enclosure",
    ),
)


@pytest.mark.parametrize("operation_id,module_name,verifier_name,boundary", CASES)
@pytest.mark.parametrize(
    "failure",
    (
        RuntimeError,
        ValueError,
        OperationExecutionTimeoutError,
        MemoryError,
        OperationResourceAdmissionError,
    ),
)
def test_verifier_preserves_operational_noncompletion(
    monkeypatch: pytest.MonkeyPatch,
    operation_id: str,
    module_name: str,
    verifier_name: str,
    boundary: str,
    failure: type[Exception],
) -> None:
    tool = next(tool for tool in BUILTIN_TOOLS if tool.operation_id == operation_id)
    payload = json.loads(json.dumps(tool.examples[0].input))
    if (
        operation_id == "formal_series.rational.reversion.compute"
        and boundary == "_compose_coefficients"
    ):
        payload["coefficients"][2] = {"num": "1", "den": "1"}
    result = tool.run(tool.request_type.model_validate_json(json.dumps(payload)))
    claim = tool.result_type.model_validate_json(result.model_dump_json())
    module = importlib.import_module("jacobian.math." + module_name)
    verifier = getattr(module, verifier_name)
    assert verifier(claim)
    error = (
        OperationResourceAdmissionError(
            location=(), code="test.resource", message="injected noncompletion"
        )
        if failure is OperationResourceAdmissionError
        else failure("injected noncompletion")
    )

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(module, boundary, fail)
    with pytest.raises(failure, match="injected noncompletion"):
        verifier(claim)


@pytest.mark.parametrize("restriction", (False, True))
@pytest.mark.parametrize("failure", (ValueError, OperationResourceAdmissionError))
def test_bernstein_verifier_preserves_operational_noncompletion(
    monkeypatch: pytest.MonkeyPatch,
    restriction: bool,
    failure: type[Exception],
) -> None:
    from jacobian.math.polynomials.bernstein import operations
    from jacobian.math.polynomials.bernstein.values import RationalBernsteinPolynomial

    tool = next(
        t
        for t in BUILTIN_TOOLS
        if t.operation_id == "polynomial.bernstein.coefficients.compute"
    )
    result = tool.run(
        tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    )
    claim = RationalBernsteinPolynomial.model_validate_json(result.model_dump_json())
    error = (
        OperationResourceAdmissionError(
            location=(), code="test.resource", message="injected noncompletion"
        )
        if failure is OperationResourceAdmissionError
        else failure("injected noncompletion")
    )

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(operations, "bernstein_coefficients", fail)
    with pytest.raises(failure, match="injected noncompletion"):
        if restriction:
            operations.verify_bernstein_restriction(claim, claim)
        else:
            operations.verify_bernstein_coefficients(claim)


EXPANDED_CASES: tuple[tuple[str, str, str, str], ...] = (
    (
        "algebra.affine_map.word_collision_profile.compute",
        "algebra.affine_map_word_collision.operations",
        "verify_word_collision_profile",
        "compute_word_collision_profile",
    ),
    (
        "convex.max_affine.evaluate",
        "analysis.convex.operations",
        "verify_max_affine_evaluation",
        "max_affine_evaluation",
    ),
    (
        "convex.max_affine.subdifferential",
        "analysis.convex.operations",
        "verify_max_affine_subdifferential",
        "max_affine_subdifferential",
    ),
    (
        "approximation.lagrange.basis.compute",
        "analysis.approximation.operations",
        "verify_lagrange_basis",
        "lagrange_basis",
    ),
    (
        "approximation.lagrange.interpolate.compute",
        "analysis.approximation.operations",
        "verify_lagrange_interpolation",
        "lagrange_interpolation",
    ),
    (
        "boolean.fourier_spectrum.compute",
        "analysis.boolean.fourier.operations",
        "verify_fourier_spectrum",
        "fourier_spectrum",
    ),
    (
        "boolean.multilinear_extension.compute",
        "analysis.boolean.fourier.operations",
        "verify_multilinear_extension",
        "multilinear_extension",
    ),
    (
        "boolean.erasure_noise.compute",
        "analysis.boolean.fourier.operations",
        "verify_erasure_noise",
        "erasure_noise",
    ),
    (
        "matrix.nullspace.compute",
        "matrices.operations",
        "verify_nullspace",
        "rank_result",
    ),
    (
        "matrix.minimal_polynomial.compute",
        "matrices.canonical_forms.operations",
        "verify_minimal_polynomial",
        "_minimal_polynomial_components",
    ),
    (
        "matrix.rational_canonical_form.compute",
        "matrices.canonical_forms.operations",
        "verify_rational_canonical_form",
        "_rational_canonical_components",
    ),
    (
        "matrix.primary_decomposition.compute",
        "matrices.canonical_forms.operations",
        "verify_primary_decomposition",
        "_primary_decomposition_components",
    ),
    (
        "matrix.real_quadratic.symmetric_spectrum.compute",
        "matrices.quadratic_spectral.operations",
        "verify_symmetric_spectrum",
        "symmetric_spectrum",
    ),
    (
        "matrix.real_quadratic.singular_spectrum.compute",
        "matrices.quadratic_spectral.operations",
        "verify_singular_spectrum",
        "singular_spectrum",
    ),
    (
        "matrix.real_quadratic.inertia.compute",
        "matrices.quadratic_spectral.operations",
        "verify_inertia",
        "inertia",
    ),
    (
        "matrix.subsystem.kronecker_product.compute",
        "matrices.subsystems.operations",
        "verify_subsystem_kronecker_product",
        "kronecker_product",
    ),
    (
        "matrix.subsystem.partial_trace.compute",
        "matrices.subsystems.operations",
        "verify_partial_trace",
        "partial_trace",
    ),
    (
        "matrix.subsystem.psd_order.decide",
        "matrices.subsystems.operations",
        "verify_psd_order",
        "psd_order",
    ),
    (
        "matrix.inertia.compute",
        "matrices.analysis.operations",
        "verify_inertia",
        "compute_inertia",
    ),
    (
        "polynomial.compute.gcd",
        "polynomials.operations",
        "verify_polynomial_gcd",
        "polynomial_gcd",
    ),
    (
        "polynomial.compute.resultant",
        "polynomials.operations",
        "verify_polynomial_resultant",
        "polynomial_resultant",
    ),
    (
        "polynomial.compute.discriminant",
        "polynomials.operations",
        "verify_polynomial_discriminant",
        "polynomial_discriminant",
    ),
    (
        "polynomial.compute.square_free_decomposition",
        "polynomials.operations",
        "verify_polynomial_square_free_decomposition",
        "polynomial_square_free_decomposition",
    ),
    (
        "polynomial.factor.compute",
        "polynomials.operations",
        "verify_polynomial_factorization",
        "polynomial_factorization",
    ),
    (
        "polynomial.unit_circle.arc_energy.compute",
        "polynomials.unit_circle.operations",
        "verify_unit_circle_arc_energy",
        "_real_cyclotomic_record",
    ),
    (
        "polynomial.unit_circle.real_symmetric_degree_one_fejer_riesz_factor.compute",
        "polynomials.unit_circle.operations",
        "verify_real_symmetric_degree_one_fejer_riesz_factor",
        "_laurent_coefficients",
    ),
    (
        "polynomial_field.scalar.gradient.compute",
        "polynomials.vector_calculus.operations",
        "verify_gradient",
        "gradient",
    ),
    (
        "polynomial_field.vector.divergence.compute",
        "polynomials.vector_calculus.operations",
        "verify_divergence",
        "divergence",
    ),
    (
        "polynomial_field.vector.curl.compute",
        "polynomials.vector_calculus.operations",
        "verify_curl",
        "curl",
    ),
    (
        "polynomial_field.scalar.laplacian.compute",
        "polynomials.vector_calculus.operations",
        "verify_laplacian",
        "laplacian",
    ),
    (
        "polynomial_field.scalar.directional_derivative.compute",
        "polynomials.vector_calculus.operations",
        "verify_directional_derivative",
        "directional_derivative",
    ),
    (
        "polynomial.interpolation.divided_differences.compute",
        "polynomials.interpolation.operations",
        "verify_divided_differences",
        "divided_differences",
    ),
    (
        "polynomial.interpolation.newton_evaluate.compute",
        "polynomials.interpolation.operations",
        "verify_newton_evaluation",
        "evaluate_newton",
    ),
    (
        "polynomial.interpolation.hermite.compute",
        "polynomials.interpolation.operations",
        "verify_hermite_interpolation",
        "hermite_interpolation",
    ),
    (
        "polynomial.map.jacobian",
        "polynomials.maps.operations",
        "verify_jacobian",
        "jacobian_matrix",
    ),
    (
        "polynomial.multivariate.divide.compute",
        "polynomials.multivariate.operations",
        "verify_multivariate_division",
        "multivariate_division",
    ),
    (
        "polynomial.multivariate.factor.compute",
        "polynomials.multivariate.operations",
        "verify_multivariate_factor",
        "multivariate_factor",
    ),
    (
        "polynomial.multivariate.subresultant_sequence.compute",
        "polynomials.multivariate.operations",
        "verify_multivariate_subresultant_sequence",
        "multivariate_subresultant_sequence",
    ),
    (
        "polynomial.real.common_interlacing_profile.compute",
        "polynomials.real_algebra.operations",
        "verify_common_interlacing_profile",
        "common_interlacing_profile",
    ),
    (
        "polynomial.sturm_chain.compute",
        "polynomials.real_algebra.operations",
        "verify_sturm_chain",
        "compute_sturm_chain",
    ),
    (
        "polynomial.root_count.compute",
        "polynomials.real_algebra.operations",
        "verify_root_count",
        "compute_root_count",
    ),
    (
        "polynomial.real.strict_sublevel_measure.compute",
        "polynomials.real_algebra.operations",
        "verify_strict_sublevel_measure",
        "compute_strict_sublevel_measure",
    ),
)

EXPANDED_CASES += (
    (
        "polynomial.ideal.groebner_basis.compute",
        "polynomials.ideals.operations",
        "verify_groebner_basis",
        "_run_sympy_kernel",
    ),
    (
        "rational_function.hermite_reduction.compute",
        "polynomials.rational_functions.operations",
        "verify_hermite_reduction",
        "hermite_reduction",
    ),
    (
        "formal_series.rational.inverse.compute",
        "polynomials.series.operations",
        "verify_inverse",
        "_cauchy_convolve",
    ),
    (
        "formal_series.rational.divide.compute",
        "polynomials.series.operations",
        "verify_divide",
        "_cauchy_convolve",
    ),
    (
        "formal_series.rational.reversion.compute",
        "polynomials.series.operations",
        "verify_reversion",
        "_compose_coefficients",
    ),
    (
        "boolean.fourier.walsh_transform.compute",
        "analysis.boolean.operations",
        "verify_walsh_transform",
        "walsh_hadamard_transform",
    ),
)

EXPANDED_CASES += (
    (
        "polynomial.multivariate.gcd.compute",
        "polynomials.multivariate.operations",
        "verify_multivariate_gcd",
        "rational_polynomial_to_sympy",
    ),
    (
        "polynomial.multivariate.resultant.compute",
        "polynomials.multivariate.operations",
        "verify_multivariate_resultant",
        "_sylvester_resultant_value",
    ),
    (
        "real_algebraic.plane_semialgebraic.component_profile.compute",
        "polynomials.real_algebra.operations",
        "verify_plane_component_profile",
        "compute_plane_component_profile",
    ),
)


@pytest.mark.parametrize(
    "operation_id,module_name,verifier_name,boundary", EXPANDED_CASES
)
@pytest.mark.parametrize(
    "failure", (ValueError, TypeError, OperationResourceAdmissionError)
)
def test_additional_verifier_preserves_operational_noncompletion(
    monkeypatch: pytest.MonkeyPatch,
    operation_id: str,
    module_name: str,
    verifier_name: str,
    boundary: str,
    failure: type[Exception],
) -> None:
    test_verifier_preserves_operational_noncompletion(
        monkeypatch, operation_id, module_name, verifier_name, boundary, failure
    )


@pytest.mark.parametrize("kind", ("smith", "membership"))
@pytest.mark.parametrize(
    "failure", (ValueError, TypeError, OperationResourceAdmissionError)
)
def test_authored_identity_verifier_preserves_operational_noncompletion(
    monkeypatch: pytest.MonkeyPatch, kind: str, failure: type[Exception]
) -> None:
    if kind == "smith":
        operation_id = "matrix.normal_form.smith.certified.compute"
        module_name = "jacobian.math.matrices.certified_snf.operations"
        verifier_name, boundary = (
            "verify_smith_normal_form_certificate",
            "matrix_determinant",
        )
    else:
        operation_id = "polynomial.ideal.membership_certificate.compute"
        module_name = "jacobian.math.polynomials.ideals.operations"
        verifier_name, boundary = (
            "verify_ideal_membership_certificate",
            "rational_polynomial_to_sympy",
        )
    tool = next(t for t in BUILTIN_TOOLS if t.operation_id == operation_id)
    result = tool.run(
        tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    )
    parsed = tool.result_type.model_validate_json(result.model_dump_json())
    claim = parsed.certificate if kind == "smith" else parsed
    module = importlib.import_module(module_name)
    verifier = getattr(module, verifier_name)
    assert verifier(claim)
    error = (
        OperationResourceAdmissionError(
            location=(), code="test.resource", message="injected noncompletion"
        )
        if failure is OperationResourceAdmissionError
        else failure("injected noncompletion")
    )

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(module, boundary, fail)
    with pytest.raises(failure, match="injected noncompletion"):
        verifier(claim)


@pytest.mark.parametrize("kind", ("inverse", "divide", "reversion"))
def test_long_trivial_series_admits_and_verifies(kind: str) -> None:
    from jacobian._exact import CanonicalRational
    from jacobian.math.polynomials.series import operations
    from jacobian.math.polynomials.series._models import TruncatedSeries

    zero = CanonicalRational(num=0, den=1)
    one = CanonicalRational(num=1, den=1)
    coefficients = (zero, one) if kind == "reversion" else (one,)
    source = TruncatedSeries(
        variable="x",
        truncation_order=513,
        coefficients=coefficients + (zero,) * (513 - len(coefficients)),
    )
    if kind == "divide":
        result = operations.divide(source, source)
    else:
        result = getattr(operations, kind)(source)
    assert getattr(operations, "verify_" + kind)(
        type(result).model_validate_json(result.model_dump_json())
    )
    tool = next(
        t
        for t in BUILTIN_TOOLS
        if t.operation_id == f"formal_series.rational.{kind}.compute"
    )
    wire_source = source.model_dump(mode="json")
    payload = (
        {"left": wire_source, "right": wire_source} if kind == "divide" else wire_source
    )
    published = tool.run(tool.request_type.model_validate_json(json.dumps(payload)))
    assert published == result


@pytest.mark.parametrize("kind", ("inverse", "divide", "reversion"))
def test_nontrivial_long_series_verification_preserves_resource_refusal(
    kind: str,
) -> None:
    from math import comb

    from jacobian._exact import CanonicalRational
    from jacobian.math.polynomials.series import operations
    from jacobian.math.polynomials.series._models import (
        SeriesDivideResult,
        SeriesInverseResult,
        SeriesReversionResult,
        TruncatedSeries,
    )

    n = 513
    zero = CanonicalRational(num=0, den=1)

    def series(values: tuple[int, ...]) -> TruncatedSeries:
        return TruncatedSeries(
            variable="x",
            truncation_order=n,
            coefficients=tuple(CanonicalRational(num=value, den=1) for value in values)
            + (zero,) * (n - len(values)),
        )

    claim: SeriesInverseResult | SeriesDivideResult | SeriesReversionResult
    if kind == "inverse":
        source = series((1, 1))
        claim = SeriesInverseResult(
            source=source,
            result=series(tuple((-1) ** i for i in range(n))),
            residual_coefficients=(zero,) * n,
        )
    elif kind == "divide":
        source = series((1, 1))
        claim = SeriesDivideResult(
            numerator=source,
            denominator=source,
            quotient=series((1,)),
            residual_coefficients=(zero,) * n,
        )
    else:
        coefficients = (
            0,
            *((-1) ** (i - 1) * (comb(2 * (i - 1), i - 1) // i) for i in range(1, n)),
        )
        claim = SeriesReversionResult(
            source=series((0, 1, 1)),
            result=series(coefficients),
            left_residual=(zero,) * n,
            right_residual=(zero,) * n,
        )
    with pytest.raises(OperationResourceAdmissionError):
        getattr(operations, "verify_" + kind)(
            type(claim).model_validate_json(claim.model_dump_json())
        )
