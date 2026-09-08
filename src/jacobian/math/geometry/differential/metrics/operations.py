"""Execute an admitted exact metric-curvature DAG on a retained chart locus."""

from __future__ import annotations

import time
from typing import Any

from pydantic_core import PydanticCustomError

from jacobian._execution import (
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.differential._recognition_process import (
    RationalFunctionRecognitionCandidate,
    recognize_canonical_rational_functions,
)
from jacobian.math.geometry.differential.metrics._dag import Expression, Node
from jacobian.math.geometry.differential.metrics._models import (
    RationalCoordinateConnection,
    RationalCoordinateMetric,
    RationalMetricCurvatureProfile,
)
from jacobian.math.geometry.differential.metrics._normalize_process import (
    cancel_fraction,
)
from jacobian.math.geometry.differential.metrics._plan import build_plan, singular
from jacobian.math.geometry.differential.values import (
    RationalCoordinateTensor,
    TensorVariance,
    canonical_locus_guards,
)
from jacobian.math.polynomials._conversions import (
    sparse_rational_polynomial_from_sympy,
    sparse_rational_polynomial_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.values import (
    RationalFunction,
    require_canonical_rational_function,
)


def _evaluate_node(
    index: int, nodes: list[Node], axis: tuple[str, ...], cache: dict[int, Any]
) -> Any:
    from sympy import QQ

    from jacobian.math.polynomials.rational_functions.gradient._kernel import (
        _differentiate_fraction,
    )

    if index in cache:
        return cache[index]
    request_checkpoint("before curvature polynomial arithmetic")
    node = nodes[index]
    if node.operation == "SOURCE":
        assert node.source is not None
        result = sparse_rational_polynomial_to_sympy(node.source, axis)
    elif node.operation == "SCALE":
        result = _evaluate_node(node.arguments[0], nodes, axis, cache).mul_ground(
            QQ(node.scalar.numerator, node.scalar.denominator)
        )
    elif node.operation == "MULTIPLY":
        result = _evaluate_node(node.arguments[0], nodes, axis, cache) * _evaluate_node(
            node.arguments[1], nodes, axis, cache
        )
    elif node.operation == "ADD":
        result = sum(
            (_evaluate_node(arg, nodes, axis, cache) for arg in node.arguments),
            cache[0],
        )
    elif node.operation == "DERIVATIVE":
        result, _ = _differentiate_fraction(
            _evaluate_node(node.arguments[0], nodes, axis, cache),
            _evaluate_node(node.arguments[1], nodes, axis, cache),
            node.axis,
        )
    else:
        raise AssertionError("unknown admitted curvature polynomial node")
    cache[index] = result
    request_checkpoint("after curvature polynomial arithmetic")
    return result


def curvature_profile(
    metric: RationalCoordinateMetric,
) -> RationalMetricCurvatureProfile:
    """Compute inverse, Levi-Civita connection, Riemann, Ricci and scalar fields.

    Every symbolic operation, source claim and canonical output is admitted
    before polynomial expansion. The input locus is intersected with det(g)!=0;
    normalization never discards inherited chart restrictions.
    """
    execution = current_request_execution()
    if execution is None:
        with request_execution(time.monotonic()):
            return curvature_profile(metric)
    deadline = execution.started_at + 120.0
    if execution.deadline is not None:
        deadline = min(deadline, execution.deadline)
    bind_request_deadline(deadline)
    request_checkpoint("before curvature admission")
    plan = build_plan(metric)
    request_checkpoint("after complete curvature admission")
    # Caller-authored field presentations have only structural validation.
    # Recognize reducedness once before relying on source-field identities.
    recognition_candidates: list[RationalFunctionRecognitionCandidate] = []
    for index, component in enumerate(metric.tensor.components):
        request_checkpoint("before metric component recognition")
        if (
            not component.numerator.terms
            or not component.variables
            or len(component.denominator.terms) == 1
        ):
            try:
                require_canonical_rational_function(component)
            except PydanticCustomError as exc:
                raise OperationDomainValidationError(
                    location=("metric",),
                    code="differential_geometry.curvature.noncanonical_source",
                    message="metric component must be a reduced canonical rational function",
                ) from exc
            continue
        recognition_candidates.append(
            RationalFunctionRecognitionCandidate(
                owner="tensor",
                component=index,
                value=component,
            )
        )
    recognition = recognize_canonical_rational_functions(
        tuple(recognition_candidates), deadline=deadline
    )
    if recognition.non_coprime is not None:
        raise OperationDomainValidationError(
            location=("metric",),
            code="differential_geometry.curvature.noncanonical_source",
            message="metric component must be a reduced canonical rational function",
        )
    from sympy import QQ, Poly

    axis = metric.tensor.coordinate_axis
    symbols = symbols_for_variables(axis)
    cache: dict[int, Any] = {
        0: Poly(0, *symbols, domain=QQ),
        1: Poly(1, *symbols, domain=QQ),
    }

    def raw(value: Expression) -> tuple[Any, Any]:
        numerator, denominator = plan.fractions[value]
        return _evaluate_node(numerator, plan.dag.nodes, axis, cache), _evaluate_node(
            denominator, plan.dag.nodes, axis, cache
        )

    determinant_guards = []
    for index in dict.fromkeys(plan.determinant.numerator):
        polynomial = _evaluate_node(index, plan.dag.nodes, axis, cache)
        if polynomial.is_zero:
            raise singular()
        determinant_guards.append(
            sparse_rational_polynomial_from_sympy(
                polynomial.monic(), axis, maximum_terms=256
            )
        )
    normalized: dict[Expression, RationalFunction] = {}

    def convert(values: tuple[Expression, ...]) -> tuple[RationalFunction, ...]:
        results = []
        for value in values:
            request_checkpoint("before curvature component normalization")
            if value not in normalized:
                numerator, denominator = raw(value)
                cancelled_num, cancelled_den = cancel_fraction(
                    numerator, denominator, deadline=deadline
                )
                normalized[value] = RationalFunction._from_kernel(
                    variables=axis,
                    numerator=sparse_rational_polynomial_from_sympy(
                        cancelled_num, axis, maximum_terms=256
                    ),
                    denominator=sparse_rational_polynomial_from_sympy(
                        cancelled_den, axis, maximum_terms=256
                    ),
                )
            results.append(normalized[value])
            request_checkpoint("after curvature component normalization")
        return tuple(results)

    inverse, connection, riemann, ricci, scalar = (
        convert(values)
        for values in (
            plan.inverse,
            plan.connection,
            plan.riemann,
            plan.ricci,
            (plan.scalar,),
        )
    )
    guards = canonical_locus_guards(
        metric.tensor.retained_nonzero_denominators,
        tuple(determinant_guards),
        component_denominators=tuple(
            value.denominator
            for values in (inverse, connection, riemann, ricci, scalar)
            for value in values
        ),
        variable_count=len(axis),
    )

    def tensor(
        components: tuple[RationalFunction, ...], variance: tuple[TensorVariance, ...]
    ) -> RationalCoordinateTensor:
        return RationalCoordinateTensor(
            coordinate_axis=axis,
            variance=variance,
            components=components,
            retained_nonzero_denominators=guards,
        )

    result = RationalMetricCurvatureProfile(
        metric=metric,
        inverse_metric=tensor(inverse, ("CONTRAVARIANT", "CONTRAVARIANT")),
        connection=RationalCoordinateConnection(
            coordinate_axis=axis,
            components=connection,
            retained_nonzero_denominators=guards,
        ),
        riemann=tensor(
            riemann, ("CONTRAVARIANT", "COVARIANT", "COVARIANT", "COVARIANT")
        ),
        ricci=tensor(ricci, ("COVARIANT", "COVARIANT")),
        scalar_curvature=tensor(scalar, ()),
    )
    request_checkpoint("after curvature profile construction")
    return result
