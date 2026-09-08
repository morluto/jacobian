"""Execute an admitted exact metric-curvature DAG on a retained chart locus."""

from __future__ import annotations

import time

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
from jacobian.math.geometry.differential.metrics._dag import Expression
from jacobian.math.geometry.differential.metrics._dag_evaluate_process import (
    evaluate_admitted_dag,
)
from jacobian.math.geometry.differential.metrics._models import (
    RationalCoordinateConnection,
    RationalCoordinateMetric,
    RationalMetricCurvatureProfile,
)
from jacobian.math.geometry.differential.metrics._plan import build_plan
from jacobian.math.geometry.differential.values import (
    RationalCoordinateTensor,
    TensorVariance,
    canonical_locus_guards,
)
from jacobian.math.polynomials.values import RationalFunction


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
    candidates = tuple(
        RationalFunctionRecognitionCandidate(
            owner="tensor", component=index, value=component
        )
        for index, component in enumerate(dict.fromkeys(metric.tensor.components))
        if component.numerator.terms
        and component.variables
        and not (
            len(component.numerator.terms) == 1
            and not any(component.numerator.terms[0].exponents)
        )
        and not (
            len(component.denominator.terms) == 1
            and component.denominator.terms[0].coefficient.as_fraction() == 1
            and component.denominator.terms[0].exponents
            == (0,) * len(component.variables)
        )
    )
    recognition = recognize_canonical_rational_functions(candidates, deadline=deadline)
    if recognition.non_coprime is not None:
        raise OperationDomainValidationError(
            location=("metric",),
            code="differential_geometry.curvature.noncanonical_source",
            message="metric component must be a reduced canonical rational function",
        )
    axis = metric.tensor.coordinate_axis
    unique_outputs = tuple(
        dict.fromkeys(
            (*plan.inverse, *plan.connection, *plan.riemann, *plan.ricci, plan.scalar)
        )
    )
    components, determinant_guards = evaluate_admitted_dag(
        plan.dag.nodes,
        axis,
        fractions=tuple(plan.fractions[value] for value in unique_outputs),
        determinants=tuple(dict.fromkeys(plan.determinant.numerator)),
        sources=(),
        deadline=deadline,
        owner="metric curvature",
        noncanonical_location=("metric",),
        noncanonical_code="differential_geometry.curvature.noncanonical_source",
        noncanonical_message="metric component must be a reduced canonical rational function",
    )
    normalized = dict(zip(unique_outputs, components, strict=True))

    def convert(values: tuple[Expression, ...]) -> tuple[RationalFunction, ...]:
        return tuple(normalized[value] for value in values)

    inverse = convert(plan.inverse)
    connection = convert(plan.connection)
    riemann = convert(plan.riemann)
    ricci = convert(plan.ricci)
    scalar = convert((plan.scalar,))
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
