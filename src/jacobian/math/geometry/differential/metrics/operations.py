"""Execute an admitted exact metric-curvature DAG on a retained chart locus."""

from __future__ import annotations

import time

from jacobian._execution import (
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.math.geometry.differential.metrics._dag_process import (
    RationalDagWorkerMessages,
    evaluate_admitted_rational_dag,
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

_CURVATURE_MESSAGES = RationalDagWorkerMessages(
    timeout_before="curvature deadline expired before DAG expansion",
    timeout_after="curvature deadline expired after payload encoding",
    timeout_during="curvature deadline expired during polynomial DAG expansion",
    cancelled_during="curvature cancelled during polynomial DAG expansion",
    start_failure="bounded curvature worker could not be started",
    malformed="bounded curvature worker returned malformed output",
    directory_prefix="jacobian-metric-curvature-",
    checkpoint_prefix="curvature",
    noncanonical_location=("metric",),
    noncanonical_code="differential_geometry.curvature.noncanonical_source",
    noncanonical_message="metric component must be a reduced canonical rational function",
)


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
    axis = metric.tensor.coordinate_axis
    unique_outputs = tuple(
        dict.fromkeys(
            (*plan.inverse, *plan.connection, *plan.riemann, *plan.ricci, plan.scalar)
        )
    )
    unique_values, determinant_guards = evaluate_admitted_rational_dag(
        plan.dag.nodes,
        [list(plan.fractions[value]) for value in unique_outputs],
        list(dict.fromkeys(plan.determinant.numerator)),
        axis,
        sources=metric.tensor.components,
        deadline=deadline,
        messages=_CURVATURE_MESSAGES,
    )
    normalized = dict(zip(unique_outputs, unique_values, strict=True))

    def convert(values: tuple[object, ...]) -> tuple[RationalFunction, ...]:
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
