"""Bounded exact pair distances using FLINT multiprecision rationals."""

from dataclasses import dataclass
from math import gcd
from time import monotonic

from flint import fmpq

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs.colorings import (
    HyperedgeColorAssignment,
    IndexedHyperedgeColoring,
)
from jacobian.math.geometry.exact._models import PointConfiguration
from jacobian.math.geometry.exact.distance_edge_coloring._models import (
    DistanceEdgeColoringResult,
)

__all__ = ["compute_distance_edge_coloring"]

_MAX_SOURCE_BITS = 8_000_000
_MAX_RESULT_BITS = 64_000_000
_MAX_WORK = 10**13
# A component below 2**_RATIONAL_BITS fits the canonical decimal carrier.
_RATIONAL_BITS = (10**MAX_CANONICAL_RATIONAL_DIGITS).bit_length() - 1


def _reject(reason: str) -> None:
    raise OperationResourceAdmissionError(
        location=("configuration",),
        code="distance_edge_coloring.resource_bound",
        message=reason,
    )


def _height(value: CanonicalRational) -> int:
    return max(abs(value.num).bit_length(), value.den.bit_length())


@dataclass
class _Budget:
    deadline: float
    work: int = 0

    def charge(
        self, amount: int = 0, *, stage: str = "during distance computation"
    ) -> None:
        request_checkpoint(stage)
        if monotonic() >= self.deadline:
            raise OperationExecutionTimeoutError(
                "exact distance request deadline expired"
            )
        self.work += amount
        if self.work > _MAX_WORK:
            _reject("exact distance arithmetic exceeds the bit-operation work bound")


def _plan(
    configuration: PointConfiguration, budget: _Budget
) -> tuple[tuple[fmpq, ...], ...]:
    """Admit bounded subtraction, then all squared sums and their output.

    Translation-invariant differences are reduced before estimating squared
    distances: equal coordinates contribute nothing, even at large absolute
    height. Subtraction is admitted separately before this normalization.
    For denominators q_i let L=lcm(q_i). Every squared sum has denominator
    dividing L² and numerator at most sum_i (abs(p_i)*L/q_i)². Positivity
    bounds all partial sums by this same numerator bound. FLINT's temporary
    products require at most twice the admitted rational component height.
    """
    points = configuration.points
    source_bits = sum(
        abs(value.num).bit_length() + value.den.bit_length()
        for point in points
        for value in point.coordinates
    )
    if source_bits > _MAX_SOURCE_BITS:
        _reject("retained source coefficients exceed the aggregate bit-storage bound")
    # Existing configuration bounds give <=2016 edges and <=40320 terms.
    # Charge every subtraction before materializing any normalized differences.
    for i, left in enumerate(points):
        for right in points[i + 1 :]:
            for a, b in zip(left.coordinates, right.coordinates, strict=True):
                if a != b:
                    height = _height(a) + _height(b) + 1
                    budget.charge(
                        8 * height if a.den == b.den == 1 else 8 * height * height
                    )
    plans: list[tuple[fmpq, ...]] = []
    result_bits = source_bits
    maximum_height = 1
    for i, left in enumerate(points):
        for right in points[i + 1 :]:
            differences = tuple(
                fmpq(a.num, a.den) - fmpq(b.num, b.den)
                for a, b in zip(left.coordinates, right.coordinates, strict=True)
                if a != b
            )
            common_denominator = 1
            for difference in differences:
                denominator = int(difference.denominator)
                h = common_denominator.bit_length() + denominator.bit_length()
                budget.charge(4 * h * h)
                factor = denominator // gcd(common_denominator, denominator)
                if common_denominator.bit_length() + factor.bit_length() > (
                    _RATIONAL_BITS // 2 + 1
                ):
                    _reject("squared-distance denominators exceed the exact carrier")
                common_denominator *= factor
            denominator_bits = 2 * (common_denominator.bit_length() - 1) + 2
            scaled_bits = max(
                (
                    abs(int(value.numerator)).bit_length()
                    + (common_denominator // int(value.denominator)).bit_length()
                    for value in differences
                ),
                default=0,
            )
            numerator_bits = 2 * scaled_bits + (len(differences) - 1).bit_length()
            height = max(1, numerator_bits, denominator_bits)
            if height > _RATIONAL_BITS:
                _reject("squared-distance coefficient growth exceeds the exact carrier")
            budget.charge(16 * len(differences) * height * height)
            result_bits += 2 * height
            maximum_height = max(maximum_height, height)
            plans.append(differences)
    # Every edge can have its own palette row; source axes, assignments and
    # graph incidences have independently bounded counts from PointConfiguration.
    if result_bits > _MAX_RESULT_BITS:
        _reject("distance palette exceeds the aggregate rational bit-storage bound")
    edge_count = len(plans)
    budget.charge(4 * edge_count * (edge_count - 1).bit_length() * maximum_height**2)
    return tuple(plans)


def _require_result_compatible_labels(configuration: PointConfiguration) -> None:
    """Reject labels the FiniteHypergraph result carrier cannot represent."""

    for point in configuration.points:
        try:
            point.label.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise OperationDomainValidationError(
                location=("configuration", "points"),
                code="distance_edge_coloring.label_encoding",
                message="point labels must be valid UTF-8",
            ) from exc


def compute_distance_edge_coloring(
    configuration: PointConfiguration,
) -> DistanceEdgeColoringResult:
    """Return every labelled pair, coloured by its exact squared distance."""
    execution = current_request_execution()
    deadline = (execution.started_at if execution is not None else monotonic()) + 60
    if execution is not None and execution.deadline is not None:
        deadline = min(deadline, execution.deadline)
    bind_request_deadline(deadline)
    _require_result_compatible_labels(configuration)
    budget = _Budget(deadline)
    budget.charge(stage="before distance admission")
    plans = _plan(configuration, budget)
    distances: list[fmpq] = []
    for differences in plans:
        budget.charge()
        distances.append(sum((value * value for value in differences), fmpq(0)))
    palette: list[fmpq] = []
    indices = [0] * len(distances)
    # Sorting and adjacent equality give O(E log E) comparisons even for
    # rational values whose numeric hashes all collide.
    for position, value in sorted(enumerate(distances), key=lambda item: item[1]):
        budget.charge()
        if not palette or value != palette[-1]:
            palette.append(value)
        indices[position] = len(palette) - 1
    labels = tuple(point.label for point in configuration.points)
    edges = tuple(
        (f"{i}:{j}", (labels[i], labels[j]))
        for i in range(len(labels))
        for j in range(i + 1, len(labels))
    )
    budget.charge(stage="before distance result construction")
    result = DistanceEdgeColoringResult(
        configuration=configuration,
        squared_distances=tuple(
            CanonicalRational(num=int(value.numerator), den=int(value.denominator))
            for value in palette
        ),
        coloring=IndexedHyperedgeColoring(
            hypergraph=FiniteHypergraph(vertices=labels, edges=edges),
            color_count=len(palette),
            assignments=tuple(
                HyperedgeColorAssignment(edge_id=edge_id, color_index=color_index)
                for (edge_id, _), color_index in zip(edges, indices, strict=True)
            ),
        ),
    )
    budget.charge(stage="after distance result construction")
    return result
