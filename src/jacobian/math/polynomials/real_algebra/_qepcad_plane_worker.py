"""One-shot exact plane-component worker backed by QEPCAD 1.74."""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable
from math import gcd, lcm

from jacobian.canonical import encode_strict_json, parse_canonical_integer
from jacobian.math.polynomials.real_algebra._plane_component_bounds import (
    MAX_PLANE_COMPONENT_PREDICTED_CELLS,
    MAX_PLANE_COMPONENT_PROJECTED_COEFFICIENT_DIGITS,
    MAX_PLANE_COMPONENT_PROJECTION_DEGREE_SUM,
    plane_projection_bound,
    plane_projection_coefficient_bound,
)
from jacobian.math.polynomials.real_algebra._plane_component_models import (
    MAX_PLANE_COMPONENTS,
    IsolatedRealPlanePoint,
    PlaneSemialgebraicSet,
    PlaneSign,
    _plane_point_identity_key,
    _PlanePointIdentityKey,
)
from jacobian.math.polynomials.real_algebra._qepcad_plane_protocol import (
    MAX_QEPCAD_CLOSURE_CELLS,
    MAX_QEPCAD_SAMPLE_CHARACTERS,
    MAX_QEPCAD_TRUE_CELLS,
    PLANE_WORKER_REQUEST_ADAPTER,
    PlaneSamplesValid,
    PlaneSampleWorkerRequest,
    QepcadPlaneCell,
    QepcadPlaneCellClosure,
    QepcadPlaneWorkerComplete,
    QepcadPlaneWorkerInvalid,
    QepcadPlaneWorkerRejected,
    QepcadPlaneWorkerRequest,
    QepcadPlaneWorkerResponse,
)
from jacobian.math.polynomials.real_algebra._qepcad_plane_samples import (
    QepcadSampleError,
    QepcadSampleLimitError,
    canonicalize_isolated_plane_point,
    parse_qepcad_plane_sample,
)
from jacobian.math.polynomials.values import RationalPolynomial
from jacobian.process import (
    BoundedWorkerDialogue,
    BoundedWorkerDialogueError,
    BoundedWorkerDialogueErrorReason,
    run_bounded_worker_dialogue,
)

_SOLUTION_PROMPT = b"Before Solution >"
_NORMALIZATION_PROMPT = b"Before Normalization >"
_MAX_FRAME_BYTES = 8 * 1024 * 1024
_QEPCAD_STDERR_BYTES = 64 * 1024
# This is one request ledger across the source CAD and its optional
# sample-classification refinement. Useful source-backed cases consume well
# below one MiB; the larger cap leaves room for exact algebraic cell samples
# without allowing each closure query to acquire a fresh output budget.
_MAX_TRANSCRIPT_BYTES = 64 * 1024 * 1024
# The structural maximum is below 9.7 million ASCII characters: at most 81
# four-atom source rows with 15-term, 480-digit cleared polynomials, plus 272
# coordinate-marker tautologies with 17-term, 512-digit polynomials. Use the
# a direct structural ceiling rather than borrowing a transport setting.
_MAX_FORMULA_CHARACTERS = 10_000_000
_TRUE_CELL_FRAME_PREFIX = "\nd-true-cells\n"
_TRUE_CELL_FRAME_SUFFIX = "\nBefore Solution >"
_CELL_HEADER = re.compile(
    r"---------- Information about the cell "
    r"\((?P<x>[1-9][0-9]*),(?P<y>[1-9][0-9]*)\) ----------\n\n"
)
_DIMENSION = re.compile(r"^Dimension\s*:\s*(?P<dimension>[012])\s*$", re.MULTILINE)
_SAMPLE_MARKER = "----------   Sample point  ---------- \n"
_CELL_FOOTER = "\n----------------------------------------------------\n"
_VERSION = re.compile(r"Version B (?P<version>[0-9]+\.[0-9]+),")
_MAX_CELL_INDEX_DIGITS = len(str(MAX_PLANE_COMPONENT_PREDICTED_CELLS))


class _QepcadProtocolError(RuntimeError):
    pass


class _UnsupportedQepcadVersionError(RuntimeError):
    pass


class _QepcadDeadlineError(RuntimeError):
    pass


class _QepcadOutputLimitError(RuntimeError):
    pass


class _QepcadCellLimitError(RuntimeError):
    pass


def _send(dialogue: BoundedWorkerDialogue, value: str) -> None:
    dialogue.send(value.encode("ascii") + b"\n")


def _read_until(dialogue: BoundedWorkerDialogue, marker: bytes) -> str:
    output = dialogue.read_until(marker, frame_limit=_MAX_FRAME_BYTES)
    try:
        return output.decode("ascii")
    except UnicodeDecodeError as exc:
        raise _QepcadProtocolError("QEPCAD output was not ASCII") from exc


def _cell_index(value: str) -> int:
    if len(value) > _MAX_CELL_INDEX_DIGITS:
        raise _QepcadCellLimitError("QEPCAD cell index exceeded its bound")
    return parse_canonical_integer(value)


def _cell_blocks(output: str) -> Iterable[tuple[tuple[int, int], str]]:
    if not output.startswith(_TRUE_CELL_FRAME_PREFIX) or not output.endswith(
        _TRUE_CELL_FRAME_SUFFIX
    ):
        raise _QepcadProtocolError("QEPCAD returned an invalid true-cell frame")
    body = output[len(_TRUE_CELL_FRAME_PREFIX) : -len(_TRUE_CELL_FRAME_SUFFIX)]
    cursor = 0
    seen: set[tuple[int, int]] = set()
    while cursor < len(body):
        match = _CELL_HEADER.match(body, cursor)
        if match is None:
            raise _QepcadProtocolError("QEPCAD returned unconsumed true-cell output")
        index = (
            _cell_index(match.group("x")),
            _cell_index(match.group("y")),
        )
        if index in seen:
            raise _QepcadProtocolError("QEPCAD repeated a true cell")
        seen.add(index)
        footer_start = body.find(_CELL_FOOTER, match.end())
        if footer_start < 0:
            raise _QepcadProtocolError("QEPCAD omitted a true-cell terminator")
        yield index, body[match.end() : footer_start]
        cursor = footer_start + len(_CELL_FOOTER)


def _parse_true_cells(output: str) -> tuple[QepcadPlaneCell, ...]:
    cells: list[QepcadPlaneCell] = []
    for index, block in _cell_blocks(output):
        dimension_matches = tuple(_DIMENSION.finditer(block))
        sample_start = block.find(_SAMPLE_MARKER)
        if (
            len(dimension_matches) != 1
            or block.count(_SAMPLE_MARKER) != 1
            or sample_start < dimension_matches[0].end()
        ):
            raise _QepcadProtocolError("QEPCAD returned an incomplete true-cell block")
        sample_start += len(_SAMPLE_MARKER)
        sample = block[sample_start:].strip()
        if not sample or len(sample) > MAX_QEPCAD_SAMPLE_CHARACTERS:
            raise _QepcadOutputLimitError("QEPCAD cell sample exceeded its bound")
        cells.append(
            QepcadPlaneCell(
                index=index,
                dimension=int(dimension_matches[0].group("dimension")),
                sample=sample,
            )
        )
        if len(cells) > MAX_QEPCAD_TRUE_CELLS:
            raise _QepcadCellLimitError("QEPCAD true-cell count exceeded its bound")
    return tuple(sorted(cells, key=lambda cell: cell.index))


def _parse_cell_indices(output: str) -> tuple[tuple[int, int], ...]:
    indices = tuple(sorted(index for index, _block in _cell_blocks(output)))
    if len(indices) > MAX_QEPCAD_CLOSURE_CELLS:
        raise _QepcadCellLimitError("QEPCAD closure-cell count exceeded its bound")
    return indices


def _command(
    dialogue: BoundedWorkerDialogue,
    command: str,
    *,
    marker: bytes = _SOLUTION_PROMPT,
) -> str:
    _send(dialogue, command)
    return _read_until(dialogue, marker)


def _set_truth_value(
    dialogue: BoundedWorkerDialogue,
    index: tuple[int, int] | None,
    value: int,
) -> None:
    rendered_index = "()" if index is None else f"({index[0]},{index[1]})"
    _send(dialogue, "set-truth-value")
    _send(dialogue, rendered_index)
    _send(dialogue, str(value))
    _read_until(dialogue, _SOLUTION_PROMPT)


def _integer_coefficients(
    polynomial: RationalPolynomial,
) -> tuple[tuple[int, tuple[int, int]], ...]:
    denominator = lcm(
        *(
            term.coefficient.as_fraction().denominator
            for term in polynomial.polynomial.terms
        )
    )
    coefficients = tuple(
        (
            term.coefficient.as_fraction().numerator
            * (denominator // term.coefficient.as_fraction().denominator),
            (term.exponents[0], term.exponents[1]),
        )
        for term in polynomial.polynomial.terms
    )
    content = 0
    for coefficient, _exponents in coefficients:
        content = gcd(content, abs(coefficient))
    return tuple(
        (coefficient // content, exponents) for coefficient, exponents in coefficients
    )


def _qepcad_polynomial(polynomial: RationalPolynomial) -> str:
    terms: list[str] = []
    for coefficient, exponents in _integer_coefficients(polynomial):
        monomial_factors = []
        for variable, exponent in zip(("x", "y"), exponents, strict=True):
            if exponent == 1:
                monomial_factors.append(variable)
            elif exponent > 1:
                monomial_factors.append(f"{variable}^{exponent}")
        monomial = " ".join(monomial_factors)
        magnitude = abs(coefficient)
        body = (
            monomial
            if monomial and magnitude == 1
            else f"{magnitude} {monomial}".strip()
        )
        if not terms:
            terms.append(f"-{body}" if coefficient < 0 else body)
        else:
            terms.append(f"{'-' if coefficient < 0 else '+'} {body}")
    return " ".join(terms)


def _atom(polynomial: str, sign: PlaneSign) -> str:
    relation = {
        PlaneSign.NEGATIVE: "<",
        PlaneSign.ZERO: "=",
        PlaneSign.POSITIVE: ">",
    }[sign]
    return f"[{polynomial} {relation} 0]"


def _join(expressions: tuple[str, ...], operator: str) -> str:
    if len(expressions) == 1:
        return expressions[0]
    return f"[{' '.join(f'{operator} {expression}' if index else expression for index, expression in enumerate(expressions))}]"


def _qepcad_formula(
    semialgebraic_set: PlaneSemialgebraicSet,
    samples: tuple[IsolatedRealPlanePoint, ...],
) -> str:
    rendered_polynomials = tuple(
        _qepcad_polynomial(polynomial) for polynomial in semialgebraic_set.polynomials
    )
    rows = tuple(
        _join(
            tuple(
                _atom(polynomial, sign)
                for polynomial, sign in zip(
                    rendered_polynomials, condition.signs, strict=True
                )
            ),
            "/\\",
        )
        for condition in semialgebraic_set.sign_conditions
    )
    source = _join(rows, "\\/")
    marker_polynomials = {
        _qepcad_polynomial(polynomial)
        for sample in samples
        for polynomial in sample.coordinate_polynomials
    }
    tautologies = tuple(
        _join(tuple(_atom(polynomial, sign) for sign in PlaneSign), "\\/")
        for polynomial in sorted(marker_polynomials)
    )
    formula = _join((source, *tautologies), "/\\") if tautologies else source
    if len(formula) > _MAX_FORMULA_CHARACTERS:
        raise _QepcadOutputLimitError("QEPCAD formula exceeded its byte bound")
    return formula


def _run_qepcad(
    request: QepcadPlaneWorkerRequest,
    formula: str,
    *,
    stdout_limit: int,
) -> tuple[
    tuple[QepcadPlaneCell, ...],
    tuple[QepcadPlaneCellClosure, ...],
    int,
]:
    def transact(
        dialogue: BoundedWorkerDialogue,
    ) -> tuple[tuple[QepcadPlaneCell, ...], tuple[QepcadPlaneCellClosure, ...]]:
        _send(dialogue, "[]")
        _send(dialogue, "(x,y)")
        _send(dialogue, "2")
        _send(dialogue, f"{formula}.")
        startup = _read_until(dialogue, _NORMALIZATION_PROMPT)
        version_match = _VERSION.search(startup)
        if version_match is None or version_match.group("version") != "1.74":
            raise _UnsupportedQepcadVersionError("unsupported QEPCAD version")

        _command(dialogue, "full-cad", marker=_NORMALIZATION_PROMPT)
        _command(dialogue, "go", marker=b"Before Projection (y) >")
        _command(dialogue, "go", marker=b"Before Choice >")
        _command(dialogue, "go")

        true_cells = _parse_true_cells(_command(dialogue, "d-true-cells"))
        closures: list[QepcadPlaneCellClosure] = []
        for cell in true_cells:
            _set_truth_value(dialogue, None, 2)
            _set_truth_value(dialogue, cell.index, 1)
            _command(dialogue, "closure2d")
            closures.append(
                QepcadPlaneCellClosure(
                    cell_index=cell.index,
                    closure_indices=_parse_cell_indices(
                        _command(dialogue, "d-true-cells")
                    ),
                )
            )

        _send(dialogue, "quit")
        return true_cells, tuple(closures)

    try:
        completed = run_bounded_worker_dialogue(
            [request.executable, "+N100000000"],
            transact,
            absolute_deadline=request.deadline_monotonic,
            environment={
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "qe": request.qepcad_root,
            },
            stdout_limit=stdout_limit,
            stderr_limit=_QEPCAD_STDERR_BYTES,
        )
    except BoundedWorkerDialogueError as exc:
        if exc.reason is BoundedWorkerDialogueErrorReason.DEADLINE_EXPIRED:
            raise _QepcadDeadlineError(
                "QEPCAD deadline expired during its adaptive transaction"
            ) from exc
        if exc.reason in {
            BoundedWorkerDialogueErrorReason.STDOUT_LIMIT,
            BoundedWorkerDialogueErrorReason.STDERR_LIMIT,
        }:
            raise _QepcadOutputLimitError(
                "QEPCAD transcript exceeded its byte bound"
            ) from exc
        if exc.reason is BoundedWorkerDialogueErrorReason.CLOSED:
            raise _QepcadProtocolError(
                "QEPCAD exited before the expected prompt"
            ) from exc
        raise OSError("QEPCAD execution failed") from exc
    true_cells, closures = completed.value
    return true_cells, closures, completed.stdout_bytes


def _components(
    true_cells: tuple[QepcadPlaneCell, ...],
    closures: tuple[QepcadPlaneCellClosure, ...],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    true_indices = {cell.index for cell in true_cells}
    adjacency = {cell: {cell} for cell in true_indices}
    for closure in closures:
        for member in true_indices.intersection(closure.closure_indices):
            adjacency[closure.cell_index].add(member)
            adjacency[member].add(closure.cell_index)
    remaining = set(true_indices)
    components: list[tuple[tuple[int, int], ...]] = []
    while remaining:
        seed = min(remaining)
        frontier = [seed]
        connected: set[tuple[int, int]] = set()
        while frontier:
            cell = frontier.pop()
            if cell in connected:
                continue
            connected.add(cell)
            frontier.extend(adjacency[cell] - connected)
        remaining -= connected
        components.append(tuple(sorted(connected)))
    if len(components) > MAX_PLANE_COMPONENTS:
        raise _QepcadCellLimitError("QEPCAD component count exceeded its bound")
    return tuple(components)


def _computed_projection(
    request: QepcadPlaneWorkerRequest,
    canonical_samples: tuple[IsolatedRealPlanePoint, ...],
    true_cells: tuple[QepcadPlaneCell, ...],
    closures: tuple[QepcadPlaneCellClosure, ...],
) -> tuple[tuple[IsolatedRealPlanePoint, ...], tuple[int | None, ...]]:
    cell_by_index = {cell.index: cell for cell in true_cells}
    representatives: list[
        tuple[IsolatedRealPlanePoint, tuple[tuple[int, int], ...]]
    ] = []
    for cells in _components(true_cells, closures):
        maximum_dimension = max(cell_by_index[cell].dimension for cell in cells)
        representative_cell = min(
            cell for cell in cells if cell_by_index[cell].dimension == maximum_dimension
        )
        representatives.append(
            (
                parse_qepcad_plane_sample(
                    cell_by_index[representative_cell].sample,
                    axis=request.request.semialgebraic_set.axis,
                ),
                cells,
            )
        )
    representatives.sort(key=lambda item: _plane_point_identity_key(item[0]))
    component_by_cell = {
        cell: component_id
        for component_id, (_representative, cells) in enumerate(representatives)
        for cell in cells
    }
    sample_keys: dict[_PlanePointIdentityKey, list[int]] = {}
    for index, sample in enumerate(canonical_samples):
        sample_keys.setdefault(_plane_point_identity_key(sample), []).append(index)
    matching_cells: dict[int, tuple[int, int]] = {}
    if sample_keys:
        for cell in true_cells:
            if cell.dimension != 0:
                continue
            point = parse_qepcad_plane_sample(
                cell.sample,
                axis=request.request.semialgebraic_set.axis,
            )
            sample_indices = sample_keys.get(_plane_point_identity_key(point))
            if sample_indices is None:
                continue
            if any(index in matching_cells for index in sample_indices):
                raise QepcadSampleError("QEPCAD repeated a supplied sample cell")
            for sample_index in sample_indices:
                matching_cells[sample_index] = cell.index

    sample_component_ids = tuple(
        component_by_cell[matching_cells[index]] if index in matching_cells else None
        for index in range(len(canonical_samples))
    )
    return (
        tuple(representative for representative, _cells in representatives),
        sample_component_ids,
    )


def _compute(request: QepcadPlaneWorkerRequest) -> QepcadPlaneWorkerResponse:
    if request.canonical_samples is not None:
        canonical_samples = request.canonical_samples
    else:
        try:
            canonical_samples = tuple(
                canonicalize_isolated_plane_point(sample)
                for sample in request.request.samples
            )
        except QepcadSampleLimitError:
            return QepcadPlaneWorkerInvalid(reason="SAMPLE_RECOGNITION_LIMIT")
        except QepcadSampleError:
            return QepcadPlaneWorkerInvalid(reason="SAMPLE_NOT_ISOLATED")

    try:
        transcript_remaining = _MAX_TRANSCRIPT_BYTES
        source_formula = _qepcad_formula(
            request.request.semialgebraic_set,
            (),
        )
        source_cells, source_closures, source_stdout_bytes = _run_qepcad(
            request,
            source_formula,
            stdout_limit=transcript_remaining,
        )
        transcript_remaining -= source_stdout_bytes
        representatives, _empty_component_ids = _computed_projection(
            request,
            (),
            source_cells,
            source_closures,
        )
        if canonical_samples:
            refinement_points = (*canonical_samples, *representatives)
            refinement_polynomials = tuple(
                {
                    encode_strict_json(polynomial.model_dump(mode="json")): polynomial
                    for point in refinement_points
                    for polynomial in point.coordinate_polynomials
                }.values()
            )
            projection_degree_sum, predicted_cells = plane_projection_bound(
                (
                    *request.request.semialgebraic_set.polynomials,
                    *refinement_polynomials,
                )
            )
            projected_coefficient_digits = plane_projection_coefficient_bound(
                (
                    *request.request.semialgebraic_set.polynomials,
                    *refinement_polynomials,
                )
            )
            if (
                projection_degree_sum > MAX_PLANE_COMPONENT_PROJECTION_DEGREE_SUM
                or predicted_cells > MAX_PLANE_COMPONENT_PREDICTED_CELLS
                or projected_coefficient_digits
                > MAX_PLANE_COMPONENT_PROJECTED_COEFFICIENT_DIGITS
            ):
                raise _QepcadCellLimitError(
                    "sample classification refinement exceeded its CAD envelope"
                )
            refinement_formula = _qepcad_formula(
                request.request.semialgebraic_set,
                refinement_points,
            )
            refinement_cells, refinement_closures, refinement_stdout_bytes = (
                _run_qepcad(
                    request,
                    refinement_formula,
                    stdout_limit=transcript_remaining,
                )
            )
            transcript_remaining -= refinement_stdout_bytes
            if transcript_remaining < 0:
                raise _QepcadOutputLimitError(
                    "QEPCAD transcript exceeded its request byte bound"
                )
            refined_representatives, refinement_component_ids = _computed_projection(
                request,
                refinement_points,
                refinement_cells,
                refinement_closures,
            )
            sample_count = len(canonical_samples)
            sample_refined_ids = refinement_component_ids[:sample_count]
            source_refined_ids = refinement_component_ids[sample_count:]
            if (
                len(refined_representatives) != len(representatives)
                or any(component_id is None for component_id in source_refined_ids)
                or len(set(source_refined_ids)) != len(representatives)
            ):
                raise _QepcadProtocolError(
                    "sample refinement changed the source component partition"
                )
            source_id_by_refined = {
                refined_id: source_id
                for source_id, refined_id in enumerate(source_refined_ids)
            }
            sample_component_ids = tuple(
                None if refined_id is None else source_id_by_refined.get(refined_id)
                for refined_id in sample_refined_ids
            )
            if any(
                refined_id is not None and source_id is None
                for refined_id, source_id in zip(
                    sample_refined_ids,
                    sample_component_ids,
                    strict=True,
                )
            ):
                raise _QepcadProtocolError(
                    "a refined sample component did not map to the source partition"
                )
        else:
            sample_component_ids = ()
    except _UnsupportedQepcadVersionError:
        return QepcadPlaneWorkerRejected(reason="UNSUPPORTED_QEPCAD_VERSION")
    except _QepcadDeadlineError:
        return QepcadPlaneWorkerRejected(reason="QEPCAD_DEADLINE_EXPIRED")
    except _QepcadCellLimitError:
        return QepcadPlaneWorkerRejected(reason="QEPCAD_CELL_LIMIT")
    except (QepcadSampleLimitError, _QepcadOutputLimitError):
        return QepcadPlaneWorkerRejected(reason="QEPCAD_OUTPUT_LIMIT")
    except (QepcadSampleError, _QepcadProtocolError):
        return QepcadPlaneWorkerRejected(reason="QEPCAD_INVALID_OUTPUT")
    except OSError:
        return QepcadPlaneWorkerRejected(reason="QEPCAD_EXECUTION_FAILED")
    return QepcadPlaneWorkerComplete(
        version="1.74",
        representatives=representatives,
        sample_component_ids=sample_component_ids,
    )


def _validate_samples(
    request: PlaneSampleWorkerRequest,
) -> PlaneSamplesValid | QepcadPlaneWorkerInvalid:
    try:
        canonical_samples = tuple(
            canonicalize_isolated_plane_point(sample) for sample in request.samples
        )
    except QepcadSampleLimitError:
        return QepcadPlaneWorkerInvalid(reason="SAMPLE_RECOGNITION_LIMIT")
    except QepcadSampleError:
        return QepcadPlaneWorkerInvalid(reason="SAMPLE_NOT_ISOLATED")
    return PlaneSamplesValid(canonical_samples=canonical_samples)


def main() -> int:
    request = PLANE_WORKER_REQUEST_ADAPTER.validate_json(
        sys.stdin.buffer.read(),
        strict=True,
    )
    response = (
        _validate_samples(request)
        if isinstance(request, PlaneSampleWorkerRequest)
        else _compute(request)
    )
    sys.stdout.write(response.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
