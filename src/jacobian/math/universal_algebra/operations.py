"""Exact native kernels over finite algebras."""

from __future__ import annotations

from itertools import product as iproduct

from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
)
from jacobian.catalog.models import OperationDomainValidationError

from ._models import (
    _HOMOMORPHISM_RESULT_RESERVE_BYTES,
    MAX_ENUMERATION_WORK,
    CongruenceResult,
    EquationCounterexample,
    EquationProfileResult,
    HomomorphismObstruction,
    HomomorphismProfileResult,
    SubalgebraResult,
    _congruence_work,
    _require_homomorphism_output_headroom,
)
from .values import (
    ApplicationTerm,
    FiniteAlgebra,
    FiniteAlgebraCarrierMap,
    FiniteAlgebraHomomorphism,
    FlatTerm,
    OperationSymbol,
    UniversalAlgebraAdmissionError,
    VariableTerm,
    _first_homomorphism_failure,
    _homomorphism_kernel_and_image,
    require_term_for_algebra,
)

__all__ = [
    "congruence_check",
    "equation_profile",
    "evaluate_term",
    "generated_subalgebra",
    "homomorphism_profile",
    "quotient",
]


def _reject(*, location: tuple[str | int, ...], code: str, message: str) -> None:
    raise OperationDomainValidationError(
        location=location, code=f"universal_algebra.{code}", message=message
    )


def _admit_evaluate(
    algebra: FiniteAlgebra, term: FlatTerm, assignment: tuple[int, ...]
) -> None:
    try:
        require_term_for_algebra(term, algebra)
    except UniversalAlgebraAdmissionError as exc:
        _reject(location=("term",), code="term_signature", message=str(exc))
    if len(assignment) != term.variable_count:
        _reject(
            location=("assignment",),
            code="assignment_coverage",
            message="assignment must cover exactly the referenced variables",
        )
    size = len(algebra.carrier)
    if any(not 0 <= value < size for value in assignment):
        _reject(
            location=("assignment",),
            code="assignment_carrier_range",
            message="assignment value out of carrier range",
        )


def _admit_equation_profile(
    algebra: FiniteAlgebra,
    left: FlatTerm,
    right: FlatTerm,
    variable_count: int,
) -> None:
    for term in (left, right):
        try:
            require_term_for_algebra(term, algebra)
        except UniversalAlgebraAdmissionError as exc:
            _reject(location=("term",), code="term_signature", message=str(exc))
    if max(left.variable_count, right.variable_count) > variable_count:
        _reject(
            location=("variable_count",),
            code="variable_coverage",
            message="variable_count must cover every referenced variable",
        )
    if len(algebra.carrier) ** variable_count > MAX_ENUMERATION_WORK:
        _reject(
            location=("variable_count",),
            code="equation_work_bound",
            message="equation profile exceeds the assignment work budget",
        )


def _admit_subalgebra(algebra: FiniteAlgebra, generators: tuple[int, ...]) -> None:
    size = len(algebra.carrier)
    if any(not 0 <= generator < size for generator in generators):
        _reject(
            location=("generators",),
            code="generator_carrier_range",
            message="generator out of carrier range",
        )
    work = sum(size**symbol.arity for symbol in algebra.operations) * size
    if work > MAX_ENUMERATION_WORK:
        _reject(
            location=("generators",),
            code="subalgebra_work_bound",
            message="subalgebra closure exceeds the operation work budget",
        )


def _admit_homomorphism(carrier_map: FiniteAlgebraCarrierMap) -> None:
    preservation_cells = sum(len(table) for table in carrier_map.source.tables)
    if preservation_cells > MAX_ENUMERATION_WORK:
        _reject(
            location=("carrier_map",),
            code="homomorphism_work_bound",
            message="homomorphism operation work exceeds the enumeration budget",
        )
    try:
        _require_homomorphism_output_headroom(carrier_map)
    except ValueError as exc:
        _reject(
            location=("carrier_map",),
            code="homomorphism_output_bound",
            message=str(exc),
        )


def _admit_partition(
    algebra: FiniteAlgebra,
    partition: tuple[tuple[int, ...], ...],
    *,
    quotient_request: bool,
) -> None:
    congruence_work = _congruence_work(algebra)
    if congruence_work > MAX_ENUMERATION_WORK:
        _reject(
            location=("algebra",),
            code="congruence_work_bound",
            message="congruence check exceeds the operation work budget",
        )
    if not quotient_request:
        return
    quotient_size = len(partition)
    quotient_table_cells = sum(
        quotient_size**operation.arity for operation in algebra.operations
    )
    quotient_work = (
        congruence_work
        + sum(len(table) for table in algebra.tables)
        + quotient_table_cells
    )
    if quotient_work > MAX_ENUMERATION_WORK:
        _reject(
            location=("partition",),
            code="quotient_work_bound",
            message="quotient construction exceeds the operation work budget",
        )
    try:
        source_bytes = len(encode_strict_json(algebra.model_dump(mode="json")))
        operation_bytes = len(
            encode_strict_json(
                [operation.model_dump(mode="json") for operation in algebra.operations]
            )
        )
        quotient_carrier_bytes = sum(
            len(encode_strict_json(f"B{index}")) + 1 for index in range(quotient_size)
        )
    except CanonicalizationError as exc:
        raise OperationDomainValidationError(
            location=("algebra",),
            code="universal_algebra.quotient_output_bound",
            message="quotient source exceeds the canonical output limit",
        ) from exc
    quotient_index_bytes = len(str(quotient_size - 1)) + 1
    predicted_bytes = (
        source_bytes
        + operation_bytes
        + quotient_carrier_bytes
        + quotient_table_cells * quotient_index_bytes
        + len(algebra.carrier) * quotient_index_bytes
        + _HOMOMORPHISM_RESULT_RESERVE_BYTES
    )
    if predicted_bytes > CanonicalLimits().max_output_bytes:
        _reject(
            location=("partition",),
            code="quotient_output_bound",
            message="canonical quotient homomorphism would exceed the output limit",
        )


def _evaluate_node(
    algebra: FiniteAlgebra,
    term: FlatTerm,
    assignment: dict[int, int],
    n: int,
    index: int,
) -> int:
    node = term.nodes[index]
    if isinstance(node, VariableTerm):
        if node.variable_id not in assignment:
            raise ValueError("incomplete assignment")
        return assignment[node.variable_id]
    if isinstance(node, ApplicationTerm):
        args = [_evaluate_node(algebra, term, assignment, n, c) for c in node.children]
        cell_index = 0
        for arg in args:
            cell_index = cell_index * n + arg
        return algebra.tables[node.operation][cell_index]
    raise AssertionError("closed term union admitted an unknown node")


def _evaluate_term_unchecked(
    algebra: FiniteAlgebra, term: FlatTerm, assignment: dict[int, int]
) -> int:
    """Evaluate a term after the caller has completed source-bound admission."""

    return _evaluate_node(algebra, term, assignment, len(algebra.carrier), term.root)


def evaluate_term(
    algebra: FiniteAlgebra, term: FlatTerm, assignment: dict[int, int]
) -> int:
    """Evaluate a source-bound term under a complete variable assignment.

    Return the exact carrier value ``t^A(alpha)``.
    """
    _admit_evaluate(algebra, term, tuple(assignment.values()))
    return _evaluate_term_unchecked(algebra, term, assignment)


def equation_profile(
    algebra: FiniteAlgebra, left: FlatTerm, right: FlatTerm, variable_count: int
) -> EquationProfileResult:
    """Evaluate ``s = t`` over all assignments.

    Return ``HOLDS`` with the satisfying assignment count, or ``FAILS`` with
    the first counterassignment and exact left/right values.
    """
    _admit_equation_profile(algebra, left, right, variable_count)
    return _equation_profile_unchecked(algebra, left, right, variable_count)


def _equation_profile_unchecked(
    algebra: FiniteAlgebra, left: FlatTerm, right: FlatTerm, variable_count: int
) -> EquationProfileResult:
    """Profile an equation after source-bound term admission."""

    n = len(algebra.carrier)
    satisfying = 0
    first_counterassignment: EquationCounterexample | None = None
    for values in iproduct(range(n), repeat=variable_count):
        assignment = dict(enumerate(values))
        lv = _evaluate_term_unchecked(algebra, left, assignment)
        rv = _evaluate_term_unchecked(algebra, right, assignment)
        if lv == rv:
            satisfying += 1
        else:
            if first_counterassignment is None:
                first_counterassignment = EquationCounterexample(
                    assignment=tuple(values),
                    left_value=lv,
                    right_value=rv,
                )
    if satisfying == n**variable_count:
        return EquationProfileResult(status="HOLDS", satisfying_count=satisfying)
    assert first_counterassignment is not None
    return EquationProfileResult(
        status="FAILS",
        satisfying_count=satisfying,
        first_counterassignment=first_counterassignment,
    )


def generated_subalgebra(
    algebra: FiniteAlgebra, generators: tuple[int, ...]
) -> SubalgebraResult:
    """Return the least subalgebra containing the generating set by finite
    closure under all basic operations and nullary constants."""
    _admit_subalgebra(algebra, generators)
    n = len(algebra.carrier)
    carrier_set = set(generators)
    for op_idx, symbol in enumerate(algebra.operations):
        if symbol.arity == 0:
            for output in algebra.tables[op_idx]:
                carrier_set.add(output)
    changed = True
    rounds = 0
    while changed:
        changed = False
        rounds += 1
        for op_idx, symbol in enumerate(algebra.operations):
            if symbol.arity == 0:
                continue
            for args in iproduct(carrier_set, repeat=symbol.arity):
                cell_index = 0
                for arg in args:
                    cell_index = cell_index * n + arg
                output = algebra.tables[op_idx][cell_index]
                if output not in carrier_set:
                    carrier_set.add(output)
                    changed = True
    sorted_carrier = sorted(carrier_set)
    return SubalgebraResult(
        generated_carrier=tuple(sorted_carrier),
        rounds=rounds,
        is_closed=set(generators) == carrier_set if generators else True,
    )


def homomorphism_profile(
    carrier_map: FiniteAlgebraCarrierMap,
) -> HomomorphismProfileResult:
    """Check one total map for preservation of every basic operation.

    Operations are checked in signature order and argument tuples in
    lexicographic carrier-position order.  A failure returns that first exact
    obstruction.  A successful result returns a checked homomorphism together
    with its canonical kernel fibers and image.
    """

    _admit_homomorphism(carrier_map)
    failure = _first_homomorphism_failure(carrier_map)
    if failure is not None:
        return HomomorphismProfileResult(
            status="NOT_A_HOMOMORPHISM",
            carrier_map=carrier_map,
            obstruction=HomomorphismObstruction(
                operation=failure.operation,
                operation_id=carrier_map.source.operations[
                    failure.operation
                ].operation_id,
                source_arguments=failure.source_arguments,
                target_arguments=failure.target_arguments,
                source_output=failure.source_output,
                mapped_source_output=failure.mapped_source_output,
                target_output=failure.target_output,
            ),
        )

    homomorphism = FiniteAlgebraHomomorphism(
        source=carrier_map.source,
        target=carrier_map.target,
        mapping=carrier_map.mapping,
    )
    kernel_partition, image = _homomorphism_kernel_and_image(carrier_map.mapping)
    injective = len(image) == len(carrier_map.source.carrier)
    surjective = len(image) == len(carrier_map.target.carrier)
    return HomomorphismProfileResult(
        status="HOMOMORPHISM",
        homomorphism=homomorphism,
        kernel_partition=kernel_partition,
        image=image,
        injective=injective,
        surjective=surjective,
        isomorphism=injective and surjective,
    )


def _compatibility_violation(
    algebra: FiniteAlgebra,
    block_of: dict[int, int],
    n: int,
    op_idx: int,
    symbol: OperationSymbol,
    x: tuple[int, ...],
    y: tuple[int, ...],
) -> CongruenceResult | None:
    if not all(block_of[x[j]] == block_of[y[j]] for j in range(symbol.arity)):
        return None
    cell_x = 0
    cell_y = 0
    for j in range(symbol.arity):
        cell_x = cell_x * n + x[j]
        cell_y = cell_y * n + y[j]
    fx = algebra.tables[op_idx][cell_x]
    fy = algebra.tables[op_idx][cell_y]
    if block_of[fx] == block_of[fy]:
        return None
    return CongruenceResult(
        is_congruence=False,
        obstruction="compatibility_violation",
        operation=op_idx,
        x=x,
        y=y,
    )


def _check_compatibility(
    algebra: FiniteAlgebra,
    block_of: dict[int, int],
    n: int,
) -> CongruenceResult | None:
    """Check congruence compatibility, returning an obstruction or None."""
    for op_idx, symbol in enumerate(algebra.operations):
        if symbol.arity == 0:
            continue
        for x in iproduct(range(n), repeat=symbol.arity):
            for j in range(symbol.arity):
                for y_elem in range(n):
                    if y_elem == x[j]:
                        continue
                    if block_of[x[j]] != block_of[y_elem]:
                        continue
                    y_list = list(x)
                    y_list[j] = y_elem
                    y: tuple[int, ...] = tuple(y_list)
                    violation = _compatibility_violation(
                        algebra, block_of, n, op_idx, symbol, x, y
                    )
                    if violation is not None:
                        return violation
    return None


def congruence_check(
    algebra: FiniteAlgebra, partition: tuple[tuple[int, ...], ...]
) -> CongruenceResult:
    """Check whether a carrier partition is a compatible equivalence
    relation (congruence).

    A congruence theta satisfies: if x_j theta y_j for every argument j, then
    f(x_1,...,x_r) theta f(y_1,...,y_r) for every basic operation.
    """
    _admit_partition(algebra, partition, quotient_request=False)
    return _congruence_check_unchecked(algebra, partition)


def _congruence_check_unchecked(
    algebra: FiniteAlgebra, partition: tuple[tuple[int, ...], ...]
) -> CongruenceResult:
    n = len(algebra.carrier)
    block_of: dict[int, int] = {}
    for block_idx, block in enumerate(partition):
        for elem in block:
            block_of[elem] = block_idx
    if len(block_of) != n:
        return CongruenceResult(
            is_congruence=False,
            obstruction="partition does not cover carrier",
        )
    result = _check_compatibility(algebra, block_of, n)
    return result if result is not None else CongruenceResult(is_congruence=True)


def quotient(
    algebra: FiniteAlgebra, partition: tuple[tuple[int, ...], ...]
) -> FiniteAlgebraHomomorphism:
    """Return the canonical quotient homomorphism ``A -> A/theta``."""
    _admit_partition(algebra, partition, quotient_request=True)
    check = _congruence_check_unchecked(algebra, partition)
    if not check.is_congruence:
        raise ValueError("partition is not a congruence")
    n = len(algebra.carrier)
    block_of: dict[int, int] = {}
    for block_idx, block in enumerate(partition):
        for elem in block:
            block_of[elem] = block_idx
    quotient_carrier = tuple(f"B{i}" for i in range(len(partition)))
    quotient_tables: list[tuple[int, ...]] = []
    for op_idx, symbol in enumerate(algebra.operations):
        if symbol.arity == 0:
            original_output = algebra.tables[op_idx][0]
            quotient_tables.append((block_of[original_output],))
        else:
            block_count = len(partition)
            # Use a representative for each block (the minimum element)
            representatives = [min(block) for block in partition]
            table = []
            for args in iproduct(range(block_count), repeat=symbol.arity):
                # Compute the operation on the representatives
                cell_index = 0
                for arg in args:
                    cell_index = cell_index * n + representatives[arg]
                output = algebra.tables[op_idx][cell_index]
                table.append(block_of[output])
            quotient_tables.append(tuple(table))
    quotient_algebra = FiniteAlgebra(
        carrier=quotient_carrier,
        operations=algebra.operations,
        tables=tuple(quotient_tables),
    )
    return FiniteAlgebraHomomorphism(
        source=algebra,
        target=quotient_algebra,
        mapping=tuple(block_of[element] for element in range(n)),
    )
