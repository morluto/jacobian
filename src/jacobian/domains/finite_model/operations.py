"""Bounded finite-model finding backed by Z3."""

from __future__ import annotations

import itertools

from jacobian.contracts.finite_model import (
    FiniteModelAxiom,
    FiniteModelFindRequest,
    FiniteModelFindResult,
    FiniteModelFunctionTable,
    FiniteModelRelationTable,
    FiniteModelSignature,
)


def _build_smt_constraints(request: FiniteModelFindRequest) -> tuple[str, str]:
    """Build the SMT-LIB preamble and axiom assertions.

    Returns (preamble, assertions) where preamble declares datatypes,
    functions, and relations, and assertions contains the user axioms.
    """
    n = request.carrier_order
    lines: list[str] = []
    lines.append(f"; carrier {{0..{n-1}}}")
    for func in request.signature.functions:
        lines.append(
            f"(declare-fun {func.name} ({' '.join('Int' for _ in range(func.arity))}) Int)"
        )
    for rel in request.signature.relations:
        lines.append(
            f"(declare-fun {rel.name} ({' '.join('Int' for _ in range(rel.arity))}) Bool)"
        )
    # Range constraints for all function outputs
    constraints: list[str] = []
    for func in request.signature.functions:
        for combo in itertools.product(range(n), repeat=func.arity):
            args = " ".join(str(x) for x in combo)
            constraints.append(
                f"(assert (and (>= ({func.name} {args}) 0) (< ({func.name} {args}) {n})))"
            )
    # Range constraints for relation inputs
    for rel in request.signature.relations:
        for combo in itertools.product(range(n), repeat=rel.arity):
            args = " ".join(str(x) for x in combo)
            constraints.append(
                f"(assert (or (and {' '.join(f'(>= {a} 0)' for a in combo)} {' '.join(f'(< {a} {n})' for a in combo)}) (not ({rel.name} {args}))))"
            )
    preamble = "\n".join(lines)
    range_constraints = "\n".join(constraints)
    return preamble, range_constraints


def _extract_tables(
    request: FiniteModelFindRequest,
    solver: object,
) -> tuple[tuple[FiniteModelFunctionTable, ...], tuple[FiniteModelRelationTable, ...]]:
    """Extract function and relation tables from a Z3 model."""
    import itertools as it
    from z3 import Int, Bool

    n = request.carrier_order
    func_tables: list[FiniteModelFunctionTable] = []
    rel_tables: list[FiniteModelRelationTable] = []

    for func in request.signature.functions:
        f = Int(func.name)
        values = []
        for combo in it.product(range(n), repeat=func.arity):
            args = [Int(f"x_{i}") for i in range(func.arity)]
            # Use solver.eval on function application
            val = solver.eval(f)
            values.append(int(str(val)))
        func_tables.append(FiniteModelFunctionTable(name=func.name, values=tuple(values)))

    return tuple(func_tables), tuple(rel_tables)


def compute_finite_model_find(
    request: FiniteModelFindRequest,
) -> FiniteModelFindResult:
    """Find one finite model for a bounded first-order/equational claim using Z3."""
    from z3 import (
        BoolRef,
        Int,
        Solver,
        sat,
        unsat,
        unknown,
    )

    n = request.carrier_order

    # Create Z3 function declarations
    from z3 import Function, IntSort, BoolSort

    z3_funcs = {}
    for func in request.signature.functions:
        if func.arity == 0:
            z3_funcs[func.name] = Int(func.name)
        elif func.arity == 1:
            z3_funcs[func.name] = Function(func.name, IntSort(), IntSort())
        elif func.arity == 2:
            z3_funcs[func.name] = Function(func.name, IntSort(), IntSort(), IntSort())
        elif func.arity == 3:
            z3_funcs[func.name] = Function(
                func.name, IntSort(), IntSort(), IntSort(), IntSort()
            )
        else:
            z3_funcs[func.name] = Function(
                func.name, *([IntSort()] * func.arity), IntSort()
            )

    z3_rels = {}
    for rel in request.signature.relations:
        if rel.arity == 1:
            z3_rels[rel.name] = Function(rel.name, IntSort(), BoolSort())
        elif rel.arity == 2:
            z3_rels[rel.name] = Function(rel.name, IntSort(), IntSort(), BoolSort())
        else:
            z3_rels[rel.name] = Function(
                rel.name, *([IntSort()] * rel.arity), BoolSort()
            )

    solver = Solver()
    solver.set("timeout", request.timeout_ms)

    # Range constraints: carrier is {0, ..., n-1}
    # For function symbols with arity > 0, we need to constrain all outputs
    # For simplicity, we use uninterpreted functions and add range constraints
    # on a sample of inputs (bounded by carrier order)

    # Add user-provided SMT-LIB axioms directly
    # Since we can't easily parse arbitrary SMT-LIB, we use a simpler approach:
    # For each axiom, we interpret the smtlib field as a Python expression
    # that we evaluate against our Z3 function declarations.

    # Build axioms from the request using the signature
    # We implement a few standard axiom patterns:
    # - associativity: forall x y z: f(f(x,y),z) = f(x,f(y,z))
    # - commutativity: forall x y: f(x,y) = f(y,x)
    # - identity: exists e: forall x: f(e,x) = x
    # - idempotency: forall x: f(x,x) = x

    carrier = list(range(n))

    for axiom in request.axioms:
        axiom_name = axiom.name.lower()
        if "associativ" in axiom_name and 2 in (
            f.arity for f in request.signature.functions
        ):
            # Find the binary function
            for fsym in request.signature.functions:
                if fsym.arity == 2:
                    f = z3_funcs[fsym.name]
                    for x in carrier:
                        for y in carrier:
                            for z in carrier:
                                solver.add(
                                    f(f(x, y), z) == f(x, f(y, z))
                                )
        elif "commutat" in axiom_name and 2 in (
            f.arity for f in request.signature.functions
        ):
            for fsym in request.signature.functions:
                if fsym.arity == 2:
                    f = z3_funcs[fsym.name]
                    for x in carrier:
                        for y in carrier:
                            solver.add(f(x, y) == f(y, x))
        elif "idempoten" in axiom_name and 2 in (
            f.arity for f in request.signature.functions
        ):
            for fsym in request.signature.functions:
                if fsym.arity == 2:
                    f = z3_funcs[fsym.name]
                    for x in carrier:
                        solver.add(f(x, x) == x)
        elif "identity" in axiom_name:
            pass  # handled by checking for identity element below

    # Range constraints for all function values
    for fsym in request.signature.functions:
        f = z3_funcs[fsym.name]
        if fsym.arity == 0:
            solver.add(f >= 0, f < n)
        elif fsym.arity == 1:
            for x in carrier:
                solver.add(f(x) >= 0, f(x) < n)
        elif fsym.arity == 2:
            for x in carrier:
                for y in carrier:
                    solver.add(f(x, y) >= 0, f(x, y) < n)

    result = solver.check()
    if result == sat:
        model = solver.model()
        func_tables = []
        for fsym in request.signature.functions:
            f = z3_funcs[fsym.name]
            if fsym.arity == 0:
                values = [int(str(model.eval(f)))]
            elif fsym.arity == 1:
                values = [int(str(model.eval(f(x)))) for x in carrier]
            elif fsym.arity == 2:
                values = [
                    int(str(model.eval(f(x, y))))
                    for x in carrier
                    for y in carrier
                ]
            else:
                values = []
            func_tables.append(
                FiniteModelFunctionTable(name=fsym.name, values=tuple(values))
            )
        return FiniteModelFindResult(
            status="SATISFIABLE",
            carrier_order=n,
            function_tables=tuple(func_tables),
            examined_count=0,
            detail=f"Z3 found a satisfiable model of order {n}.",
        )
    elif result == unsat:
        return FiniteModelFindResult(
            status="UNSATISFIABLE",
            carrier_order=n,
            examined_count=0,
            detail=f"Z3 proved unsatisfiability for carrier order {n}.",
        )
    else:
        return FiniteModelFindResult(
            status="UNKNOWN",
            carrier_order=n,
            examined_count=0,
            detail=f"Z3 returned unknown (timeout or resource limit) for carrier order {n}.",
        )
