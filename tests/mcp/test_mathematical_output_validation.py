"""SDK output validation checks data without replaying mathematical proofs."""

from __future__ import annotations

import cProfile
from types import CodeType

import pytest
from mcp.types import CallToolResult

from jacobian.catalog.catalog import Catalog
from jacobian.mcp.direct_tools import direct_operation_tools


@pytest.mark.parametrize(
    ("operation_id", "payload"),
    [
        (
            "algebraic_number.compare",
            {
                "left": {"polynomial": ["1", "0", "-2"], "real_root_index": 1},
                "right": {"polynomial": ["2", "-3"], "real_root_index": 0},
            },
        ),
        ("integer.primality.certificate.compute", {"value": "101"}),
        ("algebraic_number.add.compute", None),
        ("finite_category.opposite.compute", None),
        ("semigroup.green_relations.compute", None),
        ("cluster_algebra.seed.mutate.compute", None),
        ("code.linear.dual.compute", None),
        ("lie_algebra.chevalley_eilenberg.complex.compute", None),
        ("crossed_product.multiply.compute", None),
        ("polynomial.roots.isolate", None),
        (
            "finite_field.projective_line.enumerate",
            {
                "presentation": {
                    "characteristic": 2,
                    "modulus_coefficients": [1, 1, 0, 1],
                },
                "axis": {"name": "coordinates", "labels": ["x", "y"]},
            },
        ),
    ],
)
def test_direct_sdk_result_validation_does_not_replay_proofs(
    operation_id: str, payload: dict[str, object] | None
) -> None:
    operation = Catalog.open().operation(operation_id)
    assert operation is not None
    request = operation.request_type.model_validate(
        operation.examples[0].input if payload is None else payload
    )
    result = operation.run(request)
    wire = CallToolResult(content=[], structured_content=result.model_dump(mode="json"))
    tool = direct_operation_tools(Catalog((operation,)))[0]
    profiler = cProfile.Profile()
    projected = profiler.runcall(tool.fn_metadata.convert_result, wire)
    assert projected == wire
    assert operation.result_type.model_validate(projected.structured_content) == result
    calls = [
        entry.code
        for entry in profiler.getstats()
        if (
            isinstance(entry.code, CodeType)
            and (
                "/sympy/" in entry.code.co_filename.replace("\\", "/")
                or entry.code.co_name
                in {
                    "require_field",
                    "require_independent_basis",
                    "is_square_free",
                    "face_closure",
                    "_check_associativity",
                    "_check_category_laws",
                    "_require_jacobi",
                    "require_skew_symmetrizable",
                }
            )
        )
        or (
            operation_id == "integer.primality.certificate.compute"
            and isinstance(entry.code, str)
            and "builtins.pow" in entry.code
        )
    ]
    assert calls == []
