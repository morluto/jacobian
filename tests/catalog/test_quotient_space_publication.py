"""Catalog membership check for the finite-field quotient-space operations."""

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.matrices import finite_fields
from jacobian.math.matrices.finite_fields.operations import (
    project_quotient_vector,
)
from jacobian.math.matrices.finite_fields.quotient_spaces import (
    PrimeFieldQuotientSpace,
)


def test_quotient_construction_is_the_only_published_quotient_operation() -> None:
    tools = {tool.operation_id: tool for tool in BUILTIN_TOOLS}
    assert (
        tools["prime_field.vector_space.quotient.compute"].result_type
        is PrimeFieldQuotientSpace
    )
    # Projection is a cheap deterministic map of an existing public result, so
    # it stays a native package export instead of a discovery entry.
    assert "prime_field.vector_space.quotient.project" not in tools
    assert finite_fields.project_quotient_vector is project_quotient_vector
