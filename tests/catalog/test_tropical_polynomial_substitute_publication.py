"""Publication check for the tropical polynomial substitution example.

The published example carries rationals in the wire encoding, so decoding it
belongs in the catalog lane rather than under ``tests/math``.
"""

from jacobian.dispatch import parse_operation_input
from jacobian.math.polynomials.tropical._tools import TOOLS


def test_substitution_is_published_with_an_executable_example() -> None:
    operation = next(
        item
        for item in TOOLS
        if item.operation_id == "tropical.polynomial.substitute.compute"
    )
    assert len(operation.examples) == 1
    request = parse_operation_input(operation.request_type, operation.examples[0].input)
    result = operation.run(request)
    assert result.result.variables == ("t",)
    assert tuple(term.exponents for term in result.result.terms) == ((0,), (1,))
