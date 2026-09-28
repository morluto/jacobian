"""Publication check for the tropical polynomial power example.

This opens the catalog, so it lives in the catalog lane rather than under
``tests/math``.
"""

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials.tropical._tools import TOOLS


def test_catalogued_power_example_executes() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "tropical.polynomial.power.compute"
    )
    result = invoke_operation(tool.operation_id, tool.examples[0].input, Catalog.open())
    terms = result.output["result"]["terms"]
    assert tuple(tuple(term["exponents"]) for term in terms) == (
        (0,),
        (1,),
        (2,),
        (3,),
    )
    assert tuple(term["coefficient"]["value"] for term in terms) == tuple(
        {"num": "0", "den": "1"} for _ in range(4)
    )
