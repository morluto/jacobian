"""Koszul complex operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.koszul._models import KoszulComplexRequest
from jacobian.math.koszul.operations import koszul_complex
from jacobian.math.koszul.values import (
    MAX_KOSZUL_DEGREE,
    MAX_KOSZUL_SEQUENCE_LENGTH,
    MAX_KOSZUL_TERMS,
    MAX_KOSZUL_VARIABLES,
    KoszulComplexValue,
)


def _koszul_complex(request: KoszulComplexRequest) -> KoszulComplexValue:
    """Project a wire request into the canonical Koszul construction."""

    return koszul_complex(request.variables, request.sequence)


def _monomial(variables: list[str], exponents: list[int]) -> dict[str, Any]:
    return {
        "domain": "QQ",
        "variables": variables,
        "polynomial": {
            "terms": [
                {
                    "coefficient": {"num": "1", "den": "1"},
                    "exponents": exponents,
                }
            ]
        },
    }


_XY_RING = ["x", "y"]

_KOSZUL_XY_EXAMPLE = {
    "variables": _XY_RING,
    "sequence": [
        _monomial(_XY_RING, [1, 0]),
        _monomial(_XY_RING, [0, 1]),
    ],
}


TOOLS: MathTools = (
    MathTool(
        operation_id="koszul.complex.construct.compute",
        title="Construct the exact Koszul complex of a polynomial sequence",
        description=(
            "Construct the complete Koszul complex K(f) = R (x) Lambda(R^c) of "
            "one ordered sequence f = (f_1, ..., f_c) in R = QQ[x_1, ..., x_m] "
            "with the module fixed to the ring itself: per-degree wedge bases "
            "e_{i_1} ^ ... ^ e_{i_k} in canonical increasing order (degree k "
            "has C(c, k) basis elements), the differentials d_k(e_I) = "
            "sum_j (-1)^position f_j e_{I minus j} as exact sparse polynomial "
            "matrices bound to the ordered bases, the exactly replayed "
            "d^2 = 0 identity, and — when the ambient ring is QQ and the wedge "
            "ranks fit the shared envelope — the exact conversion to the based "
            "chain-complex value so downstream homology operations compose "
            "unchanged. Supply 'variables' as the ordered ring axis and "
            f"'sequence' as at most {MAX_KOSZUL_SEQUENCE_LENGTH} canonical "
            f"sparse QQ polynomials on that axis (at most "
            f"{MAX_KOSZUL_VARIABLES} variables, {MAX_KOSZUL_TERMS} terms, and "
            f"degree {MAX_KOSZUL_DEGREE} per variable); the empty sequence "
            "returns the identity complex R in degree 0. Based modules, "
            "DG-algebra structure, homology profiles, and sequence transforms "
            "are deferred."
        ),
        request_type=KoszulComplexRequest,
        result_type=KoszulComplexValue,
        run=_koszul_complex,
        tags=(
            "koszul-complex",
            "commutative-algebra",
            "homological-algebra",
            "chain-complex",
            "exact",
        ),
        discovery_terms=(
            "Koszul complex",
            "Koszul differential",
            "exterior algebra differential",
            "wedge basis chain complex",
            "regular sequence complex",
            "sequence-derived differential",
        ),
        examples=(
            OperationExample(
                name="koszul_complex_of_x_y",
                description=(
                    "Construct K(x, y) over QQ[x, y]: ranks (1, 2, 1) with "
                    "d_1 = [x, y] and d_2(e_12) = x e_2 - y e_1; supply the "
                    "ordered variable axis and the sequence in canonical "
                    "sparse form."
                ),
                input=_KOSZUL_XY_EXAMPLE,
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
