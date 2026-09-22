"""One-shot PARI bnf worker isolated from the MCP request process."""

from __future__ import annotations

import hashlib
import sys
from fractions import Fraction
from typing import Any

from jacobian._worker_protocol import encode_worker_result_frame
from jacobian.canonical import (
    encode_strict_json,
    format_canonical_integer,
    loads_strict_json,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.number_fields._bnf_process import (
    worker_rejection,
)
from jacobian.math.number_theory.number_fields._models import (
    _BnfWorkerRequest,
)

_POLYNOMIAL_VARIABLE = "jacobian_poly"
_BNF_VARIABLE = "jacobian_bnf"


def _as_int(value: Any) -> int:
    return int(value)


def main() -> int:
    input_bytes = sys.stdin.buffer.read()
    payload = loads_strict_json(input_bytes)
    if not isinstance(payload, dict):
        raise RuntimeError("bnf worker request must be an object")
    if "field" not in payload:
        raise RuntimeError("bnf worker request must carry a field")
    request = _BnfWorkerRequest.model_validate_json(
        encode_strict_json(payload), strict=True
    )
    digest = hashlib.sha256(input_bytes).hexdigest()
    field = request.field
    try:
        _require_pari_available()
        import cypari

        pari = cypari.pari
        # Build the PARI polynomial from validated integer coefficients. PARI's
        # own printer produces the canonical GP text of that integer polynomial;
        # no caller-supplied expression text ever reaches the evaluator.
        polynomial = pari.Polrev(list(reversed(field.coefficients_descending)))
        pari(f"{_POLYNOMIAL_VARIABLE} = {polynomial!s}")
        if not bool(pari(f"polisirreducible({_POLYNOMIAL_VARIABLE})")):
            raise OperationDomainValidationError(
                location=("field",),
                code="number_field.field_polynomial_must_be_irreducible",
                message="class and unit groups require an irreducible field polynomial",
            )
        degree = field.degree
        reduction = pari(f"polredbest({_POLYNOMIAL_VARIABLE}, 1)")
        original_root_image = reduction[1]
        presentation_inverse = _presentation_inverse(pari, original_root_image, degree)
        pari(f"{_BNF_VARIABLE} = bnfinit({_POLYNOMIAL_VARIABLE})")
        cyc = [_as_int(value) for value in pari(f"{_BNF_VARIABLE}.cyc")]
        signature = pari(f"{_BNF_VARIABLE}.sign")
        real_count = _as_int(signature[0])
        complex_count = _as_int(signature[1])
        generators = pari(f"{_BNF_VARIABLE}.gen")
        representatives: list[list[list[int]]] = []
        for index in range(len(generators)):
            hnf = pari(f"idealhnf({_BNF_VARIABLE}, {_BNF_VARIABLE}.gen[{index + 1}])")
            # The HNF columns are the ideal basis vectors in the caller power basis.
            representatives.append(
                [
                    [_as_int(hnf[row, column]) for row in range(degree)]
                    for column in range(degree)
                ]
            )
        fundamental_units = [
            _lift_coefficients(pari, unit, degree, presentation_inverse)
            for unit in pari(f"{_BNF_VARIABLE}.fu")
        ]
        torsion = pari(f"{_BNF_VARIABLE}.tu")
        torsion_order = _as_int(torsion[0])
        torsion_generator = (
            _lift_coefficients(pari, torsion[1], degree, presentation_inverse)
            if len(torsion) > 1
            else [Fraction(0)] * degree
        )
        response: dict[str, object] = {
            "kind": "complete",
            "class_number": format_canonical_integer(
                _as_int(pari(f"{_BNF_VARIABLE}.no"))
            ),
            "abelian_invariants": [format_canonical_integer(value) for value in cyc],
            "field_discriminant": format_canonical_integer(
                _as_int(pari(f"{_BNF_VARIABLE}.disc"))
            ),
            "real_embedding_count": real_count,
            "complex_embedding_pair_count": complex_count,
            "ideal_representatives": representatives,
            "rank": len(fundamental_units),
            "torsion_order": torsion_order,
            "torsion_generator": [
                {
                    "num": format_canonical_integer(value.numerator),
                    "den": format_canonical_integer(value.denominator),
                }
                for value in torsion_generator
            ],
            "fundamental_units": [
                [
                    {
                        "num": format_canonical_integer(value.numerator),
                        "den": format_canonical_integer(value.denominator),
                    }
                    for value in unit
                ]
                for unit in fundamental_units
            ],
            "request_digest": digest,
        }
    except (
        OperationDomainValidationError,
        OperationResourceAdmissionError,
    ) as exc:
        response = worker_rejection(exc, request_digest=digest)
    sys.stdout.buffer.write(encode_worker_result_frame(response))
    return 0


def _require_pari_available() -> None:
    import importlib.util

    if importlib.util.find_spec("cypari") is None:
        raise OperationResourceAdmissionError(
            location=("field",),
            code="number_field.pari_backend_unavailable",
            message="class and unit group computation requires the PARI backend",
        )


def _pari_fraction(value: Any) -> Fraction:
    return Fraction(int(value.numerator()), int(value.denominator()))


def _invert_matrix(
    matrix: tuple[tuple[Fraction, ...], ...],
) -> tuple[tuple[Fraction, ...], ...]:
    size = len(matrix)
    augmented = [
        list(row) + [Fraction(int(row_index == column)) for column in range(size)]
        for row_index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = next(
            (row for row in range(column, size) if augmented[row][column]), None
        )
        if pivot is None:
            raise RuntimeError("PARI presentation isomorphism is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column or not augmented[row][column]:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                left - factor * right
                for left, right in zip(augmented[row], augmented[column], strict=True)
            ]
    return tuple(tuple(row[size:]) for row in augmented)


def _presentation_inverse(
    pari: Any, root_image: Any, degree: int
) -> tuple[tuple[Fraction, ...], ...]:
    columns: list[tuple[Fraction, ...]] = []
    power = pari(1)
    for _ in range(degree):
        lifted = pari.lift(power)
        columns.append(
            tuple(
                _pari_fraction(pari.polcoef(lifted, index)) for index in range(degree)
            )
        )
        power *= root_image
    matrix = tuple(
        tuple(columns[column][row] for column in range(degree)) for row in range(degree)
    )
    return _invert_matrix(matrix)


def _change_coordinates(
    coordinates: tuple[Fraction, ...],
    inverse: tuple[tuple[Fraction, ...], ...],
) -> tuple[Fraction, ...]:
    return tuple(
        sum(
            (
                inverse[row][column] * coordinates[column]
                for column in range(len(coordinates))
            ),
            Fraction(0),
        )
        for row in range(len(inverse))
    )


def _lift_coefficients(
    pari: Any,
    element: Any,
    degree: int,
    inverse: tuple[tuple[Fraction, ...], ...],
) -> list[Fraction]:
    lifted = pari.lift(element)
    reduced_coordinates = tuple(
        _pari_fraction(pari.polcoef(lifted, index)) for index in range(degree)
    )
    return list(_change_coordinates(reduced_coordinates, inverse))


if __name__ == "__main__":
    raise SystemExit(main())
