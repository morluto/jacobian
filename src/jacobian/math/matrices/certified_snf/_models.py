"""Private operation contracts for certified Smith normal form."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import WithJsonSchema, model_validator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.matrices.certified_snf.values import (
    MAX_CERTIFIED_SNF_INPUT_DIGITS,
    MAX_CERTIFIED_SNF_INPUT_DIMENSION,
    CertifiedIntegerMatrix,
    SmithNormalFormCertificate,
    _integer_digits,
)


def _certified_smith_input_schema() -> JsonSchemaValue:
    """Project the producer's request bounds without creating another value type."""

    schema = CertifiedIntegerMatrix.model_json_schema()
    for field_name in ("row_count", "column_count"):
        schema["properties"][field_name].update(
            minimum=1,
            maximum=MAX_CERTIFIED_SNF_INPUT_DIMENSION,
        )
    return schema


class CertifiedSmithNormalFormRequest(StrictModel):
    matrix: Annotated[
        CertifiedIntegerMatrix,
        WithJsonSchema(_certified_smith_input_schema()),
    ]

    @model_validator(mode="after")
    def require_nonempty_bounded_input(self) -> Self:
        if (
            not 1 <= self.matrix.row_count <= MAX_CERTIFIED_SNF_INPUT_DIMENSION
            or not 1 <= self.matrix.column_count <= MAX_CERTIFIED_SNF_INPUT_DIMENSION
        ):
            raise _validation_error(
                "budget_exceeded",
                "certified Smith input must be a nonempty matrix of at most "
                f"{MAX_CERTIFIED_SNF_INPUT_DIMENSION} by "
                f"{MAX_CERTIFIED_SNF_INPUT_DIMENSION}",
            )
        if any(
            _integer_digits(value) > MAX_CERTIFIED_SNF_INPUT_DIGITS
            for row in self.matrix.entries
            for value in row
        ):
            raise _validation_error(
                "budget_exceeded",
                "certified Smith input entries may contain at most "
                f"{MAX_CERTIFIED_SNF_INPUT_DIGITS} decimal digits",
            )
        return self


class CertifiedSmithNormalFormResult(StrictModel):
    certificate: SmithNormalFormCertificate
    exactness: Literal["EXACT_INTEGER"] = "EXACT_INTEGER"
    determinism: Literal["DETERMINISTIC"] = "DETERMINISTIC"
    completeness: Literal["FULL_MATRIX_TRANSFORMATIONS"] = "FULL_MATRIX_TRANSFORMATIONS"


__all__ = [
    "CertifiedSmithNormalFormRequest",
    "CertifiedSmithNormalFormResult",
]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"matrix.{reason}", message)
