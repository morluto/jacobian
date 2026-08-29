"""Private operation contracts for certified Smith normal form."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import WithJsonSchema
from pydantic.json_schema import JsonSchemaValue

from jacobian._models import StrictModel
from jacobian.math.matrices.certified_snf.values import (
    MAX_CERTIFIED_SNF_INPUT_DIMENSION,
    CertifiedIntegerMatrix,
    SmithNormalFormCertificate,
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


class CertifiedSmithNormalFormResult(StrictModel):
    certificate: SmithNormalFormCertificate

    @classmethod
    def _from_kernel(
        cls,
        *,
        certificate: SmithNormalFormCertificate,
    ) -> Self:
        """Construct the result emitted by the trusted Smith kernel."""

        return cls.model_construct(certificate=certificate)


__all__ = [
    "CertifiedSmithNormalFormRequest",
    "CertifiedSmithNormalFormResult",
]
