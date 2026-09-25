"""Typed requests for cross-space modular-form equality."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.math.matrices.cyclic_linear._models import (
    RationalCyclotomicElement,
    RationalCyclotomicField,
)
from jacobian.math.number_theory.modular_forms.values import ModularFormCoordinates


class CyclotomicFieldEmbedding(StrictModel):
    """An explicit field embedding, specified on the source cyclotomic generator."""

    source_order: Literal[6] = Field(
        description=(
            "Cyclotomic order of the exact source form's coefficient field; this "
            "operation currently admits order 6."
        ),
    )
    target_field: RationalCyclotomicField = Field(
        description=(
            "Common exact cyclotomic target field. Both source embeddings must "
            "declare the identical target field."
        )
    )
    generator_image: RationalCyclotomicElement = Field(
        description=(
            "Image of the source field generator in target_field. It must have "
            "exact order source_order so the induced field map is injective."
        )
    )


class ModularFormGlobalEqualityRequest(StrictModel):
    """Compare same-weight exact character forms in one common coefficient field.

    Source forms currently use Q(zeta_6) coordinates in the admitted level
    13/26/39 character families. Both explicit embeddings must target the same
    Q(zeta_6) or Q(zeta_12) field, and they must induce distinct characters at
    the common level. Same-character transport/equality is owned separately.
    """

    left: ModularFormCoordinates
    left_embedding: CyclotomicFieldEmbedding
    right: ModularFormCoordinates
    right_embedding: CyclotomicFieldEmbedding


class ModularFormGlobalEqualityResult(StrictModel):
    equal: bool
