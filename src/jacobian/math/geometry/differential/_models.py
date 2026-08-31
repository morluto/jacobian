"""Public contracts for rational coordinate-tensor Lie derivatives."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.geometry.differential.values import (
    RationalCoordinateTensor,
    canonical_locus_guards,
)


def _binding_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(
        f"differential_geometry.lie_derivative.{reason}", message
    )


class RationalLieDerivativeRequest(StrictModel):
    """One rational vector field acting on one tensor over the same chart."""

    vector_field: RationalCoordinateTensor
    tensor: RationalCoordinateTensor


class RationalLieDerivativeProfile(StrictModel):
    """The exact source-bound Lie derivative on the retained intersection."""

    vector_field: RationalCoordinateTensor
    source: RationalCoordinateTensor
    lie_derivative: RationalCoordinateTensor

    @model_validator(mode="after")
    def require_structural_source_binding(self) -> Self:
        if self.vector_field.variance != ("CONTRAVARIANT",):
            raise _binding_error(
                "vector_signature",
                "Lie-derivative vector field must be rank-one contravariant",
            )
        if self.vector_field.coordinate_axis != self.source.coordinate_axis:
            raise _binding_error(
                "source_axis", "Lie-derivative sources must share one coordinate axis"
            )
        if self.lie_derivative.coordinate_axis != self.source.coordinate_axis:
            raise _binding_error(
                "result_axis", "Lie derivative must retain the source coordinate axis"
            )
        if self.lie_derivative.variance != self.source.variance:
            raise _binding_error(
                "result_variance",
                "Lie derivative must retain the source variance signature",
            )
        expected_locus = canonical_locus_guards(
            self.vector_field.retained_nonzero_denominators,
            self.source.retained_nonzero_denominators,
            component_denominators=tuple(
                component.denominator for component in self.lie_derivative.components
            ),
            variable_count=len(self.source.coordinate_axis),
        )
        if self.lie_derivative.retained_nonzero_denominators != expected_locus:
            raise _binding_error(
                "result_locus",
                "Lie derivative must retain exactly the source and result "
                "nonvanishing-locus guards",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        vector_field: RationalCoordinateTensor,
        source: RationalCoordinateTensor,
        lie_derivative: RationalCoordinateTensor,
    ) -> Self:
        """Construct a profile after owner-local exact field arithmetic."""

        return cls.model_construct(
            vector_field=vector_field,
            source=source,
            lie_derivative=lie_derivative,
        )


__all__ = ["RationalLieDerivativeProfile", "RationalLieDerivativeRequest"]
