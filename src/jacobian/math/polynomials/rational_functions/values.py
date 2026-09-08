"""Canonical coordinate maps over the rational-function field."""

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_VARIABLES,
    PolynomialVariable,
    RationalFunction,
)

MAX_RATIONAL_MAP_COMPONENTS = 4096
MAX_RATIONAL_MAP_SOURCE_TERMS = 65_536


class RationalFunctionMap(StrictModel):
    """A rational coordinate map on its common denominator-nonzero locus.

    Components correspond to the ordered target coordinates, and every
    component uses the complete ordered source axis. The common regular
    locus requires every supplied component denominator to be nonzero.
    This states no real-domain, injectivity or inverse-map claim. A future
    composition may retain a stricter construction locus separately from
    its normalized composite map.

    Parsing checks axes and structural field presentations. Consumers own
    admitted exact coprimality recognition of authored components.
    """

    source_variables: tuple[PolynomialVariable, ...] = Field(
        max_length=MAX_POLYNOMIAL_VARIABLES
    )
    target_coordinates: tuple[PolynomialVariable, ...] = Field(
        max_length=MAX_RATIONAL_MAP_COMPONENTS
    )
    components: tuple[RationalFunction, ...] = Field(
        max_length=MAX_RATIONAL_MAP_COMPONENTS
    )
    domain: Literal["COMMON_REGULAR_LOCUS"] = "COMMON_REGULAR_LOCUS"

    @model_validator(mode="before")
    @classmethod
    def require_aggregate_component_terms(cls, data: object) -> object:
        """Cap nested polynomial terms before constructing each component."""

        data = canonicalize_json_containers(data)
        if not isinstance(data, dict):
            return data
        components = data.get("components")
        if not isinstance(components, (list, tuple)):
            return data
        total = 0
        for component in components:
            if isinstance(component, RationalFunction):
                total += len(component.numerator.terms) + len(
                    component.denominator.terms
                )
            elif isinstance(component, dict):
                for key in ("numerator", "denominator"):
                    polynomial = component.get(key)
                    if isinstance(polynomial, dict):
                        terms = polynomial.get("terms")
                        if isinstance(terms, (list, tuple)):
                            total += len(terms)
                    elif hasattr(polynomial, "terms"):
                        total += len(polynomial.terms)
            if total > MAX_RATIONAL_MAP_SOURCE_TERMS:
                raise ValueError(
                    "rational-map components exceed the 65,536-term source envelope"
                )
        return data

    @model_validator(mode="after")
    def require_coordinate_axes(self) -> Self:
        if len(set(self.source_variables)) != len(self.source_variables):
            raise ValueError("rational-map source variables must be distinct")
        if len(set(self.target_coordinates)) != len(self.target_coordinates):
            raise ValueError("rational-map target coordinates must be distinct")
        if len(self.components) != len(self.target_coordinates):
            raise ValueError(
                "rational maps require one component per target coordinate"
            )
        if any(value.variables != self.source_variables for value in self.components):
            raise ValueError(
                "every rational-map component must retain the ordered source axis"
            )
        return self


__all__ = ["RationalFunctionMap"]
