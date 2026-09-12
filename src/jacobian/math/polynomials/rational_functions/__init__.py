"""Canonical rational functions and coordinate maps.

The native composition entry point accepts canonical map values directly. The
wire-only request model remains owned by the composition tool manifest.
"""

from jacobian.math.polynomials.rational_functions.composition._models import (
    RationalFunctionMapComposition,
)
from jacobian.math.polynomials.rational_functions.composition.operations import (
    compose_maps,
)
from jacobian.math.polynomials.rational_functions.values import RationalFunctionMap

__all__ = ["RationalFunctionMap", "RationalFunctionMapComposition", "compose_maps"]
