"""Typed operation arguments and exact differential GCRD result."""

from jacobian._models import StrictModel
from jacobian.math.ore_algebras._models import DifferentialOreOperator


class DifferentialOperatorGCRDRequest(StrictModel):
    """Two operators in the same rational differential-operator algebra."""

    left: DifferentialOreOperator
    right: DifferentialOreOperator


class DifferentialOperatorGCRDResult(StrictModel):
    """Monic greatest common right divisor and exact reconstruction data."""

    left: DifferentialOreOperator
    right: DifferentialOreOperator
    divisor: DifferentialOreOperator
    left_cofactor: DifferentialOreOperator
    right_cofactor: DifferentialOreOperator
    bezout_left: DifferentialOreOperator
    bezout_right: DifferentialOreOperator


__all__ = ["DifferentialOperatorGCRDRequest", "DifferentialOperatorGCRDResult"]
