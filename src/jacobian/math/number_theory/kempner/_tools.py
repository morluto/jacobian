"""Public declarations for Kempner reciprocal-series enclosures."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.number_theory.kempner._models import (
    KempnerSeriesEnclosure,
    KempnerSeriesEnclosureRequest,
)
from jacobian.math.number_theory.kempner.operations import (
    compute_kempner_series_enclosure,
)

TOOLS: MathTools = (
    MathTool(
        operation_id="number_theory.kempner_series.enclose",
        title="Enclose the reciprocal series of a Kempner digit family",
        description=(
            "Return the exact rational partial reciprocal sum over positive "
            "integers with at most D base-b digits from one proper digit "
            "subset, plus the exact geometric tail bound "
            "r*(s/b)^D/(1-s/b), as the source-bound interval "
            "[partial, partial + tail]. An empty positive family (no nonzero "
            "digit) yields [0, 0]. Admission preflights the finite numeral "
            "count, exact-rational height, and result bytes before "
            "enumeration; leading zeros never count as digits."
        ),
        request_type=KempnerSeriesEnclosureRequest,
        result_type=KempnerSeriesEnclosure,
        run=compute_kempner_series_enclosure,
        tags=("number-theory", "kempner", "reciprocal-series", "enclosure", "exact"),
        discovery_terms=(
            "Kempner series enclosure",
            "reciprocal sum over restricted digits",
            "digit-restricted harmonic sum",
        ),
        examples=(
            OperationExample(
                name="repunit_family_through_two_digits",
                description=(
                    "Enclose the base-10 repunit-digit {1} series through 2 "
                    "digits; the digit subset must be a proper canonical "
                    "subset of its base alphabet."
                ),
                input={
                    "digit_set": {"base": "10", "allowed_digits": ["1"]},
                    "cutoff": "2",
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
