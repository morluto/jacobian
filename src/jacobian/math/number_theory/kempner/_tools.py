"""Public declarations for Kempner reciprocal-series enclosures."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.number_theory.kempner._models import (
    KempnerDecimalEnclosure,
    KempnerDecimalEnclosureRequest,
)
from jacobian.math.number_theory.kempner.operations import (
    enclose_kempner_series_decimal,
)


def _run_enclosure(
    request: KempnerDecimalEnclosureRequest,
) -> KempnerDecimalEnclosure:
    return enclose_kempner_series_decimal(
        request.digit_set, request.cutoff, request.precision
    )


TOOLS: MathTools = (
    MathTool(
        operation_id="number_theory.kempner_series.enclose",
        title="Enclose the reciprocal series of a Kempner digit family",
        description=(
            "Return a source-bound rational interval for the infinite "
            "reciprocal series over positive integers with at most D base-b "
            "digits from one proper digit subset. Precision q bounds each "
            "finite reciprocal by its floor and ceiling at scale 10^q; the "
            "exact geometric tail is added above. An empty positive family "
            "(no nonzero digit) yields [0, 0]. Admission preflights the "
            "finite numeral count, traversal work, and output height; leading "
            "zeros never count as digits."
        ),
        request_type=KempnerDecimalEnclosureRequest,
        result_type=KempnerDecimalEnclosure,
        run=_run_enclosure,
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
                    "precision": 24,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
