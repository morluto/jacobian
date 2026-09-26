"""Public declarations for Kempner reciprocal-series enclosures."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.number_theory.kempner._models import (
    KempnerDecimalEnclosure,
    KempnerDecimalEnclosureRequest,
    KempnerSeriesEnclosure,
    KempnerSeriesEnclosureRequest,
)
from jacobian.math.number_theory.kempner.operations import (
    enclose_kempner_series,
    enclose_kempner_series_decimal,
)


def _run_enclosure(
    request: KempnerSeriesEnclosureRequest,
) -> KempnerSeriesEnclosure:
    return enclose_kempner_series(request.digit_set, request.cutoff)


def _run_decimal_enclosure(
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
                },
            ),
        ),
    ),
    MathTool(
        operation_id="number_theory.kempner_series.enclose_decimal",
        title="Enclose a dense Kempner series at fixed decimal precision",
        description=(
            "Return an exact rational interval for the infinite reciprocal "
            "series of a proper base-b digit family. Each finite reciprocal "
            "is rounded outward at the requested decimal precision and the "
            "exact geometric tail bound is added above. Admission bounds "
            "numerals, prefix visits, stack size, result height, and output "
            "before traversal."
        ),
        request_type=KempnerDecimalEnclosureRequest,
        result_type=KempnerDecimalEnclosure,
        run=_run_decimal_enclosure,
        tags=("number-theory", "kempner", "dense-series", "enclosure", "exact"),
        discovery_terms=(
            "dense Kempner series enclosure",
            "fixed-point reciprocal series interval",
            "high precision digit-restricted harmonic sum",
        ),
        examples=(
            OperationExample(
                name="base_four_dense_family",
                description=(
                    "Enclose the base-4 family using digits 0, 1, and 2 "
                    "through four digits at twelve decimal places."
                ),
                input={
                    "digit_set": {"base": "4", "allowed_digits": ["0", "1", "2"]},
                    "cutoff": "4",
                    "precision": 12,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
