"""Supported native combinatorics-on-words API."""

from jacobian.math.words.operations import (
    FactorAnalysis,
    PeriodAnalysis,
    apply_morphism,
    compose_morphisms,
    conjugates,
    factor_occurrences,
    factors_of_length,
    incidence_matrix,
    parikh_vector,
    periods,
    prefix_function,
    primitive_root,
)
from jacobian.math.words.values import FiniteWord, WordMorphism

__all__ = [
    "FactorAnalysis",
    "FiniteWord",
    "PeriodAnalysis",
    "WordMorphism",
    "apply_morphism",
    "compose_morphisms",
    "conjugates",
    "factor_occurrences",
    "factors_of_length",
    "incidence_matrix",
    "parikh_vector",
    "periods",
    "prefix_function",
    "primitive_root",
]
