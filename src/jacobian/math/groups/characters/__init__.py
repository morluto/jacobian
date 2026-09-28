"""Finite class-function operations."""

from jacobian.math.groups.characters._abelian_models import (
    FiniteAbelianCharacterRow,
    FiniteAbelianCharacterTableResult,
)
from jacobian.math.groups.characters._models import (
    CharacterRingElement,
    CharacterRow,
    CharacterTableResult,
    CharacterTensorDecompositionResult,
    ClassAxis,
    ClassContribution,
    ClassFunctionInductionResult,
    ClassFunctionInnerProductResult,
    ClassFunctionRestrictionResult,
    ClassPowerMapResult,
    ConjugacyClassPartition,
    CyclicCharacterRestrictionResult,
    CyclotomicValue,
    FiniteClassFunction,
    FrobeniusSchurIndicatorResult,
)
from jacobian.math.groups.characters.abelian_operations import (
    finite_abelian_character_table,
)
from jacobian.math.groups.characters.operations import (
    character_table,
    character_tensor_decomposition,
    class_function_add,
    class_function_conjugate,
    class_function_induce_from_subgroup,
    class_function_inner_product,
    class_function_pointwise_product,
    class_function_restrict_to_subgroup,
    class_function_scale,
    class_power_map,
    frobenius_schur_indicator,
    restrict_cyclic_character,
)
from jacobian.math.groups.characters.representation_ring_operations import (
    character_tensor_product,
)
from jacobian.math.groups.characters.representation_ring_operations import (
    class_function_character_decomposition as character_ring_decomposition,
)

__all__ = [
    "CharacterRingElement",
    "CharacterRow",
    "CharacterTableResult",
    "CharacterTensorDecompositionResult",
    "ClassAxis",
    "ClassContribution",
    "ClassFunctionInductionResult",
    "ClassFunctionInnerProductResult",
    "ClassFunctionRestrictionResult",
    "ClassPowerMapResult",
    "ConjugacyClassPartition",
    "CyclicCharacterRestrictionResult",
    "CyclotomicValue",
    "FiniteAbelianCharacterRow",
    "FiniteAbelianCharacterTableResult",
    "FiniteClassFunction",
    "FrobeniusSchurIndicatorResult",
    "character_ring_decomposition",
    "character_table",
    "character_tensor_decomposition",
    "character_tensor_product",
    "class_function_add",
    "class_function_conjugate",
    "class_function_induce_from_subgroup",
    "class_function_inner_product",
    "class_function_pointwise_product",
    "class_function_restrict_to_subgroup",
    "class_function_scale",
    "class_power_map",
    "finite_abelian_character_table",
    "frobenius_schur_indicator",
    "restrict_cyclic_character",
]
