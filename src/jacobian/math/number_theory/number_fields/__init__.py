"""Number field operations."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math.number_theory.number_fields._binary_power_sum import (
        BinaryPowerSumGap,
        BinaryPowerSumGapProfile,
        BinaryPowerSumValueBucket,
    )
    from jacobian.math.number_theory.number_fields._ring_of_integers import (
        NumberFieldRingOfIntegersResult,
        ring_of_integers,
    )
    from jacobian.math.number_theory.number_fields.operations import (
        binary_power_sum_gap_profile,
        compare_real_embedding_elements,
        discriminant,
        embeddings,
        verify_binary_power_sum_gap_profile,
        verify_discriminant,
    )
    from jacobian.math.number_theory.number_fields.values import (
        GaussianRational,
        NumberFieldEmbeddingProfile,
        NumberFieldRealValueEnclosure,
        RealNumberFieldEmbedding,
        SimpleNumberFieldElement,
        SimpleNumberFieldPresentation,
        SimpleNumberFieldRealEmbeddingBinding,
        SimpleNumberFieldRealEmbeddingOrder,
    )


__all__ = [
    "BinaryPowerSumGap",
    "BinaryPowerSumGapProfile",
    "BinaryPowerSumValueBucket",
    "GaussianRational",
    "NumberFieldEmbeddingProfile",
    "NumberFieldRealValueEnclosure",
    "NumberFieldRingOfIntegersResult",
    "RealNumberFieldEmbedding",
    "SimpleNumberFieldElement",
    "SimpleNumberFieldPresentation",
    "SimpleNumberFieldRealEmbeddingBinding",
    "SimpleNumberFieldRealEmbeddingOrder",
    "binary_power_sum_gap_profile",
    "compare_real_embedding_elements",
    "discriminant",
    "embeddings",
    "ring_of_integers",
    "verify_binary_power_sum_gap_profile",
    "verify_discriminant",
]


_OWNER_MODULES = {
    "BinaryPowerSumGap": "_binary_power_sum",
    "BinaryPowerSumGapProfile": "_binary_power_sum",
    "BinaryPowerSumValueBucket": "_binary_power_sum",
    "GaussianRational": "values",
    "NumberFieldEmbeddingProfile": "values",
    "NumberFieldRealValueEnclosure": "values",
    "NumberFieldRingOfIntegersResult": "_ring_of_integers",
    "RealNumberFieldEmbedding": "values",
    "SimpleNumberFieldElement": "values",
    "SimpleNumberFieldPresentation": "values",
    "SimpleNumberFieldRealEmbeddingBinding": "values",
    "SimpleNumberFieldRealEmbeddingOrder": "values",
    "ring_of_integers": "_ring_of_integers",
}


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{_OWNER_MODULES.get(name, 'operations')}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
