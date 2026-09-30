"""Domain-argument entry points for Koszul maps and sequence transport."""

from jacobian._exact import CanonicalRational
from jacobian.math.koszul._native_inputs import native_payload
from jacobian.math.koszul.homology_map import koszul_homology_map as _homology_map
from jacobian.math.koszul.module_models import (
    BasedFiniteModule,
    FiniteCommutativeAlgebra,
    ModuleKoszulChainMap,
    ModuleKoszulComplex,
    ModuleKoszulHomologyMap,
    ModuleKoszulSequenceLinearChange,
    ModuleKoszulTopHomology,
)
from jacobian.math.koszul.module_operations import (
    module_koszul_map as _chain_map,
)
from jacobian.math.koszul.module_operations import (
    module_koszul_sequence_linear_change as _linear_change,
)
from jacobian.math.koszul.module_operations import (
    module_koszul_top_homology as _top_homology,
)


def module_koszul_map(
    algebra: FiniteCommutativeAlgebra,
    source: BasedFiniteModule,
    target: BasedFiniteModule,
    sequence: tuple[tuple[CanonicalRational, ...], ...],
    map_matrix: tuple[tuple[CanonicalRational, ...], ...],
) -> ModuleKoszulChainMap:
    """Induce an exact Koszul chain map from an algebra-linear module map."""
    return _chain_map(
        native_payload(
            algebra=algebra,
            source=source,
            target=target,
            sequence=sequence,
            map_matrix=map_matrix,
        )
    )


def koszul_homology_map(chain_map: ModuleKoszulChainMap) -> ModuleKoszulHomologyMap:
    """Induce homology maps from a source-bound exact Koszul chain map."""
    return _homology_map(native_payload(chain_map=chain_map))


def module_koszul_sequence_linear_change(
    complex_value: ModuleKoszulComplex,
    change_matrix: tuple[tuple[CanonicalRational, ...], ...],
) -> ModuleKoszulSequenceLinearChange:
    """Apply an invertible coordinate matrix to a retained Koszul sequence."""
    return _linear_change(
        native_payload(complex=complex_value, change_matrix=change_matrix)
    )


def module_koszul_top_homology(
    complex_value: ModuleKoszulComplex,
) -> ModuleKoszulTopHomology:
    """Compute the top homology and annihilator of a retained Koszul complex."""
    return _top_homology(native_payload(complex=complex_value))
