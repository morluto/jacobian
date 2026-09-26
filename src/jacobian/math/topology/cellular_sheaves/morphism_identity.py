"""Identity natural transformations of finite cellular sheaves."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.cellular_sheaves._models import (
    MAX_SHEAF_MORPHISM_COMPONENT_CELLS,
    MAX_SHEAF_MORPHISM_OUTPUT_CHARS,
    FiniteCellularSheaf,
    SheafField,
    sheaf_scalar_json_bound,
)
from jacobian.math.topology.cellular_sheaves.extensions import (
    SheafMorphismResult,
)
from jacobian.math.topology.cellular_sheaves.morphism_kernel import (
    _readmit_parent_sheaf,
)


class SheafMorphismIdentityRequest(StrictModel):
    """A cellular sheaf whose identity transformation is requested."""

    sheaf: FiniteCellularSheaf


def identity_morphism(sheaf: FiniteCellularSheaf) -> SheafMorphismResult:
    """Return the identity natural transformation on ``sheaf``.

    Diagram admission and the public morphism contract are reused so a parsed
    sheaf's cover maps and naturality are checked before an identity value is
    returned. Matrix allocation follows explicit component and output bounds.
    """
    if not isinstance(sheaf, FiniteCellularSheaf):
        raise OperationDomainValidationError(
            location=("sheaf",),
            code="topology.cellular_sheaf.morphism_identity.parent_type_invalid",
            message="identity source must be a finite cellular sheaf",
        )
    try:
        _readmit_parent_sheaf(sheaf, role="identity source")
    except OperationDomainValidationError as error:
        raise OperationDomainValidationError(
            location=("sheaf",),
            code="topology.cellular_sheaf.morphism_identity.parent_diagram_not_admitted",
            message="identity source restrictions must equal its reconstructed functor diagram",
        ) from error
    ranks = {stalk.simplex: len(stalk.basis) for stalk in sheaf.stalks}
    component_cells = sum(rank * rank for rank in ranks.values())
    # Source and target are both retained in the result. Bound the full parent
    # encoding (including axes, restrictions, and labels), not just components.
    parent_json_chars = len(sheaf.model_dump_json())
    output_chars = (
        2 * parent_json_chars
        + sheaf_scalar_json_bound(component_cells, 34)
        + 64 * (len(sheaf.canonical_face_order) + component_cells)
    )
    if component_cells > MAX_SHEAF_MORPHISM_COMPONENT_CELLS:
        raise OperationResourceAdmissionError(
            location=("sheaf",),
            code="topology.cellular_sheaf.morphism_identity.component_bound",
            message="identity components exceed the admitted matrix-cell bound",
        )
    if output_chars > MAX_SHEAF_MORPHISM_OUTPUT_CHARS:
        raise OperationResourceAdmissionError(
            location=("sheaf",),
            code="topology.cellular_sheaf.morphism_identity.output_bound",
            message="identity components exceed the admitted output bound",
        )
    zero = (
        CanonicalRational.from_fraction(Fraction(0))
        if sheaf.coefficient_field is SheafField.RATIONAL
        else 0
    )
    one = (
        CanonicalRational.from_fraction(Fraction(1))
        if sheaf.coefficient_field is SheafField.RATIONAL
        else 1
    )
    components = []
    for simplex in sheaf.canonical_face_order:
        rank = ranks[simplex]
        identity = tuple(
            tuple(one if row == column else zero for column in range(rank))
            for row in range(rank)
        )
        components.append((simplex, identity))
    return SheafMorphismResult(
        source=sheaf,
        target=sheaf,
        components=tuple(components),
        natural=True,
        obstruction=None,
    )


__all__ = ["SheafMorphismIdentityRequest", "identity_morphism"]
