"""The group-lattice request must publish the envelope it enforces."""

from __future__ import annotations

from jacobian.math.affine_semigroups.group_lattice import (
    MAX_AFFINE_GROUP_LATTICE_LABEL_CHARS,
)
from jacobian.math.affine_semigroups.group_lattice_models import (
    AffineGroupLatticeRequest,
)


def test_the_configuration_field_states_the_admitted_envelope() -> None:
    """A client generating from the JSON Schema must see the real numbers.

    The shared scalar and nested configuration schemas are deliberately broader
    than this operation's envelope, so without an operation-specific description
    a schema-shaped request can fail immediately at runtime with no discoverable
    numeric contract.
    """
    description = AffineGroupLatticeRequest.model_fields["configuration"].description
    assert description is not None
    assert "1-8 row labels" in description
    assert "1-10 generator labels" in description
    assert str(10**8) in description
    assert str(MAX_AFFINE_GROUP_LATTICE_LABEL_CHARS) in description
