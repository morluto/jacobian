"""Same-fixture parity between native frame operations and catalog adapters."""

from jacobian.catalog.catalog import Catalog
from jacobian.math.topology.frames import (
    VectorFamily,
    coherence,
    frame_potential,
    gram,
)


def _fixture() -> VectorFamily:
    return VectorFamily(dimension=2, vectors=((1, 1), (1, 0), (0, 1)))


def test_catalog_frame_adapters_agree_with_native_operations() -> None:
    """The same canonical fixture must agree through every calling surface."""
    catalog = Catalog.open()
    family = _fixture()

    gram_tool = catalog.operation("frame.gram.compute")
    coherence_tool = catalog.operation("frame.coherence.compute")
    potential_tool = catalog.operation("frame.potential.compute")
    assert gram_tool is not None
    assert coherence_tool is not None
    assert potential_tool is not None

    assert gram_tool.run(gram_tool.request_type.model_validate(family)) == gram(family)
    assert coherence_tool.run(
        coherence_tool.request_type.model_validate(family)
    ) == coherence(family)
    assert potential_tool.run(
        potential_tool.request_type.model_validate(family)
    ) == frame_potential(family)
