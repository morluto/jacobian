"""Discovery vocabulary for Arb real-function enclosures."""

from jacobian.domains.analysis.operations import POINT_ENCLOSURE_CAPABILITIES


def test_point_enclosure_advertises_every_supported_function() -> None:
    operation = POINT_ENCLOSURE_CAPABILITIES[0]
    spec = operation.spec

    assert {
        "square-root",
        "sqrt",
        "logarithm",
        "log",
        "exponential",
        "exp",
        "sine",
        "sin",
        "cosine",
        "cos",
    } <= set(spec.tags)
    assert "square root, logarithm, exponential, sine, or cosine" in (
        spec.description
    )
    assert spec.invocation_examples[0].input["wall_seconds"] == 10
