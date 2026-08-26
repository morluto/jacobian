"""Dispatch composition for canonical algebraic-combinatorics values."""

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation


def test_empty_word_and_empty_alphabet_compose_through_rsk_dispatch() -> None:
    """The exact empty RSK pair reconstructs the source without an invented symbol."""
    catalog = Catalog.open()
    forward = invoke_operation(
        "tableau.rsk.word.compute",
        {
            "word": {"alphabet": [], "letters": []},
            "convention": "ROW_INSERTION_RSK_V1",
        },
        catalog,
    )

    assert forward.output["alphabet"] == []
    assert forward.output["insertion_tableau"] == {"rows": []}
    assert forward.output["recording_tableau"] == {"rows": []}
    assert forward.output["shape"] == {"parts": []}

    inverse = invoke_operation(
        "tableau.rsk.inverse_word.compute",
        {
            "pair": forward.output,
            "convention": "ROW_INSERTION_RSK_V1",
        },
        catalog,
    )

    assert inverse.output == {"alphabet": [], "letters": []}
