"""Every public forward-trace value must be reachable from the owner namespace."""

from __future__ import annotations

from jacobian.math.combinatorics.algebraic import RSKBumpStep, RSKInsertionEvent


def test_the_bump_step_type_is_exported_alongside_the_event_that_holds_it() -> None:
    """``RSKInsertionEvent.bump_path`` is typed with the forward bump step.

    Exporting only the reverse counterpart left callers unable to construct or
    annotate a complete forward trace through the authoritative domain API.
    """
    assert (
        RSKInsertionEvent.model_fields["bump_path"].annotation
        == tuple[RSKBumpStep, ...]
    )
    assert (
        "RSKBumpStep"
        in __import__("jacobian.math.combinatorics.algebraic", fromlist=["x"]).__all__
    )
