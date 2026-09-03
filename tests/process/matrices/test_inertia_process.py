"""Operational failure contracts for bounded algebraic inertia."""

from time import monotonic

import pytest

import jacobian.process as process
from jacobian.math.matrices._number_field import recognize_real_simple_number_field
from jacobian.math.matrices.analysis._inertia_process import (
    algebraic_inertia_killable,
)
from jacobian.math.number_theory.number_fields import (
    RealNumberFieldEmbedding,
    SimpleNumberFieldPresentation,
)


def test_algebraic_inertia_rejects_an_unbound_worker_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    presentation = SimpleNumberFieldPresentation(
        coefficients_descending=("1", "0", "-2")
    )
    embedding = RealNumberFieldEmbedding.model_validate(
        {
            "kind": "REAL",
            "presentation": presentation.model_dump(mode="json"),
            "root": {
                "polynomial": presentation.coefficients_descending,
                "real_root_index": 1,
            },
        }
    )
    recognized = recognize_real_simple_number_field(embedding)

    def unbound_projection(
        *args: object, **kwargs: object
    ) -> process.BoundedProcessResult:
        return process.BoundedProcessResult(
            returncode=0,
            stdout=b'{"n_negative":0,"n_positive":1,"n_zero":0,"request_digest":"bad"}',
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=False,
        )

    monkeypatch.setattr(process, "run_bounded_process", unbound_projection)
    with pytest.raises(RuntimeError, match="malformed data"):
        algebraic_inertia_killable(
            [[recognized.field.one]],
            recognized,
            regime="DIAGONAL",
            deadline=monotonic() + 10,
        )
