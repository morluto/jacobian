"""Operational failure contracts for the bounded cyclic-kernel batch."""

from __future__ import annotations

import pickle

import pytest

import jacobian.process as process
from jacobian.math.matrices.cyclic_linear._kernel_process import (
    run_cyclotomic_kernels,
)


def test_cyclic_kernel_rejects_an_unbound_worker_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unbound_projection(
        *args: object, **kwargs: object
    ) -> process.BoundedProcessResult:
        return process.BoundedProcessResult(
            returncode=0,
            stdout=pickle.dumps((b"bad", ((),))),
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=False,
        )

    monkeypatch.setattr(process, "run_bounded_process", unbound_projection)

    with pytest.raises(RuntimeError, match="malformed output"):
        run_cyclotomic_kernels(((1, 1, (((1,),),), 1),), deadline=None)
