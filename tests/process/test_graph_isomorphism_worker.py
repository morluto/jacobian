"""Process-boundary behavior for graph-isomorphism workers."""

import pytest

from jacobian import process as process_runtime
from jacobian.math.graphs.isomorphism import _vf2_process as isomorphism_operations
from jacobian.math.graphs.isomorphism._models import GraphIsomorphismRequest
from jacobian.math.graphs.isomorphism._vf2_process import decide_graph_isomorphism
from jacobian.math.graphs.isomorphism._vf2_worker import _first_isomorphism_mapping
from jacobian.process import BoundedProcessResult, ProcessResourceLimits


def test_vf2_worker_obtains_a_positive_witness_in_one_search_traversal() -> None:
    class Matcher:
        def __init__(self) -> None:
            self.traversals = 0

        def isomorphisms_iter(self):  # type: ignore[no-untyped-def]
            self.traversals += 1
            yield {1: 0, 0: 1}

    matcher = Matcher()

    assert _first_isomorphism_mapping(matcher) == [(0, 1), (1, 0)]
    assert matcher.traversals == 1


def test_timed_out_vf2_worker_is_an_operational_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        process_runtime,
        "run_bounded_process",
        lambda *_args, **_kwargs: BoundedProcessResult(
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=True,
        ),
    )

    with pytest.raises(RuntimeError):
        decide_graph_isomorphism(
            GraphIsomorphismRequest.model_validate(
                {
                    "graph_a": {
                        "vertex_count": 2,
                        "directed": False,
                        "edges": [(0, 1)],
                    },
                    "graph_b": {
                        "vertex_count": 2,
                        "directed": False,
                        "edges": [(0, 1)],
                    },
                }
            )
        )


def test_vf2_worker_has_private_cwd_and_os_resource_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}

    def complete_worker(*_args: object, **kwargs: object) -> BoundedProcessResult:
        recorded.update(kwargs)
        return BoundedProcessResult(
            returncode=0,
            stdout=b'{"ok":true,"mapping":[[0,0],[1,1]]}',
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=False,
        )

    monkeypatch.setattr(process_runtime, "run_bounded_process", complete_worker)

    result = decide_graph_isomorphism(
        GraphIsomorphismRequest.model_validate(
            {
                "graph_a": {"vertex_count": 2, "directed": False, "edges": [(0, 1)]},
                "graph_b": {"vertex_count": 2, "directed": False, "edges": [(0, 1)]},
            }
        )
    )

    assert result.status == "ISOMORPHIC"
    assert recorded["resource_limits"] == ProcessResourceLimits(
        cpu_seconds=60,
        address_space_bytes=isomorphism_operations._VF2_ADDRESS_SPACE_BYTES,
        file_size_bytes=isomorphism_operations._VF2_FILE_SIZE_BYTES,
    )
    assert str(recorded["cwd"]).split("/")[-1].startswith("jacobian-vf2-")
