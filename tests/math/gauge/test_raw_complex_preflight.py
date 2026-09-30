"""Every copied gauge-complex container is bounded before canonicalization."""

from typing import Any

import pytest
from pydantic import ValidationError

from jacobian.math.gauge import FiniteGroupGaugeComplex, _models
from jacobian.math.gauge._models import FiniteGroupGaugeComplexRequest


def _payload() -> dict[str, Any]:
    return {
        "lattice": {
            "vertices": ["a"],
            "edges": [{"edge_id": "e", "tail": "a", "head": "a"}],
        },
        "group": {"multiplication": [[0]], "identity": 0, "inverse": [0]},
        "faces": [
            {
                "face_id": "f",
                "boundary": {
                    "steps": [{"edge_id": "e", "forward": True}],
                    "basepoint": "a",
                },
            }
        ],
    }


@pytest.mark.parametrize(
    "model", (FiniteGroupGaugeComplex, FiniteGroupGaugeComplexRequest)
)
@pytest.mark.parametrize(
    "path,bad",
    (
        ((), [["bad"]]),
        (("lattice",), [["bad"]]),
        (("lattice", "vertices"), {"bad": [1]}),
        (("lattice", "vertices", 0), ["bad"]),
        (("lattice", "edges"), {"bad": [1]}),
        (("lattice", "edges", 0), ["bad"]),
        (("lattice", "edges", 0, "edge_id"), ["bad"]),
        (("group",), [["bad"]]),
        (("group", "multiplication"), {"bad": [1]}),
        (("group", "multiplication", 0), {"bad": [1]}),
        (("group", "multiplication", 0, 0), [1]),
        (("group", "identity"), [0]),
        (("group", "inverse"), {"bad": [1]}),
        (("group", "inverse", 0), [0]),
        (("faces",), {"bad": [1]}),
        (("faces", 0), ["bad"]),
        (("faces", 0, "face_id"), ["bad"]),
        (("faces", 0, "boundary"), ["bad"]),
        (("faces", 0, "boundary", "basepoint"), ["bad"]),
        (("faces", 0, "boundary", "steps"), {"bad": [1]}),
        (("faces", 0, "boundary", "steps", 0), ["bad"]),
        (("faces", 0, "boundary", "steps", 0, "edge_id"), ["bad"]),
        (("faces", 0, "boundary", "steps", 0, "forward"), [True]),
    ),
)
def test_malformed_raw_fields_are_rejected_before_copy(
    model: type[FiniteGroupGaugeComplex] | type[FiniteGroupGaugeComplexRequest],
    path: tuple[str | int, ...],
    bad: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload: Any = _payload()
    parent = payload
    if path:
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = bad
    else:
        payload = bad

    def unexpected_copy(value: object) -> object:
        raise AssertionError("malformed containers must be rejected before copying")

    monkeypatch.setattr(_models, "canonicalize_json_containers", unexpected_copy)
    with pytest.raises(ValidationError) as error:
        model.model_validate(payload)
    assert error.value.errors()[0]["type"].startswith("lattice_gauge.complex_")


def test_native_and_mixed_complex_carriers_remain_accepted() -> None:
    complex_value = FiniteGroupGaugeComplex.model_validate(_payload())
    for faces in (
        complex_value.faces,
        ({"face_id": "f", "boundary": complex_value.faces[0].boundary},),
        (
            {
                "face_id": "f",
                "boundary": {
                    "steps": complex_value.faces[0].boundary.steps,
                    "basepoint": "a",
                },
            },
        ),
    ):
        request = FiniteGroupGaugeComplexRequest.model_validate(
            {
                "lattice": complex_value.lattice,
                "group": complex_value.group,
                "faces": faces,
            }
        )
        assert request.faces == complex_value.faces
        assert (
            FiniteGroupGaugeComplex.model_validate_json(request.model_dump_json())
            == complex_value
        )


@pytest.mark.parametrize("kind", ("sequence", "mapping"))
def test_raw_container_subclasses_are_refused_before_hooks(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    class UnvisitedList(list[object]):
        def __len__(self) -> int:
            raise AssertionError("custom container length must not run")

    class UnvisitedDict(dict[str, object]):
        def __len__(self) -> int:
            raise AssertionError("custom container length must not run")

    payload = _payload()
    if kind == "sequence":
        payload["faces"] = UnvisitedList(payload["faces"])
    else:
        payload["faces"][0] = UnvisitedDict(payload["faces"][0])

    def unexpected_copy(value: object) -> object:
        raise AssertionError("noncanonical containers must not be copied")

    monkeypatch.setattr(_models, "canonicalize_json_containers", unexpected_copy)
    with pytest.raises(ValidationError):
        FiniteGroupGaugeComplex.model_validate(payload)
