"""Native and published Petri-net projections preserve Unicode axis validity."""

from __future__ import annotations

from typing import Any

import pytest

from jacobian._models import StrictModel
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import OperationRequestValidationError, invoke_operation
from jacobian.math.logic.automata.petri_nets import (
    Marking,
    PetriNet,
    marking_equation,
    operations,
    petri_net_matrices,
)

_OPERATIONS = ("petri_net.marking_equation.compute", "petri_net.matrices.compute")


def _net(place_id: str = "p", transition_id: str = "t") -> PetriNet:
    return PetriNet(
        place_count=1,
        transition_count=1,
        place_ids=(place_id,),
        transition_ids=(transition_id,),
        pre=((1,),),
        post=((1,),),
    )


def _payload(operation_id: str, net: PetriNet) -> dict[str, Any]:
    payload: dict[str, Any] = {"net": net.model_dump(mode="json")}
    if operation_id == "petri_net.marking_equation.compute":
        payload.update(
            source_marking={"tokens": [1]},
            target_marking={"tokens": [1]},
            transition_counts=[1],
        )
    return payload


def _native_result(operation_id: str, net: PetriNet) -> StrictModel:
    if operation_id == "petri_net.matrices.compute":
        return petri_net_matrices(net)
    marking = Marking(tokens=(1,))
    return marking_equation(net, marking, marking, (1,))


@pytest.mark.parametrize("operation_id", _OPERATIONS)
@pytest.mark.parametrize("axis", ("place_ids", "transition_ids"))
@pytest.mark.parametrize("surrogate", ("\ud800", "\udfff"))
def test_surrogate_axes_are_rejected_by_native_and_public_boundaries(
    operation_id: str, axis: str, surrogate: str
) -> None:
    net = _net().model_copy(update={axis: ("prefix" + surrogate + "suffix",)})
    with pytest.raises(OperationDomainValidationError) as native_error:
        _native_result(operation_id, net)
    assert native_error.value.errors()[0]["type"] == "petri_net.net_axis_encoding"

    # The published owner adapter must retain the same native admission even
    # when its request is supplied as an already parsed Python model.
    tool = Catalog.open().operation(operation_id)
    assert tool is not None
    request = tool.request_type.model_validate(_payload(operation_id, net))
    with pytest.raises(OperationDomainValidationError) as catalog_error:
        tool.run(request)
    assert catalog_error.value.errors() == native_error.value.errors()

    # Dispatch rejects invalid Unicode even earlier, at the strict JSON input
    # boundary, rather than admitting it until result serialization.
    with pytest.raises(OperationRequestValidationError) as dispatch_error:
        invoke_operation(operation_id, _payload(operation_id, net), Catalog.open())
    assert dispatch_error.value.errors()[0]["type"] == "canonicalization_error"


@pytest.mark.parametrize("operation_id", _OPERATIONS)
@pytest.mark.parametrize("axis", ("place_ids", "transition_ids"))
def test_axis_encoding_is_checked_before_retained_cell_admission(
    operation_id: str, axis: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Observe the real admission order without estimating any serialized bytes."""
    net = _net().model_copy(update={axis: ("x" * 40_000 + "\ud800",)})
    calls: list[str] = []

    def unexpected_cell_count(candidate: PetriNet) -> int:
        calls.append("cells")
        raise AssertionError("malformed axes must be rejected before cell admission")

    monkeypatch.setattr(operations, "_petri_net_retained_cells", unexpected_cell_count)
    with pytest.raises(OperationDomainValidationError) as error:
        _native_result(operation_id, net)
    assert error.value.errors()[0]["type"] == "petri_net.net_axis_encoding"
    assert calls == []


@pytest.mark.parametrize("operation_id", _OPERATIONS)
@pytest.mark.parametrize(
    "label",
    ("plain", 'é𝄞東京\n"', "é𝄞東京" * 10_000),
    ids=("plain", "unicode", "long-unicode"),
)
def test_valid_unicode_axes_preserve_native_public_results_and_cell_admission(
    operation_id: str, label: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Valid long UTF-8 labels remain admitted by retained cells, not bytes."""
    net = _net("p:" + label, "t:" + label)
    monkeypatch.setattr(operations, "MAX_MARKING_EQUATION_OUTPUT_CELLS", 64)
    monkeypatch.setattr(operations, "MAX_PETRI_NET_REVERSE_OUTPUT_CELLS", 64)
    native = _native_result(operation_id, net)
    public = invoke_operation(operation_id, _payload(operation_id, net), Catalog.open())
    assert public.output == native.model_dump(mode="json")
    assert public.output["net"]["place_ids"] == ["p:" + label]
    assert public.output["net"]["transition_ids"] == ["t:" + label]
    if operation_id == "petri_net.marking_equation.compute":
        assert public.output["formal_target"] == [1]
        assert public.output["residual"] == [0]
        assert public.output["satisfies_equation"] is True
    else:
        assert public.output["incidence"]["entries"] == [["0"]]
