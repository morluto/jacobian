from __future__ import annotations

from types import SimpleNamespace

from jacobian.family_resolver import FamilyResolver
from jacobian.selected_operation_bindings import SelectedOperationBinding


def test_family_resolver_caches_one_adapter() -> None:
    calls: list[str] = []
    adapter = SimpleNamespace(
        descriptor=SimpleNamespace(operation_id="graph.construct.explicit")
    )

    def bind(operation_id: str, _descriptor: object) -> SelectedOperationBinding:
        calls.append(operation_id)
        return SelectedOperationBinding(adapter)  # type: ignore[arg-type]

    resolver = FamilyResolver("graph", bind)
    descriptor = SimpleNamespace()
    first = resolver.resolve("graph.construct.explicit", descriptor)  # type: ignore[arg-type]
    second = resolver.resolve("graph.construct.explicit", descriptor)  # type: ignore[arg-type]
    assert first is not None
    assert second is not None
    assert first.adapter is adapter
    assert second.adapter is adapter
    assert calls == ["graph.construct.explicit"]
