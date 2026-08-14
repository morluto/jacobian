from __future__ import annotations

from jacobian.graphs.operation_resources import GraphFamilySession


def test_graph_family_session_registers_resources_once(monkeypatch) -> None:
    calls = {"resources": 0, "composition": 0}

    def fake_resources(_store: object, _schemas: object) -> object:
        calls["resources"] += 1
        return object()

    def fake_bind(
        operation_id: str,
        *_args: object,
        resources: object | None = None,
        composition: object | None = None,
        **_kwargs: object,
    ) -> object:
        del resources, composition
        return operation_id

    monkeypatch.setattr(
        "jacobian.graphs.operation_resources.register_graph_resources",
        fake_resources,
    )
    monkeypatch.setattr(
        "jacobian.graphs.operation_resources.bind_selected_graph_operation",
        fake_bind,
    )
    session = GraphFamilySession(
        store=object(),  # type: ignore[arg-type]
        schemas=object(),  # type: ignore[arg-type]
        artifacts=object(),  # type: ignore[arg-type]
        verification=object(),  # type: ignore[arg-type]
        checkers=object(),  # type: ignore[arg-type]
        catalog=object(),  # type: ignore[arg-type]
    )

    assert session.bind("graph.construct.explicit") == "graph.construct.explicit"
    assert session.bind("graph.search.atlas") == "graph.search.atlas"
    assert calls["resources"] == 1
