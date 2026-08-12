from __future__ import annotations

import ast
from pathlib import Path

_FORBIDDEN_PREFIXES = (
    "jacobian.domains",
    "networkx",
    "sympy",
)

_FORBIDDEN_BY_CHECKER = {
    "finite_posets.py": ("jacobian.contracts.posets",),
    "simplicial_topology.py": ("jacobian.contracts.topology",),
}

_INDEPENDENT_PROVIDER_BY_CHECKER = {
    "finite_field_polynomial.py": ("sympy",),
    "finite_field_rank.py": ("sympy",),
}


def test_independent_checkers_do_not_import_producer_dependencies() -> None:
    checker_root = Path(__file__).parents[3] / "src" / "jacobian_checkers"
    source_paths = sorted(checker_root.glob("*.py"))
    assert source_paths, f"no checker sources found below {checker_root}"
    violations: list[str] = []

    for source_path in source_paths:
        forbidden = (
            *_FORBIDDEN_PREFIXES,
            *_FORBIDDEN_BY_CHECKER.get(source_path.name, ()),
        )
        independent_provider = _INDEPENDENT_PROVIDER_BY_CHECKER.get(
            source_path.name, ()
        )
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            imported: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                imported = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported = (node.module,)
            for module in imported:
                if module.startswith(forbidden) and not module.startswith(
                    independent_provider
                ):
                    violations.append(f"{source_path}:{node.lineno}: {module}")

    assert violations == []
