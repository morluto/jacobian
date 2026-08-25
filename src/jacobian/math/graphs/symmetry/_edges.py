"""Canonical edge representation for graph-symmetry values."""


def canonical_edge(left: str, right: str) -> tuple[str, str]:
    """Return one undirected edge in lexical endpoint order."""

    return (left, right) if left < right else (right, left)


__all__ = ["canonical_edge"]
