"""Native mathematical APIs supported by Jacobian.

The domain packages are imported on first attribute access.  Keeping this
namespace lazy preserves the public package layout without loading optional
symbolic and integer backends for callers that use only one domain.
"""

from importlib import import_module

_SUBMODULES = frozenset(
    {
        "analysis",
        "cluster_algebras",
        "coalgebras",
        "combinatorics",
        "crossed_products",
        "dynamics",
        "finite_categories",
        "finite_dim_algebras",
        "finite_fields",
        "finite_semigroups",
        "geometry",
        "graphs",
        "groups",
        "lattices",
        "logic",
        "matrices",
        "number_theory",
        "optimization",
        "polynomials",
        "probability",
        "topology",
        "universal_algebra",
    }
)


def __getattr__(name: str) -> object:
    if name not in _SUBMODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module

__all__ = [
    "analysis",
    "cluster_algebras",
    "coalgebras",
    "combinatorics",
    "crossed_products",
    "dynamics",
    "finite_categories",
    "finite_dim_algebras",
    "finite_fields",
    "finite_semigroups",
    "geometry",
    "graphs",
    "groups",
    "lattices",
    "logic",
    "matrices",
    "number_theory",
    "optimization",
    "polynomials",
    "probability",
    "topology",
    "universal_algebra",
]
