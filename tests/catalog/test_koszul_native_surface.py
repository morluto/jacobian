"""The authoritative native surface must reach every published operation."""

from __future__ import annotations

import jacobian.math.koszul as koszul
from jacobian.math.koszul._tools import TOOLS

# Operations whose kernel is a domain function rather than an envelope adapter.
# The lambda entries in ``_tools.py`` unwrap a wire request, which is the
# catalog wrapper's documented job, so they have no name to export.
DIRECT_RUNS = (
    "koszul_homology_map",
    "module_koszul_map",
    "module_koszul_sequence_linear_change",
    "module_koszul_top_homology",
)


def test_the_published_domain_functions_are_reachable_from_the_package() -> None:
    """A native caller using the documented owner namespace must find each run.

    ``_tools.py`` published these operations while the package exported none of
    their domain functions, so they could be discovered with ``math.find`` but
    not driven natively.
    """
    missing = [name for name in DIRECT_RUNS if not hasattr(koszul, name)]
    assert missing == []
    assert all(name in koszul.__all__ for name in DIRECT_RUNS)


def test_every_published_result_carrier_is_reachable_from_the_package() -> None:
    """A consumer reading a result needs its type, not just its contents."""
    result_types = {tool.result_type for tool in TOOLS}
    missing = sorted(
        result_type.__name__
        for result_type in result_types
        if not hasattr(koszul, result_type.__name__)
    )
    assert missing == []


def test_wire_request_models_stay_out_of_the_native_surface() -> None:
    """The native API takes domain values, not wire envelopes."""
    assert not [name for name in koszul.__all__ if name.endswith("Request")]
