"""Cheap, suite-wide pytest conventions.

The root conftest is imported while pytest is collecting every test.  It must
therefore stay deliberately boring: no runtime construction, provider probes,
database connections, portfolio imports, or implementation modules belong
here.  Resource-owning fixtures live in the conftest below the tier that owns
the resource.
"""

from __future__ import annotations

pytest_plugins = (
    "tests.support.runtime_templates",
    "tests.support.runtime_instances",
)
