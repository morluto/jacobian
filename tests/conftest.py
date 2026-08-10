"""Cheap, suite-wide pytest conventions.

The root conftest is imported while pytest is collecting every test.  It must
therefore stay deliberately boring: no runtime construction, provider probes,
database connections, portfolio imports, or implementation modules belong
here.  Complete-runtime fixtures register only under owning tiers.
"""

from __future__ import annotations

pytest_plugins = ("tests.support.resource_closure_plugin",)
