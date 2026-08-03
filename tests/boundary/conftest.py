"""Fixtures that own real external boundaries.

The boundary tier is the only place where tests may deliberately start a
child process, inspect an optional provider, or open a durability-focused
SQLite store.  Fixtures are function scoped unless the value is an immutable
provider identity.  Process fixtures expose launchers so a test can choose the
exact command while teardown still owns every child it started.
"""

from __future__ import annotations

pytest_plugins = ("tests.support.runtime_fixtures",)
