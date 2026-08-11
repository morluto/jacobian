"""Boundary-tier policy and ownership documentation.

The boundary tier is the only place where tests may deliberately start a
child process, inspect an optional provider, or open a durability-focused
SQLite store.  Fixtures are function scoped unless the value is an immutable
provider identity.  Process fixtures expose launchers so a test can choose the
exact command while teardown still owns every child it started.

Runtime fixtures are registered by the storage, provider, and MCP subtrees that
consume them; process tests remain independent of complete-runtime
construction.
"""

from __future__ import annotations
