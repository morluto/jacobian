"""MCP transport boundary test configuration.

Only ``attached_complete_runtime`` is needed by this lane today
(``test_mcp_inspection_relationship_compaction``). Fresh/authorized complete
fixtures stay out of the MCP conftest so the lane does not hydrate unused
portfolio templates.
"""

from __future__ import annotations

from tests.support.complete_runtime_fixtures import (
    attached_complete_runtime,
    complete_portfolio_template,
)

__all__ = (
    "attached_complete_runtime",
    "complete_portfolio_template",
)
