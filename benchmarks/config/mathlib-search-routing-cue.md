## Task-level retrieval cue (intervention)

Do not rely on a remembered Lean declaration name alone. Before submitting,
use the connected Jacobian MCP server's public `math.find` and `math.run`
interface to discover and execute an applicable declaration lookup. Select a
declaration whose reported type matches the frozen goal. Do not substitute a
filesystem search for this lookup. This cue does not reveal an operation ID or
change the requested proof shape or verifier: correctness still depends only
on whether the submitted exact application elaborates.
