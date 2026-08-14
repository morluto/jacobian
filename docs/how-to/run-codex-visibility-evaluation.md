# Run the MCP visibility evaluation

This external diagnostic compares a no-Jacobian control with a treatment that
exposes only Jacobian's public MCP server. It measures discovery and use of
`math.find` and `math.run`; it does not replace mathematical task evaluation.

Freeze the prompt suite, model, environment, and endpoint configuration before
running either arm. Store the raw outputs with the evaluation's own metadata
and interpret them as research evidence only. The diagnostic must not add a
reasoning observer, retained workflow state, or new MCP capability to
Jacobian.
