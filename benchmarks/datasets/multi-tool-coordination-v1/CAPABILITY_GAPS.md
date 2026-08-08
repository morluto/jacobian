# Deferred capability and interface gaps

PR1 changes no product capability. The frozen trajectories support these
candidate gaps for PR2 calibration and possible PR3 treatment:

- `math.run` returns durable `artifact://` URIs in structured output but no MCP
  resource links, so Codex cannot reliably read and persist a returned
  verification-record payload.
- `matrix.normal_form.hermite` tells the agent to verify separately, while its
  discovery view advertises no related
  `matrix.normal_form.hermite.verify` capability.
- useful raw structured graph operations can be missed after an artifact-only
  graph route is surfaced.
- `reasoning.write` exposes phase-dependent optional fields in one schema;
  repeated runs supplied a caller-invented `call_id` to `BEFORE_TOOL` before
  recovering.

These are observations, not accepted implementation proposals. No capability
will change until PR2 freezes unchanged evaluation tasks and confirms which
gap recurs under calibrated mixed difficulty.
