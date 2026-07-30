"""Render the explicitly supported environment fields in a Harbor job template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MODEL_PLACEHOLDER = "${JACOBIAN_MODEL}"


def render_job_config(config: dict[str, Any], *, model: str) -> dict[str, Any]:
    """Return a copy with every explicit model placeholder resolved."""

    normalized_model = model.strip()
    if not normalized_model:
        raise ValueError("model must be non-empty")
    rendered = json.loads(json.dumps(config))
    agents = rendered.get("agents")
    if not isinstance(agents, list):
        raise ValueError("job config must contain an agents list")
    replacements = 0
    for agent in agents:
        if isinstance(agent, dict) and agent.get("model_name") == MODEL_PLACEHOLDER:
            agent["model_name"] = normalized_model
            replacements += 1
    if replacements == 0:
        raise ValueError(f"job config does not contain {MODEL_PLACEHOLDER}")
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    config = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("job config must be a JSON object")
    rendered = render_job_config(config, model=args.model)
    args.output.write_text(
        json.dumps(rendered, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
