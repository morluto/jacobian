from __future__ import annotations

from pathlib import Path

from jacobian.contracts.capabilities import CapabilityRequest
from jacobian.runtime import create_runtime

# Composition-lane admission category for architecture ratchets.
COMPOSITION_ADMISSION = "WIRING"


def test_external_adapter_loads_from_an_operator_entrypoint(tmp_path: Path) -> None:
    with create_runtime(
        tmp_path,
        capability_adapter_entrypoints=(
            "tests.component.capabilities._fixture_capabilities:create_adapter",
        ),
    ) as runtime:
        result = runtime.core.capabilities.invoke(
            CapabilityRequest(
                capability_id="fixture.increment",
                input={"value": 4},
            )
        )
        assert result.output == {"value": 5}
