"""Native solver crashes must remain inside request-owned tool workers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap


def test_concurrent_z3_tools_do_not_load_z3_in_the_service_host() -> None:
    script = textwrap.dedent(
        """
        import json
        import sys
        from concurrent.futures import ThreadPoolExecutor

        from jacobian.catalog.catalog import Catalog

        catalog = Catalog.open()
        maximum_cut = catalog.operation("graph.cut.maximum.compute")
        discrepancy = catalog.operation("discrepancy.theory.optimum.compute")
        assert maximum_cut is not None
        assert discrepancy is not None

        maximum_cut_payload = {
            "graph": {
                "vertices": [str(index) for index in range(12)],
                "edges": sorted(
                    {
                        tuple(sorted((str(index), str((index + 1) % 12))))
                        for index in range(12)
                    }
                    | {
                        tuple(sorted((str(index), str((index + 3) % 12))))
                        for index in range(12)
                    }
                ),
            }
        }
        discrepancy_payload = {
            "set_system": {
                "ground_set_size": 15,
                "sets": [
                    sorted({index, (index + step) % 15, (index + 2 * step) % 15})
                    for step in range(1, 6)
                    for index in range(15)
                ],
            }
        }

        def invoke(index):
            operation, payload = (
                (maximum_cut, maximum_cut_payload)
                if index % 2
                else (discrepancy, discrepancy_payload)
            )
            request = operation.request_type.model_validate(payload)
            return operation.run(request).model_dump(mode="json")

        with ThreadPoolExecutor(max_workers=16) as executor:
            results = tuple(executor.map(invoke, range(16)))

        assert all(result for result in results)
        assert "z3" not in sys.modules
        print(json.dumps({"results": len(results), "host_loaded_z3": False}))
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env={**os.environ, "SYMPY_GROUND_TYPES": "python"},
        text=True,
        timeout=180,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "results": 16,
        "host_loaded_z3": False,
    }
