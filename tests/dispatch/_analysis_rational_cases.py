"""Inventory of ordinary matrix-analysis rational request positions."""

from dataclasses import dataclass
from typing import Any

from tests.dispatch._rational_request_cases import dense, ratio


@dataclass(frozen=True)
class AnalysisRationalCase:
    operation: str
    position: str

    def payload(self, value: dict[str, Any]) -> dict[str, Any]:
        if "completion" in self.operation:
            return {
                "matrix": {
                    "graph": {"vertex_count": 1, "edges": []},
                    "specified_entries": [{"row": 0, "column": 0, "value": value}],
                }
            }
        payload: dict[str, Any] = {"matrix": dense(value)}
        if "collatz" in self.operation:
            payload["vector"] = [ratio("1", "1")]
            if self.position == "vector":
                payload["matrix"] = dense(ratio("1", "1"))
                payload["vector"] = [value]
        if "decompose" in self.operation:
            payload["graph"] = {"vertex_count": 1, "edges": []}
        return payload


ANALYSIS_RATIONAL_CASES = (
    AnalysisRationalCase("matrix.inertia.compute", "matrix"),
    AnalysisRationalCase("matrix.collatz_wielandt.quotient_profile.compute", "matrix"),
    AnalysisRationalCase("matrix.collatz_wielandt.quotient_profile.compute", "vector"),
    AnalysisRationalCase("matrix.chordal_psd.decompose", "matrix"),
    AnalysisRationalCase("matrix.chordal_psd_completion.compute", "specified_entries"),
)
