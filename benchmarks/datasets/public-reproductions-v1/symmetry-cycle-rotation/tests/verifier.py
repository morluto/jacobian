import json
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    load_submission,
    normalize_reward_file,
    witness_list_is_bound,
)

W = Path("/app")
E = Path("/tests")


def _norm_edges(orbits):
    if not isinstance(orbits, list):
        return None
    norm: list[tuple[tuple[str, str], ...]] = []
    for orbit in orbits:
        if not isinstance(orbit, list):
            return None
        normalized_orbit: list[tuple[str, str]] = []
        for edge in orbit:
            if (
                not isinstance(edge, list)
                or len(edge) != 2
                or not all(type(vertex) is str for vertex in edge)
            ):
                return None
            normalized_orbit.append(tuple(sorted(edge)))
        norm.append(tuple(sorted(normalized_orbit)))
    return sorted(norm)


def _norm_vertices(orbits):
    return sorted([sorted(orbit) for orbit in orbits])


def _math(s, x, e):
    r = s.get("result", {})
    vo = r.get("vertex_orbits")
    eo = r.get("edge_orbits")
    if not isinstance(vo, list) or not isinstance(eo, list):
        return False
    return _norm_vertices(vo) == _norm_vertices(
        e["expected_vertex_orbits"]
    ) and _norm_edges(eo) == _norm_edges(e["expected_edge_orbits"])


def main():
    s = load_submission()
    x = json.loads(next(E.glob("*input*.json")).read_text())
    e = json.loads((E / "expected.json").read_text())
    math_correct = _math(s, x, e)
    correct = bool(math_correct)
    good = bool(witness_list_is_bound(s["witness"]))
    protocol_ok = s is not None
    reward = aggregate_reward(
        correctness=correct,
        witness_validity=good,
        protocol_ok=protocol_ok,
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "witness_validity": float(good),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
