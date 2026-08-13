from __future__ import annotations

from pathlib import Path

import pytest
from tests.boundary.providers.external_sat.external_sat_support import (
    open_verified_unsat_services,
)
from tests.support.operations import invoke_operation as _invoke

from jacobian.contracts.results import ExecutionStatus
from jacobian.contracts.sat import SatProofArtifact
from jacobian.providers.external_solver_runtime import (
    CADICAL_VERSION,
    DRAT_TRIM_RELEASE_TAG,
    cadical_provider_runtime,
    drat_trim_provider_runtime,
)


def test_cadical_text_proof_replays_in_pinned_drat_trim(
    tmp_path: Path,
) -> None:
    cadical = cadical_provider_runtime()
    drat_trim = drat_trim_provider_runtime()
    if cadical.version != CADICAL_VERSION:
        pytest.skip(f"requires pinned CaDiCaL {CADICAL_VERSION}")
    if drat_trim.version != DRAT_TRIM_RELEASE_TAG:
        pytest.skip(f"requires pinned DRAT-trim {DRAT_TRIM_RELEASE_TAG}")
    with open_verified_unsat_services(tmp_path / "state") as runtime:
        cnf = runtime.core.sat.put_cnf(
            variable_names=(
                "p1h1",
                "p1h2",
                "p2h1",
                "p2h2",
                "p3h1",
                "p3h2",
            ),
            clauses=(
                (1, 2),
                (3, 4),
                (5, 6),
                (-1, -3),
                (-1, -5),
                (-3, -5),
                (-2, -4),
                (-2, -6),
                (-4, -6),
            ),
        )

        produced = _invoke(
            runtime,
            "sat.unsat_proof.find",
            {
                "cnf_uri": cnf.artifact_uri,
                "resource_budget": {"wall_seconds": 5},
            },
        )
        assert produced.execution.status is ExecutionStatus.COMPLETED
        assert produced.output["status"] == "PROOF_PRODUCED"
        proof_uri = produced.output["proof_uri"]

        verified = _invoke(
            runtime,
            "sat.unsat_proof.verify",
            {"proof_uri": proof_uri},
        )
        assert verified.execution.status is ExecutionStatus.COMPLETED
        assert verified.output["status"] == "VERIFIED_UNSAT"
        assert verified.output["conclusion"] == "TRUE"
        assert verified.verification_record_uri is not None

        stored_proof = SatProofArtifact.model_validate(
            runtime.core.store.get(proof_uri).payload
        )
        empty_proof = runtime.core.sat.put_proof(
            cnf_uri=cnf.artifact_uri,
            proof=b"",
            producer=stored_proof.producer,
            resource_budget=stored_proof.resource_budget,
        )
        rejected = _invoke(
            runtime,
            "sat.unsat_proof.verify",
            {"proof_uri": empty_proof.artifact_uri},
        )
        assert rejected.output["status"] == "REJECTED"
        assert rejected.output["conclusion"] == "UNKNOWN"
        assert rejected.verification_record_uri is None

        raw_proof = stored_proof.raw_bytes()
        assert raw_proof.endswith(b"0\n")
        unsupported_contradiction = runtime.core.sat.put_proof(
            cnf_uri=cnf.artifact_uri,
            proof=b"0\n",
            producer=stored_proof.producer,
            resource_budget=stored_proof.resource_budget,
        )
        unsupported_replay = _invoke(
            runtime,
            "sat.unsat_proof.verify",
            {"proof_uri": unsupported_contradiction.artifact_uri},
        )
        assert unsupported_replay.output["status"] == "REJECTED"
        assert unsupported_replay.output["conclusion"] == "UNKNOWN"
        assert unsupported_replay.verification_record_uri is None

        concatenated_proof = runtime.core.sat.put_proof(
            cnf_uri=cnf.artifact_uri,
            proof=raw_proof + b"1 0\n",
            producer=stored_proof.producer,
            resource_budget=stored_proof.resource_budget,
        )
        concatenated_replay = _invoke(
            runtime,
            "sat.unsat_proof.verify",
            {"proof_uri": concatenated_proof.artifact_uri},
        )
        assert concatenated_replay.output["status"] == "REJECTED"
        assert concatenated_replay.output["conclusion"] == "UNKNOWN"
        assert concatenated_replay.verification_record_uri is None

        satisfiable = runtime.core.sat.put_cnf(
            variable_names=("x",),
            clauses=((1,),),
        )
        cross_bound = runtime.core.sat.put_proof(
            cnf_uri=satisfiable.artifact_uri,
            proof=stored_proof.raw_bytes(),
            producer=stored_proof.producer,
            resource_budget=stored_proof.resource_budget,
        )
        cross_replay = _invoke(
            runtime,
            "sat.unsat_proof.verify",
            {"proof_uri": cross_bound.artifact_uri},
        )
        assert cross_replay.output["status"] == "REJECTED"
        assert cross_replay.output["conclusion"] == "UNKNOWN"
        assert cross_replay.verification_record_uri is None
