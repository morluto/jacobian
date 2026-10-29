"""CLI output follows the inspected wire schema, including nested aliases."""

from __future__ import annotations

import json

from jsonschema import Draft202012Validator
from typer.testing import CliRunner

from jacobian.cli import app


def test_cli_pauli_output_matches_inspection_and_composes_unchanged() -> None:
    runner = CliRunner()
    operation_id = "quantum.pauli.qubit.from_labels.compute"
    payload = {
        "register": {"qubit_ids": ["q0"]},
        "labels": ["Y"],
        "phase": 2,
    }

    inspected = runner.invoke(app, ["inspect", operation_id])
    produced = runner.invoke(app, ["run", operation_id, "--json", json.dumps(payload)])

    assert inspected.exit_code == produced.exit_code == 0
    output = json.loads(produced.stdout)["output"]
    Draft202012Validator(json.loads(inspected.stdout)["output_schema"]).validate(output)
    assert output["source"] == payload
    assert output["pauli"]["phase_free"]["register"] == payload["register"]
    recovered = runner.invoke(
        app,
        [
            "run",
            "quantum.pauli.qubit.to_labels.compute",
            "--json",
            json.dumps({"pauli": output["pauli"]}),
        ],
    )
    assert recovered.exit_code == 0
    assert json.loads(recovered.stdout)["output"]["labels"] == ["Y"]
    assert json.loads(recovered.stdout)["output"]["phase"] == 2
