FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
# Exact verifier for a strict-monotonicity missing-continuity audit.
LABEL jacobian.task="jacobian/monotone-inverse-continuity-audit" \
      jacobian.checksum="99c4c13afb73f041f404451a6ccce6f2f5d5fd08f191558b821cb994d01bee92"
COPY expected.json input.json public_contract.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/monotone-inverse-continuity-audit"'
