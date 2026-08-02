FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
# Exact radical-elimination audit verifier.
LABEL jacobian.task="jacobian/radical-system-uniqueness-audit" \
      jacobian.checksum="c91958559099aa6aec573d3070c9b220cdbb4692d7cb6935c84971fe3d926de9"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/radical-system-uniqueness-audit"'
