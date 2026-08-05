FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
# Exact finite semantic verifier for binder shadowing and subset-sum drift.
LABEL jacobian.task="jacobian/extremal-subset-sum-semantic-audit" \
      jacobian.checksum="504201199e3c4337a68e23f081ce608c98103aa10de877744c3e5893d70638c9"
COPY expected.json input.json test.sh verifier.py verifier_support.py public_contract.json /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/extremal-subset-sum-semantic-audit"'
