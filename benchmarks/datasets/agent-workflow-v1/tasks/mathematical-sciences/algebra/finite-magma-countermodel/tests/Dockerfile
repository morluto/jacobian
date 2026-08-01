FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
LABEL jacobian.task="jacobian/agent-workflow-v1-finite-magma-countermodel" \
      jacobian.checksum="605ffe3995dc3422112e366a6a50da63165d92e1e7b73c1ed324c280028ecc62"
COPY expected.json agent-workflow-v1-finite-magma-countermodel-input.json test.sh verifier.py verifier_support.py /tests/
COPY agent-workflow-v1-finite-magma-countermodel-input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/agent-workflow-v1-finite-magma-countermodel"'
