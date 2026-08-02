FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
LABEL jacobian.task="jacobian/well-total-domination-counterexample" \
      jacobian.checksum="5e9b8da4f20037fdfc41a8c5d0258d76d55fe27f5c681f17f16e6799ce4c8b92"
COPY expected.json agent-workflow-v1-well-total-domination-counterexample-input.json test.sh verifier.py verifier_support.py /tests/
COPY agent-workflow-v1-well-total-domination-counterexample-input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/well-total-domination-counterexample"'
