FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
LABEL jacobian.task="jacobian/agent-workflow-v1-well-total-domination-counterexample" \
      jacobian.checksum="f835ba12422a127fc1fb9a3c3cc80c2722f9f0d052ed0904cc8323e52f387398"
COPY expected.json agent-workflow-v1-well-total-domination-counterexample-input.json test.sh verifier.py verifier_support.py /tests/
COPY agent-workflow-v1-well-total-domination-counterexample-input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/agent-workflow-v1-well-total-domination-counterexample"'
