FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
RUN python -m pip install --no-cache-dir attrs==26.1.0 jsonschema==4.26.0 jsonschema-specifications==2025.9.1 referencing==0.37.0 rpds-py==2026.6.3 typing-extensions==4.16.0
LABEL jacobian.task="jacobian/rational-linear-solution" \
      jacobian.checksum="2845aa0f01709d7bc8844baf09ad916ba07d16aa33e33100e17f25e2f7f1db48"
COPY expected.json agent-workflow-v1-rational-linear-solution-input.json test.sh verifier.py verifier_support.py public_contract.json /tests/
COPY agent-workflow-v1-rational-linear-solution-input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/rational-linear-solution"'
