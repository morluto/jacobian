FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
RUN python -m pip install --no-cache-dir attrs==26.1.0 jsonschema==4.26.0 jsonschema-specifications==2025.9.1 referencing==0.37.0 rpds-py==2026.6.3 typing-extensions==4.16.0
LABEL jacobian.task="jacobian/sat-witness" \
      jacobian.checksum="b17df843cb1839fd9680ba97b7787743d97dfe49debf365605c9917bbebfcb11"
COPY authorized_records.json expected.json agent-workflow-v1-sat-witness-input.json test.sh verifier.py verifier_support.py public_contract.json /tests/
COPY agent-workflow-v1-sat-witness-input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/sat-witness"'
