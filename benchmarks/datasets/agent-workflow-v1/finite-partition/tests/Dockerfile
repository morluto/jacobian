FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
LABEL jacobian.task="jacobian/finite-partition" \
      jacobian.checksum="98a64529712eb9b6b04e44757b5353289bf853b2e0aec07514b1c657a6124121"
COPY authorized_record.json expected.json agent-workflow-v1-finite-partition-input.json test.sh verifier.py verifier_support.py /tests/
COPY agent-workflow-v1-finite-partition-input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/finite-partition"'
