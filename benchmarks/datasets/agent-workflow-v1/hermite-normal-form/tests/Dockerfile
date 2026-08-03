FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
LABEL jacobian.task="jacobian/hermite-normal-form" \
      jacobian.checksum="0314d8b282c1efe803755cc542f270485ab593e3860c30d5cc3ae159b1a51698"
COPY authorized_record.json expected.json agent-workflow-v1-hermite-normal-form-input.json test.sh verifier.py verifier_support.py /tests/
COPY agent-workflow-v1-hermite-normal-form-input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/hermite-normal-form"'
