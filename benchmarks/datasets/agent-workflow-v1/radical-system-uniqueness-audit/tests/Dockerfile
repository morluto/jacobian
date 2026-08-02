FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
# Exact radical-elimination audit verifier.
LABEL jacobian.task="jacobian/radical-system-uniqueness-audit" \
      jacobian.checksum="f95799d589856ec28a9e30ba369ba819688c2dda31b5015fb6e7a96a44b1a2e3"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/radical-system-uniqueness-audit"'
