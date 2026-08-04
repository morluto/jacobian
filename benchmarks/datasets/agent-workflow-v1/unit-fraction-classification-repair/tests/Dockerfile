FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
# Exact verifier for a unit-fraction classification repair.
LABEL jacobian.task="jacobian/unit-fraction-classification-repair" \
      jacobian.checksum="4c709dc00ca65f24292afcf9fa06b4277297078d75050055a7a66de607fdb6bb"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/unit-fraction-classification-repair"'
