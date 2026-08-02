FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
# Bounded exhaustive audit verifier.
LABEL jacobian.task="jacobian/calendar-good-days-audit" \
      jacobian.checksum="4be90105d3f9cf5e552f68eab3471dab45659554c7bab4a5e160e4f28ce7fecd"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/calendar-good-days-audit"'
