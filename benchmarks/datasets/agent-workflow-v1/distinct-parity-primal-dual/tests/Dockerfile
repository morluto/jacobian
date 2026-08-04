FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
# Exact verifier for a parity-constrained primal/dual certificate.
LABEL jacobian.task="jacobian/distinct-parity-primal-dual" \
      jacobian.checksum="72b4c33be80387166be0e75ffc9be24a8cc3cf8bfaed2914f10106623c5f506a"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/distinct-parity-primal-dual"'
