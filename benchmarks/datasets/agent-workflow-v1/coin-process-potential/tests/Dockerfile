FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
LABEL jacobian.checksum="60fc80578c3015bcdd767d2c373e1b953b6b45275524b4d4592605a5aac86714"
# Exact finite-state potential and stopping-time verifier.
LABEL jacobian.task="jacobian/coin-process-potential"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
