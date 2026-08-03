FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
LABEL jacobian.checksum="d8ca69b0481cc8d131c5fdfa789a9df058a46e4d7847dfc85317f802499d605f"
# Exact finite-state potential and stopping-time verifier.
LABEL jacobian.task="jacobian/coin-process-potential"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
