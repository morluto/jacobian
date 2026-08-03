FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
LABEL jacobian.checksum="3ca8764e63f84f9b5b38ff1c4955fb31810339c5584f052bc09e4c2f695e5679"
# Exact finite-state potential and stopping-time verifier.
LABEL jacobian.task="jacobian/coin-process-potential"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
