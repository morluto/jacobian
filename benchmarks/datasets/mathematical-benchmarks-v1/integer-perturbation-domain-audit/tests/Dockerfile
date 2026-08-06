FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
# Exact semantic verifier for an integer-to-natural perturbation-domain mismatch.
LABEL jacobian.task="jacobian/integer-perturbation-domain-audit" \
      jacobian.checksum="2966e9c203e433f9b01a850a6676b83cb8f5250b5988ffe5930b7d1a63643c11"
COPY expected.json input.json test.sh verifier.py verifier_support.py public_contract.json /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
