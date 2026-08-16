FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
# Exhaustive finite-algebra and rational-point verifier.
LABEL jacobian.task="jacobian/finite-scheme-rational-points-audit" \
      jacobian.checksum="8dc0631ae5231c4a8f2ff56171ac90488917a10feeba2b01a4eb0443602d32e1"
RUN python -m pip install --no-cache-dir attrs==26.1.0 jsonschema==4.26.0 jsonschema-specifications==2025.9.1 referencing==0.37.0 rpds-py==2026.6.3 typing-extensions==4.16.0
COPY expected.json input.json test.sh verifier.py verifier_support.py public_contract.json verifier_contract.json /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
