FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
RUN python -m pip install --no-cache-dir attrs==26.1.0 jsonschema==4.26.0 jsonschema-specifications==2025.9.1 referencing==0.37.0 rpds-py==2026.6.3 typing-extensions==4.16.0
# Exact semantic verifier for an integer-to-natural perturbation-domain mismatch.
LABEL jacobian.task="jacobian/integer-perturbation-domain-audit" \
      jacobian.checksum="12e3d17839765b958111051e7c1d1f4bf98e98c9ea4e8c0940cc9234071b09f1"
COPY expected.json input.json test.sh verifier.py verifier_support.py public_contract.json /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
