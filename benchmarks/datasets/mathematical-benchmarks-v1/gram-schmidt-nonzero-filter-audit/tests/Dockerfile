FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="25fe39f24a874be4a5b06f72775e7e7eeef810a8d1ec3682d8fa291759e6b543"
RUN python -m pip install --no-cache-dir attrs==26.1.0 jsonschema==4.26.0 jsonschema-specifications==2025.9.1 referencing==0.37.0 rpds-py==2026.6.3 typing-extensions==4.16.0
COPY test.sh verifier.py verifier_support.py public_contract.json input.json expected.json /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
