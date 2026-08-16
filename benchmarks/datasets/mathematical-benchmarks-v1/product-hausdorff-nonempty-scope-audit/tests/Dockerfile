FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="da5a603cbbd38b23683a5df968d35bd6e371b3f9033fccae231f844d43fe3a24"
RUN python -m pip install --no-cache-dir attrs==26.1.0 jsonschema==4.26.0 jsonschema-specifications==2025.9.1 referencing==0.37.0 rpds-py==2026.6.3 typing-extensions==4.16.0
COPY test.sh verifier.py verifier_support.py public_contract.json input.json expected.json /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
