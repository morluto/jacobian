FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
RUN python -m pip install --no-cache-dir attrs==26.1.0 jsonschema==4.26.0 jsonschema-specifications==2025.9.1 referencing==0.37.0 rpds-py==2026.6.3 typing-extensions==4.16.0
LABEL jacobian.checksum="0dfe5a61530d60960f11a7f1353b6d6a069bb1ef3273eaa7d72bbba61b3d4032"
COPY expected.json input.json test.sh verifier.py verifier_support.py public_contract.json /tests/
RUN chmod +x /tests/test.sh
