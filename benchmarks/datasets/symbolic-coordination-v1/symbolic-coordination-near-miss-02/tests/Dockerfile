FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="cb8a11da9e91e49ac3b9376577832fe719b9dcd75ea90ca6feb7278e540e6ca3"
RUN python -m pip install --no-cache-dir attrs==26.1.0 jsonschema==4.26.0 jsonschema-specifications==2025.9.1 referencing==0.37.0 rpds-py==2026.6.3 typing-extensions==4.16.0
COPY verifier.py verifier_support.py public_contract.json input.json test.sh /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
