FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.task="jacobian/illumination-strictness-audit" jacobian.checksum="66d9cd09a4c23e3f4dc9259cffdf4067a4fde657cf6963788578743a69fd692b"
RUN python -m pip install --no-cache-dir attrs==26.1.0 jsonschema==4.26.0 jsonschema-specifications==2025.9.1 referencing==0.37.0 rpds-py==2026.6.3 typing-extensions==4.16.0
COPY input.json public_contract.json test.sh verifier.py verifier_contract.json verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
