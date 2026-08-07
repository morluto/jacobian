FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
RUN python -m pip install --no-cache-dir attrs==26.1.0 jsonschema==4.26.0 jsonschema-specifications==2025.9.1 referencing==0.37.0 rpds-py==2026.6.3 typing-extensions==4.16.0
LABEL jacobian.checksum="5ad96c74b9ba96d5891d4e1d5de6bc95b50fb7d7d7b7cfb829bcd1d54c009244"
COPY expected.json input.json test.sh verifier.py verifier_support.py public_contract.json /tests/
RUN chmod +x /tests/test.sh
