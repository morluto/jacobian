FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
RUN python -m pip install --no-cache-dir attrs==26.1.0 jsonschema==4.26.0 jsonschema-specifications==2025.9.1 referencing==0.37.0 rpds-py==2026.6.3 typing-extensions==4.16.0
LABEL jacobian.checksum="87e0b0098e84b0c66ed59bf218df3faa6ca7d4575824194b425c5ae8f39c4aa5"
LABEL jacobian.task="jacobian/metamath-syllogism-repair"
COPY expected.json input.json test.sh verifier.py verifier_support.py proof_replay.py public_contract.json /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
