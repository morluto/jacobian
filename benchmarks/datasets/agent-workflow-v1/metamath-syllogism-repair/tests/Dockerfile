FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
RUN python -m pip install --no-cache-dir attrs==26.1.0 jsonschema==4.26.0 jsonschema-specifications==2025.9.1 referencing==0.37.0 rpds-py==2026.6.3 typing-extensions==4.16.0
LABEL jacobian.checksum="80465dafb3de4907c83ccfa4a9ff0c02b5cb60eb2b8990c16ece1a0d66dce25f"
LABEL jacobian.task="jacobian/metamath-syllogism-repair"
COPY expected.json input.json test.sh verifier.py verifier_support.py proof_replay.py public_contract.json /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
