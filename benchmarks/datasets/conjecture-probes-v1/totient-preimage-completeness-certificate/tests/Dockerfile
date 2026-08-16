FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.task="jacobian/totient-preimage-completeness-certificate" \
      jacobian.checksum="fdafc251371b2d1a156feea3edade4a3187433c0d30d78e1bcb4cfb10aec1c1b"
RUN python -m pip install --no-cache-dir attrs==26.1.0 jsonschema==4.26.0 jsonschema-specifications==2025.9.1 referencing==0.37.0 rpds-py==2026.6.3 typing-extensions==4.16.0
COPY input.json public_contract.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && python -c 'import json; assert json.load(open("/tests/input.json"))["task_id"] == "jacobian/totient-preimage-completeness-certificate"'
