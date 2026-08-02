FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="ee4c0fdcd663e9ca3fd4291e9daae9372252d6c5f256a7ccc820aa31a0871837"
LABEL jacobian.task="jacobian/rsa-exponent-domain-audit"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/rsa-exponent-domain-audit"'
