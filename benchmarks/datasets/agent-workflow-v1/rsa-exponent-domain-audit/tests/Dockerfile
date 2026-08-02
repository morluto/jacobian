FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="90bb9de59aadb3be7d39dc6ce63d738934a4fb4af2043c90b7841d4eefb02848"
LABEL jacobian.task="jacobian/rsa-exponent-domain-audit"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/rsa-exponent-domain-audit"'
