FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="0b6056a1b03d9fd60a1a5da891a6a5a8138d8bf2cae5c3e899a6d3611ebf132e"
LABEL jacobian.task="jacobian/emerald-path-family-audit"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/emerald-path-family-audit"'
