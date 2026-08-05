FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="a113dd58f2c8cf1019c50026cd1cbc453cfbf131aeeeca21229a02811aa8ea95"
LABEL jacobian.task="jacobian/emerald-path-family-audit"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/emerald-path-family-audit"'
