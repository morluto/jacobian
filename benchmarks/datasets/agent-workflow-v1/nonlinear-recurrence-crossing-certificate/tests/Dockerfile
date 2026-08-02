FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="7112f7d2453462c240bc4ee5d5df4a5c41a8f2a6c9da323aa3c78fe79819f6ef"
LABEL jacobian.task="jacobian/nonlinear-recurrence-crossing-certificate"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/nonlinear-recurrence-crossing-certificate"'
