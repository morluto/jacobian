FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="75bcff2bd56feb44e69a753d4bb9372c7d1ee2c3fc99799062acdf1d6fe7003d"
LABEL jacobian.task="jacobian/nonlinear-recurrence-crossing-certificate"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/nonlinear-recurrence-crossing-certificate"'
