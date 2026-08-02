FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="ef189d753e56df8816153ad64690bd1ed12468b98eaf14150998ffe5897412c1"
LABEL jacobian.task="jacobian/nonlinear-recurrence-crossing-certificate"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/nonlinear-recurrence-crossing-certificate"'
