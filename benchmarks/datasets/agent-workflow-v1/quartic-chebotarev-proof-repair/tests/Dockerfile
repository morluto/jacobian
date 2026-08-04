FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="ea2efe0ed49aa43260351a7fab7c3b7a9aaa62ced59fc2ccdfc3803453061328"
LABEL jacobian.task="jacobian/quartic-chebotarev-proof-repair"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/quartic-chebotarev-proof-repair"'
