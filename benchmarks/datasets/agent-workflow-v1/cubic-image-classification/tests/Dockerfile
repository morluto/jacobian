FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="188d51930c7bafebb659afa2d9d75c9388616f39636c18216114d9823fef83f6"
LABEL jacobian.task="jacobian/cubic-image-classification"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/cubic-image-classification"'
