FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="5c72a2d296e082a74f36ab7e454e0af7f9d2f3befa90cef9667a439061a8bc4e"
LABEL jacobian.task="jacobian/cubic-image-classification"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/cubic-image-classification"'
