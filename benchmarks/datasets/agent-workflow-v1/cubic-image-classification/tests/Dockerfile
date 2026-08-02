FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="9ce11b14414d6b1e4777609379327cd44d327864ae9b7edd76dd86eb380d8e64"
LABEL jacobian.task="jacobian/cubic-image-classification"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/cubic-image-classification"'
