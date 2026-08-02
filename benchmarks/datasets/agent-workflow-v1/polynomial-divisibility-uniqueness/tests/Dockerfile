FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="022704b9996683b75c27afd9f6606662118fbf960445b60f99ded9016cf48730"
LABEL jacobian.task="jacobian/polynomial-divisibility-uniqueness"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/polynomial-divisibility-uniqueness"'
