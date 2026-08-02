FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="9321b7a1f599d7e0c33e09f53fa3892ebaa81ce13c18030fe356aeeba17a9d25"
LABEL jacobian.task="jacobian/polynomial-divisibility-uniqueness"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/polynomial-divisibility-uniqueness"'
