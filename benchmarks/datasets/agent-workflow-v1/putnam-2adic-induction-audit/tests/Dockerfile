FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="d3f883a8b37f87a727225fe10d5a3d5124eb66865f789dc28bb2866bd36b1fde"
LABEL jacobian.task="jacobian/putnam-2adic-induction-audit"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/putnam-2adic-induction-audit"'
