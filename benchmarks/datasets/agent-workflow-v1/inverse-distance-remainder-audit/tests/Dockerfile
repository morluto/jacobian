FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="06f7604a2ead62d622bb99a8f978dc28d739079419edd3820d345b85bf074e7d"
LABEL jacobian.task="jacobian/inverse-distance-remainder-audit"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/inverse-distance-remainder-audit"'
