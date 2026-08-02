FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="f7f587be759f4f28aa08e0b74c38b39e52881969c1a402ac73f207cf0983b7de"
LABEL jacobian.task="jacobian/subset-incidence-determinant"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/subset-incidence-determinant"'
