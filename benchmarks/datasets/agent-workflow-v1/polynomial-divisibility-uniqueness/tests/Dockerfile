FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="9077a806e1359c4bb17c9ae2c2da2da47ec397bfecb571c73dc902acf6b4e003"
LABEL jacobian.task="jacobian/polynomial-divisibility-uniqueness"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/polynomial-divisibility-uniqueness"'
