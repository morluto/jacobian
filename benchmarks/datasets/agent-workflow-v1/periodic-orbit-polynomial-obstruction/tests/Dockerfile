FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="2783b332575b6c4b7340ba27802dbd8df3895270235da23649cac36eef11da8e"
LABEL jacobian.task="jacobian/periodic-orbit-polynomial-obstruction"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && \
    python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/periodic-orbit-polynomial-obstruction"'
