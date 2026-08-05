FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="724d7be408d20a6c7db39e828e9959a572612d8a4deaeb8404dc90f2bfc7d1a2"
LABEL jacobian.task="jacobian/apollonius-gap-repair"
COPY expected.json input.json test.sh verifier.py verifier_support.py public_contract.json /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/apollonius-gap-repair"'
