FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="ed202b8009a9a23d980d6e90ea4baeeefe553884e35c20953b7232a9031f94ca"
LABEL jacobian.task="jacobian/apollonius-gap-repair"
COPY expected.json input.json test.sh verifier.py verifier_support.py public_contract.json /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/apollonius-gap-repair"'
