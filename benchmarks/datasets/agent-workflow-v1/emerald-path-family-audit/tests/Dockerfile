FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="87be2737ea0c24b80790d401547ff19139f4937a67a489cb2a7b43fe1e7b4614"
LABEL jacobian.task="jacobian/emerald-path-family-audit"
COPY expected.json input.json test.sh verifier.py verifier_support.py public_contract.json /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/emerald-path-family-audit"'
