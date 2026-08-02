FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="924b40c173985845f2058a09a7f1ec1a4ba462eb35fb89924b0c5dc85bad9af8"
LABEL jacobian.task="jacobian/lean-transitive-axiom-audit"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/lean-transitive-axiom-audit"'
