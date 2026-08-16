FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
RUN python -m pip install --no-cache-dir attrs==26.1.0 jsonschema==4.26.0 jsonschema-specifications==2025.9.1 referencing==0.37.0 rpds-py==2026.6.3 typing-extensions==4.16.0
LABEL jacobian.checksum="39426242a4c3fa249ec1e4c9e15b94965fe6a64636ad63c2afab05895203cba2"
LABEL jacobian.task="jacobian/apollonius-gap-repair"
COPY expected.json input.json test.sh verifier.py verifier_support.py public_contract.json /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/apollonius-gap-repair"'
