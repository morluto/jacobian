FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
RUN python -m pip install --no-cache-dir attrs==26.1.0 jsonschema==4.26.0 jsonschema-specifications==2025.9.1 referencing==0.37.0 rpds-py==2026.6.3 sympy==1.14.0 typing-extensions==4.16.0
LABEL jacobian.checksum="8c491b56fd1a56bd917c45fe9b5371a860919e9e5ee311a3580ff492fecc213a"
LABEL jacobian.task="jacobian/fiber-dimension-semicontinuity-repair"
COPY expected.json input.json test.sh verifier.py verifier_support.py public_contract.json verifier_contract.json /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh && python -c 'import json; assert json.load(open("/tests/expected.json"))["task_id"] == "jacobian/fiber-dimension-semicontinuity-repair"'
