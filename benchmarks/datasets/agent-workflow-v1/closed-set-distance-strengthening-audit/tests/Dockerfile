FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="0753423221afe4ffd5fa48d6a06f00fbd0e447d7acec1019c9b48d5aba13c0f7"
LABEL jacobian.task="jacobian/closed-set-distance-strengthening-audit"
COPY expected.json input.json public_contract.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
