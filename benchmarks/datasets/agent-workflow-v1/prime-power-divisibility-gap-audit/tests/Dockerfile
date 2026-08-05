FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="8874a329c376849a53abc2fc75a638fd116129dc293d901ab60bfe00290a546c"
LABEL jacobian.task="jacobian/prime-power-divisibility-gap-audit"
COPY expected.json input.json test.sh verifier.py verifier_support.py public_contract.json /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
