FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="608d1ffba6797e3620755ab315647503067874bac027da6dbc40c0da780e472d"
LABEL jacobian.task="jacobian/prime-power-divisibility-gap-audit"
COPY expected.json input.json test.sh verifier.py verifier_support.py public_contract.json /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
