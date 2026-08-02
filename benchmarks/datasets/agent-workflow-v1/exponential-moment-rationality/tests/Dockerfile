FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="7f390d88c36509a160e93bd5cc805829c099cdc16d2fd64083e50f7168b09388"
LABEL jacobian.task="jacobian/exponential-moment-rationality"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
