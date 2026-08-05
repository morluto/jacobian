FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="c5bbe25d4c957c74ba6315a33517d13999c938e9a3da7ea4a85cf212281ea303"
LABEL jacobian.task="jacobian/prime-power-divisibility-gap-audit"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
