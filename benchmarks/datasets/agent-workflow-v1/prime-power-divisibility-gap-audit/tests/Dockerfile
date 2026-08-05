FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="26c0808b685f1b8dc6ab4f2161bc5ca2f8f335dd30209a3482785ef6851affeb"
LABEL jacobian.task="jacobian/prime-power-divisibility-gap-audit"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
