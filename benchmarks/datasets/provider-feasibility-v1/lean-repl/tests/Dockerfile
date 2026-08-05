FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="ebfda507aca2e042afbd7e346c25a2b75cd0ee14cc4c5d2bf107ffa1f47f56c9"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
RUN chmod +x /tests/test.sh
