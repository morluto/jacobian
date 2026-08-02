FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="c2dc05e6ee238d4e29382b75ccc4ecea6b1de537f6ae84ba9da58cf1df94883b"
LABEL jacobian.checksum="placeholder"
LABEL jacobian.task="jacobian/cyclotomic-reciprocity-certificate"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
