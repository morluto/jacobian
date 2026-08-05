FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="7c7caedf2e8f9fab8f8bf8ecd8020b1e4e61cbd168448eebf31dedd8ca6d3ce8"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
RUN chmod +x /tests/test.sh
