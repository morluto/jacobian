FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="1efa89b44fadd20c73ad52c6cb615a0e0653aec9fc9a93926bf0acf3bfe46241"
LABEL jacobian.task="jacobian/exact-farkas-ldl-slice"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
