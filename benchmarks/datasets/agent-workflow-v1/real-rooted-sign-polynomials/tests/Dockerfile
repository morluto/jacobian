FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="9a2c22ca43ef035240fd725dd55f3459dffe1c8374ae54f2323b1a567fa17afd"
LABEL jacobian.task="jacobian/real-rooted-sign-polynomials"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
