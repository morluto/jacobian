FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="f6838e1242f27211fc622f4093ee9e7633c26e40989a17437eaaf28f4b887595"
LABEL jacobian.task="jacobian/valuation-gcd-quantifier-audit"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
