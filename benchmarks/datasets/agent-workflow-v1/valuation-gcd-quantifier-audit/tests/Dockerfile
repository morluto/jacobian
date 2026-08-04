FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="9ca4b0c552078ef5ed0748dbe0237db90fb55a7f98c1337811f2680b10b6b5c9"
LABEL jacobian.task="jacobian/valuation-gcd-quantifier-audit"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
