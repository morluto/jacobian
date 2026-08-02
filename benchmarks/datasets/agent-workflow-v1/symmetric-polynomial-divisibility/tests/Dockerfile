FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="fb2fe4c78185b7ca8480a6df010e3044bffd55ae43d2cfc2da33b35850aeb305"
LABEL jacobian.task="jacobian/symmetric-polynomial-divisibility"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
