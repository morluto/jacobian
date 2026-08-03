FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="92940be96eb71a89a70068d92cbe61ce5503aadc2c547c35df35a82fc3cabd73"
LABEL jacobian.task="jacobian/reciprocal-polynomial-classification"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
