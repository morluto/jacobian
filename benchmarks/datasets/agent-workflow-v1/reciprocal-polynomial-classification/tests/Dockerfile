FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
LABEL jacobian.checksum="db7a10246de064757522bf4f7d223633d45c524a8674b1a4361613d2c43d57f3"
LABEL jacobian.task="jacobian/reciprocal-polynomial-classification"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
