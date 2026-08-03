FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
LABEL jacobian.checksum="49e2316c3811d9f7723942650007c09606c902573d780ea54a9b6312ef456fe6"
WORKDIR /tests
COPY verifier.py verifier_support.py input.json expected.json test.sh /tests/
RUN chmod +x /tests/test.sh
ENV PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["/tests/test.sh"]
