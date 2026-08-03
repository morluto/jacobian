FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
LABEL jacobian.checksum="a48ef57370e3643272f39ba732a9953d13e0af19a71f96a28ee9efe79de27ab4"
WORKDIR /tests
COPY verifier.py verifier_support.py input.json expected.json test.sh /tests/
RUN chmod +x /tests/test.sh
ENV PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["/tests/test.sh"]
