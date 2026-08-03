FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
LABEL jacobian.checksum="1adcfc820aa983639dd0737420bf95c4b34a870f340462b5305ee8ee7a7486b8"
# Exact integral chain-complex and lattice-index verifier.
LABEL jacobian.task="jacobian/rp2-homology-lattice"
COPY expected.json input.json test.sh verifier.py verifier_support.py /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
