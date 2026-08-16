FROM ghcr.io/astral-sh/uv:0.11.28-python3.12-trixie-slim

ARG JACOBIAN_REVISION=unknown
ARG JACOBIAN_VERSION=0+unknown
ARG JACOBIAN_SOURCE_DIRTY=false
LABEL org.opencontainers.image.source=https://github.com/morluto/jacobian
LABEL org.opencontainers.image.revision=$JACOBIAN_REVISION
LABEL org.opencontainers.image.version=$JACOBIAN_VERSION
LABEL io.jacobian.source-dirty=$JACOBIAN_SOURCE_DIRTY

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY lean ./lean

RUN uv sync --locked --no-dev

EXPOSE 8000

ENTRYPOINT ["uv", "run", "--no-sync", "jacobian-remote-mcp"]
CMD ["--help"]
