FROM ghcr.io/astral-sh/uv:0.11.28-python3.12-trixie-slim@sha256:3137a0b606f65a74ee0245f43dae219b09e8af98fc37fef20841cbceef35a646

ARG JACOBIAN_REVISION=unknown
ARG JACOBIAN_VERSION=0+unknown
ARG JACOBIAN_SOURCE_DIRTY=false
LABEL org.opencontainers.image.source=https://github.com/morluto/jacobian
LABEL org.opencontainers.image.revision=$JACOBIAN_REVISION
LABEL org.opencontainers.image.version=$JACOBIAN_VERSION
LABEL io.jacobian.source-dirty=$JACOBIAN_SOURCE_DIRTY
ENV JACOBIAN_REVISION=$JACOBIAN_REVISION

WORKDIR /app

ARG SINGULAR_DEBIAN_VERSION=1:4.4.1+ds-2
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      "singular=${SINGULAR_DEBIAN_VERSION}" \
    && rm -rf /var/lib/apt/lists/* \
    && test "$(Singular -q --execute 'system("version");quit;' | tr -d '[:space:]')" = "44100"

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev

EXPOSE 8000

ENTRYPOINT ["uv", "run", "--no-sync", "jacobian-remote-mcp"]
CMD ["--help"]
