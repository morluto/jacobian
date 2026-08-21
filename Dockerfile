FROM ghcr.io/astral-sh/uv:0.11.28-python3.12-trixie-slim

ARG JACOBIAN_REVISION=unknown
ARG JACOBIAN_VERSION=0+unknown
ARG JACOBIAN_SOURCE_DIRTY=false
ARG LEAN_VERSION=4.31.0
ARG LEAN_ARCHIVE_SHA256=07a633cc8d9151cbc08825ea4cdda50d4b02a2c9cb852c0131b13046f49cad7f
ARG MATHLIB_MANIFEST_SHA256=2e3e4f23e695c64bd3eac9d210a7e0aa6ce9a270495aaa10442a019ea303d679
LABEL org.opencontainers.image.source=https://github.com/morluto/jacobian
LABEL org.opencontainers.image.revision=$JACOBIAN_REVISION
LABEL org.opencontainers.image.version=$JACOBIAN_VERSION
LABEL io.jacobian.source-dirty=$JACOBIAN_SOURCE_DIRTY

WORKDIR /app

ARG SINGULAR_DEBIAN_VERSION=1:4.4.1+ds-2
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      git \
      libgmp10 \
      "singular=${SINGULAR_DEBIAN_VERSION}" \
      zstd \
    && rm -rf /var/lib/apt/lists/* \
    && test "$(Singular -q --execute 'system("version");quit;' | tr -d '[:space:]')" = "44100"

RUN mkdir -p /opt/lean \
    && curl -fsSL "https://github.com/leanprover/lean4/releases/download/v${LEAN_VERSION}/lean-${LEAN_VERSION}-linux.tar.zst" \
      -o /opt/lean/lean.tar.zst \
    && printf '%s  %s\n' "$LEAN_ARCHIVE_SHA256" /opt/lean/lean.tar.zst | sha256sum -c - \
    && tar --use-compress-program=unzstd -xf /opt/lean/lean.tar.zst -C /opt/lean \
    && rm /opt/lean/lean.tar.zst

ENV PATH=/opt/lean/lean-4.31.0-linux/bin:$PATH

COPY pyproject.toml uv.lock README.md ./
COPY lean ./lean

RUN cd lean \
    && lake update \
    && printf '%s  %s\n' "$MATHLIB_MANIFEST_SHA256" lake-manifest.json | sha256sum -c - \
    && lake exe cache get \
    && test -f .lake/packages/mathlib/.lake/build/lib/lean/Mathlib.olean

COPY src ./src

RUN uv sync --locked --no-dev

ENV JACOBIAN_MATHLIB_ROOT=/app/lean

EXPOSE 8000

ENTRYPOINT ["uv", "run", "--no-sync", "jacobian-remote-mcp"]
CMD ["--help"]
