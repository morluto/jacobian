FROM gcc:14.2.0-bookworm@sha256:b99b86a28812b1e6453a231a947dc43d76fe192788a12f344a9b568bf9f5d24c AS cake_lpr_build

WORKDIR /cake-lpr

ADD --checksum=sha256:8e30d84fdcb2177aa5571d7fa6661a2fae5ecfd56baa0ce49c65f9233a9f87cb \
  https://raw.githubusercontent.com/tanyongkiam/cake_lpr/a36874a8b750b43fe4b385b8ddbf5b033e46a3fa/basis_ffi.c \
  basis_ffi.c
ADD --checksum=sha256:2f3af32d55083839b3fa0e693afd817679c0b8944bef41def05a8b0ec72b7d4a \
  https://raw.githubusercontent.com/tanyongkiam/cake_lpr/a36874a8b750b43fe4b385b8ddbf5b033e46a3fa/cake_lpr.S \
  cake_lpr.S
ADD --checksum=sha256:eb1e4ff71900d55d384ff18afad5aae48417ce0fff25231565862c42b45e5dbc \
  https://raw.githubusercontent.com/tanyongkiam/cake_lpr/a36874a8b750b43fe4b385b8ddbf5b033e46a3fa/LICENSE \
  LICENSE.cake_lpr

RUN gcc -O2 basis_ffi.c cake_lpr.S -o cake_lpr -std=c99 \
    && printf '%s\n' \
      'format=jacobian.cake-lpr/v1' \
      'upstream_commit=a36874a8b750b43fe4b385b8ddbf5b033e46a3fa' \
      'basis_ffi.c=8e30d84fdcb2177aa5571d7fa6661a2fae5ecfd56baa0ce49c65f9233a9f87cb' \
      'cake_lpr.S=2f3af32d55083839b3fa0e693afd817679c0b8944bef41def05a8b0ec72b7d4a' \
      > cake-lpr.manifest

FROM ghcr.io/astral-sh/uv:0.11.28-python3.12-trixie-slim@sha256:3137a0b606f65a74ee0245f43dae219b09e8af98fc37fef20841cbceef35a646

ARG JACOBIAN_REVISION=unknown
ARG JACOBIAN_VERSION=0+unknown
ARG JACOBIAN_SOURCE_DIRTY=false
LABEL org.opencontainers.image.source=https://github.com/morluto/jacobian
LABEL org.opencontainers.image.revision=$JACOBIAN_REVISION
LABEL org.opencontainers.image.version=$JACOBIAN_VERSION
LABEL io.jacobian.source-dirty=$JACOBIAN_SOURCE_DIRTY

WORKDIR /app

COPY --from=cake_lpr_build /cake-lpr/cake_lpr /usr/local/bin/cake_lpr
COPY --from=cake_lpr_build /cake-lpr/cake-lpr.manifest /usr/local/share/jacobian/cake-lpr.manifest
COPY --from=cake_lpr_build /cake-lpr/LICENSE.cake_lpr /usr/local/share/licenses/cake_lpr/LICENSE

ARG SINGULAR_DEBIAN_VERSION=1:4.4.1+ds-2
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      "singular=${SINGULAR_DEBIAN_VERSION}" \
    && rm -rf /var/lib/apt/lists/* \
    && test "$(Singular -q --execute 'system("version");quit;' | tr -d '[:space:]')" = "44100"

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY lean ./lean

RUN uv sync --locked --no-dev

EXPOSE 8000

ENTRYPOINT ["uv", "run", "--no-sync", "jacobian-remote-mcp"]
CMD ["--help"]
