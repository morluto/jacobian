FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential ca-certificates curl libgmp-dev m4 && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/provider && \
    curl -fsSL https://github.com/cddlib/cddlib/releases/download/0.94n/cddlib-0.94n.tar.gz -o /opt/provider/cddlib.tar.gz && \
    curl -fsSL https://files.pythonhosted.org/packages/d2/25/85e2e17491a4d51f796f7212a13cef7d6d3b2967dfba104627a465224f06/pycddlib-3.0.2.tar.gz -o /opt/provider/pycddlib.tar.gz && \
    printf '%s  %s\n' b87ee07ba2c1d0ab92a3e4eccacdf568f981a095a392e3b9efd7e7e4a9e125b1 /opt/provider/cddlib.tar.gz | sha256sum -c - && \
    printf '%s  %s\n' 50f43f039818ab97c8cb44345eed9a365dae66789ce11cc8c95a74134a8907a7 /opt/provider/pycddlib.tar.gz | sha256sum -c -
RUN tar -xzf /opt/provider/cddlib.tar.gz -C /opt/provider && \
    cd /opt/provider/cddlib-0.94n && \
    ./configure --prefix=/usr/local && \
    make -j2 && make install && ldconfig && \
    CFLAGS=-I/usr/local/include LDFLAGS=-L/usr/local/lib uv pip install --system /opt/provider/pycddlib.tar.gz
WORKDIR /app
COPY benchmarks/__init__.py /opt/benchmarks/__init__.py
COPY benchmarks/tooling/__init__.py /opt/benchmarks/tooling/__init__.py
COPY benchmarks/tooling/command_runner.py /opt/benchmarks/tooling/command_runner.py
COPY benchmarks/tooling/spike_utils.py /opt/benchmarks/tooling/spike_utils.py
COPY tests/fixtures/providers/cddlib/input.json tests/fixtures/providers/cddlib/pin.json tests/fixtures/providers/cddlib/spike.py /app/
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/opt
