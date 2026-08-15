FROM ghcr.io/astral-sh/uv:0.8.4-python3.12-bookworm-slim@sha256:dc7e1d08f8ca979826ec0b68b31c783e0b35b568be6a078d4cbedf38c4cc085e
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential ca-certificates curl libboost-dev libgmp-dev libmpfr-dev xz-utils && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/provider && \
    curl -fsSL https://github.com/CGAL/cgal/releases/download/v6.2/CGAL-6.2-library.tar.xz -o /opt/provider/cgal.tar.xz && \
    printf '%s  %s\n' f75d2b2486842fb47222c780c7241c262d392802d0811143c4e9b539972ee55b /opt/provider/cgal.tar.xz | sha256sum -c - && \
    tar -xJf /opt/provider/cgal.tar.xz -C /opt/provider
WORKDIR /app
COPY benchmarks/__init__.py /opt/benchmarks/__init__.py
COPY benchmarks/tooling/__init__.py /opt/benchmarks/tooling/__init__.py
COPY benchmarks/tooling/command_runner.py /opt/benchmarks/tooling/command_runner.py
COPY benchmarks/tooling/spike_utils.py /opt/benchmarks/tooling/spike_utils.py
COPY tests/fixtures/providers/cgal/cgal_delaunay_spike.cpp tests/fixtures/providers/cgal/input.json tests/fixtures/providers/cgal/cgal_delaunay_pin.json tests/fixtures/providers/cgal/spike.py /app/
RUN g++ -std=c++17 -O2 -I/opt/provider/CGAL-6.2/include /app/cgal_delaunay_spike.cpp -lgmp -lmpfr -o /opt/provider/cgal-spike
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/opt
