#!/bin/sh
set -eu
mkdir -p /app/evidence
python /app/spike.py \
  --python-executable /usr/local/bin/python \
  --wheel /opt/provider/regina-7.4.1-cp312-cp312-manylinux_2_28_x86_64.whl \
  --source-archive /opt/provider/regina-source.tar.gz \
  --pin /app/pin.json \
  --output /app/evidence/provider-report.json
python /solution/solve.py
