#!/bin/sh
set -eu
mkdir -p /app/evidence
cp /solution/submission.json /app/submission.json
cp /solution/syzygy-certificate.json /app/evidence/syzygy-certificate.json
