#!/bin/sh
set -eu
mkdir -p /app/evidence
cp /solution/submission.json /app/submission.json
cp /solution/certificate.json /app/evidence/certificate.json
cp /solution/answer.txt /app/answer.txt
