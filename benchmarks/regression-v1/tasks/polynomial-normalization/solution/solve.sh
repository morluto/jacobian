#!/bin/sh
set -eu
mkdir -p /app/evidence
cp /solution/submission.json /app/submission.json
cp /solution/answer.txt /app/evidence/answer.txt
cp /solution/verification-record.json /app/evidence/verification-record.json
