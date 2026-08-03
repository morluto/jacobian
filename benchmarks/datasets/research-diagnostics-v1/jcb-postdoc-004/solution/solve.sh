#!/bin/sh
set -eu
mkdir -p /app/evidence
cp /solution/submission.json /app/submission.json
cp /solution/counterexample.json /app/evidence/counterexample.json
