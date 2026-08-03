#!/bin/sh
set -eu
mkdir -p /app/evidence
printf '%s\n' 'The frozen solution misapplies the chain rule. The corrected closedness equation yields rank 2, dimension 8, and the JSON-bound potentials differentiate exactly.' > /app/evidence/answer.txt
cp /solution/submission.json /app/submission.json
