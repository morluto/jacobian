#!/bin/sh
set -eu
mkdir -p /app/evidence
cp /solution/answer.txt /app/evidence/answer.txt
python - <<'PY'
import hashlib, json
from pathlib import Path
submission = json.loads(Path('/solution/submission.json').read_text())
submission['evidence'][0]['sha256'] = 'sha256:' + hashlib.sha256(Path('/app/evidence/answer.txt').read_bytes()).hexdigest()
Path('/app/submission.json').write_text(json.dumps(submission, separators=(',', ':')))
PY
