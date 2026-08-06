#!/bin/sh
set -eu
mkdir -p /app/evidence
cp /solution/answer.txt /app/evidence/answer.txt
python - <<'PY'
import hashlib, json
from pathlib import Path
source = json.loads(Path('/solution/submission.json').read_text())
digest = hashlib.sha256(Path('/app/evidence/answer.txt').read_bytes()).hexdigest()
source['evidence'][0]['sha256'] = 'sha256:' + digest
Path('/app/submission.json').write_text(json.dumps(source, separators=(',', ':')))
PY
