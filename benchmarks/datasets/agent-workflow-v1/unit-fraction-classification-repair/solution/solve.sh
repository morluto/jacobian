#!/bin/sh
set -eu
mkdir -p /app/evidence
cp /solution/submission.json /app/submission.json
python - <<'PY'
import hashlib,json
from pathlib import Path
s=json.loads(Path('/app/submission.json').read_text()); e={'schema_version':'1','task_id':s['task_id'],'result':s['result'],'limitations':s['limitations']}; raw=json.dumps(e,separators=(',',':')).encode(); Path('/app/evidence/unit-fraction-repair.json').write_bytes(raw); s['evidence'][0]['sha256']='sha256:'+hashlib.sha256(raw).hexdigest(); Path('/app/submission.json').write_text(json.dumps(s,separators=(',',':')))
PY
cp /solution/answer.txt /app/answer.txt
