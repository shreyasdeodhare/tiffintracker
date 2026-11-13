set -Eeuo pipefail
PY=/var/tmp/tiffintrack-verify-venv/bin/python
"$PY" - <<'PY'
import os,subprocess
from dotenv import dotenv_values
values=dotenv_values('/home/support/tiffintrack/.env')
env=dict(os.environ)
env['WA_SESSION_KEY']=values['WA_SESSION_KEY']
subprocess.run(['runuser','-u','postgres','--preserve-environment','--','node','/home/support/tiffintrack-final-auth.mjs'],env=env,check=True)
PY
# Run the installed backup script with a separate protected output location.
runuser -u postgres -- env BACKUP_DIR=/var/backups/tiffintrack/final-acceptance-20261112 PGHOST=/var/run/postgresql PGUSER=postgres PGDATABASE=tiffintrack /bin/bash /usr/local/sbin/tiffintrack-backup
sha256sum -c /var/backups/tiffintrack/final-acceptance-20261112/tiffintrack-*.dump.sha256
stat -c '%a %U %n' /var/backups/tiffintrack/final-acceptance-20261112/*.dump
echo SESSION_RECOVERY_AND_INSTALLED_BACKUP_VERIFIED
