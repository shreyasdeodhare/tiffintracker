set -Eeuo pipefail
STAGE=/var/tmp/tiffintrack-final-acceptance-20261112
PY=/var/tmp/tiffintrack-verify-venv/bin/python
cd "$STAGE"
name=tiffintrack_final_perf_test
if runuser -u postgres -- psql -Atqc "SELECT 1 FROM pg_database WHERE datname='$name'" | grep -q 1;then echo REFUSING_EXISTING_PERF_DATABASE;exit 1;fi
runuser -u postgres -- createdb "$name"
runuser -u postgres -- env PYTHONPATH="$STAGE" "$PY" - <<'PY'
from app import create_app
from flask_migrate import upgrade
with create_app({'SECRET_KEY':'isolated-performance-'*3,'SQLALCHEMY_DATABASE_URI':'postgresql+psycopg:///tiffintrack_final_perf_test'}).app_context():
 upgrade()
PY
runuser -u postgres -- env TEST_POSTGRES_URL=postgresql+psycopg:///tiffintrack_final_perf_test "$PY" -m scripts.benchmark >/home/support/tiffintrack-final-performance.json
runuser -u postgres -- env PYTHONPATH="$STAGE" TEST_POSTGRES_URL=postgresql+psycopg:///tiffintrack_final_perf_test "$PY" tests/benchmark_whatsapp.py >/home/support/tiffintrack-final-inbox-performance.json
"$PY" - <<'PY'
import json
from pathlib import Path
for name in ['performance','inbox-performance']:
 data=json.loads(Path('/home/support/tiffintrack-final-'+name+'.json').read_text())
 if 'results' in data:assert all(v['errors']==0 for v in data['results'].values())
 else:assert data['errors']==0
 print(name,json.dumps(data))
PY
name=tiffintrack_final_backup_script_restore_test
if runuser -u postgres -- psql -Atqc "SELECT 1 FROM pg_database WHERE datname='$name'" | grep -q 1;then echo REFUSING_EXISTING_SCRIPT_RESTORE_DATABASE;exit 1;fi
runuser -u postgres -- createdb "$name"
file=$(find /var/backups/tiffintrack/final-acceptance-20261112 -maxdepth 1 -name 'tiffintrack-*.dump' | sort | tail -1)
runuser -u postgres -- pg_restore --exit-on-error --no-owner --no-acl -d "$name" "$file"
runuser -u postgres -- env PYTHONPATH="$STAGE" "$PY" - <<'PY'
from app import create_app
from app.extensions import db
from app.application.reporting import period_report
from app.application.exports import counts_workbook
from app.domain.orders import reporting_week
from app.services import local_today
from openpyxl import load_workbook
with create_app({'SECRET_KEY':'installed-backup-restore-'*3,'SQLALCHEMY_DATABASE_URI':'postgresql+psycopg:///tiffintrack_final_backup_script_restore_test'}).app_context():
 assert db.session.execute(db.text('select version_num from alembic_version')).scalar()=='0005_participants'
 report=period_report(*reporting_week(local_today()))
 assert report['reconciled']
 assert load_workbook(counts_workbook(local_today(),'all')).sheetnames==['Daily','Weekly','Monthly']
 print('INSTALLED_BACKUP_SCRIPT_DUMP_RESTORE_AND_APPLICATION_VERIFIED')
PY
echo FINAL_ISOLATED_PERFORMANCE_AND_SCRIPT_BACKUP_RESTORE_COMPLETE
