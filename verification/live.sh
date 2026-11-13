set -Eeuo pipefail
cd /home/support/tiffintrack
docker compose exec -T -e PYTHONPATH=/app tiffintrack python - < /home/support/tiffintrack-final-live.py
/var/tmp/tiffintrack-verify-venv/bin/python - <<'PY'
import json,subprocess
from dotenv import dotenv_values
values=dotenv_values('/home/support/tiffintrack/.env')
logs=subprocess.run(['docker','logs','--since','1h','tiffintrack'],capture_output=True,text=True,check=True)
content=logs.stdout+logs.stderr
secrets=[v for k,v in values.items() if v and len(v)>=12 and (any(x in k for x in ('PASSWORD','TOKEN','SECRET','SESSION_KEY')) or k=='DATABASE_URL')]
assert not any(v in content for v in secrets),'Known protected configuration value appears in logs'
print('LOG_REVIEW',json.dumps({'lines':len(content.splitlines()),'component_exit_lines':content.count(' exited code='),'processing_errors':content.count('WhatsApp event processing failed'),'publication_storage_errors':content.count('weekly_publication_storage_error'),'known_secret_leaks':0}))
PY
for path in /auth/login /admin/orders /admin/operations /admin/whatsapp/export.xlsx;do
 curl --cacert /etc/tiffintrack/tls/server.crt -s -o /dev/null -w "$path %{http_code} %{time_total}s\n" "https://192.168.1.101$path"
done
code=$(curl --cacert /etc/tiffintrack/tls/server.crt -s -o /dev/null -w '%{http_code}' -X POST https://192.168.1.101/orders)
test "$code" = 400
echo CSRF_UNTOKENED_POST_REJECTED_400
curl --cacert /etc/tiffintrack/tls/server.crt -sI https://192.168.1.101/auth/login | grep -Ei 'HTTP/|content-security-policy|x-frame-options|strict-transport-security|set-cookie|x-content-type-options' | sed -E 's/(session=)[^;]+/\1[redacted]/'
docker stats --no-stream --format 'cpu={{.CPUPerc}} memory={{.MemUsage}} pids={{.PIDs}}' tiffintrack
echo FINAL_LIVE_ACCEPTANCE_READ_ONLY_CHECKS_COMPLETE
