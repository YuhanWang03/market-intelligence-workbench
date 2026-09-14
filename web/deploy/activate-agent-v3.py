"""Activate the Agent V3 API on the authorized VPS without a separate page."""
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import urllib.request
import time

root=Path('/root/market-intelligence-workbench')
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
backup=Path('/root')/('agent-v3-backup-'+stamp)
backup.mkdir()
nginx=Path('/etc/nginx/sites-enabled/hedge-fund-web.conf').resolve(strict=True)
shutil.copy2(nginx,backup/'nginx.conf')
unit=Path('/etc/systemd/system/hedge-fund-agent-v3.service')
if unit.exists():
    shutil.copy2(unit,backup/'service')
shutil.copy2(root/'web/deploy/hedge-fund-agent-v3.service',unit)
subprocess.run(['systemctl','daemon-reload'],check=True)
subprocess.run(['systemctl','start','hedge-fund-agent-v3'],check=True)
for _ in range(40):
    try:
        with urllib.request.urlopen('http://127.0.0.1:8104/health',timeout=2) as response:
            if response.status==200:
                break
    except OSError:
        time.sleep(1)
else:
    raise RuntimeError('V3 did not become healthy; nginx unchanged. Backup: '+str(backup))
original=nginx.read_text()
if 'location /api/agent-v3/' in original:
    raise RuntimeError('Existing V3 routes require manual review; nginx unchanged')
anchor='    location /api/ {'
if anchor not in original:
    raise RuntimeError('Expected API location missing; nginx unchanged')
routes='''    location /api/agent-v3/ {
        proxy_pass http://127.0.0.1:8104;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 90s;
    }
'''
nginx.write_text(original.replace(anchor,routes+anchor,1))
try:
    subprocess.run(['nginx','-t'],check=True)
    subprocess.run(['systemctl','reload','nginx'],check=True)
except BaseException:
    shutil.copy2(backup/'nginx.conf',nginx)
    subprocess.run(['nginx','-t'],check=True)
    subprocess.run(['systemctl','reload','nginx'],check=True)
    raise
subprocess.run(['systemctl','enable','hedge-fund-agent-v3'],check=True)
print('Backup:',backup)
print('V3 API activated; original web and workbench were not restarted.')
