#!/usr/bin/env bash
# Redeploy the web backend + ai-workbench frontend (+ scheduler) on the VPS.
#
#   ssh root@<vps>
#   cd /root/market-intelligence-workbench && bash web/deploy/redeploy.sh [git-ref] [--frontend] [--scheduler] [--agent-v3]
#
# git-ref defaults to origin/main. The script fast-forwards the checked-out
# branch to it, restarts the FastAPI backend and waits (up to 60 s) for
# /api/health, rebuilds the vinext workbench when ai-workbench/ changed (or
# --frontend), restarts the scheduler when v2/scheduler/ changed (or
# --scheduler), and syncs + restarts the isolated Agent V3 service when any
# code it imports changed (or --agent-v3). Exits on the first error.
set -euo pipefail

main() {
REPO=/root/market-intelligence-workbench
REF=origin/main
FORCE_FRONTEND=0
FORCE_SCHEDULER=0
FORCE_AGENT_V3=0
for arg in "$@"; do
  case "$arg" in
    --frontend) FORCE_FRONTEND=1 ;;
    --scheduler) FORCE_SCHEDULER=1 ;;
    --agent-v3) FORCE_AGENT_V3=1 ;;
    *) REF="$arg" ;;
  esac
done
cd "$REPO"

wait_http() {  # url, seconds
  local url=$1 secs=${2:-60} i
  for ((i = 0; i < secs; i++)); do
    if curl -sf -o /dev/null "$url"; then return 0; fi
    sleep 1
  done
  return 1
}

if [ -n "${REDEPLOY_SINCE:-}" ]; then
  # Second pass after the script updated itself: the tree is already at $REF.
  BEFORE=$REDEPLOY_SINCE
  AFTER=$(git rev-parse HEAD)
  echo "== git: already at $AFTER (re-run of the updated script)"
else
  echo "== git: fetch + fast-forward to $REF"
  git fetch origin --prune
  BEFORE=$(git rev-parse HEAD)
  git merge --ff-only "$REF"
  AFTER=$(git rev-parse HEAD)
  echo "   $BEFORE -> $AFTER"
  if [ "$BEFORE" = "$AFTER" ]; then echo "   nothing new"; fi
fi
CHANGED=$(git diff --name-only "$BEFORE" "$AFTER" || true)

# When the update touched this script, finish the deploy with the new logic
# rather than the copy that was loaded at start (the first run after adding
# the Agent V3 section never executed it for exactly this reason).
if [ -z "${REDEPLOY_SINCE:-}" ] && echo "$CHANGED" | grep -q '^web/deploy/redeploy.sh$'; then
  echo "== redeploy.sh changed in this update; re-running the new script"
  REDEPLOY_SINCE=$BEFORE exec bash "$REPO/web/deploy/redeploy.sh" "$@"
fi

if echo "$CHANGED" | grep -q '^web/deploy/nginx-web.conf$'; then
  echo "== nginx: web/deploy/nginx-web.conf changed. NOT installed automatically —"
  echo "   diff it against /etc/nginx/sites-available/hedge-fund-web.conf and port the"
  echo "   changed lines by hand (the live file carries the Certbot TLS block)."
fi

echo "== backend: restart hedge-fund-web"
sudo systemctl restart hedge-fund-web
if wait_http http://127.0.0.1:8100/api/health 60; then
  echo "   /api/health ok"
else
  echo "   backend not healthy after 60 s — tail logs/web.err"; tail -n 30 logs/web.err; exit 1
fi

if [ "$FORCE_FRONTEND" = 1 ] || echo "$CHANGED" | grep -q '^ai-workbench/'; then
  echo "== frontend: rebuild ai-workbench"
  (cd ai-workbench && npm ci --no-audit --no-fund && npm run build)
  sudo systemctl restart hedge-fund-workbench
  if wait_http http://127.0.0.1:3000/ 60; then
    echo "   workbench :3000 ok"
  else
    echo "   workbench not serving after 60 s — tail logs/workbench.err"; tail -n 30 logs/workbench.err; exit 1
  fi
else
  echo "== frontend: ai-workbench unchanged, skipping build (use --frontend to force)"
fi

if [ "$FORCE_SCHEDULER" = 1 ] || echo "$CHANGED" | grep -q '^v2/scheduler/'; then
  echo "== scheduler: restart hedge-fund-scheduler"
  sudo systemctl restart hedge-fund-scheduler
else
  echo "== scheduler: unchanged, not restarted (use --scheduler to force)"
fi

# Agent V3 is a separate process with its own venv: a backend restart does not
# reach it. It imports v2/agent_v2 + v2/agent_common and the backend's auth,
# so a change to any of those leaves it serving stale code (seen once as a
# 401 after the owner-login change, then a missing tavily after the restart).
V3_VENV=$REPO/.venv-agent-v3
V3_LOCK=v2/agent_v3/requirements-business.lock
V3_CODE='^v2/agent_v3/|^v2/agent_v2/|^v2/agent_common/|^web/backend/app/auth.py|^web/backend/app/config.py|^web/backend/app/routers/agent_v3.py|^web/deploy/agent-v3-server.py'
if systemctl list-unit-files hedge-fund-agent-v3.service --no-legend 2>/dev/null | grep -q hedge-fund-agent-v3; then
  if [ "$FORCE_AGENT_V3" = 1 ] || echo "$CHANGED" | grep -Eq "$V3_CODE"; then
    if [ "$FORCE_AGENT_V3" = 1 ] || echo "$CHANGED" | grep -q "^$V3_LOCK$" || ! "$V3_VENV/bin/python" -c 'import langgraph, tavily, pandas, bs4' 2>/dev/null; then
      echo "== agent-v3: sync $V3_VENV from $V3_LOCK"
      "$V3_VENV/bin/pip" install --quiet -r "$V3_LOCK"
    fi
    echo "== agent-v3: restart hedge-fund-agent-v3"
    sudo systemctl restart hedge-fund-agent-v3
    if wait_http http://127.0.0.1:8104/health 60; then
      echo "   /health ok"
    else
      echo "   agent-v3 not healthy after 60 s — journalctl -u hedge-fund-agent-v3 -n 30"; journalctl -u hedge-fund-agent-v3 -n 30 --no-pager; exit 1
    fi
  else
    echo "== agent-v3: unchanged, not restarted (use --agent-v3 to force)"
  fi
fi

echo "== done. services:"
systemctl --no-pager --plain list-units 'hedge-fund-*' | sed -n '1,8p'
}

# Everything is parsed before the first command runs, so a `git merge` that
# rewrites this file mid-deploy cannot alter the run in progress.
main "$@"
