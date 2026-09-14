#!/usr/bin/env bash
# Redeploy the web backend + ai-workbench frontend (+ scheduler) on the VPS.
#
#   ssh root@<vps>
#   cd /root/market-intelligence-workbench && bash web/deploy/redeploy.sh [git-ref] [--frontend] [--scheduler]
#
# git-ref defaults to origin/main. The script fast-forwards the checked-out
# branch to it, restarts the FastAPI backend and waits (up to 60 s) for
# /api/health, rebuilds the vinext workbench when ai-workbench/ changed (or
# --frontend), and restarts the scheduler when v2/scheduler/ changed (or
# --scheduler). Exits on the first error.
set -euo pipefail

REPO=/root/market-intelligence-workbench
REF=origin/main
FORCE_FRONTEND=0
FORCE_SCHEDULER=0
for arg in "$@"; do
  case "$arg" in
    --frontend) FORCE_FRONTEND=1 ;;
    --scheduler) FORCE_SCHEDULER=1 ;;
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

echo "== git: fetch + fast-forward to $REF"
git fetch origin --prune
BEFORE=$(git rev-parse HEAD)
git merge --ff-only "$REF"
AFTER=$(git rev-parse HEAD)
echo "   $BEFORE -> $AFTER"
if [ "$BEFORE" = "$AFTER" ]; then echo "   nothing new"; fi
CHANGED=$(git diff --name-only "$BEFORE" "$AFTER" || true)

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

echo "== done. services:"
systemctl --no-pager --plain list-units 'hedge-fund-*' | sed -n '1,8p'
