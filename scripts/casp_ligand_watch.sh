#!/usr/bin/env bash
# Cron wrapper for scripts/casp_ligand_watch.py.
# Reads the Slack webhook from a file OUTSIDE the repo so the secret never
# lands in crontab, commits, or logs.
#
# Setup (repo-local, gitignored):
#   printf '%s\n' 'https://hooks.slack.com/services/XXX/YYY/ZZZ' > config/casp_slack_webhook
#   chmod 600 config/casp_slack_webhook
#
# Crontab (every 30 min):
#   */30 * * * * /home/jaemin/project/CASP17/scripts/casp_ligand_watch.sh
set -euo pipefail

REPO="/home/jaemin/project/CASP17"
WEBHOOK_FILE="${CASP_SLACK_WEBHOOK_FILE:-$REPO/config/casp_slack_webhook}"
LOG="$REPO/experiments/logs/casp_ligand_watch.log"

mkdir -p "$(dirname "$LOG")"

if [[ ! -r "$WEBHOOK_FILE" ]]; then
  echo "$(date -Is) ERROR: webhook file not readable: $WEBHOOK_FILE" >> "$LOG"
  exit 1
fi

export CASP_SLACK_WEBHOOK
CASP_SLACK_WEBHOOK="$(tr -d '[:space:]' < "$WEBHOOK_FILE")"

cd "$REPO"
{
  echo "$(date -Is) run"
  python3 scripts/casp_ligand_watch.py
} >> "$LOG" 2>&1
