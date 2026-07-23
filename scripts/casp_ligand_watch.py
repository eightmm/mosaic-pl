#!/usr/bin/env python3
"""Watch the CASP17 ligand target list and Slack-notify on new targets.

Scrapes https://predictioncenter.org/casp17/targetlist.cgi?view=ligand,
diffs the target set against a local state file, and POSTs a message to a
Slack incoming webhook for every target not seen before.

Stdlib only (urllib/json/re) so it runs under the system python3 in cron
without activating any venv.

Env:
  CASP_SLACK_WEBHOOK   Slack incoming webhook URL (required to actually send;
                       without it the script prints what it would send).

Usage:
  CASP_SLACK_WEBHOOK=https://hooks.slack.com/services/... \
    python3 scripts/casp_ligand_watch.py
  # first run seeds state silently (no flood); add --notify-first to send all.

Exit codes: 0 ok (with or without new targets), 1 fetch/parse error.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

LIST_URL = "https://predictioncenter.org/casp17/targetlist.cgi?view=ligand"
TARGET_URL = "https://predictioncenter.org/casp17/target.cgi?id={id}&view=ligand"
STATE_PATH = Path(__file__).resolve().parent.parent / "experiments" / ".casp_ligand_seen.json"
UA = "casp17-ligand-watch/1.0 (+local cron)"

# Row pattern: a target.cgi?id=<num> link whose visible text is the target
# name (e.g. R2390). One regex over the raw HTML captures (id, name) pairs.
ROW_RE = re.compile(
    r'target\.cgi\?id=(\d+)[^"]*"[^>]*>\s*([RTHDML]2\d{3})\s*<',
    re.IGNORECASE,
)


def fetch(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def parse_targets(html: str) -> dict[str, int]:
    """Return {name: numeric_id} for every ligand-view target row."""
    out: dict[str, int] = {}
    for tid, name in ROW_RE.findall(html):
        out[name.upper()] = int(tid)
    return out


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"seen": {}}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def slack_post(webhook: str, text: str) -> None:
    payload = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        webhook, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Slack webhook returned {resp.status}")


def message_for(name: str, tid: int) -> str:
    return (
        f":dna: *New CASP17 ligand target: {name}*\n"
        f"{TARGET_URL.format(id=tid)}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--notify-first",
        action="store_true",
        help="send notifications even on the first run (default: seed silently)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print messages instead of posting; do not update state",
    )
    args = ap.parse_args()

    webhook = os.environ.get("CASP_SLACK_WEBHOOK", "").strip()

    try:
        html = fetch(LIST_URL)
        current = parse_targets(html)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR fetch/parse: {e}", file=sys.stderr)
        return 1
    if not current:
        print("ERROR: parsed 0 targets (page layout changed?)", file=sys.stderr)
        return 1

    state = load_state()
    seen = state.get("seen", {})
    first_run = not seen

    new = {n: i for n, i in current.items() if n not in seen}

    if new and (not first_run or args.notify_first):
        for name in sorted(new):
            msg = message_for(name, new[name])
            if args.dry_run or not webhook:
                tag = "DRY-RUN" if args.dry_run else "NO-WEBHOOK"
                print(f"[{tag}] would post:\n{msg}\n")
            else:
                slack_post(webhook, msg)
                print(f"posted: {name}")
    elif new and first_run:
        print(f"first run: seeding {len(new)} targets silently "
              f"(use --notify-first to send)")
    else:
        print("no new targets")

    if not args.dry_run:
        seen.update(current)  # values are ids; harmless to refresh
        state["seen"] = seen
        save_state(state)

    return 0


if __name__ == "__main__":
    sys.exit(main())
