#!/usr/bin/env python3
"""Tiny Slack incoming-webhook text notifier (stdlib only).

Shared by the LG submission builders so that finishing an LG file posts a
summary to Slack. Incoming webhooks are TEXT only — no file upload (that
needs a bot token; see scripts/slack_upload_file.py).

Webhook resolution (first hit wins):
  1. $CASP_SLACK_WEBHOOK
  2. config/casp_slack_webhook  (repo-local, gitignored)

``notify(text)`` is a no-op (returns False) when no webhook is configured,
so callers can wire it unconditionally without breaking offline runs.

CLI:  python3 scripts/slack_notify.py "message text"
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_WEBHOOK_FILE = _REPO / "config" / "casp_slack_webhook"


def webhook_url() -> str | None:
    env = os.environ.get("CASP_SLACK_WEBHOOK", "").strip()
    if env:
        return env
    if _WEBHOOK_FILE.is_file():
        v = _WEBHOOK_FILE.read_text().strip()
        return v or None
    return None


def notify(text: str) -> bool:
    """POST ``text`` to the configured Slack webhook. Returns True on a 200.
    Returns False (no raise) when unconfigured or on any network error, so a
    notification failure never breaks the build that triggered it."""
    url = webhook_url()
    if not url:
        return False
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps({"text": text}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: slack_notify.py <message>", file=sys.stderr)
        return 2
    ok = notify(" ".join(sys.argv[1:]))
    print("sent" if ok else "not sent (no webhook or error)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
