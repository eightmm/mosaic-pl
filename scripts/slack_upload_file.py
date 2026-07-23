#!/usr/bin/env python3
"""Upload a file to Slack via the external-upload API (3-step, 2024+).

Incoming webhooks CANNOT upload files (text only); this needs a Slack
**bot token** (xoxb-...) with the `files:write` scope, and the bot must be
a member of the target channel.

Steps: files.getUploadURLExternal -> POST bytes -> files.completeUploadExternal

Stdlib only (urllib) so it runs under system python3 in cron.

Credentials (off-repo or repo-local gitignored file, one per line):
  config/casp_slack_bot_token     -> xoxb-...
  config/casp_slack_channel_id    -> Cxxxxxxxx  (channel the bot is in)
Or via env: SLACK_BOT_TOKEN, SLACK_CHANNEL_ID.

Usage:
  python3 scripts/slack_upload_file.py FILE [--title T] [--comment C] [--dry-run]

--dry-run validates token/channel/file presence and prints the plan WITHOUT
calling Slack. Exit 0 ok, 2 missing creds/file, 1 API error.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOKEN_FILE = REPO / "config" / "casp_slack_bot_token"
CHANNEL_FILE = REPO / "config" / "casp_slack_channel_id"
API = "https://slack.com/api"


def _read(path: Path) -> str | None:
    if path.is_file():
        v = path.read_text().strip()
        return v or None
    return None


def get_token() -> str | None:
    return (os.environ.get("SLACK_BOT_TOKEN", "").strip() or _read(TOKEN_FILE))


def get_channel() -> str | None:
    return (os.environ.get("SLACK_CHANNEL_ID", "").strip() or _read(CHANNEL_FILE))


def _api_get(method: str, token: str, params: dict) -> dict:
    url = f"{API}/{method}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _api_post(method: str, token: str, payload: dict) -> dict:
    data = urllib.parse.urlencode(payload).encode()
    req = urllib.request.Request(
        f"{API}/{method}",
        data=data,
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _post_bytes(upload_url: str, blob: bytes, filename: str) -> None:
    ctype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    req = urllib.request.Request(upload_url, data=blob,
                                 headers={"Content-Type": ctype})
    with urllib.request.urlopen(req, timeout=120) as r:
        if r.status != 200:
            raise RuntimeError(f"byte upload returned {r.status}")


def upload(path: Path, token: str, channel: str,
           title: str, comment: str) -> dict:
    blob = path.read_bytes()
    # 1. reserve an upload URL
    step1 = _api_get("files.getUploadURLExternal", token,
                     {"filename": path.name, "length": len(blob)})
    if not step1.get("ok"):
        raise RuntimeError(f"getUploadURLExternal: {step1.get('error')}")
    upload_url, file_id = step1["upload_url"], step1["file_id"]
    # 2. PUT/POST the raw bytes
    _post_bytes(upload_url, blob, path.name)
    # 3. complete + share to channel
    files = json.dumps([{"id": file_id, "title": title}])
    payload = {"files": files, "channel_id": channel}
    if comment:
        payload["initial_comment"] = comment
    step3 = _api_post("files.completeUploadExternal", token, payload)
    if not step3.get("ok"):
        raise RuntimeError(f"completeUploadExternal: {step3.get('error')}")
    return step3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", type=Path)
    ap.add_argument("--title", default=None)
    ap.add_argument("--comment", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = args.file
    title = args.title or path.name

    if not path.is_file():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 2
    token, channel = get_token(), get_channel()
    missing = []
    if not token:
        missing.append("bot token (config/casp_slack_bot_token or $SLACK_BOT_TOKEN)")
    if not channel:
        missing.append("channel id (config/casp_slack_channel_id or $SLACK_CHANNEL_ID)")

    if args.dry_run:
        print("DRY-RUN — would upload:")
        print(f"  file    : {path}  ({path.stat().st_size} bytes)")
        print(f"  title   : {title}")
        print(f"  comment : {args.comment or '(none)'}")
        print(f"  token   : {'present' if token else 'MISSING'}")
        print(f"  channel : {channel if channel else 'MISSING'}")
        if missing:
            print("  -> NOT ready; missing: " + "; ".join(missing))
            return 2
        print("  -> ready: creds present, file readable. (no API call made)")
        return 0

    if missing:
        print("ERROR missing: " + "; ".join(missing), file=sys.stderr)
        return 2
    try:
        res = upload(path, token, channel, title, args.comment)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR upload: {e}", file=sys.stderr)
        return 1
    fid = (res.get("files") or [{}])[0].get("id", "?")
    print(f"uploaded: {path.name} (file_id={fid}) -> channel {channel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
