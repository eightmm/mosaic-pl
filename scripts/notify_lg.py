#!/usr/bin/env python3
"""Post-process a finished LG submission: build the viewer HTML + a
self-contained single-file copy, then Slack-notify a summary.

Pipeline (semi-automatic mode — Artifact publish stays a manual agent step):
    LG  ->  viz/<target>.html         (build_target_viz.py, with clash badges)
        ->  viz/standalone/<target>.html  (3Dmol inlined, hostable single file)
        ->  Slack webhook text summary    (slack_notify.py)

The Slack message carries the per-MODEL clash flags parsed from the freshly
built viewer payload, so a clashing RNA cofold pose is visible in chat without
opening the viewer.

Usage:
    uv run python scripts/notify_lg.py --lg experiments/CASP17/submissions/R2390_LCDD.lg
    # --no-slack  : build HTML only, skip the webhook
    # --no-standalone : skip the inlined single-file copy

A missing webhook is a no-op (slack_notify returns False); HTML is still built.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import slack_notify  # noqa: E402

ASSET = REPO / "viz" / "assets" / "3Dmol-min.js"


def _target_of(lg: Path) -> str:
    return lg.stem.replace("_LCDD", "")


def _ligand_name(lg: Path) -> str:
    for ln in lg.read_text().splitlines():
        if ln.startswith("LIGAND"):
            parts = ln.split()
            if len(parts) >= 3:
                return parts[2]
            break
    return "LIG"


def _extract_data(html: str) -> dict | None:
    m = re.search(r"const DATA = (\{.*?\});", html, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _inline_standalone(target: str, viz_dir: Path) -> Path | None:
    src = viz_dir / f"{target}.html"
    if not (src.exists() and ASSET.exists()):
        return None
    out_dir = viz_dir / "standalone"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{target}.html"
    out.write_text(src.read_text().replace(
        '<script src="assets/3Dmol-min.js"></script>',
        "<script>\n" + ASSET.read_text() + "\n</script>"))
    return out


def _summary(lg: Path, data: dict | None) -> str:
    target = _target_of(lg)
    lig = _ligand_name(lg)
    if not data or not data.get("models"):
        return f":page_facing_up: *LG built: {target}* (ligand {lig})\n`{lg}`"
    models = data["models"]
    n = len(models)
    lines = [f":page_facing_up: *LG built: {target}* — {n} MODEL, ligand {lig}"]
    any_clash = False
    for m in models:
        lg0 = (m.get("ligands") or [{}])[0]
        ls = lg0.get("lscore")
        cl = m.get("clash") or {}
        nclash = cl.get("n")
        if nclash:
            any_clash = True
            badge = f":x: clash {nclash} (min {cl.get('min')}Å)"
        elif nclash == 0:
            badge = f":white_check_mark: no clash (min {cl.get('min')}Å)"
        else:
            badge = ""
        lsr = f"{ls:.3f}" if isinstance(ls, (int, float)) else "—"
        lines.append(f"  • MODEL {m.get('model')}: LSCORE {lsr}  {badge}")
    aff = models[0].get("affinity")
    if aff:
        lines.append(f"AFFNTY {aff}")
    if any_clash:
        lines.append(":warning: some MODELs have ligand–receptor clashes (cofold pose jammed into receptor)")
    lines.append(f"`{lg}`")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lg", type=Path, required=True)
    ap.add_argument("--viz-dir", type=Path, default=REPO / "viz")
    ap.add_argument("--no-slack", action="store_true")
    ap.add_argument("--no-standalone", action="store_true")
    ap.add_argument("--template-structs", type=int, default=10, metavar="N",
                    help="Embed top-N template structures as cartoon+ligand "
                         "overlays (USalign re-align). 0 = centroid dots only. "
                         "Default 10 — templates show the protein fold, not dots.")
    args = ap.parse_args()

    lg = args.lg.resolve()
    if not lg.exists():
        print(f"ERROR: LG not found: {lg}", file=sys.stderr)
        return 2
    target = _target_of(lg)

    # 1. build the viewer for this target (clash badges included)
    r = subprocess.run(
        ["uv", "run", "python", str(REPO / "scripts" / "build_target_viz.py"),
         "--lg", str(lg), "--output-dir", str(args.viz_dir),
         "--template-structs", str(args.template_structs)],
        cwd=str(REPO), capture_output=True, text=True)
    if r.returncode != 0:
        print(f"ERROR build_target_viz:\n{r.stderr}", file=sys.stderr)
        return 1
    html_path = args.viz_dir / f"{target}.html"
    print(f"built {html_path}")

    # 2. standalone single-file copy (hostable / uploadable)
    if not args.no_standalone:
        sa = _inline_standalone(target, args.viz_dir)
        if sa:
            print(f"built {sa}")

    # 3. Slack summary
    data = _extract_data(html_path.read_text()) if html_path.exists() else None
    if args.no_slack:
        print("slack skipped (--no-slack)")
    else:
        ok = slack_notify.notify(_summary(lg, data))
        print("slack sent" if ok else "slack not sent (no webhook or error)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
