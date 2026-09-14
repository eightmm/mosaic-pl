#!/usr/bin/env python3
"""Build a single-file 3Dmol viewer for a CASP ``PFRMAT TS`` solvent submission.

The LG viewer (``notify_lg.py``) is built around ``LIGAND``/``LSCORE`` blocks and
cannot read a TS file, but an ordered-solvent target is exactly the case that
needs looking at: five models that differ only in *which* water and ion sites
they claim, where a table of counts says nothing about whether the sites sit in
the grooves or float in bulk.

So the viewer is organised around the solvent, not the fold:

* one model at a time, switchable, with the RNA as context;
* solvent drawn per species in the conventional element colours, each toggleable;
* a **support** filter — how many independent donor structures place a site —
  which is the only quality signal these predictions carry, and reading it in
  space is the point (consensus sites cluster in the ion core, singletons
  scatter over the surface);
* buried/exposed split by distance to the RNA, so "is this a real pocket site"
  is answerable without measuring.

Usage:
    uv run python scripts/make_ts_viewer.py \
        --ts experiments/CASP17/R2386/submissions/R2386_LCDD.ts \
        --out viz/R2386.html
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ASSET = REPO / "viz" / "assets" / "3Dmol-min.js"

#: Conventional element colours — the same ones a crystallographer sees in
#: PyMOL/Coot, so the picture reads without a legend lookup.
SPECIES = {
    "MG": {"label": "Mg²⁺", "color": "#2f9e44"},
    "K": {"label": "K⁺", "color": "#9c6ade"},
    "NA": {"label": "Na⁺", "color": "#3b82c4"},
    "HOH": {"label": "H₂O", "color": "#d64545"},
}


def parse_ts(path: Path) -> tuple[list[dict], list[str]]:
    """Split a TS file into per-MODEL records plus the header lines."""
    models: list[dict] = []
    header: list[str] = []
    cur: dict | None = None
    for line in path.read_text().splitlines():
        if line.startswith("MODEL"):
            cur = {"index": line.split()[1], "atom": [], "het": [], "method": []}
            continue
        if line.startswith("ENDMDL") or line.startswith("END"):
            if cur is not None:
                models.append(cur)
                cur = None
            continue
        if cur is None:
            if line.startswith(("PFRMAT", "TARGET", "AUTHOR", "METHOD", "REMARK")):
                header.append(line)
            continue
        if line.startswith("ATOM"):
            cur["atom"].append(line)
        elif line.startswith("HETATM"):
            cur["het"].append(line)
        elif line.startswith(("PARENT", "REMARK")):
            cur["method"].append(line)
    return models, header


def _xyz(line: str) -> tuple[float, float, float]:
    return float(line[30:38]), float(line[38:46]), float(line[46:54])


def superpose_models(models: list[dict]) -> list[tuple[str, float, float]]:
    """Put every MODEL in MODEL 1's frame, in place.

    A TS MODEL is a self-contained snapshot, so nothing forces the MODELs to
    share a frame — R2386 MODELs 4 and 5 sit ~150 Å from MODEL 1 while being
    the same structure (1.7 and 0.9 Å once superposed). Left alone, switching
    MODELs throws the molecule out of view and no solvent comparison is
    possible.

    Fits on the shared RNA backbone keys and carries each MODEL's solvent along
    with its RNA, so every solvent-to-RNA distance is preserved exactly. The TS
    submission itself is never rewritten — display only; CASP superposes each
    model independently when scoring.

    Returns ``[(index, removed_frame_offset, residual_rmsd)]`` for reporting.
    """
    import numpy as np

    def keyed(m):
        d = {}
        for ln in m["atom"]:
            try:
                d[(ln[21], int(ln[22:26]), ln[12:16].strip())] = _xyz(ln)
            except ValueError:
                continue
        return d

    if len(models) < 2:
        return []
    ref = keyed(models[0])
    report: list[tuple[str, float, float]] = []
    if not ref:
        return report
    for m in models[1:]:
        cur = keyed(m)
        shared = sorted(set(ref) & set(cur))
        if len(shared) < 3:
            continue
        P = np.asarray([cur[k] for k in shared], float)
        Q = np.asarray([ref[k] for k in shared], float)
        raw = float(np.sqrt(((P - Q) ** 2).sum(1).mean()))

        def fit(sel):
            a, b = P[sel], Q[sel]
            ac, bc = a.mean(0), b.mean(0)
            U, _S, Vt = np.linalg.svd((a - ac).T @ (b - bc))
            rot = Vt.T @ np.diag(
                [1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
            return rot, bc - rot @ ac

        # Fit on the part the MODELs actually share. A model may carry regions
        # built rather than observed - a disordered terminus completed from
        # different sources in different MODELs - and those disagree by tens of
        # angstroms. Including them drags the fit off the body: on R2386 a
        # straight all-atom fit reports 7-10 A where the structured core agrees
        # to 0.6-1.7. So reject outliers and refit until the set settles.
        sel = np.ones(len(P), bool)
        R, t = fit(sel)
        for _ in range(5):
            dev = np.linalg.norm(P @ R.T + t - Q, axis=1)
            keep = dev < max(2.0, 2.5 * float(np.median(dev)))
            if keep.sum() < max(20, 0.5 * len(P)) or (keep == sel).all():
                break
            sel = keep
            R, t = fit(sel)
        fitted = float(np.sqrt(((P[sel] @ R.T + t - Q[sel]) ** 2).sum(1).mean()))
        report.append((m["index"], raw - fitted, fitted))
        if raw - fitted < 0.05:      # already in MODEL 1's frame
            continue
        for key in ("atom", "het"):
            out = []
            for ln in m[key]:
                try:
                    v = (R @ np.asarray(_xyz(ln), float)) + t
                except ValueError:
                    out.append(ln)
                    continue
                out.append(f"{ln[:30]}{v[0]:8.3f}{v[1]:8.3f}{v[2]:8.3f}{ln[54:]}")
            m[key] = out
    return report


def summarise(model: dict) -> dict:
    """Per-model counts, support histogram and buried/exposed split."""
    import numpy as np

    rna = np.array([_xyz(ln) for ln in model["atom"]], dtype=float)
    counts: dict[str, int] = {}
    supports: list[int] = []
    buried = 0
    het_meta = []
    for ln in model["het"]:
        name = ln[17:20].strip().upper()
        counts[name] = counts.get(name, 0) + 1
        try:
            b = float(ln[60:66])
        except ValueError:
            b = 0.0
        p = np.asarray(_xyz(ln), dtype=float)
        d = float(np.linalg.norm(rna - p, axis=1).min()) if len(rna) else 99.0
        if d <= 4.0:
            buried += 1
        supports.append(int(round(b)))
        try:
            serial = int(ln[6:11])
        except ValueError:
            serial = 0
        het_meta.append({"name": name, "b": round(b, 1), "d": round(d, 2),
                         "s": serial})

    resnums = [int(ln[22:26]) for ln in model["atom"]]
    return {
        "index": model["index"],
        "n_rna_atoms": len(model["atom"]),
        "n_residues": len(set(resnums)),
        "res_lo": min(resnums) if resnums else 0,
        "res_hi": max(resnums) if resnums else 0,
        "counts": counts,
        "n_solvent": len(model["het"]),
        "buried": buried,
        "b_mean": round(sum(supports) / len(supports), 1) if supports else 0.0,
        "het": het_meta,
    }


def build_html(ts_path: Path, target: str, inline_asset: bool) -> str:
    models, header = parse_ts(ts_path)
    if not models:
        raise SystemExit(f"{ts_path}: no MODEL blocks")

    for idx, removed, resid in superpose_models(models):
        if removed >= 0.05:
            print(f"  MODEL {idx}: superposed onto MODEL 1 "
                  f"(removed {removed:.1f} Å of frame offset, "
                  f"{resid:.2f} Å real difference remains)")

    method = " ".join(ln[7:].strip() for ln in header if ln.startswith("METHOD"))
    stats = [summarise(m) for m in models]
    payload = [
        {
            "index": m["index"],
            "pdb": "\n".join(m["atom"] + m["het"]) + "\nEND\n",
            "stats": {k: v for k, v in s.items() if k != "het"},
            "het": s["het"],
        }
        for m, s in zip(models, stats)
    ]
    max_b = max((max((h["b"] for h in s["het"]), default=100.0) for s in stats),
                default=100.0)

    script_tag = ('<script>\n' + ASSET.read_text() + '\n</script>'
                  if inline_asset and ASSET.exists()
                  else '<script src="assets/3Dmol-min.js"></script>')

    species_rows = "\n".join(
        f'<label class="sp"><input type="checkbox" data-sp="{k}" checked>'
        f'<span class="dot" style="background:{v["color"]}"></span>{v["label"]}'
        f'<span class="n" id="n-{k}">0</span></label>'
        for k, v in SPECIES.items()
    )

    return f"""<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{target} Solvent Models</title>
{script_tag}
<style>
:root {{
  color-scheme: light dark;
  --ground:#eef1f2; --surface:#ffffff; --ink:#131a1d;
  --muted:#5d6b71; --rule:#d5dcde; --accent:#0f6f7a; --shadow:rgba(15,30,35,.09);
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --ground:#0f1417; --surface:#161d21; --ink:#e6edef; --muted:#93a3a9;
    --rule:#263136; --accent:#5cc9d6; --shadow:rgba(0,0,0,.4);
  }}
}}
:root[data-theme="dark"] {{
  --ground:#0f1417; --surface:#161d21; --ink:#e6edef; --muted:#93a3a9;
  --rule:#263136; --accent:#5cc9d6; --shadow:rgba(0,0,0,.4);
}}
:root[data-theme="light"] {{
  --ground:#eef1f2; --surface:#ffffff; --ink:#131a1d; --muted:#5d6b71;
  --rule:#d5dcde; --accent:#0f6f7a; --shadow:rgba(15,30,35,.09);
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--ground); color:var(--ink);
  font:14px/1.5 ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
}}
header {{
  padding:14px 18px; border-bottom:1px solid var(--rule); background:var(--surface);
  display:flex; flex-wrap:wrap; gap:6px 18px; align-items:baseline;
}}
h1 {{ margin:0; font-size:17px; letter-spacing:.02em; }}
.sub {{ color:var(--muted); font-size:12px; }}
main {{ display:grid; grid-template-columns:288px 1fr; min-height:calc(100vh - 56px); }}
@media (max-width:820px) {{ main {{ grid-template-columns:1fr; }} #view {{ height:62vh; }} }}
aside {{
  border-right:1px solid var(--rule); background:var(--surface);
  padding:14px; display:flex; flex-direction:column; gap:16px;
  overflow-y:auto; max-height:calc(100vh - 56px);
}}
.grp {{ display:flex; flex-direction:column; gap:7px; }}
.hd {{
  font-size:10px; letter-spacing:.13em; text-transform:uppercase;
  color:var(--muted); border-bottom:1px solid var(--rule); padding-bottom:5px;
}}
#view {{ position:relative; height:calc(100vh - 56px); }}
button.m {{
  font:inherit; font-size:12px; padding:6px 9px; cursor:pointer; text-align:left;
  background:transparent; color:var(--ink);
  border:1px solid var(--rule); border-radius:3px;
}}
button.m:hover {{ border-color:var(--accent); }}
button.m[aria-pressed="true"] {{
  background:var(--accent); border-color:var(--accent); color:var(--surface);
}}
button.m:focus-visible, input:focus-visible, select:focus-visible {{
  outline:2px solid var(--accent); outline-offset:2px;
}}
label.sp {{ display:flex; align-items:center; gap:7px; font-size:12px; cursor:pointer; }}
.dot {{ width:11px; height:11px; border-radius:50%; flex:none; }}
.n {{ margin-left:auto; color:var(--muted); font-variant-numeric:tabular-nums; }}
table.st {{ width:100%; border-collapse:collapse; font-size:11.5px; }}
table.st td {{ padding:2.5px 0; }}
table.st td:last-child {{ text-align:right; font-variant-numeric:tabular-nums; }}
table.st td:first-child {{ color:var(--muted); }}
input[type=range] {{ width:100%; accent-color:var(--accent); }}
select {{
  font:inherit; font-size:12px; padding:4px 6px; background:var(--surface);
  color:var(--ink); border:1px solid var(--rule); border-radius:3px; width:100%;
}}
.note {{ font-size:11px; color:var(--muted); line-height:1.45; }}
.bar {{ height:4px; background:var(--rule); border-radius:2px; overflow:hidden; }}
.bar > i {{ display:block; height:100%; background:var(--accent); }}
</style>

<header>
  <h1>{target}</h1>
  <span class="sub">{len(models)} solvent models · PFRMAT TS</span>
  <span class="sub" style="flex:1 1 100%">{method}</span>
</header>

<main>
  <aside>
    <div class="grp">
      <div class="hd">Model</div>
      <div id="models" style="display:flex;flex-direction:column;gap:5px"></div>
    </div>

    <div class="grp">
      <div class="hd">Solvent species</div>
      {species_rows}
    </div>

    <div class="grp">
      <div class="hd">Support filter</div>
      <input type="range" id="bmin" min="0" max="{int(math.ceil(max_b))}" value="0" step="1">
      <div class="note">show sites with confidence &ge; <b id="bval">0</b>
        <span id="bkept"></span></div>
    </div>

    <div class="grp">
      <div class="hd">Placement</div>
      <select id="place">
        <option value="all">all sites</option>
        <option value="near">contacting RNA (&le; 4 Å)</option>
        <option value="far">bulk (&gt; 4 Å)</option>
      </select>
    </div>

    <div class="grp">
      <div class="hd">RNA</div>
      <select id="rna">
        <option value="cartoon">cartoon</option>
        <option value="cartoon+line">cartoon + line</option>
        <option value="line">line</option>
        <option value="surface">cartoon + surface</option>
        <option value="hide">hide</option>
      </select>
    </div>

    <div class="grp">
      <div class="hd">This model</div>
      <table class="st"><tbody id="stats"></tbody></table>
      <div class="bar"><i id="burbar" style="width:0%"></i></div>
      <div class="note" id="burnote"></div>
    </div>

    <div class="grp">
      <div class="hd">View</div>
      <button class="m" id="reset">reset camera</button>
      <button class="m" id="theme">toggle theme</button>
    </div>
  </aside>

  <div id="view"></div>
</main>

<script>
const MODELS = {json.dumps(payload)};
const SPECIES = {json.dumps(SPECIES)};
const view = $3Dmol.createViewer(document.getElementById("view"), {{}});
let cur = 0;
const on = new Set(Object.keys(SPECIES));

function bg() {{
  const dark = getComputedStyle(document.documentElement)
    .getPropertyValue("--surface").trim();
  view.setBackgroundColor(dark);
}}

function draw(keepCam) {{
  const m = MODELS[cur];
  view.removeAllModels();
  view.removeAllSurfaces();
  const mdl = view.addModel(m.pdb, "pdb");

  const style = document.getElementById("rna").value;
  if (style !== "hide") {{
    const s = {{}};
    if (style.startsWith("cartoon") || style === "surface")
      s.cartoon = {{ color: "spectrum", opacity: style === "surface" ? 0.9 : 1 }};
    if (style.includes("line") || style === "line")
      s.line = {{ colorscheme: "default" }};
    mdl.setStyle({{ hetflag: false }}, s);
    if (style === "surface")
      view.addSurface($3Dmol.SurfaceType.VDW,
        {{ opacity: 0.45, color: "white" }}, {{ hetflag: false }});
  }} else {{
    mdl.setStyle({{ hetflag: false }}, {{}});
  }}

  const bmin = +document.getElementById("bmin").value;
  const place = document.getElementById("place").value;
  const counts = {{}};
  const keep = {{}};
  let shown = 0;
  Object.keys(SPECIES).forEach(k => {{ counts[k] = 0; keep[k] = []; }});

  m.het.forEach(h => {{
    counts[h.name] = (counts[h.name] || 0) + 1;
    const okP = place === "all" || (place === "near" ? h.d <= 4 : h.d > 4);
    if (!(on.has(h.name) && h.b >= bmin && okP)) return;
    (keep[h.name] = keep[h.name] || []).push(h.s);
    shown++;
  }});

  // Hide every solvent atom first, then re-style only the serials that survive
  // the filters — one call per species instead of one per atom (a union model
  // carries ~850, and per-atom setStyle makes the slider crawl).
  mdl.setStyle({{ hetflag: true }}, {{}});
  Object.keys(SPECIES).forEach(k => {{
    if (!keep[k] || !keep[k].length) return;
    mdl.setStyle(
      {{ serial: keep[k] }},
      {{ sphere: {{ radius: k === "HOH" ? 0.42 : 0.78, color: SPECIES[k].color }} }}
    );
  }});

  Object.keys(SPECIES).forEach(k => {{
    const el = document.getElementById("n-" + k);
    if (el) el.textContent = counts[k] || 0;
  }});
  document.getElementById("bkept").textContent = " — " + shown + " shown";

  bg();
  if (!keepCam) view.zoomTo({{ hetflag: false }});
  view.render();
}}

function stats() {{
  const s = MODELS[cur].stats;
  const rows = [
    ["RNA residues", s.n_residues + " (" + s.res_lo + "-" + s.res_hi + ")"],
    ["solvent sites", s.n_solvent],
    ["mean confidence", s.b_mean],
  ];
  Object.entries(s.counts).forEach(([k, v]) => rows.push(["&nbsp;&nbsp;" + k, v]));
  document.getElementById("stats").innerHTML =
    rows.map(r => "<tr><td>" + r[0] + "</td><td>" + r[1] + "</td></tr>").join("");
  const pct = s.n_solvent ? Math.round(100 * s.buried / s.n_solvent) : 0;
  document.getElementById("burbar").style.width = pct + "%";
  document.getElementById("burnote").textContent =
    s.buried + " of " + s.n_solvent + " sites contact the RNA (" + pct + "%)";
}}

const holder = document.getElementById("models");
MODELS.forEach((m, i) => {{
  const b = document.createElement("button");
  b.className = "m";
  b.setAttribute("aria-pressed", i === 0 ? "true" : "false");
  b.innerHTML = "MODEL " + m.index +
    ' <span style="color:var(--muted)">· ' + m.stats.n_solvent + " sites</span>";
  b.onclick = () => {{
    cur = i;
    [...holder.children].forEach((c, j) =>
      c.setAttribute("aria-pressed", j === i ? "true" : "false"));
    stats(); draw(true);
  }};
  holder.appendChild(b);
}});

document.querySelectorAll("input[data-sp]").forEach(cb => {{
  cb.onchange = () => {{
    cb.checked ? on.add(cb.dataset.sp) : on.delete(cb.dataset.sp);
    draw(true);
  }};
}});
document.getElementById("bmin").oninput = e => {{
  document.getElementById("bval").textContent = e.target.value;
  draw(true);
}};
document.getElementById("place").onchange = () => draw(true);
document.getElementById("rna").onchange = () => draw(true);
document.getElementById("reset").onclick = () => draw(false);
document.getElementById("theme").onclick = () => {{
  const r = document.documentElement;
  const now = r.getAttribute("data-theme")
    || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  r.setAttribute("data-theme", now === "dark" ? "light" : "dark");
  bg(); view.render();
}};
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {{
  if (!document.documentElement.getAttribute("data-theme")) {{ bg(); view.render(); }}
}});

stats();
draw(false);
</script>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target", type=str, default=None)
    ap.add_argument("--no-inline", action="store_true",
                    help="reference viz/assets/3Dmol-min.js instead of inlining it "
                         "(smaller file, only works served from viz/)")
    args = ap.parse_args()

    target = args.target or args.ts.stem.replace("_LCDD", "")
    html = build_html(args.ts, target, inline_asset=not args.no_inline)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f"built {args.out}  ({len(html)/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
