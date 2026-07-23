"""Per-scorer Top-1/Best-5 evaluation.

Picks the top-1 (and best-of-5) pose by EACH individual scorer:

  - rmsd_pred  (smaller pRMSD better)        — current production default
  - lscore     (1 - prob_above_2A)           — derived from RMSD-Pred
  - ba_pred    (BA-Pred pKd, higher better)
  - iptm       (cofold ipTM, higher better)
  - ptm        (cofold pTM, higher better)
  - plddt      (cofold pLDDT, higher better)
  - conf       (cofold ranking/confidence score, higher better)
  - boltz_aff  (Boltz pose-level affinity log_kd_nM, lower better; weighted by binder_prob)
  - oracle     (best possible — picks the actual lowest crystal-RMSD pose)

For each run:
  1. USalign(cofold receptor → crystal) once.
  2. For every staged pose SDF, transform every conformer into crystal frame
     and compute heavy-atom RMSD against the experimental ligand.
  3. Join with BA-Pred / RMSD-Pred TSVs and per-pose cofold confidence
     (per-pose for cofold poses; the docking-anchor cofold's confidence for
     docking poses; None for template poses).

Outputs:
  - per_pose_scores.csv
  - per_metric_summary.txt
"""
from __future__ import annotations

import csv
import json
import math
import signal
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from compute_submission_scores import (  # noqa: E402
    collect_pose_scores,
)
from evaluate import (  # noqa: E402
    cif_path_for,
    dump_crystal_protein_pdb,
    load_crystal_ligand,
    load_targets,
    run_usalign,
    _largest_template_matching_fragment,
    _reassign_with_fallback,
    _sanitize_loose,
    _mcs_rmsd,
)
from casp17.geometry import pose_rmsd, transform_mol  # noqa: E402

SUBMISSIONS = REPO / "experiments" / "submissions"
RUNS = REPO / "experiments" / "runs"
WORK = ROOT / "_per_metric_work"


# ---------- Cofold confidence ----------

def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _confidence_for_cif(cif: Path, model: str) -> dict | None:
    """Return dict with normalized keys: iptm, ptm, plddt, conf.

    pLDDT is normalized to the [0, 100] scale (Boltz returns [0,1] → ×100).
    ``conf`` comes from confidence_score (boltz) or ranking_score (protenix/af3).
    """
    if model in ("boltz2", "boltz2x"):
        # boltz_input_model_<i>_aligned.cif → confidence_boltz_input_model_<i>.json
        i = cif.stem.replace("_aligned", "").rsplit("_model_", 1)[-1]
        conf_json = cif.parent / f"confidence_boltz_input_model_{i}.json"
        d = _read_json(conf_json)
        if d is None:
            return None
        plddt = d.get("complex_plddt")
        return {
            "iptm": d.get("iptm"),
            "ptm": d.get("ptm"),
            "plddt": (plddt * 100.0) if plddt is not None else None,
            "conf": d.get("confidence_score"),
        }
    if model == "protenix":
        # *_sample_<i>_aligned.cif → *_summary_confidence_sample_<i>.json
        stem = cif.stem.replace("_aligned", "")
        # split off "_sample_<i>"
        if "_sample_" not in stem:
            return None
        prefix, _sep, idx = stem.rpartition("_sample_")
        conf_json = cif.parent / f"{prefix}_summary_confidence_sample_{idx}.json"
        d = _read_json(conf_json)
        if d is None:
            return None
        return {
            "iptm": d.get("iptm"),
            "ptm": d.get("ptm"),
            "plddt": d.get("plddt"),
            "conf": d.get("ranking_score"),
        }
    if model == "alphafold3":
        # CIF in seed-X_sample-Y/ → {name}_seed-X_sample-Y_summary_confidences.json in same dir
        # CIF stem: 10sl_input_seed-42_sample-3_aligned (model run uses {target}_input prefix).
        stem = cif.stem.replace("_aligned", "")
        conf_json = cif.parent / f"{stem}_summary_confidences.json"
        if not conf_json.exists():
            # Fallback: glob in the same directory
            cand = list(cif.parent.glob("*_summary_confidences.json"))
            if cand:
                conf_json = cand[0]
        d = _read_json(conf_json)
        if d is None:
            return None
        return {
            "iptm": d.get("iptm"),
            "ptm": d.get("ptm"),
            "plddt": None,  # AF3 summary doesn't carry an aggregate plddt
            "conf": d.get("ranking_score"),
        }
    return None


def _build_cofold_confidence_map(run_dir: Path) -> dict[str, list[dict | None]]:
    """{tool_short: [conf_dict_per_pose_idx]} matching the staging order."""
    model_subdirs = {
        "boltz2": "boltz2",
        "boltz2x": "boltz2x",
        "protenix": "protenix",
        "af3": "alphafold3",
    }
    out: dict[str, list[dict | None]] = {}
    for short, subdir in model_subdirs.items():
        cifs = sorted((run_dir / "outputs" / subdir).rglob("*_aligned.cif"))
        out[short] = [_confidence_for_cif(c, subdir) for c in cifs]
    return out


def _docking_anchor_confidence(run_dir: Path) -> dict | None:
    summary = run_dir / "inputs" / "docking" / "docking_prep_summary.json"
    if not summary.exists():
        return None
    try:
        d = json.loads(summary.read_text())
    except Exception:
        return None
    cif_path = d.get("cofolding_structure")
    model = d.get("cofolding_model")
    if not cif_path or not model:
        return None
    p = Path(cif_path)
    if not p.exists():
        return None
    # model in summary is "boltz2"/"boltz2x"/"protenix"/"alphafold3"
    return _confidence_for_cif(p, model if model != "af3" else "alphafold3")


# ---------- Boltz pose affinity ----------

def _build_boltz_affinity_map(run_dir: Path) -> dict[tuple[str, int], dict]:
    """{(model_short, cofold_pose_idx): {value, binder_prob}}.

    pose_idx matches the staging order: same `sorted(rglob("*_aligned.cif"))`
    as `_stage_cofolding_poses`. Per-CIF affinity comes from the sibling
    affinity_*.json (key affinity_pred_value) or sample-numbered variants
    (affinity_pred_value{i}) co-located with the same predictions/ subdir.
    """
    out: dict[tuple[str, int], dict] = {}
    for short, subdir in (("boltz2", "boltz2"), ("boltz2x", "boltz2x")):
        cifs = sorted((run_dir / "outputs" / subdir).rglob("*_aligned.cif"))
        for n, cif in enumerate(cifs):
            i = cif.stem.replace("_aligned", "").rsplit("_model_", 1)[-1]
            try:
                model_idx = int(i)
            except Exception:
                continue
            # Affinity JSON in same predictions dir
            aff_json = next(cif.parent.glob("affinity_*.json"), None)
            if aff_json is None:
                continue
            try:
                d = json.loads(aff_json.read_text())
            except Exception:
                continue
            # Prefer per-sample variant matching this model index
            val_k = f"affinity_pred_value{model_idx}" if model_idx > 0 else "affinity_pred_value"
            prob_k = f"affinity_probability_binary{model_idx}" if model_idx > 0 else "affinity_probability_binary"
            val = d.get(val_k, d.get("affinity_pred_value"))
            prob = d.get(prob_k, d.get("affinity_probability_binary"))
            if val is None or prob is None:
                continue
            out[(short, n)] = {"value": float(val), "binder_prob": float(prob)}
    return out


# ---------- Geometry helpers ----------

def _conformer_at(sdf: Path, idx: int) -> Chem.Mol | None:
    try:
        supp = Chem.SDMolSupplier(str(sdf), removeHs=True, sanitize=False)
        for i, m in enumerate(supp):
            if i == idx:
                return m
    except Exception:
        return None
    return None


def _compute_rmsd(pred_mol: Chem.Mol, ref_lig: Chem.Mol, eff_template: Chem.Mol) -> float | None:
    pred_raw = _largest_template_matching_fragment(pred_mol, eff_template)
    if pred_raw is None:
        return None
    pred_recast, _ = _reassign_with_fallback(pred_raw, eff_template)
    if pred_recast is None:
        pred_recast = _sanitize_loose(pred_raw)
    if pred_recast is None:
        return None
    try:
        r = pose_rmsd(pred_recast, ref_lig)
        if not math.isnan(r):
            return float(r)
    except Exception:
        pass
    return _mcs_rmsd(pred_recast, ref_lig)


# ---------- Per-pose row ----------

@dataclass
class PoseRow:
    target: str
    seq_zone: str
    source: str
    pose_name: str
    ba_pred_pkd: float | None
    prmsd: float | None
    lscore: float | None
    iptm: float | None
    ptm: float | None
    plddt: float | None
    conf: float | None
    boltz_aff: float | None      # log_kd_nM (smaller = stronger binder)
    boltz_binder_prob: float | None
    true_rmsd: float | None


# Map pose source → cofold model (for confidence lookup)
_COFOLD_PREFIX = {
    "cofold_boltz2_": "boltz2",
    "cofold_boltz2x_": "boltz2x",
    "cofold_protenix_": "protenix",
    "cofold_af3_": "af3",
}


def _pose_n(pose_name: str, source: str) -> int | None:
    """Extract the cofold pose index from a pose_name like cofold_boltz2_L_3_3.

    Cofold staging writes each conformer as `{source}_{n}`. RMSD-Pred adds
    a trailing `_{record_idx}` which `_canonicalize` collapses, so we
    typically see `cofold_boltz2_L_{n}_{n}`. The first numeric token after
    the source prefix is the cofold pose index.
    """
    if not pose_name.startswith(source + "_"):
        return None
    tail = pose_name[len(source) + 1:].split("_")
    if not tail:
        return None
    if tail[0].isdigit():
        return int(tail[0])
    return None


def evaluate_run(target: str, meta: dict, work_dir: Path) -> list[PoseRow]:
    cif = cif_path_for(target)
    if cif is None:
        return []
    crystal_pdb = work_dir / f"{target}_crystal.pdb"
    if not crystal_pdb.exists() and not dump_crystal_protein_pdb(cif, crystal_pdb):
        return []

    run_dir = RUNS / f"{target}_input"
    # novel2025_runs layout is nested: <RUNS>/<target>_input/<target>_input/
    nested = run_dir / f"{target}_input"
    if nested.exists():
        run_dir = nested
    if not run_dir.exists():
        return []
    cofold_pdb = run_dir / "inputs" / "docking" / "receptor.pdb"
    if not cofold_pdb.exists():
        return []

    us = run_usalign(cofold_pdb, crystal_pdb)
    if us is None:
        return []
    R, t, _tm, _rmsd_prot = us

    # Per-ligand reference table — match prep['ligands'] (which carry the
    # lig_id used in pose filenames: L / L2 / L3 …) to a candidate CCD by
    # canonical SMILES. Multi-cofactor targets (FAD+NAP+ligand-of-interest)
    # used to feed every pose through the FIRST candidate ref, causing
    # atom-count mismatch → MCS fallback at 30s/pose, and occasionally
    # FindMCS hangs past its own timeout in the C-extension. Now each
    # pose's lig_id picks the correct ref.
    prep_path = run_dir / "inputs" / "docking" / "docking_prep_summary.json"
    prep = json.loads(prep_path.read_text()) if prep_path.exists() else {}
    cand_canon = []
    for ccd_, smi_ in zip(meta.get("candidate_ccd_codes") or [],
                          meta.get("candidate_smiles") or []):
        tmpl_ = Chem.MolFromSmiles(smi_) if smi_ else None
        if tmpl_ is None:
            continue
        try:
            canon_ = Chem.MolToSmiles(tmpl_)
        except Exception:
            canon_ = smi_
        cand_canon.append((ccd_, tmpl_, canon_))
    lig_id_to_ref: dict[str, tuple] = {}
    for lig in (prep.get("ligands") or []):
        lig_id = str(lig.get("id") or "L")
        lig_smi = lig.get("smiles")
        if not lig_smi or not cand_canon:
            continue
        lig_mol = Chem.MolFromSmiles(lig_smi)
        if lig_mol is None:
            continue
        try:
            lig_canon = Chem.MolToSmiles(lig_mol)
        except Exception:
            lig_canon = lig_smi
        match = next((c for c in cand_canon if c[2] == lig_canon), None)
        if match is None:
            # Fallback: closest by heavy-atom count.
            match = min(cand_canon,
                        key=lambda c: abs(c[1].GetNumAtoms() - lig_mol.GetNumAtoms()))
        ccd_m, tmpl_m, _ = match
        ref_lig_, eff_ = load_crystal_ligand(cif, ccd_m, tmpl_m)
        if ref_lig_ is not None:
            lig_id_to_ref[lig_id] = (ref_lig_, eff_)
    if not lig_id_to_ref:
        # Legacy single-ligand fallback — load the first candidate.
        smi = (meta.get("candidate_smiles") or [""])[0]
        ccd = (meta.get("candidate_ccd_codes") or [""])[0]
        if not smi or not ccd:
            return []
        template = Chem.MolFromSmiles(smi)
        if template is None:
            return []
        ref_lig_, eff_ = load_crystal_ligand(cif, ccd, template)
        if ref_lig_ is None:
            return []
        lig_id_to_ref["L"] = (ref_lig_, eff_)

    cofold_conf = _build_cofold_confidence_map(run_dir)
    docking_conf = _docking_anchor_confidence(run_dir)
    boltz_aff = _build_boltz_affinity_map(run_dir)

    poses = collect_pose_scores(run_dir)
    # Dedup legacy (pre multi-ligand refactor) entries: when both `cofold_af3`
    # and `cofold_af3_L` exist, the bare key duplicates the same SDFs and would
    # double-count the run. Keep only the `_<lig_id>`-suffixed variant when
    # available.
    sources = {p.source for p in poses}
    suffixed_stems: set[str] = set()
    for s in sources:
        for lig_id in ("L", "L1", "L2", "L3", "L4", "L5"):
            if s.endswith(f"_{lig_id}"):
                suffixed_stems.add(s[: -(len(lig_id) + 1)])
    poses = [p for p in poses if p.source not in suffixed_stems]
    rows: list[PoseRow] = []
    rmsd_cache: dict[tuple[str, int], float | None] = {}

    for p in poses:
        sdf = p.pose_file
        if sdf is None or str(sdf) in ("", "."):
            continue
        if sdf.suffix.lower() != ".sdf":
            sib = sdf.with_suffix(".sdf")
            if not sib.exists():
                continue
            sdf = sib
        if not sdf.exists():
            continue
        # Recover record index
        idx = 0
        tail = p.pose_name.rsplit("_", 1)[-1]
        if tail.isdigit():
            idx = int(tail)

        # Cofold confidence + boltz affinity
        iptm = ptm = plddt = conf = None
        b_aff = b_prob = None
        cofold_short = None
        cofold_n = None
        for prefix, short in _COFOLD_PREFIX.items():
            if p.source.startswith(prefix):
                cofold_short = short
                break
        if cofold_short is not None:
            cofold_n = _pose_n(p.pose_name, p.source)
            cmap = cofold_conf.get(cofold_short, [])
            if cofold_n is not None and cofold_n < len(cmap) and cmap[cofold_n] is not None:
                c = cmap[cofold_n]
                iptm, ptm, plddt, conf = c.get("iptm"), c.get("ptm"), c.get("plddt"), c.get("conf")
            if cofold_short in ("boltz2", "boltz2x") and cofold_n is not None:
                af = boltz_aff.get((cofold_short, cofold_n))
                if af is not None:
                    # log_kd_nM = log10(IC50_uM) + 3 (approx; same convention as Boltz output → nM)
                    # Boltz reports log10(IC50_uM); we convert to log10(Kd_nM) = value + 3
                    b_aff = af["value"] + 3.0
                    b_prob = af["binder_prob"]
        else:
            # Docking pose → use the picked cofold's confidence (constant within the run)
            if docking_conf is not None:
                iptm, ptm, plddt, conf = (
                    docking_conf.get("iptm"),
                    docking_conf.get("ptm"),
                    docking_conf.get("plddt"),
                    docking_conf.get("conf"),
                )

        # True RMSD against crystal — pick ref by pose's ligand id so a
        # multi-cofactor target's NAP pose is scored against NAP ref, not
        # FAD ref (atom-count mismatch → MCS fallback → 30s/pose hangs).
        lig_id_match = None
        for _lid in ("L1", "L2", "L3", "L4", "L5", "L"):
            if p.source.endswith(f"_{_lid}"):
                lig_id_match = _lid
                break
        if lig_id_match is None:
            lig_id_match = "L"
        ref_pair = lig_id_to_ref.get(lig_id_match) or next(iter(lig_id_to_ref.values()), None)
        if ref_pair is None:
            continue
        ref_lig, eff_template = ref_pair

        cache_key = (str(sdf), idx, lig_id_match)
        if cache_key in rmsd_cache:
            rmsd = rmsd_cache[cache_key]
        else:
            mol = _conformer_at(sdf, idx)
            if mol is None:
                rmsd = None
            else:
                # Apply USalign cofold→crystal transform
                transform_mol(mol, R, t)
                rmsd = _compute_rmsd(mol, ref_lig, eff_template)
            rmsd_cache[cache_key] = rmsd

        rows.append(PoseRow(
            target=target,
            seq_zone=meta["seq_zone"],
            source=p.source,
            pose_name=p.pose_name,
            ba_pred_pkd=p.ba_pred_pkd,
            prmsd=p.rmsd_pred,
            lscore=p.lscore,
            iptm=iptm, ptm=ptm, plddt=plddt, conf=conf,
            boltz_aff=b_aff, boltz_binder_prob=b_prob,
            true_rmsd=rmsd,
        ))
    return rows


# ---------- Aggregation ----------

def aggregate(rows: list[PoseRow]) -> dict:
    by_target: dict[str, list[PoseRow]] = {}
    zone_of: dict[str, str] = {}
    for r in rows:
        by_target.setdefault(r.target, []).append(r)
        zone_of[r.target] = r.seq_zone

    # Each scorer key → (label, sort_key)
    BIG = 1e9
    NEG = -1e9
    scorers = {
        "rmsd_pred":  ("Lower pRMSD",       lambda r: (r.prmsd if r.prmsd is not None else BIG)),
        "lscore":     ("Higher LSCORE",     lambda r: -(r.lscore if r.lscore is not None else NEG)),
        "ba_pred":    ("Higher pKd",        lambda r: -(r.ba_pred_pkd if r.ba_pred_pkd is not None else NEG)),
        "iptm":       ("Higher ipTM",       lambda r: -(r.iptm if r.iptm is not None else NEG)),
        "ptm":        ("Higher pTM",        lambda r: -(r.ptm if r.ptm is not None else NEG)),
        "plddt":      ("Higher pLDDT",      lambda r: -(r.plddt if r.plddt is not None else NEG)),
        "conf":       ("Higher conf score", lambda r: -(r.conf if r.conf is not None else NEG)),
        "boltz_aff":  ("Lower log10(Kd_nM)",
                       lambda r: (r.boltz_aff if (r.boltz_aff is not None and (r.boltz_binder_prob or 0) >= 0.5) else BIG)),
        "oracle":     ("Lowest TRUE RMSD",  lambda r: (r.true_rmsd if r.true_rmsd is not None else BIG)),
    }

    summary: dict[str, dict] = {}
    for sc_id, (label, key) in scorers.items():
        zones: dict[str, dict] = {}
        for tgt, prs in by_target.items():
            valid = [r for r in prs if r.true_rmsd is not None]
            if not valid:
                continue
            ordered = sorted(valid, key=key)
            top1 = ordered[0].true_rmsd
            best5 = min(r.true_rmsd for r in ordered[:5])
            z = zone_of[tgt]
            d = zones.setdefault(z, {"n": 0, "top1_lt2": 0, "best5_lt2": 0, "top1_lt1": 0,
                                     "top1_vals": [], "best5_vals": []})
            d["n"] += 1
            if top1 < 2: d["top1_lt2"] += 1
            if top1 < 1: d["top1_lt1"] += 1
            if best5 < 2: d["best5_lt2"] += 1
            d["top1_vals"].append(top1)
            d["best5_vals"].append(best5)
        all_d = {"n": 0, "top1_lt2": 0, "best5_lt2": 0, "top1_lt1": 0,
                 "top1_vals": [], "best5_vals": []}
        for d in zones.values():
            for k in ("n", "top1_lt2", "best5_lt2", "top1_lt1"):
                all_d[k] += d[k]
            all_d["top1_vals"].extend(d["top1_vals"])
            all_d["best5_vals"].extend(d["best5_vals"])
        summary[sc_id] = {"label": label, "zones": zones, "total": all_d}
    return summary


def _med(vals: list[float]) -> float:
    vs = sorted(vals)
    return vs[len(vs) // 2] if vs else float("nan")


@contextmanager
def _time_limit(seconds: int):
    """SIGALRM-based timeout — kills runaway RDKit MCS calls per target.

    Note: not signal-safe inside C extensions; raises TimeoutError on the
    next Python bytecode boundary, which is sufficient for RDKit's
    Python-callable loops.
    """
    def _handler(signum, frame):
        raise TimeoutError("per-target timeout")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def main():
    global SUBMISSIONS, RUNS, WORK
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--submissions-dir", type=Path, default=SUBMISSIONS)
    parser.add_argument("--runs-dir", type=Path, default=RUNS)
    parser.add_argument("--output-csv", type=Path,
                        default=ROOT / "per_pose_scores.csv")
    parser.add_argument("--work-dir", type=Path, default=WORK)
    args = parser.parse_args()
    SUBMISSIONS = args.submissions_dir
    RUNS = args.runs_dir
    WORK = args.work_dir
    print(f"submissions: {SUBMISSIONS}")
    print(f"runs:        {RUNS}")
    print(f"output csv:  {args.output_csv}")
    print(f"work dir:    {WORK}")

    targets = load_targets()
    lgs = sorted(SUBMISSIONS.glob("*_input.lg"))
    novel_lgs = [p for p in lgs if not p.name.startswith("L10")]

    WORK.mkdir(exist_ok=True, parents=True)
    args.output_csv.parent.mkdir(exist_ok=True, parents=True)
    csv_path = args.output_csv
    # Resume support: read already-completed targets from CSV
    done: set[str] = set()
    header_row = ["target", "seq_zone", "source", "pose_name",
                  "ba_pred_pkd", "prmsd", "lscore",
                  "iptm", "ptm", "plddt", "conf",
                  "boltz_aff_log10_kd_nM", "boltz_binder_prob",
                  "true_rmsd"]
    if csv_path.exists():
        with open(csv_path) as f:
            r = csv.reader(f)
            try:
                next(r)
                for row in r:
                    if row:
                        done.add(row[0])
            except StopIteration:
                pass
    print(f"Scoring {len(novel_lgs)} runs per metric (resume: {len(done)} done, {len(novel_lgs) - len(done)} remaining)...")
    mode = "a" if csv_path.exists() and done else "w"
    all_rows: list[PoseRow] = []
    with open(csv_path, mode, newline="") as f:
        writer = csv.writer(f)
        if mode == "w":
            writer.writerow(header_row)
        for i, lg in enumerate(novel_lgs):
            tgt = lg.stem.replace("_input", "")
            if tgt not in targets:
                continue
            if tgt in done:
                continue
            try:
                with _time_limit(180):
                    rs = evaluate_run(tgt, targets[tgt], WORK)
            except TimeoutError:
                print(f"  {tgt}  TIMEOUT (>180s)")
                f.flush()
                continue
            except Exception as e:
                print(f"  {tgt}  CRASH: {e!r}")
                f.flush()
                continue
            for r in rs:
                writer.writerow([r.target, r.seq_zone, r.source, r.pose_name,
                                 r.ba_pred_pkd, r.prmsd, r.lscore,
                                 r.iptm, r.ptm, r.plddt, r.conf,
                                 r.boltz_aff, r.boltz_binder_prob,
                                 r.true_rmsd])
            all_rows.extend(rs)
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(novel_lgs)}] last: {tgt} ({len(rs)} poses)", flush=True)

    # Aggregation reads back the full CSV so resumed runs see all rows.
    all_rows = []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            def _f(k):
                v = row.get(k, "")
                return float(v) if v not in ("", "None") else None
            all_rows.append(PoseRow(
                target=row["target"],
                seq_zone=row["seq_zone"],
                source=row["source"],
                pose_name=row["pose_name"],
                ba_pred_pkd=_f("ba_pred_pkd"),
                prmsd=_f("prmsd"),
                lscore=_f("lscore"),
                iptm=_f("iptm"),
                ptm=_f("ptm"),
                plddt=_f("plddt"),
                conf=_f("conf"),
                boltz_aff=_f("boltz_aff_log10_kd_nM"),
                boltz_binder_prob=_f("boltz_binder_prob"),
                true_rmsd=_f("true_rmsd"),
            ))
    summary = aggregate(all_rows)
    out_lines: list[str] = []
    n_targets = len({r.target for r in all_rows})
    out_lines.append(f"Per-scorer evaluation over {n_targets} targets")
    out_lines.append(f"Total per-pose rows: {len(all_rows)}")
    out_lines.append("")
    header = (f"  {'scorer':<11}  {'label':<22}  {'zone':<8} {'n':>4} "
              f"{'top1<2':>7} {'best5<2':>8} {'top1<1':>7} {'med_t1':>7} {'med_b5':>7}")
    for sc_id, info in summary.items():
        out_lines.append(header)
        out_lines.append("  " + "-" * (len(header) - 2))
        for zone in ("novel", "remote", "related"):
            d = info["zones"].get(zone)
            if not d or d["n"] == 0:
                continue
            out_lines.append(
                f"  {sc_id:<11}  {info['label']:<22}  {zone:<8} {d['n']:>4} "
                f"{d['top1_lt2']:>7} {d['best5_lt2']:>8} {d['top1_lt1']:>7} "
                f"{_med(d['top1_vals']):>7.2f} {_med(d['best5_vals']):>7.2f}"
            )
        d = info["total"]
        if d["n"]:
            out_lines.append(
                f"  {sc_id:<11}  {info['label']:<22}  {'TOTAL':<8} {d['n']:>4} "
                f"{d['top1_lt2']:>7} {d['best5_lt2']:>8} {d['top1_lt1']:>7} "
                f"{_med(d['top1_vals']):>7.2f} {_med(d['best5_vals']):>7.2f}"
            )
        out_lines.append("")
    out_text = "\n".join(out_lines)
    print(out_text)
    summary_path = args.output_csv.with_name("per_metric_summary.txt")
    summary_path.write_text(out_text + "\n")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
