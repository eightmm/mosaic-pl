#!/usr/bin/env python3
"""Aggregate scores from pipeline outputs into submission-ready values.

Collects and ensembles:
- Boltz affinity predictions (all seeds × samples)
- BA-Pred pKd (all poses from all docking tools)
- RMSD-Pred pRMSD (all poses)

Produces:
- Best pose selection (by RMSD-Pred quality)
- LSCORE per pose (from RMSD-Pred)
- AFFNTY for whole complex (log-space ensemble)

Usage:
    python compute_submission_scores.py --run-dir experiments/runs/L2001_input
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PoseScore:
    source: str              # e.g. "vina_cofolding_L", "autodock_gpu_swinsite_L2"
    pose_file: Path
    pose_name: str           # e.g. "vina_cofolding_L_seed_42_3"
    ba_pred_pkd: float | None = None
    rmsd_pred: float | None = None
    rmsd_gt_2a_prob: float | None = None
    docking_score: float | None = None  # kcal/mol
    ligand_id: str | None = None   # "L", "L2", ... — derived from source tool key
    # Cofold confidence (per-pose for cofold sources; docking-anchor cofold's
    # value for vina/adg/pxdock; ``None`` for template poses).
    iptm: float | None = None
    ptm: float | None = None
    plddt: float | None = None
    conf: float | None = None
    # Boltz per-pose affinity (cofold-Boltz poses only).
    boltz_aff_log_kd_nM: float | None = None
    boltz_binder_prob: float | None = None

    @property
    def lscore(self) -> float | None:
        """Convert RMSD-Pred output to [0, 1] confidence."""
        if self.rmsd_gt_2a_prob is None:
            return None
        return max(0.0, min(1.0, 1.0 - self.rmsd_gt_2a_prob))

    @property
    def log_kd_nM(self) -> float | None:
        """Convert BA-Pred pKd to log10(Kd in nM)."""
        if self.ba_pred_pkd is None:
            return None
        # pKd = -log10(Kd in M); log10(Kd in nM) = 9 - pKd
        return 9.0 - self.ba_pred_pkd

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "pose_file": str(self.pose_file),
            "pose_name": self.pose_name,
            "ba_pred_pkd": self.ba_pred_pkd,
            "rmsd_pred": self.rmsd_pred,
            "rmsd_gt_2a_prob": self.rmsd_gt_2a_prob,
            "lscore": self.lscore,
            "log_kd_nM": self.log_kd_nM,
            "ligand_id": self.ligand_id,
            "iptm": self.iptm,
            "ptm": self.ptm,
            "plddt": self.plddt,
            "conf": self.conf,
            "boltz_aff_log_kd_nM": self.boltz_aff_log_kd_nM,
            "boltz_binder_prob": self.boltz_binder_prob,
        }


def _infer_ligand_id(tool: str, known_ligand_ids: list[str] | None = None) -> str | None:
    """Infer ligand id from a tool key.

    The multi-ligand refactor appends ``_{lig_id}`` to every tool key
    (``vina_cofolding_L``, ``cofold_boltz2_L2``, ``template_1abc_vina_L``).
    Match the longest known ligand id at the end of the tool string to
    disambiguate ``L`` vs ``L2`` suffixes.
    """
    if not known_ligand_ids:
        return None
    # Match longest first (L2 should win over L on "cofold_boltz2_L2").
    for lig_id in sorted(known_ligand_ids, key=len, reverse=True):
        if tool == lig_id or tool.endswith(f"_{lig_id}"):
            return lig_id
    return None


def _discover_ligand_ids_from_summary(run_dir: Path) -> list[str]:
    """Read ligand ids from ``inputs/docking/docking_prep_summary.json``."""
    summary = run_dir / "inputs" / "docking" / "docking_prep_summary.json"
    if not summary.exists():
        return []
    try:
        data = json.loads(summary.read_text())
    except Exception:
        return []
    return [str(lig.get("id")) for lig in (data.get("ligands") or []) if lig.get("id")]


# ---------- Cofold confidence loading ----------
#
# Each cofold model writes one CIF per (seed × diffusion_sample) and a sibling
# confidence/affinity JSON. ``run_post_analysis.py`` stages those CIFs into
# ``analysis/poses/cofold_{model}_{lig}_{n}.sdf`` in ``sorted(rglob("*_aligned.cif"))``
# order, so ``n`` is the index into that sorted list. We rebuild the same
# list here to map (model, n) → confidence dict.

_COFOLD_MODEL_SUBDIRS = {
    "boltz2": "boltz2",
    "boltz2x": "boltz2x",
    "protenix": "protenix",
    "af3": "alphafold3",
}

# Source-prefix → cofold model short name (matches `_stage_cofolding_poses`).
_COFOLD_SOURCE_PREFIXES = {
    "cofold_boltz2_": "boltz2",
    "cofold_boltz2x_": "boltz2x",
    "cofold_protenix_": "protenix",
    "cofold_af3_": "af3",
}


def _read_json_safe(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _confidence_for_cif(cif: Path, model: str) -> dict | None:
    """Return ``{iptm, ptm, plddt, conf}`` for a single cofold CIF.

    pLDDT is normalized to the [0, 100] scale (Boltz returns [0,1] → ×100;
    Protenix already on [0,100]; AF3 has no aggregate plddt).
    """
    if model in ("boltz2", "boltz2x"):
        i = cif.stem.replace("_aligned", "").rsplit("_model_", 1)[-1]
        d = _read_json_safe(cif.parent / f"confidence_boltz_input_model_{i}.json")
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
        stem = cif.stem.replace("_aligned", "")
        if "_sample_" not in stem:
            return None
        prefix, _sep, idx = stem.rpartition("_sample_")
        d = _read_json_safe(cif.parent / f"{prefix}_summary_confidence_sample_{idx}.json")
        if d is None:
            return None
        return {
            "iptm": d.get("iptm"),
            "ptm": d.get("ptm"),
            "plddt": d.get("plddt"),
            "conf": d.get("ranking_score"),
        }
    if model == "alphafold3":
        stem = cif.stem.replace("_aligned", "")
        conf_json = cif.parent / f"{stem}_summary_confidences.json"
        if not conf_json.exists():
            cand = list(cif.parent.glob("*_summary_confidences.json"))
            if cand:
                conf_json = cand[0]
        d = _read_json_safe(conf_json)
        if d is None:
            return None
        return {
            "iptm": d.get("iptm"),
            "ptm": d.get("ptm"),
            "plddt": None,
            "conf": d.get("ranking_score"),
        }
    return None


def _build_cofold_confidence_map(run_dir: Path) -> dict[str, list[dict | None]]:
    """``{model_short: [conf_per_pose_idx]}`` matching the staging order."""
    out: dict[str, list[dict | None]] = {}
    for short, subdir in _COFOLD_MODEL_SUBDIRS.items():
        cifs = sorted((run_dir / "outputs" / subdir).rglob("*_aligned.cif"))
        out[short] = [_confidence_for_cif(c, subdir) for c in cifs]
    return out


def _build_boltz_pose_affinity(run_dir: Path) -> dict[tuple[str, int], dict]:
    """``{(model_short, pose_idx): {value, binder_prob}}`` for Boltz cofold poses.

    Per-pose affinity comes from the sibling ``affinity_*.json`` (key
    ``affinity_pred_value``) or sample-numbered variants ``affinity_pred_value{i}``
    co-located with the same ``predictions/`` subdir.
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
            aff_json = next(cif.parent.glob("affinity_*.json"), None)
            if aff_json is None:
                continue
            d = _read_json_safe(aff_json)
            if d is None:
                continue
            val_k = f"affinity_pred_value{model_idx}" if model_idx > 0 else "affinity_pred_value"
            prob_k = f"affinity_probability_binary{model_idx}" if model_idx > 0 else "affinity_probability_binary"
            val = d.get(val_k, d.get("affinity_pred_value"))
            prob = d.get(prob_k, d.get("affinity_probability_binary"))
            if val is None or prob is None:
                continue
            out[(short, n)] = {"value": float(val), "binder_prob": float(prob)}
    return out


def _docking_anchor_confidence(run_dir: Path) -> dict | None:
    """Confidence of the cofold structure used as docking anchor (constant across docked poses)."""
    summary = run_dir / "inputs" / "docking" / "docking_prep_summary.json"
    if not summary.exists():
        return None
    d = _read_json_safe(summary)
    if d is None:
        return None
    cif_path = d.get("cofolding_structure")
    model = d.get("cofolding_model")
    if not cif_path or not model:
        return None
    p = Path(cif_path)
    if not p.exists():
        return None
    return _confidence_for_cif(p, model if model != "af3" else "alphafold3")


def _cofold_pose_idx(pose_name: str, model_short: str, ligand_id: str | None) -> int | None:
    """Recover ``n`` from a staged cofold pose name.

    Staging name is ``cofold_{model}_{lig}_{n}`` (or legacy ``cofold_{model}_{n}``);
    RMSD-Pred appends ``_{record_idx}`` which ``_canonicalize`` collapses to one
    record token. The first numeric token after the source prefix is ``n``.
    """
    candidates = [f"cofold_{model_short}_{ligand_id}_", f"cofold_{model_short}_"] if ligand_id else [f"cofold_{model_short}_"]
    for prefix in candidates:
        if pose_name.startswith(prefix):
            tail = pose_name[len(prefix):].split("_")
            if tail and tail[0].isdigit():
                return int(tail[0])
    return None


def _attach_pose_confidence(
    run_dir: Path,
    pose_scores: list[PoseScore],
) -> None:
    """Populate iptm/ptm/plddt/conf and Boltz pose affinity in-place.

    - cofold_{model}_* poses get per-pose values from the staging-order maps.
    - Non-cofold poses (vina/adg/pxdock) inherit the docking-anchor cofold's
      values (constant — useful as a "model confidence" prior, NOT for ranking
      among docked poses since it doesn't vary).
    - Template poses get nothing.
    """
    cofold_conf = _build_cofold_confidence_map(run_dir)
    boltz_aff = _build_boltz_pose_affinity(run_dir)
    anchor = _docking_anchor_confidence(run_dir)

    for p in pose_scores:
        # Identify cofold model from source prefix.
        model_short = None
        for prefix, m in _COFOLD_SOURCE_PREFIXES.items():
            if p.source.startswith(prefix) or p.source == prefix.rstrip("_"):
                model_short = m
                break

        if model_short is not None:
            n = _cofold_pose_idx(p.pose_name, model_short, p.ligand_id)
            if n is not None:
                conf_list = cofold_conf.get(model_short) or []
                if 0 <= n < len(conf_list) and conf_list[n] is not None:
                    c = conf_list[n]
                    p.iptm = c.get("iptm")
                    p.ptm = c.get("ptm")
                    p.plddt = c.get("plddt")
                    p.conf = c.get("conf")
                if model_short in ("boltz2", "boltz2x"):
                    aff = boltz_aff.get((model_short, n))
                    if aff:
                        p.boltz_aff_log_kd_nM = aff["value"] + 3.0  # log10(IC50 μM) → log10(Kd nM)
                        p.boltz_binder_prob = aff["binder_prob"]
            continue

        # Non-cofold pose → inherit docking anchor (constant, but populated for downstream).
        if p.source.startswith("template_"):
            continue
        if anchor is not None:
            p.iptm = anchor.get("iptm")
            p.ptm = anchor.get("ptm")
            p.plddt = anchor.get("plddt")
            p.conf = anchor.get("conf")


@dataclass
class BoltzAffinity:
    source: str              # e.g. "boltz2/seed_42"
    affinity_value: float    # log10(IC50 μM)
    binder_prob: float       # [0, 1]

    @property
    def log_kd_nM(self) -> float:
        """Convert Boltz affinity_pred_value to log10(Kd in nM).

        Boltz predicts log10(IC50 μM), approximate Kd ≈ IC50 for competitive binders.
        log10(IC50 nM) = log10(IC50 μM) + 3
        """
        return self.affinity_value + 3.0

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "affinity_value": self.affinity_value,
            "binder_prob": self.binder_prob,
            "log_kd_nM": self.log_kd_nM,
        }


@dataclass
class AggregatedScores:
    pose_scores: list[PoseScore] = field(default_factory=list)
    boltz_affinities: list[BoltzAffinity] = field(default_factory=list)
    best_pose: PoseScore | None = None
    ensemble_affinity_nM: float | None = None
    ensemble_log_kd_nM: float | None = None
    ensemble_details: dict = field(default_factory=dict)


def parse_ba_pred_tsv(path: Path) -> dict[str, float]:
    """Parse BA-Pred output TSV → {pose_name: pKd}."""
    if not path.exists():
        return {}
    result = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            name = row.get("Name", "").strip()
            if not name:
                continue
            try:
                result[name] = float(row["pKd"])
            except (ValueError, KeyError):
                pass
    return result


def parse_rmsd_pred_tsv(path: Path) -> dict[str, tuple[float, float]]:
    """Parse RMSD-Pred output TSV → {pose_name: (pRMSD, prob_gt_2A)}."""
    if not path.exists():
        return {}
    result = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            name = row.get("Name", "").strip()
            if not name:
                continue
            try:
                prmsd = float(row["pRMSD"])
                prob = float(row["Is_Above_2A"])
                result[name] = (prmsd, prob)
            except (ValueError, KeyError):
                pass
    return result


def collect_boltz_affinities(run_dir: Path) -> list[BoltzAffinity]:
    """Collect all Boltz affinity predictions across seeds and samples."""
    results = []

    for model_name in ("boltz2", "boltz2x"):
        # Check both multi-seed and flat layouts
        patterns = [
            run_dir / "outputs" / model_name / "seed_*/boltz_results_*/predictions/*/affinity_*.json",
            run_dir / "outputs" / model_name / "boltz_results_*/predictions/*/affinity_*.json",
        ]
        for pattern_path in patterns:
            for affinity_json in Path(pattern_path.anchor).glob(str(pattern_path.relative_to(pattern_path.anchor))):
                try:
                    data = json.loads(affinity_json.read_text())
                except Exception:
                    continue

                # Main prediction
                val = data.get("affinity_pred_value")
                prob = data.get("affinity_probability_binary")
                if val is not None and prob is not None:
                    # Determine source label (seed_X or flat)
                    parts = affinity_json.relative_to(run_dir).parts
                    seed_label = "default"
                    for p in parts:
                        if p.startswith("seed_"):
                            seed_label = p
                            break
                    source = f"{model_name}/{seed_label}"
                    results.append(BoltzAffinity(
                        source=source,
                        affinity_value=float(val),
                        binder_prob=float(prob),
                    ))

                # Numbered variants (affinity_pred_value1, 2, ...)
                for i in range(1, 10):
                    val_k = f"affinity_pred_value{i}"
                    prob_k = f"affinity_probability_binary{i}"
                    if val_k in data and prob_k in data:
                        parts = affinity_json.relative_to(run_dir).parts
                        seed_label = "default"
                        for p in parts:
                            if p.startswith("seed_"):
                                seed_label = p
                                break
                        source = f"{model_name}/{seed_label}/sample_{i}"
                        results.append(BoltzAffinity(
                            source=source,
                            affinity_value=float(data[val_k]),
                            binder_prob=float(data[prob_k]),
                        ))

    return results


def collect_pose_scores(
    run_dir: Path,
    known_ligand_ids: list[str] | None = None,
) -> list[PoseScore]:
    """Collect all pose scores from BA-Pred and RMSD-Pred TSVs.

    Each PoseScore is resolved to the concrete staged file that contains that
    specific pose, plus a record index. ``run_post_analysis.py`` stages every
    docked input under ``outputs/analysis/poses/{stem}.sdf`` (one SDF per
    original pose source, e.g. ``vina_seed_42.sdf``) and BA-Pred/RMSD-Pred
    name poses as ``{stem}_{record_index}``, so the pose_name alone is enough
    to recover both the file and the record.

    ``known_ligand_ids`` (optional): when provided, each PoseScore gets its
    ``ligand_id`` populated by matching the tool key suffix. Without this,
    ``ligand_id`` stays ``None`` and downstream per-ligand grouping is a no-op.
    """
    if known_ligand_ids is None:
        known_ligand_ids = _discover_ligand_ids_from_summary(run_dir)
    analysis_dir = run_dir / "outputs" / "analysis"
    if not analysis_dir.exists():
        return []

    # Map docking tool name → (ba_pred tsv, rmsd_pred tsv)
    tool_files: dict[str, dict[str, Path]] = {}
    for ba_tsv in analysis_dir.glob("ba_pred_*.tsv"):
        key = ba_tsv.stem.replace("ba_pred_", "")
        tool_files.setdefault(key, {})["ba"] = ba_tsv
    for rmsd_tsv in analysis_dir.glob("rmsd_pred_*.tsv"):
        key = rmsd_tsv.stem.replace("rmsd_pred_", "")
        tool_files.setdefault(key, {})["rmsd"] = rmsd_tsv

    staged_dir = run_dir / "outputs" / "analysis" / "poses"

    def _canonicalize(pose_name: str) -> str:
        """Normalize a BA-Pred/RMSD-Pred pose name to ``{stem}_{record}``.

        RMSD-Pred appends a record index to the SDF record's ``_Name``; BA-Pred
        does not. Older cofolding runs staged with ``SetProp("_Name", key_N)``
        therefore produced mismatched names (``cofold_af3_0`` in BA-Pred,
        ``cofold_af3_0_0`` in RMSD-Pred). Collapse by walking from the longest
        possible stem down to find which prefix actually matches a staged file;
        everything after that prefix except the first numeric token is
        discarded. No-op for tools whose names already match a staged file.
        """
        parts = pose_name.split("_")
        for i in range(len(parts), 0, -1):
            prefix = "_".join(parts[:i])
            for ext in (".sdf", ".pdbqt", ".dlg", ".mol2"):
                if (staged_dir / f"{prefix}{ext}").exists():
                    if i == len(parts):
                        return pose_name
                    # Keep one record index after the resolved stem.
                    tail = parts[i] if parts[i].isdigit() else "0"
                    return f"{prefix}_{tail}"
        return pose_name

    pose_scores = []
    for tool, files in tool_files.items():
        raw_ba = parse_ba_pred_tsv(files.get("ba", Path())) if "ba" in files else {}
        raw_rmsd = parse_rmsd_pred_tsv(files.get("rmsd", Path())) if "rmsd" in files else {}

        # Normalize and merge (last-writer-wins per canonical name; ordered
        # traversal means we keep the first-seen value deterministically).
        ba_data: dict[str, float] = {}
        for k, v in raw_ba.items():
            ba_data.setdefault(_canonicalize(k), v)
        rmsd_data: dict[str, tuple[float, float]] = {}
        for k, v in raw_rmsd.items():
            rmsd_data.setdefault(_canonicalize(k), v)

        tool_ligand_id = _infer_ligand_id(tool, known_ligand_ids)
        all_names = set(ba_data.keys()) | set(rmsd_data.keys())
        for name in sorted(all_names):
            ba = ba_data.get(name)
            rmsd_info = rmsd_data.get(name, (None, None))
            pose_file = _resolve_pose_file(run_dir, tool, name) or Path("")
            pose_scores.append(PoseScore(
                source=tool,
                pose_file=pose_file,
                pose_name=name,
                ba_pred_pkd=ba,
                rmsd_pred=rmsd_info[0],
                rmsd_gt_2a_prob=rmsd_info[1],
                ligand_id=tool_ligand_id,
            ))

    # Dedup legacy (pre multi-ligand refactor) entries: when both ``tool`` and
    # ``tool_{lig_id}`` exist, the bare-stem key duplicates the same SDFs and
    # would double-count the run (and inflate consensus self-matches in the new
    # ranker). Keep only the ``_{lig_id}``-suffixed variant when available.
    if known_ligand_ids:
        sources = {p.source for p in pose_scores}
        suffixed_stems: set[str] = set()
        for s in sources:
            for lig_id in known_ligand_ids:
                if s.endswith(f"_{lig_id}"):
                    suffixed_stems.add(s[: -(len(lig_id) + 1)])
        pose_scores = [p for p in pose_scores if p.source not in suffixed_stems]

    _attach_pose_confidence(run_dir, pose_scores)
    return pose_scores


def poses_by_ligand(
    pose_scores: list[PoseScore],
    known_ligand_ids: list[str] | None = None,
) -> dict[str, list[PoseScore]]:
    """Partition pose_scores by ligand_id.

    Returns ``{lig_id: [PoseScore, ...]}`` for every id in ``known_ligand_ids``
    (missing ids map to empty lists). PoseScores with no ligand_id are bucketed
    under the first known id (backwards-compat path for legacy single-ligand
    pose files without the ``_{id}`` suffix).
    """
    if known_ligand_ids is None:
        known_ligand_ids = []
    out: dict[str, list[PoseScore]] = {lig: [] for lig in known_ligand_ids}
    primary = known_ligand_ids[0] if known_ligand_ids else "L"
    out.setdefault(primary, [])
    for p in pose_scores:
        lig = p.ligand_id or primary
        out.setdefault(lig, []).append(p)
    return out


def _split_pose_name(pose_name: str) -> tuple[str, int | None]:
    """Split ``{stem}_{index}`` into ``(stem, index)``. Returns (name, None) if
    no trailing ``_<digits>`` suffix is present."""
    if "_" not in pose_name:
        return pose_name, None
    stem, _, tail = pose_name.rpartition("_")
    if tail.isdigit():
        return stem, int(tail)
    return pose_name, None


def _resolve_pose_file(run_dir: Path, tool: str, pose_name: str) -> Path | None:
    """Find the concrete file that contains the pose named ``pose_name``.

    Prefers the staged SDFs written by ``run_post_analysis.py`` under
    ``outputs/analysis/poses/``. Falls back to legacy flat layouts for
    backwards compatibility with older runs.
    """
    stem, _ = _split_pose_name(pose_name)
    staged_dir = run_dir / "outputs" / "analysis" / "poses"

    # Preferred: staged SDF matching the pose stem exactly
    candidate = staged_dir / f"{stem}.sdf"
    if candidate.exists():
        return candidate
    # Same stem but original format
    for ext in (".pdbqt", ".dlg", ".mol2"):
        c = staged_dir / f"{stem}{ext}"
        if c.exists():
            return c

    # Protenix-Dock multi-record SDF (one file, many poses).
    # Multi-ligand refactor renames the tool key to ``protenix_dock_{lig_id}``
    # (e.g. ``protenix_dock_L``, ``protenix_dock_L2``); each ligand has its
    # own ``poses_{lig_id}.sdf``. Legacy single-ligand layout used the bare
    # ``protenix_dock`` key + ``poses.sdf``.
    if tool.startswith("protenix_dock"):
        pxdock_dir = run_dir / "outputs" / "protenix_dock"
        # Multi-ligand: poses_L.sdf, poses_L2.sdf
        if tool != "protenix_dock":
            lig_id = tool[len("protenix_dock_"):]
            multi = pxdock_dir / f"poses_{lig_id}.sdf"
            if multi.exists():
                return multi
        # Legacy / fallback: poses.sdf
        legacy_pxdock = pxdock_dir / "poses.sdf"
        if legacy_pxdock.exists():
            return legacy_pxdock

    # Legacy flat layouts (single-seed pipelines before multi-seed refactor)
    legacy = [
        run_dir / "outputs" / tool / "docked.sdf",
        run_dir / "outputs" / tool / "docked.pdbqt",
        run_dir / "outputs" / tool / "docking.sdf",
        run_dir / "outputs" / tool / "docking.dlg",
    ]
    for c in legacy:
        if c.exists():
            return c
    return None


# ---------- RRF + consensus pose ranker ----------
#
# Motivation: a single scorer (lscore, plddt, iptm, ...) tops out around 24 %
# top-1 native-rate on novel2025, while the oracle ceiling is ~59 %. Most of
# the gap is "right pose exists but no scorer finds it consistently". RRF
# combines several independently-trained / independently-derived scorers, then
# a cross-family consensus modifier nudges toward poses that *also* show up in
# other tracks (validated: N_fam_consensus ≥ 4 → 62 % native; ≥ 7 → 89 %).
#
# Design choices
# - K = 60 (canonical RRF hyperparameter; not tuned here).
# - pLDDT is normalized within each cofold model via percentile-rank, so
#   boltz2x's lower absolute pLDDT range doesn't unfairly penalise it.
# - Cofold-only scorers contribute 0 for non-cofold poses, giving cofold
#   poses a small natural advantage (matches the empirical hit-rate gap).
# - Consensus support is computed in the cofold/docking receptor frame using
#   the staged SDFs (no re-alignment): pairs are pre-filtered by centroid
#   distance, surviving pairs get full RDKit RMS. Cross-family only.
# - lig_align bonus: template_lig_align is the highest-recall track (24.3 %)
#   but no scorer picks its hits; small additive boost recovers some.

import math
from collections import defaultdict


def _source_family(src: str) -> str:
    if src.startswith("cofold_boltz2x_"):  return "cofold_boltz2x"
    if src.startswith("cofold_boltz2_"):   return "cofold_boltz2"
    if src.startswith("cofold_protenix_"): return "cofold_protenix"
    if src.startswith("cofold_af3_"):      return "cofold_af3"
    if src.startswith("vina_cofolding"):       return "vina_cofold"
    if src.startswith("vina_p2rank"):          return "vina_p2rank"
    if src.startswith("vina_swinsite"):        return "vina_swinsite"
    # Template-consensus pocket variants share one family — the box source is
    # the same kind of signal regardless of which cluster index it points at,
    # so cross-family consensus treats them as a single bucket.
    if src.startswith("vina_template_consensus"):       return "vina_template_consensus"
    if src.startswith("autodock_gpu_template_consensus"): return "adg_template_consensus"
    if src.startswith("autodock_gpu_cofolding"): return "adg_cofold"
    if src.startswith("autodock_gpu_p2rank"):    return "adg_p2rank"
    if src.startswith("autodock_gpu_swinsite"):  return "adg_swinsite"
    if src.startswith("protenix_dock"):          return "pxdock"
    if "template_" in src and "_lig_align_" in src: return "template_lig_align"
    if "template_" in src and "_vina_" in src:      return "template_vina"
    if "template_" in src and "_adg_"  in src:      return "template_adg"
    return src


def _is_cofold(p: PoseScore) -> bool:
    return any(p.source.startswith(pre) for pre in _COFOLD_SOURCE_PREFIXES)


def _cofold_model_short(p: PoseScore) -> str | None:
    for prefix, m in _COFOLD_SOURCE_PREFIXES.items():
        if p.source.startswith(prefix):
            return m
    return None


def _is_lig_align(p: PoseScore) -> bool:
    return "template_" in p.source and "_lig_align_" in p.source


def _rrf_score(poses: list[PoseScore], k: int = 60) -> dict[int, float]:
    """RRF over {lscore, plddt(percentile-normalized), iptm, ptm, conf}.

    Returns ``{id(pose) → rrf_score}``. Scorers' pose pools differ:

    - lscore: all poses (every pose has RMSD-Pred output)
    - plddt: cofold poses only, ranked WITHIN each cofold model so absolute
      scale differences (Boltz [0..1]×100 vs Protenix [0..100]) don't matter
    - iptm / ptm / conf: cofold poses only (degenerate elsewhere — docked
      poses inherit the constant docking-anchor value)
    """
    scores: dict[int, float] = {id(p): 0.0 for p in poses}

    def _accumulate(ranked: list[PoseScore]):
        for r, p in enumerate(ranked):
            scores[id(p)] += 1.0 / (k + r + 1)

    # 1) lscore — all poses, descending
    _accumulate(sorted(
        [p for p in poses if p.lscore is not None],
        key=lambda q: -q.lscore,
    ))

    cofold_poses = [p for p in poses if _is_cofold(p)]

    # 2) plddt — within-model rank to neutralize scale handicap
    by_model: dict[str | None, list[PoseScore]] = defaultdict(list)
    for p in cofold_poses:
        if p.plddt is not None:
            by_model[_cofold_model_short(p)].append(p)
    for ps in by_model.values():
        _accumulate(sorted(ps, key=lambda q: -q.plddt))

    # 3) iptm / ptm / conf — global rank within cofold pool
    for attr in ("iptm", "ptm", "conf"):
        ranked = sorted(
            [p for p in cofold_poses if getattr(p, attr) is not None],
            key=lambda q: -getattr(q, attr),
        )
        _accumulate(ranked)

    return scores


def _pose_coord_array(mol):
    """Heavy-atom coords as ``(N, 3)`` numpy array, or ``None`` if mol invalid."""
    import numpy as np  # local — keep top-level imports unchanged
    if mol is None:
        return None
    try:
        conf = mol.GetConformer()
        coords = np.empty((mol.GetNumAtoms(), 3), dtype=np.float64)
        for i in range(mol.GetNumAtoms()):
            pt = conf.GetAtomPosition(i)
            coords[i] = (pt.x, pt.y, pt.z)
        return coords
    except Exception:
        return None


def _atom_index_rmsd(a, b) -> float | None:
    """Direct atom-index heavy-atom RMSD between two ``(N, 3)`` arrays.

    Same atom count required. No symmetry correction — for the consensus
    check we only need "approximately the same pose"; a few hundred ms of
    CalcRMS substructure-matching per pair is too expensive at batch scale.
    Atom orderings may differ across docking tools but the typical noise
    (~0.5 Å) is well below the 2 Å consensus threshold.
    """
    if a is None or b is None or a.shape != b.shape:
        return None
    import numpy as np
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


def _consensus_support(
    poses: list[PoseScore],
    threshold: float = 2.0,
    centroid_slack: float | None = None,
    mol_cache: dict | None = None,
) -> dict[int, int]:
    """For each pose, count cross-family poses within ``threshold`` Å.

    Per-family staging produces SDFs in a single common receptor frame
    (cofold and docked poses live in their respective cofold/anchor frame;
    template poses are pre-aligned to the cofold frame by the staging step),
    so we don't re-align — heavy-atom RMSD is the raw positional difference.

    Centroid pre-filter (default ``threshold + 1 Å``) skips obvious non-pairs.
    Heavy-atom RMSD via numpy (no symmetry correction — cheap, see
    ``_atom_index_rmsd``).
    """
    if centroid_slack is None:
        centroid_slack = threshold + 1.0
    if mol_cache is None:
        mol_cache = {}

    import numpy as np
    coords_arr: dict[int, "np.ndarray"] = {}
    centroids: dict[int, "np.ndarray"] = {}
    for p in poses:
        mol = _load_pose_mol(p, mol_cache)
        arr = _pose_coord_array(mol)
        if arr is None:
            continue
        coords_arr[id(p)] = arr
        centroids[id(p)] = arr.mean(axis=0)

    fams = {id(p): _source_family(p.source) for p in poses}
    support = {id(p): 0 for p in poses}

    pids = list(coords_arr.keys())
    n = len(pids)
    for i in range(n):
        pid_i = pids[i]
        ai = coords_arr[pid_i]; ci = centroids[pid_i]
        fi = fams[pid_i]; n_atoms_i = ai.shape[0]
        for j in range(i + 1, n):
            pid_j = pids[j]
            if fams[pid_j] == fi:
                continue
            aj = coords_arr[pid_j]
            if aj.shape[0] != n_atoms_i:
                continue
            cj = centroids[pid_j]
            d = ci - cj
            if abs(d[0]) > centroid_slack or abs(d[1]) > centroid_slack or abs(d[2]) > centroid_slack:
                continue
            r = float(np.sqrt(((ai - aj) ** 2).sum(axis=1).mean()))
            if r < threshold:
                support[pid_i] += 1
                support[pid_j] += 1
    return support


def _final_ranker_score(
    poses: list[PoseScore],
    consensus_threshold: float = 2.0,
    consensus_weight: float = 0.1,
    lig_align_bonus: float = 0.2,
    mol_cache: dict | None = None,
) -> dict[int, float]:
    """Combined RRF × consensus modifier × lig_align bonus.

    final(i) = RRF(i) · (1 + consensus_weight · log(1 + support(i))) + bonus(i)

    bonus(i) = ``lig_align_bonus`` × lscore(i) if pose is template_lig_align
    (recovers some of the high-recall lig_align hits that no scorer picks).
    """
    if mol_cache is None:
        mol_cache = {}
    rrf = _rrf_score(poses)
    support = _consensus_support(poses, threshold=consensus_threshold, mol_cache=mol_cache)
    out: dict[int, float] = {}
    for p in poses:
        s = rrf.get(id(p), 0.0)
        sup = support.get(id(p), 0)
        s *= 1.0 + consensus_weight * math.log1p(sup)
        if _is_lig_align(p) and p.lscore is not None:
            s += lig_align_bonus * p.lscore
        out[id(p)] = s
    return out


# ---------- Cluster-then-pick ranker (option A) ----------
#
# Aggregate first, then rank: cluster all poses by 2 Å heavy-atom RMSD into
# basins, score each basin by max(lscore) × log(1 + n_families), pick the
# best basin, then pick the lowest-pRMSD member as representative.
#
# Why this should beat per-pose RRF:
# - decoy basins (Q3 in validate_consensus.py: 67 % false-positive at the
#   per-pose consensus level) get filtered because a basin's max_lscore is
#   typically low when no member is actually native;
# - top-5 reps come from 5 separate clusters → ≥ 2 Å diversity by construction.

def _build_pose_clusters(
    poses: list[PoseScore],
    threshold: float = 2.0,
    mol_cache: dict | None = None,
) -> list[list[PoseScore]]:
    """Single-link cluster poses by heavy-atom RMSD with ``threshold`` link.

    Same coordinate-frame assumption as ``_consensus_support`` (cofold/docking
    poses live in their own cofold/anchor frame; template poses are pre-aligned
    in staging). Centroid pre-filter + numpy direct-RMSD (no symmetry — the
    same SMILES across docking tools may differ in atom ordering by ~ 0.5 Å,
    well below the 2 Å link distance).

    Returns one list-of-PoseScore per cluster. Poses without loadable coords
    are dropped from clustering and won't appear in the result.
    """
    import numpy as np
    if mol_cache is None:
        mol_cache = {}

    arrs: dict[int, "np.ndarray"] = {}
    centroids: dict[int, "np.ndarray"] = {}
    valid: list[PoseScore] = []
    for p in poses:
        mol = _load_pose_mol(p, mol_cache)
        arr = _pose_coord_array(mol)
        if arr is None:
            continue
        arrs[id(p)] = arr
        centroids[id(p)] = arr.mean(axis=0)
        valid.append(p)

    n = len(valid)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    slack = threshold + 1.0
    for i in range(n):
        ai = arrs[id(valid[i])]; ci = centroids[id(valid[i])]
        for j in range(i + 1, n):
            aj = arrs[id(valid[j])]
            if aj.shape[0] != ai.shape[0]:
                continue
            cj = centroids[id(valid[j])]
            d = ci - cj
            if abs(d[0]) > slack or abs(d[1]) > slack or abs(d[2]) > slack:
                continue
            r = float(np.sqrt(((ai - aj) ** 2).sum(axis=1).mean()))
            if r < threshold:
                union(i, j)

    buckets: dict[int, list[PoseScore]] = defaultdict(list)
    for i, p in enumerate(valid):
        buckets[find(i)].append(p)
    return list(buckets.values())


def _cluster_quality(cluster: list[PoseScore]) -> float:
    """``max(lscore in cluster) × log(1 + n_distinct_families)``.

    Returns 0 when no member has a usable lscore (the cluster cannot be
    ranked against others; it falls behind any cluster that does).
    """
    lscores = [p.lscore for p in cluster if p.lscore is not None]
    if not lscores:
        return 0.0
    n_fam = len({_source_family(p.source) for p in cluster})
    return max(lscores) * math.log(1 + n_fam)


# ---------- Cluster quality variants for ablation ----------

def _cq_size(cluster: list[PoseScore]) -> float:
    """Pure consensus: cluster size (n_members). No score quality involved."""
    return float(len(cluster))


def _cq_n_fam(cluster: list[PoseScore]) -> float:
    """Pure cross-family consensus: distinct source families in cluster."""
    return float(len({_source_family(p.source) for p in cluster}))


def _cq_max_lscore(cluster: list[PoseScore]) -> float:
    """Cluster as diversity filter only — pick by best member's lscore."""
    lscores = [p.lscore for p in cluster if p.lscore is not None]
    return max(lscores) if lscores else 0.0


def _cq_sum_lscore(cluster: list[PoseScore]) -> float:
    """Quantity × quality: sum of lscore over members. Both consensus AND quality."""
    return sum(p.lscore for p in cluster if p.lscore is not None)


def _cq_size_x_max_lscore(cluster: list[PoseScore]) -> float:
    """Product variant: n_members × max_lscore. Like sum but max-quality."""
    lscores = [p.lscore for p in cluster if p.lscore is not None]
    if not lscores:
        return 0.0
    return len(cluster) * max(lscores)


def _select_with_cluster_quality(
    poses: list[PoseScore],
    quality_fn,
    k: int,
    threshold: float = 2.0,
    mol_cache: dict | None = None,
) -> list[PoseScore]:
    """Generic top-k cluster picker. Reuses _build_pose_clusters + _cluster_representative."""
    if k <= 0 or not poses:
        return []
    if mol_cache is None:
        mol_cache = {}
    clusters = _build_pose_clusters(poses, threshold=threshold, mol_cache=mol_cache)
    if not clusters:
        cand = [p for p in poses if p.lscore is not None]
        return sorted(cand, key=lambda p: -(p.lscore or 0.0))[:k] or poses[:k]
    sorted_clusters = sorted(clusters, key=quality_fn, reverse=True)
    return [_cluster_representative(c) for c in sorted_clusters[:k]]


def select_top_k_cluster_by(
    poses: list[PoseScore],
    quality: str = "lscore_x_fam",
    k: int = 5,
    threshold: float = 2.0,
    mol_cache: dict | None = None,
) -> list[PoseScore]:
    """Top-k clusters by named quality function.

    Variants:
      - ``lscore_x_fam`` (default; same as ``select_top_k_cluster``):
        ``max_lscore × log(1 + n_fam)``
      - ``size``:           cluster size (pure consensus)
      - ``n_fam``:          distinct source families (pure cross-family)
      - ``max_lscore``:     best member's lscore (cluster as diversity-only)
      - ``sum_lscore``:     sum of lscores (quantity × quality, soft)
      - ``size_x_lscore``:  n_members × max_lscore (quantity × quality, hard)
    """
    fn_map = {
        "lscore_x_fam":  _cluster_quality,
        "size":          _cq_size,
        "n_fam":         _cq_n_fam,
        "max_lscore":    _cq_max_lscore,
        "sum_lscore":    _cq_sum_lscore,
        "size_x_lscore": _cq_size_x_max_lscore,
    }
    fn = fn_map.get(quality)
    if fn is None:
        raise ValueError(f"Unknown cluster quality: {quality!r} (choose from {sorted(fn_map)})")
    return _select_with_cluster_quality(poses, fn, k=k, threshold=threshold, mol_cache=mol_cache)


def _cluster_representative(cluster: list[PoseScore]) -> PoseScore:
    """Pick the cluster member to actually submit.

    Order: smallest pRMSD (lscore tie-break) → max lscore → first member.
    """
    primary = [p for p in cluster if p.rmsd_pred is not None]
    if primary:
        return min(
            primary,
            key=lambda p: (p.rmsd_pred, -(p.lscore if p.lscore is not None else 0.0)),
        )
    cand = [p for p in cluster if p.lscore is not None]
    if cand:
        return max(cand, key=lambda p: p.lscore or 0.0)
    return cluster[0]


def select_best_pose_cluster(
    poses: list[PoseScore],
    threshold: float = 2.0,
    mol_cache: dict | None = None,
) -> PoseScore | None:
    """Cluster-then-pick top-1 (option A — see module-level docstring above)."""
    if not poses:
        return None
    if mol_cache is None:
        mol_cache = {}
    clusters = _build_pose_clusters(poses, threshold=threshold, mol_cache=mol_cache)
    if not clusters:
        cand = [p for p in poses if p.lscore is not None]
        if cand:
            return max(cand, key=lambda p: p.lscore or 0.0)
        return poses[0]
    best = max(clusters, key=_cluster_quality)
    return _cluster_representative(best)


def select_top_k_cluster(
    poses: list[PoseScore],
    k: int = 5,
    threshold: float = 2.0,
    mol_cache: dict | None = None,
) -> list[PoseScore]:
    """Top-k cluster representatives by cluster_quality.

    Each rep comes from a distinct cluster, so they are ≥ ``threshold`` Å
    apart by construction — no extra diversity loop needed.
    """
    if k <= 0 or not poses:
        return []
    if mol_cache is None:
        mol_cache = {}
    clusters = _build_pose_clusters(poses, threshold=threshold, mol_cache=mol_cache)
    if not clusters:
        cand = [p for p in poses if p.lscore is not None]
        return sorted(cand, key=lambda p: -(p.lscore or 0.0))[:k] or poses[:k]
    sorted_clusters = sorted(clusters, key=_cluster_quality, reverse=True)
    return [_cluster_representative(c) for c in sorted_clusters[:k]]


# ---------- Cascaded filter ranker (option B) ----------
#
# Conservative ranker: keep lscore as primary signal (single best top-1 SR
# at 25 %, only beaten by cofold pLDDT which is restricted to cofold pool).
# Use weak secondaries only inside lscore's top-K neighborhood, so the
# strong primary signal isn't diluted.
#
# Promotion rule for template_lig_align: this track has 24 % family-recall
# but no scorer picks its hits. If a lig_align candidate sits within
# `lig_align_margin` of the lscore top, we promote it.

def _lscore_top_k_filter(poses: list[PoseScore], top_k: int) -> list[PoseScore]:
    cand = [p for p in poses if p.lscore is not None]
    cand.sort(key=lambda p: -(p.lscore or 0.0))
    return cand[:top_k]


def select_best_pose_cascaded(
    poses: list[PoseScore],
    top_k_filter: int = 50,
    lig_align_margin: float = 0.1,
) -> PoseScore | None:
    """Top-1 under cascaded filter (option B — see header above)."""
    if not poses:
        return None
    top = _lscore_top_k_filter(poses, top_k_filter)
    if not top:
        ba = [p for p in poses if p.ba_pred_pkd is not None]
        if ba:
            return max(ba, key=lambda p: p.ba_pred_pkd or 0.0)
        return poses[0]
    top_lscore = top[0].lscore or 0.0

    # Lig_align promotion: in the top-K neighborhood, prefer lig_align if
    # within margin (Track 3 has the highest family recall but is never
    # picked by any single scorer).
    lig_aligns = [p for p in top if _is_lig_align(p) and p.lscore is not None]
    if lig_aligns:
        best_la = max(lig_aligns, key=lambda p: p.lscore or 0.0)
        if top_lscore - (best_la.lscore or 0.0) < lig_align_margin:
            return best_la

    # pLDDT tie-break: lscore-primary, plddt secondary (cofold only carries plddt).
    return max(top, key=lambda p: ((p.lscore or 0.0), (p.plddt or 0.0)))


def select_top_k_cascaded(
    poses: list[PoseScore],
    k: int = 5,
    top_k_filter: int = 50,
    lig_align_margin: float = 0.1,
    rmsd_threshold: float = 2.0,
    mol_cache: dict | None = None,
) -> list[PoseScore]:
    """Top-k under cascaded filter, with ≥ ``rmsd_threshold`` heavy-atom diversity.

    Score order = same lscore-primary / plddt-secondary as ``select_best_pose_cascaded``,
    plus a +``lig_align_margin`` lscore boost for lig_align poses (so they can
    appear in the top-5 even if not the absolute best by lscore).
    """
    if k <= 0 or not poses:
        return []
    if mol_cache is None:
        mol_cache = {}
    top = _lscore_top_k_filter(poses, top_k_filter)
    if not top:
        ba = [p for p in poses if p.ba_pred_pkd is not None]
        return sorted(ba, key=lambda p: -(p.ba_pred_pkd or 0.0))[:k] or poses[:k]

    def combined(p: PoseScore) -> tuple:
        boost = lig_align_margin if _is_lig_align(p) else 0.0
        return ((p.lscore or 0.0) + boost, (p.plddt or 0.0))

    ordered = sorted(top, key=combined, reverse=True)
    selected: list[PoseScore] = []
    selected_mols: list = []
    for cand in ordered:
        cand_mol = _load_pose_mol(cand, mol_cache)
        if not selected:
            selected.append(cand); selected_mols.append(cand_mol)
            if len(selected) == k:
                break
            continue
        if cand_mol is None:
            continue
        too_close = False
        for prev in selected_mols:
            if prev is None:
                continue
            r = _pose_pair_rmsd(cand_mol, prev)
            if r is not None and r < rmsd_threshold:
                too_close = True
                break
        if not too_close:
            selected.append(cand); selected_mols.append(cand_mol)
            if len(selected) == k:
                break
    return selected


def select_best_pose(poses: list[PoseScore]) -> PoseScore | None:
    """Pick the pose with highest lscore (RMSD-Pred 1 − P(>2 Å)).

    Production ranker chosen after a full ablation against RRF + consensus,
    cluster-then-pick (6 quality variants), and cascaded filter on the
    novel2025 489-target batch — see ``docs/pose_ranker_design.md``.

    lscore is RMSD-Pred's direct prediction of "is this pose < 2 Å from
    native"; mixing it with indirect proxies (pLDDT / ipTM / pTM / conf /
    ba_pred / consensus support) consistently underperformed.

    Fallback ladder:
      1. lscore present anywhere → max(lscore)
      2. otherwise, max BA-Pred pKd (partial-data corner case, noisy)
      3. otherwise, first pose
    """
    if not poses:
        return None
    cand = [p for p in poses if p.lscore is not None]
    if cand:
        return max(cand, key=lambda p: p.lscore or 0.0)
    scored = [p for p in poses if p.ba_pred_pkd is not None]
    if scored:
        return max(scored, key=lambda p: p.ba_pred_pkd or 0.0)
    return poses[0]


def select_best_pose_pRMSD_legacy(poses: list[PoseScore]) -> PoseScore | None:
    """Legacy ranker (pre-2026-04 prod): smallest pRMSD, lscore tie-break, ba_pred fallback.

    Kept callable for ablation. The default ``select_best_pose`` is now
    lscore-based (best in the v4 full-489 ablation; legacy was -2.7 %p top-1).
    """
    primary = [p for p in poses if p.rmsd_pred is not None]
    if primary:
        return min(
            primary,
            key=lambda p: (
                p.rmsd_pred,
                -(p.lscore if p.lscore is not None else 0.0),
            ),
        )
    scored = [p for p in poses if p.lscore is not None]
    if not scored:
        # Fallback: pick best by BA-Pred
        scored = [p for p in poses if p.ba_pred_pkd is not None]
        if not scored:
            return poses[0] if poses else None
        return max(scored, key=lambda p: p.ba_pred_pkd or 0)
    return max(scored, key=lambda p: p.lscore or 0)


def select_best_pose_rrf_legacy(poses: list[PoseScore]) -> PoseScore | None:
    """Legacy RRF + consensus + lig_align bonus ranker (Apr 2026 attempt).

    Kept callable for ablation. Underperformed lscore on stride=10 sample
    (22.5 % vs 37 %); see ``docs/pose_ranker_design.md`` Ablation §1 for the
    failure analysis (cofold-confidence redundancy + decoy-basin false pos).
    """
    if not poses:
        return None
    mol_cache: dict = {}
    final = _final_ranker_score(poses, mol_cache=mol_cache)
    rankable = [p for p in poses if final.get(id(p), 0.0) > 0.0]
    if rankable:
        return max(rankable, key=lambda p: final[id(p)])
    scored = [p for p in poses if p.ba_pred_pkd is not None]
    if scored:
        return max(scored, key=lambda p: p.ba_pred_pkd or 0.0)
    return poses[0]


def _load_pose_mol(pose: "PoseScore", mol_cache: dict):
    """Load the RDKit Mol for a specific pose (file + record index), with caching.

    Returns ``None`` if the pose file is missing or cannot be parsed. All poses
    live as records inside a single staged SDF (``analysis/poses/{stem}.sdf``)
    or inside ``protenix_dock/poses.sdf``; we advance the SDMolSupplier to the
    correct record and keep the Mol in a per-call cache so repeated accesses
    during greedy selection are free.
    """
    from rdkit import Chem  # type: ignore

    if not pose.pose_file or str(pose.pose_file) in ("", "."):
        return None
    if not pose.pose_file.exists():
        return None

    stem, rec_idx = _split_pose_name(pose.pose_name)
    if rec_idx is None:
        rec_idx = 0

    cache_key = (str(pose.pose_file), rec_idx)
    if cache_key in mol_cache:
        return mol_cache[cache_key]

    suffix = pose.pose_file.suffix.lower()
    source_path = pose.pose_file
    if suffix in (".pdbqt", ".dlg"):
        # Staging produces a parallel .sdf for every pdbqt/dlg; prefer that.
        sdf_sibling = pose.pose_file.with_suffix(".sdf")
        if sdf_sibling.exists():
            source_path = sdf_sibling

    def _read(sanitize: bool):
        try:
            supplier = Chem.SDMolSupplier(str(source_path), removeHs=True, sanitize=sanitize)
            for i, m in enumerate(supplier):
                if i == rec_idx:
                    return m
        except Exception:
            pass
        return None

    mol = _read(sanitize=True)
    if mol is None:
        mol = _read(sanitize=False)

    # Sanitized mols allow RDKit's CalcRMS/GetSubstructMatch to work across
    # pose sources with different atom orderings (e.g. meeko vs PxDock).
    mol_cache[cache_key] = mol
    return mol


def _pose_pair_rmsd(mol_a, mol_b) -> float | None:
    """Heavy-atom RMSD between two pose conformers *in the same frame*.

    Poses come from docking tools that all use the receptor coordinate frame,
    so we do not re-align — this is the raw positional difference. RDKit's
    ``rdMolAlign.CalcRMS`` handles symmetry-equivalent atoms by trying
    substructure matches. If the two mols have different atom counts (which
    should not happen for the same ligand) we return ``None``.
    """
    if mol_a is None or mol_b is None:
        return None
    if mol_a.GetNumAtoms() != mol_b.GetNumAtoms():
        return None
    try:
        from rdkit.Chem import rdMolAlign  # type: ignore
        return float(rdMolAlign.CalcRMS(mol_a, mol_b))
    except Exception:
        # Fallback: direct atom-index RMSD (no symmetry correction)
        try:
            ca = mol_a.GetConformer()
            cb = mol_b.GetConformer()
            n = mol_a.GetNumAtoms()
            s = 0.0
            for i in range(n):
                pa = ca.GetAtomPosition(i)
                pb = cb.GetAtomPosition(i)
                s += (pa.x - pb.x) ** 2 + (pa.y - pb.y) ** 2 + (pa.z - pb.z) ** 2
            return (s / n) ** 0.5
        except Exception:
            return None


def select_diverse_top_k(
    poses: list[PoseScore],
    k: int = 5,
    rmsd_threshold: float = 2.0,
) -> list[PoseScore]:
    """Greedy top-k by lscore with ≥ ``rmsd_threshold`` heavy-atom RMSD diversity.

    Production ranker chosen after a full ablation against RRF + consensus,
    cluster-then-pick (6 quality variants), and cascaded filter on the
    novel2025 489-target batch — see ``docs/pose_ranker_design.md``.

    Algorithm:
        1. Sort poses by lscore descending.
        2. Take the top-1 unconditionally.
        3. Walk the sorted list; accept the next candidate only if its
           heavy-atom RMSD to every already-selected pose is
           ``>= rmsd_threshold``.
        4. Stop at ``k`` selections or when no candidate satisfies diversity.

    If fewer than ``k`` diverse poses exist, the list is shorter than ``k``.

    Falls back to BA-Pred pKd descending if no pose has lscore — a
    partial-data corner case where RMSD-Pred output is missing entirely.

    Diversity check uses numpy direct heavy-atom RMSD (no symmetry correction
    — same-SMILES different-conformer pairs across docking tools have
    consistent enough atom ordering that the sub-Å noise is well below the
    2 Å diversity threshold). RDKit ``CalcRMS`` was 10× slower in the same
    setup.
    """
    if k <= 0 or not poses:
        return []

    cand = [p for p in poses if p.lscore is not None]
    if cand:
        ordered = sorted(cand, key=lambda p: -(p.lscore or 0.0))
    else:
        scored = [p for p in poses if p.ba_pred_pkd is not None]
        if not scored:
            return poses[:k]
        ordered = sorted(scored, key=lambda p: p.ba_pred_pkd or 0.0, reverse=True)

    mol_cache: dict = {}
    selected: list[PoseScore] = []
    selected_mols: list = []

    for c in ordered:
        cand_mol = _load_pose_mol(c, mol_cache)
        if not selected:
            selected.append(c)
            selected_mols.append(cand_mol)
            if len(selected) == k:
                break
            continue
        if cand_mol is None:
            continue
        too_close = False
        for prev_mol in selected_mols:
            if prev_mol is None:
                continue
            r = _pose_pair_rmsd(cand_mol, prev_mol)
            if r is not None and r < rmsd_threshold:
                too_close = True
                break
        if not too_close:
            selected.append(c)
            selected_mols.append(cand_mol)
            if len(selected) == k:
                break

    return selected


def _pose_centroid(p: "PoseScore", mol_cache: dict):
    import numpy as np
    arr = _pose_coord_array(_load_pose_mol(p, mol_cache))
    return None if arr is None else arr.mean(axis=0)


def select_diverse_top_k_anchored(
    poses: list["PoseScore"],
    anchors: list,
    k: int = 5,
    rmsd_threshold: float = 2.0,
    bonus_weight: float = 0.15,
    lam: float = 6.0,
    coverage_dist: float = 8.0,
    ligand_heavy: int | None = None,
    size_gain: float = 1.5,
    size_lo: int = 15,
    size_hi: int = 35,
    max_anchorless: int | None = 2,
) -> list["PoseScore"]:
    """``select_diverse_top_k`` with research/template anchors folded in.

    1. Soft: each pose gets ``bonus_weight · size · weight · exp(-d/lam)`` added
       to its lscore (d = ligand-centroid → nearest anchor), so poses in a known
       functional pocket rank up. ``size`` = ``1 + size_gain·f`` where
       ``f = clamp((ligand_heavy - size_lo)/(size_hi - size_lo), 0, 1)`` — a
       **larger ligand pulls harder toward the template** (a big ligand's
       template binding mode is more diagnostic than its LSCORE; CASP meeting
       request). Small ligands (f≈0) keep LSCORE-dominated ranking.
    2. Hard coverage: if no selected MODEL lands within ``coverage_dist`` Å of an
       anchor, the best-scoring pose that *does* is force-promoted into the MODEL
       set (evicting the weakest, diversity kept when possible). With ≥2 major
       ``template_site`` anchors this splits the MODELs across both sites.
    3. Final order: MODELs are returned sorted by effective score, so for a large
       ligand a template-anchored pose leads (MODEL 1) even at a lower raw
       LSCORE, and the meeting's per-target reordering falls out.

    Falls back to plain ``select_diverse_top_k`` when there are no anchors or no
    lscore-bearing poses (fail-open — submission unchanged).
    """
    import numpy as np
    if not anchors or k <= 0 or not poses:
        return select_diverse_top_k(poses, k=k, rmsd_threshold=rmsd_threshold)
    cand = [p for p in poses if p.lscore is not None]
    if not cand:
        return select_diverse_top_k(poses, k=k, rmsd_threshold=rmsd_threshold)

    mol_cache: dict = {}
    apts = [np.asarray(a.xyz, dtype=float) for a in anchors]
    aw = [float(getattr(a, "weight", 1.0)) for a in anchors]

    # ligand-size gain: derive heavy-atom count from a pose mol if not supplied
    if ligand_heavy is None:
        m0 = _load_pose_mol(cand[0], mol_cache)
        try:
            ligand_heavy = m0.GetNumHeavyAtoms() if m0 is not None else size_lo
        except Exception:
            ligand_heavy = size_lo
    f = max(0.0, min(1.0, (ligand_heavy - size_lo) / float(max(1, size_hi - size_lo))))
    size_mult = 1.0 + size_gain * f

    def nearest(cen):
        if cen is None:
            return None, None
        ds = [float(np.linalg.norm(cen - ap)) for ap in apts]
        j = int(np.argmin(ds))
        return ds[j], j

    eff: dict[int, float] = {}
    dmin: dict[int, float] = {}
    for p in cand:
        cen = _pose_centroid(p, mol_cache)
        d, j = nearest(cen)
        dmin[id(p)] = d if d is not None else 1e9
        bonus = 0.0 if d is None else bonus_weight * size_mult * aw[j] * math.exp(-d / lam)
        eff[id(p)] = (p.lscore or 0.0) + bonus

    # SET membership is LSCORE-driven (keep the genuinely best diverse poses);
    # the template `eff` bonus only drives coverage-promotion choice + final
    # ORDER, so a strong template pose leads without evicting good non-template
    # poses (meeting T2413: template→MODEL 1 but keep old 1-3).
    ordered = sorted(cand, key=lambda p: -(p.lscore or 0.0))
    selected, selected_mols = [], []
    for c in ordered:
        cmol = _load_pose_mol(c, mol_cache)
        if not selected:
            selected.append(c); selected_mols.append(cmol)
        else:
            if cmol is None:
                continue
            if all((pm is None) or ((_pose_pair_rmsd(cmol, pm) or 9e9) >= rmsd_threshold)
                   for pm in selected_mols):
                selected.append(c); selected_mols.append(cmol)
        if len(selected) == k:
            break

    # hard coverage: every anchor should have a MODEL within coverage_dist.
    # Process HIGH-WEIGHT anchors first (template/site before individual catalytic
    # residues) and, when evicting, never remove the *sole* cover of another
    # anchor — so covering a minor anchor can't un-cover the top template (fixes
    # the multi-anchor thrashing seen on targets with many catalytic residues).
    promote_floor = 0.05  # don't force a worthless pose just to tick an anchor

    def _covers(p, ai):
        cen = _pose_centroid(p, mol_cache)
        return cen is not None and float(np.linalg.norm(cen - apts[ai])) <= coverage_dist

    for ai in sorted(range(len(anchors)), key=lambda i: -aw[i]):
        if any(_covers(p, ai) for p in selected):
            continue
        near_pool = [p for p in cand if p not in selected and _covers(p, ai)
                     and (p.lscore or 0.0) >= promote_floor]
        if not near_pool:
            continue  # docking never sampled this site → can't force (flag upstream)
        promote = max(near_pool, key=lambda p: eff[id(p)])
        if len(selected) < k:
            selected.append(promote)
            continue
        # evict weakest-eff MODEL that (a) is not the sole cover of another anchor
        # and (b) is weaker than the pose we'd add
        def _sole_cover(p):
            for aj in range(len(anchors)):
                if aj == ai:
                    continue
                covs = [q for q in selected if _covers(q, aj)]
                if len(covs) == 1 and covs[0] is p:
                    return True
            return False
        evictable = [p for p in selected
                     if not _sole_cover(p) and eff[id(p)] < eff[id(promote)]]
        if not evictable:
            continue  # can't add without un-covering a stronger anchor → skip
        victim = min(evictable, key=lambda p: eff[id(p)])
        selected[selected.index(victim)] = promote

    # hedge: cap how many MODELs may sit at NO anchor at all.
    #
    # The coverage pass above is anchor-driven — it fills an *uncovered anchor*
    # and gives up when docking never sampled there. Nothing removes the mirror
    # case: a pose that covers no anchor, contradicting both the research
    # briefing and the template consensus, yet surviving on raw LSCORE alone
    # (T2455: 3 of 5 MODELs 13-18 Å from the only anchor; T2451: 3 of 5).
    #
    # Keeping `max_anchorless` of them is deliberate — the prior can be wrong
    # (T2451's c-di-GMP pocket is an explicit, unconfirmed homology hypothesis),
    # so the strongest off-anchor pose stays as the hedge. The rest are swapped
    # for the best-scoring anchor-covering poses that keep MODEL diversity.
    if max_anchorless is not None and anchors:
        def _covers_any(p):
            return any(_covers(p, ai) for ai in range(len(anchors)))

        stray = [p for p in selected if not _covers_any(p)]
        n_drop = len(stray) - max_anchorless
        if n_drop > 0:
            # weakest LSCORE strays go first — the top-scoring one is the hedge
            stray.sort(key=lambda p: (p.lscore or 0.0))
            pool = [p for p in cand
                    if p not in selected and _covers_any(p)
                    and (p.lscore or 0.0) >= promote_floor]

            # Replacement preference: a pose covering an anchor the MODEL set
            # still leaves UNCOVERED beats one piling onto an already-covered
            # site. Ranking by `eff` alone deepens the pocket that already won
            # and can leave the strongest template site empty — T2451 kept
            # site1/91% (392 unique PDB) uncovered while MODELs sat at 1-3 PDB
            # clusters, because every at-site pose there scores below 0.15.
            def _rank(p, unc):
                gain = max((aw[ai] for ai in unc if _covers(p, ai)), default=0.0)
                return (-gain, -eff[id(p)])

            for victim in stray[:n_drop]:
                keep = [s for s in selected if s is not victim]
                unc = [ai for ai in range(len(anchors))
                       if not any(_covers(p, ai) for p in keep)]
                pool.sort(key=lambda p: _rank(p, unc))
                keep_mols = [_load_pose_mol(s, mol_cache) for s in keep]
                repl = None
                for c in pool:
                    cm = _load_pose_mol(c, mol_cache)
                    if cm is None:
                        continue
                    if all((pm is None)
                           or ((_pose_pair_rmsd(cm, pm) or 9e9) >= rmsd_threshold)
                           for pm in keep_mols):
                        repl = c
                        break
                if repl is None:
                    break  # no anchored pose left that stays diverse — keep stray
                selected[selected.index(victim)] = repl
                pool.remove(repl)

    # multi-site balance: with >=2 major template_site anchors, ensure each site
    # gets at least `min_per_site` MODELs so the submission splits across both
    # sites (e.g. 3:2, not 4:1) — CASP meeting "2개 site에 나눠서 뽑기". Fail-open.
    try:
        site_pts = [apts[i] for i, a in enumerate(anchors)
                    if getattr(a, "kind", "") == "template_site"]
        if len(site_pts) >= 2 and k >= 2:
            min_per_site = 2 if k >= 4 else 1

            def _site_of(p):
                cen = _pose_centroid(p, mol_cache)
                if cen is None:
                    return None
                return int(np.argmin([float(np.linalg.norm(cen - sp)) for sp in site_pts]))

            def _diverse_ok(cm):
                return all((pm is None) or ((_pose_pair_rmsd(cm, pm) or 9e9) >= rmsd_threshold)
                           for pm in (_load_pose_mol(s, mol_cache) for s in selected))

            for si in range(len(site_pts)):
                while sum(1 for p in selected if _site_of(p) == si) < min_per_site:
                    pool = sorted((p for p in cand if p not in selected and _site_of(p) == si),
                                  key=lambda p: -eff[id(p)])
                    promoted = next((c for c in pool
                                     if (_load_pose_mol(c, mol_cache) is not None)
                                     and _diverse_ok(_load_pose_mol(c, mol_cache))), None)
                    if promoted is None:
                        break
                    evictable = [p for p in selected
                                 if (_site_of(p) is not None)
                                 and sum(1 for q in selected if _site_of(q) == _site_of(p)) > min_per_site]
                    if evictable:
                        victim = min(evictable, key=lambda p: eff[id(p)])
                        selected[selected.index(victim)] = promoted
                    elif len(selected) < k:
                        selected.append(promoted)
                    else:
                        break
    except Exception:
        pass

    # final MODEL order = effective score desc (template-boosted). For a large
    # ligand this lifts a template-anchored pose to MODEL 1 even at lower raw
    # LSCORE; force-promoted site poses no longer sit forced at the tail.
    selected = sorted(selected[:k], key=lambda p: -eff.get(id(p), p.lscore or 0.0))
    return selected


def select_diverse_top_k_rrf_legacy(
    poses: list[PoseScore],
    k: int = 5,
    rmsd_threshold: float = 2.0,
) -> list[PoseScore]:
    """Legacy RRF + consensus + lig_align bonus top-k (Apr 2026 attempt).

    Kept callable for ablation. Replaced by ``select_diverse_top_k`` (lscore-
    primary) after the v4 full-489 ablation showed it lost on best-5 SR.
    """
    if k <= 0 or not poses:
        return []
    mol_cache: dict = {}
    final = _final_ranker_score(poses, mol_cache=mol_cache)
    rankable = [p for p in poses if final.get(id(p), 0.0) > 0.0]
    if rankable:
        ordered = sorted(rankable, key=lambda p: -final[id(p)])
    else:
        scored = [p for p in poses if p.ba_pred_pkd is not None]
        if not scored:
            return poses[:k]
        ordered = sorted(scored, key=lambda p: p.ba_pred_pkd or 0.0, reverse=True)
    selected: list[PoseScore] = []
    selected_mols: list = []
    for c in ordered:
        cand_mol = _load_pose_mol(c, mol_cache)
        if not selected:
            selected.append(c); selected_mols.append(cand_mol)
            if len(selected) == k: break
            continue
        if cand_mol is None:
            continue
        too_close = False
        for prev_mol in selected_mols:
            if prev_mol is None: continue
            r = _pose_pair_rmsd(cand_mol, prev_mol)
            if r is not None and r < rmsd_threshold:
                too_close = True; break
        if not too_close:
            selected.append(c); selected_mols.append(cand_mol)
            if len(selected) == k: break
    return selected


def ensemble_affinity(
    poses: list[PoseScore],
    boltz: list[BoltzAffinity],
    boltz_binder_threshold: float = 0.5,
) -> tuple[float | None, dict]:
    """Compute ensemble affinity in log10(Kd nM) space.

    Strategy:
    1. Collect all BA-Pred log_kd_nM values → median
    2. Collect all Boltz log_kd_nM values (only binder_prob >= threshold) → median
    3. Average the two medians (equal weight)
    4. Return 10^avg as Kd in nM

    Returns (log_kd_nM, details_dict).
    """
    details = {
        "ba_pred_values": [],
        "boltz_values": [],
        "boltz_filtered_out": [],
        "strategy": "log-space ensemble median",
    }

    ba_logs = [p.log_kd_nM for p in poses if p.log_kd_nM is not None]
    if ba_logs:
        details["ba_pred_values"] = ba_logs
        details["ba_pred_median"] = _median(ba_logs)
        details["ba_pred_n"] = len(ba_logs)

    boltz_filtered = []
    for b in boltz:
        if b.binder_prob >= boltz_binder_threshold:
            boltz_filtered.append(b.log_kd_nM)
        else:
            details["boltz_filtered_out"].append({
                "source": b.source,
                "binder_prob": b.binder_prob,
                "reason": f"binder_prob < {boltz_binder_threshold}",
            })

    if boltz_filtered:
        details["boltz_values"] = boltz_filtered
        details["boltz_median"] = _median(boltz_filtered)
        details["boltz_n"] = len(boltz_filtered)

    # Combine
    components = []
    if ba_logs:
        components.append(_median(ba_logs))
    if boltz_filtered:
        components.append(_median(boltz_filtered))

    if not components:
        details["warning"] = "No affinity data available"
        return None, details

    ensemble_log = sum(components) / len(components)
    details["ensemble_log_kd_nM"] = ensemble_log
    details["ensemble_kd_nM"] = 10 ** ensemble_log
    details["num_components"] = len(components)
    return ensemble_log, details


def _median(values: list[float]) -> float:
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return sorted_vals[n // 2]
    return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2


def aggregate(run_dir: Path) -> AggregatedScores:
    """Run all collection and aggregation steps."""
    pose_scores = collect_pose_scores(run_dir)
    boltz_affinities = collect_boltz_affinities(run_dir)
    best = select_best_pose(pose_scores)
    log_kd, details = ensemble_affinity(pose_scores, boltz_affinities)
    kd_nM = 10 ** log_kd if log_kd is not None else None
    return AggregatedScores(
        pose_scores=pose_scores,
        boltz_affinities=boltz_affinities,
        best_pose=best,
        ensemble_affinity_nM=kd_nM,
        ensemble_log_kd_nM=log_kd,
        ensemble_details=details,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate pipeline scores for submission.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON path (default: <run-dir>/outputs/submission_scores.json)")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output = args.output or (run_dir / "outputs" / "submission_scores.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Aggregating scores from {run_dir}")
    scores = aggregate(run_dir)

    print(f"\nPose scores: {len(scores.pose_scores)}")
    for p in scores.pose_scores[:5]:
        print(f"  {p.source}/{p.pose_name}: pKd={p.ba_pred_pkd}, "
              f"pRMSD={p.rmsd_pred}, LSCORE={p.lscore}")

    print(f"\nBoltz affinities: {len(scores.boltz_affinities)}")
    for b in scores.boltz_affinities[:5]:
        print(f"  {b.source}: val={b.affinity_value:.3f}, binder_prob={b.binder_prob:.3f}")

    if scores.best_pose:
        bp = scores.best_pose
        print(f"\nBest pose: {bp.source}/{bp.pose_name}")
        print(f"  LSCORE: {bp.lscore}")
        print(f"  BA-Pred pKd: {bp.ba_pred_pkd}")
        print(f"  pRMSD: {bp.rmsd_pred}")

    if scores.ensemble_affinity_nM is not None:
        print("\nEnsemble affinity:")
        print(f"  log10(Kd nM): {scores.ensemble_log_kd_nM:.3f}")
        print(f"  Kd: {scores.ensemble_affinity_nM:.3g} nM")

    # Write JSON
    result = {
        "run_dir": str(run_dir),
        "pose_scores": [p.to_dict() for p in scores.pose_scores],
        "boltz_affinities": [b.to_dict() for b in scores.boltz_affinities],
        "best_pose": scores.best_pose.to_dict() if scores.best_pose else None,
        "ensemble_log_kd_nM": scores.ensemble_log_kd_nM,
        "ensemble_kd_nM": scores.ensemble_affinity_nM,
        "ensemble_details": scores.ensemble_details,
    }
    output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(f"\nScores written to {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
