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
    source: str              # e.g. "vina", "autodock_gpu/seed_42", "template/1cgh/vina"
    pose_file: Path
    pose_name: str           # e.g. "docked_0"
    ba_pred_pkd: float | None = None
    rmsd_pred: float | None = None
    rmsd_gt_2a_prob: float | None = None
    docking_score: float | None = None  # kcal/mol

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
        }


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


def collect_pose_scores(run_dir: Path) -> list[PoseScore]:
    """Collect all pose scores from BA-Pred and RMSD-Pred TSVs.

    Each PoseScore is resolved to the concrete staged file that contains that
    specific pose, plus a record index. ``run_post_analysis.py`` stages every
    docked input under ``outputs/analysis/poses/{stem}.sdf`` (one SDF per
    original pose source, e.g. ``vina_seed_42.sdf``) and BA-Pred/RMSD-Pred
    name poses as ``{stem}_{record_index}``, so the pose_name alone is enough
    to recover both the file and the record.
    """
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

    pose_scores = []
    for tool, files in tool_files.items():
        ba_data = parse_ba_pred_tsv(files.get("ba", Path())) if "ba" in files else {}
        rmsd_data = parse_rmsd_pred_tsv(files.get("rmsd", Path())) if "rmsd" in files else {}

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
            ))

    return pose_scores


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

    # Protenix-Dock multi-record SDF (one file, many poses)
    pxdock_sdf = run_dir / "outputs" / "protenix_dock" / "poses.sdf"
    if tool == "protenix_dock" and pxdock_sdf.exists():
        return pxdock_sdf

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


def select_best_pose(poses: list[PoseScore]) -> PoseScore | None:
    """Pick the pose with smallest predicted pRMSD (tie-break: highest LSCORE)."""
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
    """Greedy diversity-aware top-k selection.

    Algorithm:
        1. Keep only poses with a usable LSCORE.
        2. Sort by LSCORE descending.
        3. Start with the top-1 pose.
        4. Walk the sorted list; accept the next candidate only if its
           heavy-atom RMSD to every already-selected pose is ``>= rmsd_threshold``.
        5. Stop when ``k`` poses are selected or no candidate satisfies the
           constraint.

    If fewer than ``k`` diverse poses exist, the list is shorter than ``k``.

    Ordering priority:
        1. RMSD-Pred ``pRMSD`` ascending (primary — smaller predicted pose RMSD
           to the native frame is better).
        2. ``LSCORE`` descending as tie-break (equivalent to ``prob_gt_2A``
           ascending).
        3. If neither is available, fall back to BA-Pred ``pKd`` descending.

    LSCORE is still written into the LG MODEL header as the "0..1 confidence"
    readout — only the selection ordering is driven by pRMSD.
    """
    if k <= 0:
        return []

    # Tier 1: poses with a real pRMSD value.
    primary = [p for p in poses if p.rmsd_pred is not None]
    if primary:
        ordered = sorted(
            primary,
            key=lambda p: (
                p.rmsd_pred,                                  # smaller pRMSD first
                -(p.lscore if p.lscore is not None else 0.0), # larger LSCORE first
            ),
        )
    else:
        # Tier 2: no pRMSD anywhere → fall back to BA-Pred pKd descending so a
        # submission can still be produced from partial data.
        scored = [p for p in poses if p.ba_pred_pkd is not None]
        if not scored:
            return poses[:k]
        ordered = sorted(scored, key=lambda p: p.ba_pred_pkd or 0.0, reverse=True)

    mol_cache: dict = {}
    selected: list[PoseScore] = []
    selected_mols: list = []

    for cand in ordered:
        cand_mol = _load_pose_mol(cand, mol_cache)
        # If we cannot load a mol for diversity checking, still allow the first
        # pick so we never return an empty list when data exists.
        if not selected:
            selected.append(cand)
            selected_mols.append(cand_mol)
            if len(selected) == k:
                break
            continue

        if cand_mol is None:
            # Cannot verify diversity, skip defensively.
            continue

        too_close = False
        for prev_mol in selected_mols:
            if prev_mol is None:
                continue
            r = _pose_pair_rmsd(cand_mol, prev_mol)
            if r is None:
                continue
            if r < rmsd_threshold:
                too_close = True
                break

        if not too_close:
            selected.append(cand)
            selected_mols.append(cand_mol)
            if len(selected) == k:
                break

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
