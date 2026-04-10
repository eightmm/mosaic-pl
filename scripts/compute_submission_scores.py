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
import math
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
    """Collect all pose scores from BA-Pred and RMSD-Pred TSVs."""
    analysis_dir = run_dir / "outputs" / "analysis"
    if not analysis_dir.exists():
        return []

    # Map docking tool name → (ba_pred tsv, rmsd_pred tsv, pose file)
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

        # Find the corresponding pose file
        pose_file = _find_pose_file(run_dir, tool)

        all_names = set(ba_data.keys()) | set(rmsd_data.keys())
        for name in sorted(all_names):
            ba = ba_data.get(name)
            rmsd_info = rmsd_data.get(name, (None, None))
            pose_scores.append(PoseScore(
                source=tool,
                pose_file=pose_file or Path(""),
                pose_name=name,
                ba_pred_pkd=ba,
                rmsd_pred=rmsd_info[0],
                rmsd_gt_2a_prob=rmsd_info[1],
            ))

    return pose_scores


def _find_pose_file(run_dir: Path, tool: str) -> Path | None:
    """Locate the docked pose file for a given tool name."""
    candidates = [
        run_dir / "outputs" / tool / "docked.sdf",
        run_dir / "outputs" / tool / "docked.pdbqt",
        run_dir / "outputs" / tool / "docking.sdf",
        run_dir / "outputs" / tool / "docking.dlg",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback: search
    for ext in (".sdf", ".pdbqt", ".dlg"):
        for f in (run_dir / "outputs").rglob(f"*{tool}*{ext}"):
            return f
    return None


def select_best_pose(poses: list[PoseScore]) -> PoseScore | None:
    """Pick the pose with highest LSCORE (lowest RMSD-Pred prob > 2A)."""
    scored = [p for p in poses if p.lscore is not None]
    if not scored:
        # Fallback: pick best by BA-Pred
        scored = [p for p in poses if p.ba_pred_pkd is not None]
        if not scored:
            return poses[0] if poses else None
        return max(scored, key=lambda p: p.ba_pred_pkd or 0)
    return max(scored, key=lambda p: p.lscore or 0)


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
        print(f"\nEnsemble affinity:")
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
