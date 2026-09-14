#!/usr/bin/env python3
"""Unified pose-scoring harness for a CASP17 ligand-series fragment run.

Runs three protein-ligand pose scorers over EVERY modeled pose of one fragment
run and emits a single per-pose CSV:

  * AKScore2  -> full ``akscore2`` (lower=better, docking-score-informed) plus
                the pure-GNN ``akscore2_ens`` (higher=better).
  * GenScore  -> ``score`` (higher=better).
  * EquiScore -> ``test_pred`` (higher=better).

Pose identity is preserved by assigning every pose a unique id
``<fragment>__<sdf_stem>__m<modelIdx>`` and writing it as the SDF molecule title
in one merged multi-mol SDF that is fed to every scorer. Because two of the three
scorers (AKScore2, EquiScore) name poses by *input-file-index* rather than title,
the robust join key is the merged-SDF row index ``i`` which all three echo back
(AKScore2: ``merged_poses_<i>``, GenScore: ``<title>-<i>``,
EquiScore pocket file: ``merged_poses_Compound_<i>``).

AKScore2 full score
-------------------
``akscore2 = akscore2_ens + 0.65 * docking_score`` (lower=better). The docking
score is obtained per pose WITHOUT re-mapping DLG indices, because the pipeline
already embeds it in each staged pose's ``<meeko>`` SDF tag ``free_energy``:
  * AutoDock-GPU poses  -> AutoDock free energy (from the .dlg)      -> ak_source=dlg
  * Vina poses          -> Vina affinity (from the pdbqt REMARK)     -> ak_source=vina_embedded
  * cofolding poses     -> no docking score embedded; rescored live with
                           Vina --score_only (vina python API)       -> ak_source=vina_score_only
  * rescore failed      -> ak_score/ak_dock = NaN, ak_ens kept       -> ak_source=ens_only

Run it with an interpreter that has rdkit + pandas + numpy, e.g. the AKScore2
venv python:

  uv run python scripts/score_poses.py \
      --run-dir experiments/ligand_series/L01/holo/L010001 \
      --device cpu

Only CPU is available on the master node; ``--device cuda`` is a pass-through for
GPU compute nodes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception as exc:  # pragma: no cover
    sys.exit(f"[score_poses] rdkit is required to run this driver: {exc!r}")


# --------------------------------------------------------------------------- #
# Tool locations (overridable by env var or CLI flag)
# --------------------------------------------------------------------------- #
# Scorer repos live inside this repo at external/scorers/<Name>. Paths are
# derived from the repo root so the tree stays relocatable; each is still
# overridable by env var.
_REPO = Path(__file__).resolve().parents[1]
_SCORERS = _REPO / "external" / "scorers"
DEF_AK_PY = os.environ.get("AKSCORE2_PY", str(_SCORERS / "AKScore2/.venv/bin/python"))
DEF_GEN_PY = os.environ.get("GENSCORE_PY", str(_SCORERS / "GenScore/.venv/bin/python"))
DEF_GEN_DIR = os.environ.get("GENSCORE_DIR", str(_SCORERS / "GenScore"))
DEF_EQUI_PY = os.environ.get("EQUISCORE_PY", str(_SCORERS / "EquiScore/.venv/bin/python"))
DEF_EQUI_DIR = os.environ.get("EQUISCORE_DIR", str(_SCORERS / "EquiScore"))
DEF_VINA_PY = os.environ.get("VINA_PY", str(_REPO / ".venvs/protenix-dock/bin/python"))
DEF_MK_PREP = os.environ.get(
    "MK_PREPARE_LIGAND",
    str(_REPO / ".venvs/protenix-dock/bin/mk_prepare_ligand.py"),
)
DEF_GEN_MODEL = os.environ.get(
    "GENSCORE_MODEL", str(_SCORERS / "GenScore/trained_models/GatedGCN_0.5_1.pth")
)
DEF_EQUI_WEIGHT = os.environ.get(
    "EQUISCORE_WEIGHT",
    str(_SCORERS / "EquiScore/workdir/official_weight/save_model_screen.pt"),
)

POCKET_CUTOFF = 10.0        # GenScore union-pocket cutoff (Angstrom)
VINA_BOX = 25.0             # Vina --score_only box edge (Angstrom)
AK_DOCK_WEIGHT = 0.65       # akscore2 = akscore2_ens + 0.65 * docking_score


def log(msg: str) -> None:
    print(f"[score_poses] {msg}", flush=True)


def parse_source_seed(stem: str):
    """Return (source, seed) parsed from an sdf stem.

    ``autodock_gpu_cofolding_1_L_seed_101`` -> ("autodock_gpu_cofolding_1_L", 101)
    ``cofold_boltz2_L``                     -> ("cofold_boltz2_L", None)
    """
    if "_seed_" in stem:
        src, _, tail = stem.rpartition("_seed_")
        try:
            return src, int(tail)
        except ValueError:
            return stem, None
    return stem, None


def dock_kind_for(stem: str) -> str:
    """Classify a pose source for docking-score provenance."""
    s = stem.lower()
    if s.startswith("autodock_gpu") or "_adg_" in s or s.endswith("_adg_l"):
        return "dlg"
    if s.startswith("vina") or "_vina_" in s or s.endswith("_vina_l"):
        return "vina_embedded"
    if s.startswith("cofold"):
        return "cofold"
    return "other"


def read_meeko_free_energy(mol) -> float:
    """Return the meeko free_energy tag if present, else NaN."""
    for prop in ("meeko", "meeko_json"):
        if mol.HasProp(prop):
            try:
                data = json.loads(mol.GetProp(prop))
                fe = data.get("free_energy")
                if fe is not None:
                    return float(fe)
            except Exception:
                pass
    return math.nan


def _free_port() -> int:
    """An actually-free TCP port (asked of the kernel, not guessed)."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sk:
        sk.bind(("127.0.0.1", 0))
        return int(sk.getsockname()[1])


def run_cmd(cmd, cwd=None, env=None, timeout=None):
    """Run a subprocess, capture stdout+stderr, return (rc, output)."""
    log("$ " + " ".join(str(c) for c in cmd) + (f"   (cwd={cwd})" if cwd else ""))
    proc = subprocess.run(
        [str(c) for c in cmd],
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout


# --------------------------------------------------------------------------- #
# Pose discovery + merged SDF construction
# --------------------------------------------------------------------------- #
def ensure_explicit_hydrogens(mol, pose_id: str):
    """Give a pose explicit hydrogens with 3D coordinates.

    Docked poses come back from meeko with every hydrogen present, but the
    cofolding poses are written heavy-atom only. AKScore2 prepares ligands
    through meeko, which needs the hydrogens, so those poses silently scored as
    NaN — every cofold pose of every fragment. Adding them here keeps the pose
    set uniform; GenScore strips hydrogens itself, so it is unaffected.
    """
    if any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        return mol
    try:
        with_h = Chem.AddHs(mol, addCoords=True)
        return with_h if with_h.GetNumConformers() else mol
    except Exception as exc:
        log(f"WARN could not add hydrogens to {pose_id}: {type(exc).__name__}: {exc}")
        return mol


def select_pose_files(sdf_files: list[Path], reduce_mode: str) -> list[Path]:
    """Optionally score only a subset of the staged pose files.

    ``seed1`` keeps one docking seed per binding-site source (plus every
    seedless file, i.e. the cofold and template poses). Measured on 65 scored
    L01 fragments: ~840 -> ~220 poses, and the fused (RRF) fragment ranking is
    unchanged (spearman 1.00 with a p90 aggregate). It is a STAGE-1 shortcut
    only: ~38% of distinct binding modes are dropped, so pose-level work
    (stage-2 models) must go back to the full set, which stays on disk.
    """
    if reduce_mode == "none":
        return sdf_files
    if reduce_mode != "seed1":
        raise ValueError(f"unknown --reduce mode: {reduce_mode}")
    by_source: dict[str, list[tuple[int | None, Path]]] = {}
    seedless: list[Path] = []
    for f in sdf_files:
        source, seed = parse_source_seed(f.stem)
        if seed is None:
            seedless.append(f)
        else:
            by_source.setdefault(source, []).append((seed, f))
    kept = list(seedless)
    for source, items in by_source.items():
        kept.append(sorted(items)[0][1])          # lowest seed id, deterministic
    return sorted(kept)


def build_merged_sdf(run_dir: Path, fragment: str, scoring_dir: Path,
                     reduce_mode: str = "none"):
    """Read every pose, build the merged SDF, and return per-pose records.

    Returns (records, merged_sdf_path, cofold_sdf_path_or_None, reflig_path).
    ``records`` is a list of dicts (one per model, parsed and unparsed) carrying
    fragment/pose_id/source/seed/model_idx/parsed/merged_index/ak_dock/ak_source.
    """
    pose_dir = run_dir / "outputs" / "analysis" / "poses"
    sdf_files = sorted(pose_dir.glob("*.sdf"))
    if not sdf_files:
        sys.exit(f"[score_poses] no *.sdf poses under {pose_dir}")
    if reduce_mode != "none":
        before = len(sdf_files)
        sdf_files = select_pose_files(sdf_files, reduce_mode)
        log(f"--reduce {reduce_mode}: {before} -> {len(sdf_files)} pose files")

    merged_sdf = scoring_dir / "merged_poses.sdf"
    cofold_sdf = scoring_dir / "cofold_poses.sdf"
    reflig_pdb = scoring_dir / "union_reflig.pdb"

    writer = Chem.SDWriter(str(merged_sdf))
    cofold_writer = Chem.SDWriter(str(cofold_sdf))
    reflig = open(reflig_pdb, "w")

    records = []
    merged_index = 0
    reflig_serial = 0
    n_cofold = 0
    failed = []

    for sdf in sdf_files:
        stem = sdf.stem
        source, seed = parse_source_seed(stem)
        kind = dock_kind_for(stem)
        supplier = Chem.SDMolSupplier(str(sdf), removeHs=False, sanitize=True)
        for idx, mol in enumerate(supplier):
            pose_id = f"{fragment}__{stem}__m{idx}"
            rec = {
                "fragment": fragment,
                "pose_id": pose_id,
                "source": source,
                "seed": seed,
                "model_idx": idx,
                "parsed": mol is not None,
                "merged_index": None,
                "ak_dock": math.nan,
                "ak_source": None,
            }
            if mol is None:
                log(f"WARN could not parse {stem} model {idx} (RDKit returned None) -- skipping")
                failed.append(pose_id)
                records.append(rec)
                continue

            mol = ensure_explicit_hydrogens(mol, pose_id)
            mol.SetProp("_Name", pose_id)
            writer.write(mol)
            rec["merged_index"] = merged_index
            merged_index += 1

            # docking score for AKScore2 full score
            fe = read_meeko_free_energy(mol)
            if not math.isnan(fe):
                rec["ak_dock"] = fe
                rec["ak_source"] = "dlg" if kind == "dlg" else "vina_embedded"
            elif kind == "cofold":
                cofold_writer.write(mol)
                n_cofold += 1
                rec["ak_source"] = "pending_vina"  # resolved after rescore
            else:
                rec["ak_source"] = "ens_only"

            # contribute heavy atoms to the union reflig (GenScore pocket)
            conf = mol.GetConformer()
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() <= 1:
                    continue
                p = conf.GetAtomPosition(atom.GetIdx())
                reflig_serial += 1
                el = atom.GetSymbol()
                reflig.write(
                    "HETATM%5d %-4s%1s%3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n"
                    % (reflig_serial % 100000, el[:4], "", "LIG", "L", 1, "", p.x, p.y, p.z, 1.0, 0.0, el[:2])
                )
            records.append(rec)

    writer.close()
    cofold_writer.close()
    reflig.write("END\n")
    reflig.close()

    log(
        f"discovered {len(sdf_files)} sdf files; {merged_index} poses merged; "
        f"{len(failed)} unparseable; {n_cofold} cofold poses need vina --score_only"
    )
    return records, merged_sdf, (cofold_sdf if n_cofold else None), reflig_pdb


# --------------------------------------------------------------------------- #
# AKScore2 (akscore2_ens on the exact staged conformation)
# --------------------------------------------------------------------------- #
def run_akscore2(ak_py, receptor_pdb, merged_sdf, scoring_dir, device):
    ak_tsv = scoring_dir / "akscore2_raw.tsv"
    env = dict(os.environ)
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [
        ak_py, "-m", "akscore2.inference",
        "-r", receptor_pdb, "-l", merged_sdf, "-o", ak_tsv,
        "--device", device, "--batch_size", "8",
        # match the CPUs the scheduler actually gave us: over-requesting cores
        # is what kept extra workers (and therefore free GPUs) off the node
        "--ncpu", os.environ.get("SLURM_CPUS_PER_TASK", "4"),
    ]
    rc, out = run_cmd(cmd, env=env)
    if rc != 0:
        raise RuntimeError(f"AKScore2 failed (rc={rc}):\n{out[-3000:]}")
    df = pd.read_csv(ak_tsv, sep="\t")
    ens = {}
    for _, row in df.iterrows():
        name = str(row["names"])
        try:
            i = int(name.rsplit("_", 1)[1])
        except (ValueError, IndexError):
            continue
        ens[i] = pd.to_numeric(row.get("akscore2_ens"), errors="coerce")
    return ens


# --------------------------------------------------------------------------- #
# Vina --score_only rescore for cofolding poses (protenix-dock venv)
# --------------------------------------------------------------------------- #
_VINA_HELPER = r'''
import json, sys, math
import numpy as np
from vina import Vina

manifest = json.load(open(sys.argv[1]))
receptor = manifest["receptor_pdbqt"]
box = float(manifest["box"])
out = {}
v = Vina(sf_name="vina", verbosity=0)
v.set_receptor(receptor)
for pid, pdbqt in manifest["poses"].items():
    try:
        xs = [[float(l[30:38]), float(l[38:46]), float(l[46:54])]
              for l in open(pdbqt) if l.startswith(("ATOM", "HETATM"))]
        c = list(np.array(xs).mean(0))
        v.set_ligand_from_file(pdbqt)
        v.compute_vina_maps(center=c, box_size=[box, box, box])
        e = v.score()
        out[pid] = {"score": float(e[0]), "ok": True, "err": ""}
    except Exception as exc:
        out[pid] = {"score": None, "ok": False, "err": repr(exc)[:200]}
json.dump(out, open(sys.argv[2], "w"))
'''


def run_vina_rescore(vina_py, mk_prep, cofold_sdf, receptor_pdbqt, scoring_dir, work):
    """Prepare cofold pdbqt (meeko multimol) + Vina --score_only. Returns {pose_id: score}."""
    if not Path(receptor_pdbqt).is_file():
        log(f"WARN receptor pdbqt missing ({receptor_pdbqt}); cofold poses -> ens_only")
        return {}
    pdbqt_dir = work / "cofold_pdbqt"
    if pdbqt_dir.exists():
        shutil.rmtree(pdbqt_dir)
    pdbqt_dir.mkdir(parents=True)
    rc, out = run_cmd(
        [vina_py, mk_prep, "-i", cofold_sdf, "--multimol_outdir", pdbqt_dir],
        timeout=600,
    )
    if rc != 0:
        log(f"WARN mk_prepare_ligand failed (rc={rc}); cofold poses -> ens_only\n{out[-1500:]}")
        return {}
    poses = {p.stem: str(p) for p in sorted(pdbqt_dir.glob("*.pdbqt"))}
    if not poses:
        log("WARN no cofold pdbqt produced; cofold poses -> ens_only")
        return {}
    manifest = {"receptor_pdbqt": str(receptor_pdbqt), "box": VINA_BOX, "poses": poses}
    man_path = work / "vina_manifest.json"
    res_path = work / "vina_scores.json"
    man_path.write_text(json.dumps(manifest))
    helper = work / "_vina_rescore.py"
    helper.write_text(_VINA_HELPER)
    rc, out = run_cmd([vina_py, helper, man_path, res_path], timeout=1800)
    if rc != 0 or not res_path.is_file():
        log(f"WARN vina score_only failed (rc={rc}); cofold poses -> ens_only\n{out[-1500:]}")
        return {}
    res = json.loads(res_path.read_text())
    scores = {}
    for pid, d in res.items():
        if d.get("ok"):
            scores[pid] = d["score"]
        else:
            log(f"WARN vina score_only failed for {pid}: {d.get('err')}")
    return scores


# --------------------------------------------------------------------------- #
# GenScore (union pocket pre-extraction + GatedGCN score)
# --------------------------------------------------------------------------- #
_POCKET_HELPER = r'''
import sys
from GenScore.feats.extract_pocket_prody import extract_pocket
receptor, reflig, workdir, cutoff = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
extract_pocket(receptor, reflig, cutoff, protname="receptor", workdir=workdir)
'''


def run_genscore(gen_py, gen_dir, model, receptor_pdb, reflig_pdb, merged_sdf, scoring_dir, work, device):
    # 1) pre-extract a union pocket (reflig must live in workdir for the .pdb path)
    reflig_in_work = work / "union_reflig.pdb"
    shutil.copyfile(reflig_pdb, reflig_in_work)
    helper = work / "_extract_pocket.py"
    helper.write_text(_POCKET_HELPER)
    env = dict(os.environ)
    env["PYTHONPATH"] = gen_dir + os.pathsep + env.get("PYTHONPATH", "")
    rc, out = run_cmd(
        [gen_py, helper, receptor_pdb, reflig_in_work, str(work), str(POCKET_CUTOFF)],
        env=env, timeout=1200,
    )
    pocket_pdb = work / ("receptor_pocket_%s.pdb" % POCKET_CUTOFF)
    if rc != 0 or not pocket_pdb.is_file():
        raise RuntimeError(f"GenScore pocket extraction failed (rc={rc}):\n{out[-3000:]}")

    # 2) score
    out_prefix = scoring_dir / "genscore_raw"
    env2 = dict(os.environ)
    if device == "cpu":
        env2["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [
        gen_py, "genscore.py",
        "-p", pocket_pdb, "-l", merged_sdf,
        "-e", "gatedgcn", "-m", model, "-o", out_prefix,
    ]
    rc, out = run_cmd(cmd, cwd=str(Path(gen_dir) / "example"), env=env2, timeout=3600)
    out_csv = Path(str(out_prefix) + ".csv")
    if rc != 0 or not out_csv.is_file():
        raise RuntimeError(f"GenScore scoring failed (rc={rc}):\n{out[-3000:]}")
    df = pd.read_csv(out_csv)
    scores = {}
    for _, row in df.iterrows():
        gid = str(row["id"])
        try:
            i = int(gid.rsplit("-", 1)[1])
        except (ValueError, IndexError):
            continue
        scores[i] = pd.to_numeric(row.get("score"), errors="coerce")
    return scores


# --------------------------------------------------------------------------- #
# EquiScore (get_pocket -> Screening)
# --------------------------------------------------------------------------- #
def run_equiscore(equi_py, equi_dir, weight, receptor_pdb, merged_sdf, scoring_dir, work, device):
    tmp_sdfs = work / "equi_sdfs"
    pockets_parent = work / "equi_pockets_parent"
    pockets_name = "tmp_pockets"
    tmp_pockets = pockets_parent / pockets_name
    for d in (tmp_sdfs, pockets_parent):
        if d.exists():
            shutil.rmtree(d)
    tmp_sdfs.mkdir(parents=True)
    tmp_pockets.mkdir(parents=True)

    get_pocket = str(Path(equi_dir) / "get_pocket" / "get_pocket.py")
    rc, out = run_cmd(
        [
            equi_py, get_pocket,
            "--docking_result", merged_sdf,
            "--recptor_pdb", receptor_pdb,
            "--single_sdf_save_path", tmp_sdfs,
            "--pocket_save_dir", tmp_pockets,
            "--process_num", "8",
        ],
        cwd=str(work),   # get_pocket writes BS_tmp_*.pdb into cwd
        timeout=3600,
    )
    n_pockets = len(list(tmp_pockets.glob("*")))
    if n_pockets == 0:
        raise RuntimeError(f"EquiScore get_pocket produced no pockets (rc={rc}):\n{out[-3000:]}")

    equi_csv = scoring_dir / "equiscore_raw.csv"
    ngpu = "0" if device == "cpu" else "1"
    env = dict(os.environ)
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    rc, out = run_cmd(
        [
            equi_py, "Screening.py",
            # Fresh rendezvous port per invocation: EquiScore spins up
            # torch.distributed even for world_size=1 and defaults to 29505, so
            # concurrent workers collide; a per-job port is not enough either
            # because the previous fragment leaves the port in TIME_WAIT.
            "--MASTER_PORT", str(_free_port()),
            "--ngpu", ngpu, "--test",
            "--test_path", str(pockets_parent),
            "--test_name", pockets_name,
            "--pred_save_path", str(equi_csv),
            "--save_model", weight,
        ],
        cwd=str(equi_dir),   # top-level imports need repo on sys.path
        env=env,
        timeout=7200,
    )
    if rc != 0 or not equi_csv.is_file():
        raise RuntimeError(f"EquiScore Screening failed (rc={rc}):\n{out[-3000:]}")
    df = pd.read_csv(equi_csv)
    scores = {}
    for _, row in df.iterrows():
        base = os.path.basename(str(row["test_sample_path"]))
        try:
            i = int(base.rsplit("_", 1)[1])
        except (ValueError, IndexError):
            continue
        scores[i] = pd.to_numeric(row.get("test_pred"), errors="coerce")
    return scores, n_pockets


# --------------------------------------------------------------------------- #
def _collect_boltz_affinity(run_dir: Path) -> dict:
    """Boltz-2 affinity + binder probability for this fragment.

    Boltz writes one ``affinity_*.json`` per cofolding model/seed. These are
    fragment-level (not per-pose) predictions, so they go in the summary rather
    than the per-pose CSV. ``affinity_pred_value`` is a log(IC50)-style number
    (LOWER = stronger binder); ``affinity_probability_binary`` is a binder
    probability in [0,1] (HIGHER = more likely a binder).
    """
    per_model: dict[str, dict] = {}
    for jf in sorted(run_dir.glob("outputs/boltz*/seed_*/**/affinity_*.json")):
        try:
            d = json.loads(jf.read_text())
        except Exception:
            continue
        # .../outputs/<model>/seed_<N>/... -> "boltz2x_seed_101"
        parts = jf.relative_to(run_dir / "outputs").parts
        key = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else jf.stem
        per_model[key] = {
            "affinity_pred_value": d.get("affinity_pred_value"),
            "affinity_probability_binary": d.get("affinity_probability_binary"),
        }
    vals = [v["affinity_pred_value"] for v in per_model.values()
            if isinstance(v["affinity_pred_value"], (int, float))]
    probs = [v["affinity_probability_binary"] for v in per_model.values()
             if isinstance(v["affinity_probability_binary"], (int, float))]
    return {
        "per_model": per_model,
        "n_models": len(per_model),
        "mean_affinity_pred_value": (sum(vals) / len(vals)) if vals else None,
        "mean_binder_prob": (sum(probs) / len(probs)) if probs else None,
        "max_binder_prob": max(probs) if probs else None,
        "direction": {
            "affinity_pred_value": "lower_is_better",
            "affinity_probability_binary": "higher_is_better",
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="fragment run dir (e.g. .../L01/holo/L010001)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="compute device (cuda = GPU nodes)")
    ap.add_argument("--genscore-model", default=DEF_GEN_MODEL)
    ap.add_argument("--equiscore-weight", default=DEF_EQUI_WEIGHT)
    ap.add_argument("--ak-py", default=DEF_AK_PY)
    ap.add_argument("--gen-py", default=DEF_GEN_PY)
    ap.add_argument("--gen-dir", default=DEF_GEN_DIR)
    ap.add_argument("--equi-py", default=DEF_EQUI_PY)
    ap.add_argument("--equi-dir", default=DEF_EQUI_DIR)
    ap.add_argument("--vina-py", default=DEF_VINA_PY)
    ap.add_argument("--mk-prepare", default=DEF_MK_PREP)
    ap.add_argument("--skip", default="", help="comma list of tools to skip: akscore2,genscore,equiscore")
    ap.add_argument("--vina-rescore", action="store_true",
                    help="rescore cofold poses with vina --score_only so they get a "
                         "full ak_score (off by default: ~2%% of poses, no effect on "
                         "the fragment ranking, costs ~16 s/fragment)")
    ap.add_argument("--reduce", default="none", choices=["none", "seed1"],
                    help="score a subset of poses: 'seed1' keeps one docking seed "
                         "per binding-site source (~4x cheaper, fused ranking "
                         "unchanged; stage-1 only — drops ~38%% of binding modes)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    fragment = run_dir.name
    receptor_pdb = str(run_dir / "inputs" / "docking" / "receptor.pdb")
    receptor_pdbqt = str(run_dir / "inputs" / "docking" / "receptor.pdbqt")
    if not Path(receptor_pdb).is_file():
        sys.exit(f"[score_poses] receptor not found: {receptor_pdb}")

    scoring_dir = run_dir / "outputs" / "scoring"
    scoring_dir.mkdir(parents=True, exist_ok=True)
    work = scoring_dir / "_work"
    work.mkdir(parents=True, exist_ok=True)

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    summary = {
        "fragment": fragment,
        "run_dir": str(run_dir),
        "device": args.device,
        "reduce": args.reduce,
        "genscore_model": args.genscore_model,
        "equiscore_weight": args.equiscore_weight,
        "elapsed_sec": {},
        "errors": {},
        "score_directions": {
            "ak_score": "lower_is_better",
            # ak_ens is a docking-score-like energy: lower is better. It is summed
            # with ak_dock (also lower-is-better) to form ak_score, so all three
            # share one direction.
            "ak_ens": "lower_is_better",
            "ak_dock": "lower_is_better",
            "gen_score": "higher_is_better",
            "equi_pred": "higher_is_better",
        },
    }

    # ---- build merged SDF ------------------------------------------------- #
    t0 = time.time()
    records, merged_sdf, cofold_sdf, reflig_pdb = build_merged_sdf(run_dir, fragment, scoring_dir, args.reduce)
    summary["elapsed_sec"]["build_merged"] = round(time.time() - t0, 1)
    merged = str(merged_sdf)
    idx2pid = {r["merged_index"]: r["pose_id"] for r in records if r["merged_index"] is not None}
    pid2rec = {r["pose_id"]: r for r in records}
    n_parsed = len(idx2pid)
    n_failed = sum(1 for r in records if not r["parsed"])
    summary["poses_in"] = len(records)
    summary["poses_parsed"] = n_parsed
    summary["poses_failed_parse"] = n_failed
    summary["failed_parse_ids"] = [r["pose_id"] for r in records if not r["parsed"]]

    # ---- vina --score_only rescore for cofold poses ---------------------- #
    # Off by default: cofold poses are ~2% of a fragment's poses and never reach
    # the ak_score top decile, so dropping their docking score leaves the
    # fragment ranking identical (spearman 1.000 over 108 fragments) while
    # saving ~16 s of the ~79 s per fragment. Re-enable with --vina-rescore when
    # per-pose ak_score is wanted for cofold poses (e.g. stage-2 pose work).
    ak_ens = {}
    if cofold_sdf is not None and args.vina_rescore and "akscore2" not in skip:
        t0 = time.time()
        try:
            vscores = run_vina_rescore(
                args.vina_py, args.mk_prepare, str(cofold_sdf), receptor_pdbqt, scoring_dir, work
            )
        except Exception as exc:
            log(f"WARN vina rescore raised {exc!r}; cofold -> ens_only")
            vscores = {}
        summary["elapsed_sec"]["vina_rescore"] = round(time.time() - t0, 1)
        for r in records:
            if r.get("ak_source") == "pending_vina":
                pid = r["pose_id"]
                if pid in vscores:
                    r["ak_dock"] = vscores[pid]
                    r["ak_source"] = "vina_score_only"
                else:
                    r["ak_source"] = "ens_only"
    else:
        for r in records:
            if r.get("ak_source") == "pending_vina":
                r["ak_source"] = "ens_only"

    # ---- AKScore2 --------------------------------------------------------- #
    if "akscore2" not in skip:
        t0 = time.time()
        try:
            ak_ens = run_akscore2(args.ak_py, receptor_pdb, merged, scoring_dir, args.device)
        except Exception as exc:
            summary["errors"]["akscore2"] = repr(exc)[:500]
            log(f"ERROR AKScore2: {exc!r}")
        summary["elapsed_sec"]["akscore2"] = round(time.time() - t0, 1)

    # ---- GenScore --------------------------------------------------------- #
    gen_scores = {}
    if "genscore" not in skip:
        t0 = time.time()
        try:
            gen_scores = run_genscore(
                args.gen_py, args.gen_dir, args.genscore_model, receptor_pdb,
                str(reflig_pdb), merged, scoring_dir, work, args.device,
            )
        except Exception as exc:
            summary["errors"]["genscore"] = repr(exc)[:500]
            log(f"ERROR GenScore: {exc!r}")
        summary["elapsed_sec"]["genscore"] = round(time.time() - t0, 1)

    # ---- EquiScore -------------------------------------------------------- #
    equi_scores = {}
    if "equiscore" not in skip:
        t0 = time.time()
        try:
            equi_scores, _ = run_equiscore(
                args.equi_py, args.equi_dir, args.equiscore_weight, receptor_pdb,
                merged, scoring_dir, work, args.device,
            )
        except Exception as exc:
            summary["errors"]["equiscore"] = repr(exc)[:500]
            log(f"ERROR EquiScore: {exc!r}")
        summary["elapsed_sec"]["equiscore"] = round(time.time() - t0, 1)

    # ---- assemble unified CSV -------------------------------------------- #
    rows = []
    for r in records:
        i = r["merged_index"]
        ens = ak_ens.get(i, math.nan) if i is not None else math.nan
        dock = r["ak_dock"]
        ak_source = r["ak_source"] or "ens_only"
        if not (isinstance(ens, float) and math.isnan(ens)) and not math.isnan(dock):
            ak_score = float(ens) + AK_DOCK_WEIGHT * float(dock)
        else:
            ak_score = math.nan
        rows.append({
            "fragment": r["fragment"],
            "pose_id": r["pose_id"],
            "source": r["source"],
            "seed": r["seed"] if r["seed"] is not None else math.nan,
            "model_idx": r["model_idx"],
            "ak_score": ak_score,
            "ak_ens": ens,
            "ak_dock": dock,
            "ak_source": ak_source,
            "gen_score": gen_scores.get(i, math.nan) if i is not None else math.nan,
            "equi_pred": equi_scores.get(i, math.nan) if i is not None else math.nan,
        })
    df = pd.DataFrame(rows, columns=[
        "fragment", "pose_id", "source", "seed", "model_idx",
        "ak_score", "ak_ens", "ak_dock", "ak_source", "gen_score", "equi_pred",
    ])
    out_csv = scoring_dir / "pose_scores.csv"
    df.to_csv(out_csv, index=False, na_rep="NaN")

    # ---- summary counts --------------------------------------------------- #
    def n_real(col):
        return int(df[col].notna().sum())

    all3 = int((df["ak_score"].notna() & df["gen_score"].notna() & df["equi_pred"].notna()).sum())
    # "scored by all 3" using ak_ens as AKScore2 presence (ak_score can be NaN for ens_only)
    all3_ens = int((df["ak_ens"].notna() & df["gen_score"].notna() & df["equi_pred"].notna()).sum())
    summary["scored_counts"] = {
        "ak_score": n_real("ak_score"),
        "ak_ens": n_real("ak_ens"),
        "gen_score": n_real("gen_score"),
        "equi_pred": n_real("equi_pred"),
        "all3_full_akscore": all3,
        "all3_using_ak_ens": all3_ens,
    }
    summary["ak_source_breakdown"] = {
        k: int(v) for k, v in df["ak_source"].value_counts().items()
    }
    summary["output_csv"] = str(out_csv)
    summary["boltz_affinity"] = _collect_boltz_affinity(run_dir)
    (scoring_dir / "scoring_summary.json").write_text(json.dumps(summary, indent=2))

    log(f"wrote {out_csv} ({len(df)} rows)")
    log(f"scored: ak_ens={summary['scored_counts']['ak_ens']} "
        f"ak_score={summary['scored_counts']['ak_score']} "
        f"gen={summary['scored_counts']['gen_score']} "
        f"equi={summary['scored_counts']['equi_pred']} "
        f"all3(full)={all3} all3(ak_ens)={all3_ens}")
    log("ak_source breakdown: " + json.dumps(summary["ak_source_breakdown"]))
    if summary["errors"]:
        log("ERRORS: " + json.dumps(summary["errors"]))


if __name__ == "__main__":
    main()
