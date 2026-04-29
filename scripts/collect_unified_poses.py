#!/usr/bin/env python3
"""Collect unified pose archive from one or more CASP17 run directories.

Per target, produces::

    <output>/<target>/
        poses.sdf           - multi-record SDF of every loadable pose
        manifest.csv        - one row per pose with parsed source / scores /
                              pose-file path / oracle markers
        receptors/
            cofold_best.pdb              # cofold model picked by docking-prep
            cofold_samples/<id>.pdb      # one PDB per cofold sample (~100/target)
            template_<pdb>.pdb           # Track 2/3 receptors (when present)
            crystal.pdb                  # ground-truth protein (when --rcsb-dir
                                         # supplied and target's CIF is found)
        crystal_ligand_<lid>.sdf         # ground-truth ligand (when crystal)

Batch mode (pass ``--runs-dir``) walks every ``*_input`` subdirectory and
emits ``<output>/_summary.csv`` — the concatenation of every per-target
manifest with one extra ``target`` column. That's the single CSV to
load into pandas for cross-target analysis (zone breakdown, oracle vs
top-1 SR, source-family contributions, etc.).

Pose loading reuses ``compute_submission_scores._load_pose_mol`` so all
cofold / vina / adg / pxdock / template tracks go through the same
RDKit-based reader. Failures are logged and skipped per-pose so a single
broken file doesn't kill the whole target.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

# Reuse the existing pose-score collector + mol loader. They handle
# every variant + multi-ligand layout already.
from compute_submission_scores import (  # noqa: E402
    PoseScore,
    collect_pose_scores,
    _load_pose_mol,
)


# --------------------------------------------------------------------------- #
# Source / pose-name parsers                                                  #
# --------------------------------------------------------------------------- #

_BOX_SOURCES = ("cofolding_", "swinsite", "p2rank", "template_consensus_")
_RE_SEED = re.compile(r"_seed[_-](\d+)")
_RE_POSE_TAIL = re.compile(r"_(\d+)$")
_RE_SAMPLE = re.compile(r"sample[_-](\d+)")
_RE_TEMPLATE = re.compile(r"^template_([0-9a-zA-Z]+)_(vina|adg|autodock_gpu|pxdock|protenix_dock|lig_align)")


def parse_source(source: str, ligand_id: str | None) -> dict[str, str | None]:
    """Crack a tool key like ``vina_template_consensus_3_L`` into structured
    bits the manifest can index by. Returns a dict with::

        family         coarse bucket (cofold_boltz2, vina, adg, pxdock,
                       template_vina, template_adg, template_lig_align, ...)
        box_source     binding-site source for vina/adg variants
                       (cofolding / swinsite / p2rank / template_consensus_N)
        template_pdb   PDB id when the pose came from Track 2/3 docking
        track          0 = cofold itself, 1 = cofold-receptor docking,
                       2 = template-box docking, 3 = lig-align
    """
    family: str
    box_source: str | None = None
    template_pdb: str | None = None
    track = 1  # docking-on-cofold default; overridden per pattern

    # Strip trailing ligand id once so the remaining tokens line up cleanly.
    base = source
    if ligand_id and base.endswith("_" + ligand_id):
        base = base[: -(len(ligand_id) + 1)]

    if base.startswith("cofold_"):
        family = base  # cofold_boltz2 / cofold_boltz2x / cofold_protenix / cofold_af3
        track = 0
    elif base.startswith("template_"):
        m = _RE_TEMPLATE.match(base)
        if m:
            template_pdb = m.group(1)
            method = m.group(2)
            if method in ("autodock_gpu", "adg"):
                family = "template_adg"
            elif method == "vina":
                family = "template_vina"
            elif method in ("pxdock", "protenix_dock"):
                family = "template_pxdock"
            elif method == "lig_align":
                family = "template_lig_align"
                track = 3
            else:
                family = f"template_{method}"
            track = 3 if method == "lig_align" else 2
        else:
            family = "template_unknown"
            track = 2
    elif base.startswith("vina_"):
        family = "vina"
        box_source = base[len("vina_"):] or None
    elif base.startswith("autodock_gpu_"):
        family = "adg"
        box_source = base[len("autodock_gpu_"):] or None
    elif base.startswith("protenix_dock") or base == "pxdock":
        family = "pxdock"
    else:
        family = base

    return {
        "family": family,
        "box_source": box_source,
        "template_pdb": template_pdb,
        "track": track,
    }


def parse_pose_name(pose_name: str) -> dict[str, int | None]:
    """Pull seed + pose index out of a name like
    ``vina_cofolding_L_seed_42_3`` or ``cofold_boltz2_L_seed_42_sample_0``.
    """
    seed: int | None = None
    pose_idx: int | None = None
    sample: int | None = None
    m = _RE_SEED.search(pose_name)
    if m:
        seed = int(m.group(1))
    sm = _RE_SAMPLE.search(pose_name)
    if sm:
        sample = int(sm.group(1))
    pm = _RE_POSE_TAIL.search(pose_name)
    if pm:
        pose_idx = int(pm.group(1))
    return {"seed": seed, "pose_idx": pose_idx, "sample_idx": sample}


# --------------------------------------------------------------------------- #
# Receptor extraction helpers                                                 #
# --------------------------------------------------------------------------- #

def _strip_ligand_to_pdb(cif_path: Path, out_pdb: Path) -> bool:
    """Write the protein-only PDB of a CIF (drops non-polymers + waters)."""
    import gemmi
    try:
        st = gemmi.read_structure(str(cif_path))
    except Exception as e:
        print(f"[receptors] gemmi failed on {cif_path}: {e}", flush=True)
        return False
    for model in st:
        polymer_chains = []
        for chain in model:
            polymer = chain.get_polymer()
            if polymer and len(polymer) > 0:
                polymer_chains.append(chain.name)
        # Remove everything that's not a polymer chain (ligands, waters,
        # unknown metals). Done in two passes because gemmi mutates in
        # place and we need the chain list snapshot first.
        for chain in list(model):
            if chain.name not in polymer_chains:
                model.remove_chain(chain.name)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    try:
        st.write_pdb(str(out_pdb))
    except Exception as e:
        print(f"[receptors] write_pdb failed for {cif_path}: {e}", flush=True)
        return False
    return True


def _gather_cofold_cifs(run_dir: Path) -> list[tuple[str, Path]]:
    """Return ``(stem, cif_path)`` pairs for every aligned cofold sample.

    Stem follows ``{model}_seed_{N}_sample_{M}`` convention so the cofold
    pose key ``cofold_{model}_{lid}_seed_{N}_sample_{M}`` can join to it
    by string slicing.
    """
    pairs: list[tuple[str, Path]] = []
    out_dir = run_dir / "outputs"

    # boltz2 / boltz2x: outputs/{model}/seed_{N}/boltz_results_*/predictions/.../boltz_input_model_{M}_aligned.cif
    for model in ("boltz2", "boltz2x"):
        model_dir = out_dir / model
        if not model_dir.exists():
            continue
        for seed_dir in sorted(model_dir.glob("seed_*")):
            try:
                seed_n = int(seed_dir.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            for cif in sorted(seed_dir.rglob("*_aligned.cif")):
                m = re.search(r"model_(\d+)_aligned\.cif$", cif.name)
                if not m:
                    continue
                sample_idx = int(m.group(1))
                stem = f"{model}_seed_{seed_n}_sample_{sample_idx}"
                pairs.append((stem, cif))

    # protenix: outputs/protenix/seed_{N}/<name>/seed_{N}/predictions/<name>_sample_{M}_aligned.cif
    px_dir = out_dir / "protenix"
    if px_dir.exists():
        for cif in sorted(px_dir.rglob("*_sample_*_aligned.cif")):
            seed_match = re.search(r"seed_(\d+)", str(cif))
            sample_match = re.search(r"sample_(\d+)_aligned\.cif$", cif.name)
            if not (seed_match and sample_match):
                continue
            seed_n = int(seed_match.group(1))
            sample_idx = int(sample_match.group(1))
            stem = f"protenix_seed_{seed_n}_sample_{sample_idx}"
            pairs.append((stem, cif))

    # alphafold3: outputs/alphafold3/{name}_seed-{N}_sample-{M}_model_aligned.cif
    af3_dir = out_dir / "alphafold3"
    if af3_dir.exists():
        for cif in sorted(af3_dir.rglob("*_seed-*_sample-*_model_aligned.cif")):
            seed_match = re.search(r"seed-(\d+)", cif.name)
            sample_match = re.search(r"sample-(\d+)", cif.name)
            if not (seed_match and sample_match):
                continue
            stem = f"alphafold3_seed_{seed_match.group(1)}_sample_{sample_match.group(1)}"
            pairs.append((stem, cif))

    return pairs


def _gather_template_receptors(run_dir: Path, output_dir: Path) -> dict[str, str]:
    """Copy each Track 2/3 template receptor PDB into ``output_dir/receptors``
    and return ``{template_pdb: relative_path}``."""
    mapping: dict[str, str] = {}
    td = run_dir / "outputs" / "template_docking"
    if not td.exists():
        return mapping
    rec_dir = output_dir / "receptors"
    rec_dir.mkdir(parents=True, exist_ok=True)
    for child in sorted(td.iterdir()):
        if not child.is_dir():
            continue
        # Template directory layout: outputs/template_docking/{pdb_id}/
        # Look for receptor.pdb in that dir or in any sub-template_docking_summary
        candidates = list(child.rglob("receptor.pdb"))
        if not candidates:
            continue
        src = candidates[0]
        dst = rec_dir / f"template_{child.name}.pdb"
        try:
            shutil.copy2(src, dst)
            mapping[child.name] = str(dst.relative_to(output_dir))
        except Exception as e:
            print(f"[receptors] template copy failed for {child.name}: {e}", flush=True)
    return mapping


def _resolve_cofold_best(run_dir: Path, output_dir: Path) -> Path | None:
    """Honour ``docking_prep_summary.cofolding_structure`` when available, else
    pick the first aligned CIF we find. Strips ligand → PDB."""
    summary = run_dir / "inputs" / "docking" / "docking_prep_summary.json"
    cif_path: Path | None = None
    if summary.exists():
        try:
            data = json.loads(summary.read_text())
            cofold_cif = data.get("cofolding_structure")
            if cofold_cif and Path(cofold_cif).exists():
                cif_path = Path(cofold_cif)
        except Exception:
            pass
    if cif_path is None:
        # Fallback: first aligned CIF we can find.
        candidates = sorted((run_dir / "outputs").rglob("*_aligned.cif"))
        if candidates:
            cif_path = candidates[0]
    if cif_path is None:
        return None
    out_pdb = output_dir / "receptors" / "cofold_best.pdb"
    return out_pdb if _strip_ligand_to_pdb(cif_path, out_pdb) else None


def _resolve_crystal(
    target: str, rcsb_dir: Path | None, output_dir: Path,
    candidate_ccds: list[str],
) -> tuple[Path | None, list[Path]]:
    """If ``rcsb_dir`` points at the standard hash-bucket layout, locate the
    target CIF (``<rcsb_dir>/<bucket>/<pdb>.cif[.gz]``), strip ligand →
    crystal protein PDB, and extract each candidate ligand instance →
    one SDF per CCD. Returns ``(crystal_protein_pdb, [ligand_sdfs])``.
    """
    if rcsb_dir is None:
        return None, []
    pdb_id = target.replace("_input", "").lower()
    if len(pdb_id) != 4:
        return None, []
    bucket = pdb_id[1:3]
    cif_paths = [rcsb_dir / bucket / f"{pdb_id}.cif", rcsb_dir / bucket / f"{pdb_id}.cif.gz"]
    cif_path = next((c for c in cif_paths if c.exists()), None)
    if cif_path is None:
        return None, []

    work_dir = output_dir / "_crystal_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    if cif_path.suffix == ".gz":
        plain_cif = work_dir / f"{pdb_id}.cif"
        if not plain_cif.exists():
            with gzip.open(cif_path, "rb") as f_in, open(plain_cif, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        cif_path = plain_cif

    crystal_pdb = output_dir / "receptors" / "crystal.pdb"
    ok = _strip_ligand_to_pdb(cif_path, crystal_pdb)

    ligand_sdfs: list[Path] = []
    if candidate_ccds:
        try:
            import gemmi
            from rdkit import Chem
            st = gemmi.read_structure(str(cif_path))
            for model in st:
                for chain in model:
                    for residue in chain:
                        if residue.name.upper() not in candidate_ccds:
                            continue
                        # Build a minimal PDB block then RDKit parses it
                        # leniently. Caller can re-bond-order against the
                        # SMILES later if needed.
                        pdb_block = ["HEADER    crystal ligand"]
                        for atom in residue:
                            pdb_block.append(
                                "HETATM%5d  %-3s %3s %1s%4d    %8.3f%8.3f%8.3f  1.00  0.00          %s"
                                % (
                                    atom.serial, atom.name, residue.name, chain.name,
                                    residue.seqid.num,
                                    atom.pos.x, atom.pos.y, atom.pos.z,
                                    atom.element.name.rjust(2),
                                )
                            )
                        pdb_block.append("END")
                        mol = Chem.MolFromPDBBlock("\n".join(pdb_block), sanitize=False)
                        if mol is None:
                            continue
                        sdf_out = output_dir / f"crystal_ligand_{residue.name}_{chain.name}_{residue.seqid.num}.sdf"
                        with Chem.SDWriter(str(sdf_out)) as w:
                            w.write(mol)
                        ligand_sdfs.append(sdf_out)
                break
        except Exception as e:
            print(f"[crystal] ligand extract failed for {target}: {e}", flush=True)
    return (crystal_pdb if ok else None), ligand_sdfs


# --------------------------------------------------------------------------- #
# Per-target driver                                                           #
# --------------------------------------------------------------------------- #

def _receptor_for_pose(
    pose: PoseScore,
    parsed: dict,
    cofold_sample_map: dict[str, str],
    cofold_best_rel: str | None,
    template_recs: dict[str, str],
) -> tuple[str, str | None]:
    """Map a pose to the right receptor file (relative to <output>/<target>)."""
    if parsed["family"].startswith("cofold_"):
        # cofold_<model>_{lid}_seed_<N>_sample_<M>
        model = parsed["family"].split("_", 1)[1]
        # Pose name carries the seed/sample.
        seed_match = _RE_SEED.search(pose.pose_name)
        sample_match = _RE_SAMPLE.search(pose.pose_name)
        if seed_match and sample_match:
            stem_alts = []
            stem_alts.append(f"{model}_seed_{seed_match.group(1)}_sample_{sample_match.group(1)}")
            if model == "af3":
                stem_alts.append(f"alphafold3_seed_{seed_match.group(1)}_sample_{sample_match.group(1)}")
            for stem in stem_alts:
                if stem in cofold_sample_map:
                    return "cofold_sample", cofold_sample_map[stem]
        return "cofold_sample", None
    if parsed["template_pdb"]:
        rel = template_recs.get(parsed["template_pdb"])
        if rel:
            return "template", rel
        return "template", None
    if cofold_best_rel:
        return "cofold_best", cofold_best_rel
    return "unknown", None


def per_target(
    run_dir: Path,
    output_root: Path,
    rcsb_dir: Path | None,
    skip_receptors: bool = False,
) -> dict | None:
    target = run_dir.name
    output_dir = output_root / target
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        poses = collect_pose_scores(run_dir)
    except Exception as e:
        print(f"[{target}] collect_pose_scores failed: {e}", flush=True)
        return None
    if not poses:
        print(f"[{target}] no poses; skipping", flush=True)
        return None

    # --- receptors ----------------------------------------------------------
    cofold_sample_map: dict[str, str] = {}
    cofold_best_rel: str | None = None
    template_recs: dict[str, str] = {}
    crystal_pdb = None
    crystal_ligs: list[Path] = []
    if not skip_receptors:
        # Cofold sample receptors
        rec_dir = output_dir / "receptors" / "cofold_samples"
        rec_dir.mkdir(parents=True, exist_ok=True)
        for stem, cif in _gather_cofold_cifs(run_dir):
            out_pdb = rec_dir / f"{stem}.pdb"
            if _strip_ligand_to_pdb(cif, out_pdb):
                cofold_sample_map[stem] = str(out_pdb.relative_to(output_dir))
        # cofold_best
        best_pdb = _resolve_cofold_best(run_dir, output_dir)
        if best_pdb:
            cofold_best_rel = str(best_pdb.relative_to(output_dir))
        # Track 2/3 template receptors
        template_recs = _gather_template_receptors(run_dir, output_dir)
        # Crystal (from RCSB; only when --rcsb-dir supplied and target maps)
        # Pull candidate CCDs from the input YAML.
        candidate_ccds = _read_candidate_ccds(run_dir)
        crystal_pdb, crystal_ligs = _resolve_crystal(
            target, rcsb_dir, output_dir, candidate_ccds
        )

    # --- pose extraction ---------------------------------------------------
    from rdkit import Chem  # type: ignore
    from rdkit.RDLogger import DisableLog
    DisableLog("rdApp.*")

    sdf_path = output_dir / "poses.sdf"
    rows: list[dict] = []
    n_skipped = 0
    mol_cache: dict = {}

    with Chem.SDWriter(str(sdf_path)) as writer:
        for pose in poses:
            mol = _load_pose_mol(pose, mol_cache)
            if mol is None:
                n_skipped += 1
                continue
            parsed = parse_source(pose.source, pose.ligand_id)
            name_parts = parse_pose_name(pose.pose_name)
            rec_kind, rec_file = _receptor_for_pose(
                pose, parsed, cofold_sample_map, cofold_best_rel, template_recs
            )
            record_idx = len(rows)

            # Tag every pose with its key metadata so SDF readers can recover
            # provenance without the manifest. Cheap on disk and useful in
            # PyMOL / ChimeraX where the SDF is the only artefact loaded.
            tagged = Chem.Mol(mol)
            tagged.SetProp("_Name", pose.pose_name)
            tagged.SetProp("target", target)
            tagged.SetProp("source", pose.source)
            tagged.SetProp("source_family", parsed["family"])
            if pose.ligand_id:
                tagged.SetProp("ligand_id", pose.ligand_id)
            if parsed["box_source"]:
                tagged.SetProp("box_source", parsed["box_source"])
            if parsed["template_pdb"]:
                tagged.SetProp("template_pdb", parsed["template_pdb"])
            tagged.SetProp("track", str(parsed["track"]))
            for k, v in name_parts.items():
                if v is not None:
                    tagged.SetProp(k, str(v))
            for k, v in (
                ("lscore", pose.lscore),
                ("ba_pred_pkd", pose.ba_pred_pkd),
                ("rmsd_pred", pose.rmsd_pred),
                ("iptm", pose.iptm),
                ("ptm", pose.ptm),
                ("plddt", pose.plddt),
                ("conf", pose.conf),
                ("boltz_aff_log_kd_nM", pose.boltz_aff_log_kd_nM),
                ("boltz_binder_prob", pose.boltz_binder_prob),
            ):
                if v is not None:
                    tagged.SetProp(k, f"{float(v):.6g}")
            if rec_file:
                tagged.SetProp("receptor_file", rec_file)
                tagged.SetProp("receptor_kind", rec_kind)
            writer.write(tagged)

            rows.append({
                "target": target,
                "ligand_id": pose.ligand_id,
                "source": pose.source,
                "source_family": parsed["family"],
                "box_source": parsed["box_source"],
                "template_pdb": parsed["template_pdb"],
                "track": parsed["track"],
                "pose_name": pose.pose_name,
                "seed": name_parts["seed"],
                "sample_idx": name_parts["sample_idx"],
                "pose_idx": name_parts["pose_idx"],
                "sdf_record_idx": record_idx,
                "lscore": pose.lscore,
                "ba_pred_pkd": pose.ba_pred_pkd,
                "rmsd_pred": pose.rmsd_pred,
                "log_kd_nM": pose.log_kd_nM,
                "iptm": pose.iptm,
                "ptm": pose.ptm,
                "plddt": pose.plddt,
                "conf": pose.conf,
                "boltz_aff_log_kd_nM": pose.boltz_aff_log_kd_nM,
                "boltz_binder_prob": pose.boltz_binder_prob,
                "receptor_kind": rec_kind,
                "receptor_file": rec_file,
                "pose_file": str(pose.pose_file) if pose.pose_file else "",
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"[{target}] all poses failed to load (n_skipped={n_skipped})", flush=True)
        return None

    # Per-(target, ligand) markers — easy boolean filters in pandas.
    df["is_top1_lscore_per_ligand"] = False
    if "lscore" in df.columns:
        for lid, sub in df.groupby("ligand_id", dropna=False):
            valid = sub["lscore"].dropna()
            if valid.empty:
                continue
            top_row = valid.idxmax()
            df.loc[top_row, "is_top1_lscore_per_ligand"] = True

    df.to_csv(output_dir / "manifest.csv", index=False)
    print(
        f"[{target}] wrote {len(df)} poses to {sdf_path} "
        f"(skipped {n_skipped} unloadable; "
        f"{len(cofold_sample_map)} cofold receptors; "
        f"{len(template_recs)} template receptors; "
        f"crystal={'yes' if crystal_pdb else 'no'})",
        flush=True,
    )
    return {
        "target": target,
        "n_poses": len(df),
        "n_skipped": n_skipped,
        "n_cofold_receptors": len(cofold_sample_map),
        "n_template_receptors": len(template_recs),
        "has_crystal": bool(crystal_pdb),
    }


def _read_candidate_ccds(run_dir: Path) -> list[str]:
    """Best-effort: parse the input YAML for CCD codes that should match the
    crystal. Empty list if anything fails — crystal extraction then just
    skips the per-CCD ligand SDF (still writes crystal protein)."""
    yaml_path = run_dir / "inputs" / "boltz_input.yaml"
    if not yaml_path.exists():
        return []
    try:
        import yaml
        data = yaml.safe_load(yaml_path.read_text()) or {}
    except Exception:
        return []
    out: list[str] = []
    for entry in data.get("sequences", []) or []:
        if isinstance(entry, dict) and "ligand" in entry:
            ccd = (entry["ligand"] or {}).get("ccd")
            if ccd:
                out.append(str(ccd).upper())
    return out


# --------------------------------------------------------------------------- #
# Batch / rollup                                                              #
# --------------------------------------------------------------------------- #

def _per_target_worker(args: tuple) -> dict | None:
    run_dir, output_root, rcsb_dir, skip_receptors = args
    return per_target(run_dir, output_root, rcsb_dir, skip_receptors)


def batch(
    runs_dir: Path,
    output_root: Path,
    rcsb_dir: Path | None,
    skip_receptors: bool,
    n_workers: int,
    targets_glob: str = "*_input",
) -> None:
    targets = sorted(p for p in runs_dir.glob(targets_glob) if p.is_dir())
    if not targets:
        print(f"[batch] no targets matched {targets_glob} under {runs_dir}")
        return
    print(f"[batch] processing {len(targets)} targets with {n_workers} workers")
    output_root.mkdir(parents=True, exist_ok=True)

    args_list = [(t, output_root, rcsb_dir, skip_receptors) for t in targets]
    summaries: list[dict] = []
    if n_workers > 1:
        with Pool(n_workers) as pool:
            for s in pool.imap_unordered(_per_target_worker, args_list, chunksize=1):
                if s is not None:
                    summaries.append(s)
    else:
        for a in args_list:
            s = _per_target_worker(a)
            if s is not None:
                summaries.append(s)

    # --- rollup: concat every per-target manifest into one summary CSV -----
    rollup_rows: list[pd.DataFrame] = []
    for t in targets:
        m = output_root / t.name / "manifest.csv"
        if m.exists():
            try:
                rollup_rows.append(pd.read_csv(m))
            except Exception as e:
                print(f"[rollup] failed to read {m}: {e}")
    if rollup_rows:
        big = pd.concat(rollup_rows, ignore_index=True)
        big.to_csv(output_root / "_summary.csv", index=False)
        print(f"[batch] _summary.csv written: {len(big):,} pose rows across {big['target'].nunique()} targets")
    else:
        print("[batch] no per-target manifests produced; skipping summary")

    # Persist a small batch-level overview JSON so downstream notebooks can
    # see counts without re-loading the giant CSV.
    overview = {
        "n_targets_attempted": len(targets),
        "n_targets_emitted": len(summaries),
        "per_target_summary": summaries,
    }
    (output_root / "_batch_overview.json").write_text(json.dumps(overview, indent=2))


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Single run directory (e.g. experiments/runs/22mj_input).")
    parser.add_argument("--runs-dir", type=Path, default=None,
                        help="Parent of many run directories (batch mode).")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="Output root; defaults to experiments/poses_unified.")
    parser.add_argument("--rcsb-dir", type=Path, default=None,
                        help="RCSB raw mmCIF dir for crystal pose extraction.")
    parser.add_argument("--skip-receptors", action="store_true",
                        help="Skip cofold/template/crystal receptor extraction.")
    parser.add_argument("--workers", type=int, default=max(1, min(cpu_count() // 2, 8)),
                        help="Parallel workers in batch mode.")
    parser.add_argument("--targets-glob", type=str, default="*_input",
                        help="Glob pattern for target dirs under --runs-dir.")
    args = parser.parse_args()

    output_root = args.output_root or (REPO / "experiments" / "poses_unified")
    rcsb_dir = args.rcsb_dir.expanduser() if args.rcsb_dir else None

    if args.run_dir is not None:
        per_target(args.run_dir, output_root, rcsb_dir, args.skip_receptors)
    elif args.runs_dir is not None:
        batch(args.runs_dir, output_root, rcsb_dir, args.skip_receptors,
              args.workers, args.targets_glob)
    else:
        parser.error("provide --run-dir (single) or --runs-dir (batch)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
