#!/usr/bin/env python3
"""Generate per-fragment holo (protein+ligand) runs for CASP17 ligand-series.

For a ligand-series target (L01 / L02) this fans the fixed receptor out against
its whole fragment library, one full-pipeline run per fragment:

    receptor(A) + fragment(L) + affinity(binder=L)
        -> template-search + cofolding (Boltz2/2x + Protenix + AF3)
        -> docking + post-analysis + LG submission

Every fragment shares the *same* receptor, so the AF3 MSA (jackhmmer/hmmsearch)
is computed once and reused: we symlink the apo run's ``*_data.json`` into each
fragment run's ``outputs/msa_pipeline/`` and rely on the msa-bridge reuse guard
(``script_builder._emit_msa_pipeline_bridge``) to skip jackhmmer when a
``*_data.json`` is already present. Only GPU inference runs per fragment.

Outputs under ``experiments/ligand_series/<T>/holo/``:
    _yamls/<fid>.yaml                per-fragment common input
    <fid>/...                        prepared run (scripts, inputs, outputs)
    _manifest.txt                    one wrapper path per line (fragment order)
    submit_holo_chunkNN.sbatch.sh    SLURM array driver(s), <=1000 tasks each

Nothing is submitted; run the emitted sbatch script(s) yourself.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from casp17.models import load_common_input, load_runner_config  # noqa: E402
from casp17.orchestrator import prepare_stage_wrapper_run  # noqa: E402

BASE = REPO / "inputs/ligand_series"
EXP = REPO / "experiments/ligand_series"
MAX_ARRAY = 1000  # SLURM MaxArraySize=1001 -> indices 0..1000 (1001 tasks)


def fmt_time(value) -> str:
    """SLURM --time as HH:MM:SS. A bare integer (screen_config uses seconds) is
    otherwise read by SLURM as *minutes* (the known 60x gotcha), so convert it."""
    s = str(value).strip()
    if ":" in s:
        return s
    secs = int(s)
    h, rem = divmod(secs, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def read_receptor(target: str) -> str:
    lines = (BASE / target / "receptor.fasta").read_text().splitlines()
    return "".join(ln.strip() for ln in lines if ln.strip() and not ln.startswith(">"))


def read_ligands(target: str, csv_path: Path | None = None) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    src = csv_path or (BASE / target / "ligands.csv")
    for line in src.read_text().splitlines():
        line = line.strip()
        if not line or "," not in line:
            continue
        lid, smi = line.split(",", 1)
        out.append((lid.strip(), smi.strip()))
    return out


def apo_data_json(target: str) -> Path | None:
    p = (
        EXP / target / f"{target}_apo" / "outputs" / "msa_pipeline"
        / f"{target}_apo" / f"{target}_apo_data.json"
    )
    return p if p.exists() else None


# Receptor residues the target page declares as chemically modified. Boltz reads
# these keys verbatim from the unified YAML; adapters translate them to Protenix
# ``ptmType``/``ptmPosition`` and the same pair for AF3.
#   L02: "Cysteine C253 in the protein structure is in the oxidized state (CSO)"
#        (predictioncenter.org/casp17/target.cgi?id=222)
TARGET_MODIFICATIONS: dict[str, list[tuple[int, str]]] = {
    "L01": [],
    "L02": [(253, "CSO")],
}


def parse_modifications(spec: str | None, target: str) -> list[tuple[int, str]]:
    """``"253:CSO,17:SEP"`` -> [(253, 'CSO'), (17, 'SEP')]. None -> target default."""
    if spec is None:
        return TARGET_MODIFICATIONS.get(target, [])
    spec = spec.strip()
    if not spec or spec.lower() == "none":
        return []
    out = []
    for item in spec.split(","):
        pos, _, ccd = item.partition(":")
        out.append((int(pos), ccd.strip().upper()))
    return out


def write_yaml(
    path: Path,
    fid: str,
    seq: str,
    smi: str,
    seed: int,
    modifications: list[tuple[int, str]] | None = None,
) -> None:
    # Matches the holo target format (see inputs/T2409.yaml): protein chain A +
    # ligand chain L + affinity binder. No `msa:` field on the protein so the
    # bridge distributes the reused MSA into it.
    # Single-quote SMILES: many start with '[' (YAML flow-seq) or contain
    # special chars. Backslash is literal inside single quotes (double any ').
    smi_q = "'" + smi.replace("'", "''") + "'"
    mod_block = ""
    for pos, ccd in modifications or []:
        if mod_block == "":
            mod_block = "    modifications:\n"
        mod_block += f"    - position: {pos}\n      ccd: {ccd}\n"
    path.write_text(
        "version: 1\n"
        f"name: {fid}\n"
        f"seed: {seed}\n"
        "sequences:\n"
        "- protein:\n"
        "    id: A\n"
        f"    sequence: {seq}\n"
        f"{mod_block}"
        "- ligand:\n"
        "    id: L\n"
        f"    smiles: {smi_q}\n"
        "properties:\n"
        "- affinity:\n"
        "    binder: L\n"
    )


# Stage-2 complexes are not "receptor + one fragment" any more. The 2026-08-07
# answer files add, per binder, how many copies of the fragment bind and a
# receptor cofactor that every structure in the series carries — L01 a Zn2+,
# L02 an SFG (sinefungin). Modelling either away changes the pocket, so the
# stage-2 YAML carries N ligand copies plus the cofactor as a CCD entity.
def _is_ccd_code(code: str) -> bool:
    """A PDB chemical-component id (<=3 alphanumerics), e.g. ZN or SFG.

    The answer files name the fragments with long in-house codes (HTX00050570),
    so length alone separates "give this by CCD" from "give this by SMILES".
    A cofactor passed by CCD keeps the deposited atom naming and stereochemistry
    that every tool already agrees on; the fragments have no CCD entry at all.
    """
    return bool(re.fullmatch(r"[A-Z0-9]{1,3}", code.upper()))


def read_stage2(csv_path: Path) -> list[dict]:
    """Parse `<target>.smiles.stage2.csv`.

    Header repeats `ligand code, canonical_smiles, stoichiometry` per component,
    so the columns must be read positionally — `csv.DictReader` collapses the
    duplicates and silently loses every component after the first.
    """
    import csv as _csv
    rows = list(_csv.reader(csv_path.read_text().splitlines()))
    if not rows:
        return []
    out = []
    for row in rows[1:]:
        if not row or not row[0].strip():
            continue
        comps = []
        for i in range(1, len(row) - 2, 3):
            code, smi, sto = row[i].strip(), row[i + 1].strip(), row[i + 2].strip()
            if not code:
                continue
            comps.append({"code": code, "smiles": smi, "n": int(sto or 1)})
        if comps:
            out.append({"id": row[0].strip(), "components": comps})
    return out


def write_yaml_stage2(path: Path, fid: str, seq: str, comps: list[dict], seed: int,
                      modifications: list[tuple[int, str]] | None = None) -> None:
    """One entity per component; the fragment keeps chain ids L1..LN.

    The first component is the binder (the answer file lists it first).

    Two constraints the first version of this got wrong, both of which failed
    silently and cost a full 109-run screen:

    - **Chain ids stay two characters.** The old ``f"X{ci}{k+1}"`` produced
      ``X11`` for the first copy of the second component, and gemmi refuses to
      write a three-character chain to PDB (``chain name too long for the PDB
      format``). That killed docking preparation for every run, and the wrapper
      swallowed it as "docking prep failed, continuing".
    - **Affinity needs a single-copy binder.** Boltz rejects the whole input
      with ``Cannot compute affinity for a ligand that has multiple copies!``
      and skips it — producing no structures at all, with exit status 0. So the
      property is emitted only when the binder is present once; a multi-copy
      run keeps its structures and loses only the affinity number.
    """
    mod_block = ""
    for pos, ccd in modifications or []:
        if mod_block == "":
            mod_block = "    modifications:\n"
        mod_block += f"    - position: {pos}\n      ccd: {ccd}\n"
    body = ["version: 1", f"name: {fid}", f"seed: {seed}", "sequences:",
            "- protein:", "    id: A", f"    sequence: {seq}"]
    if mod_block:
        body.append(mod_block.rstrip("\n"))
    binder_id = None
    binder_copies = 0
    cofactor_ids = iter(f"{p}{d}" for p in "XYZW" for d in range(1, 10))
    for ci, comp in enumerate(comps):
        ids = ([f"L{k + 1}" for k in range(comp["n"])] if ci == 0
               else [next(cofactor_ids) for _ in range(comp["n"])])
        if ci == 0:
            binder_id, binder_copies = ids[0], len(ids)
        id_field = ids[0] if len(ids) == 1 else "[" + ", ".join(ids) + "]"
        body += ["- ligand:", f"    id: {id_field}"]
        # Cofactors go in by CCD (ZN, SFG) so every tool builds the deposited
        # chemistry; the fragments have no CCD entry and keep their SMILES, which
        # is also what Boltz affinity needs.
        if _is_ccd_code(comp["code"]) or not comp["smiles"]:
            body.append(f"    ccd: {comp['code'].upper()}")
        else:
            smi_q = "'" + comp["smiles"].replace("'", "''") + "'"
            body.append(f"    smiles: {smi_q}")
    if binder_copies == 1:
        body += ["properties:", "- affinity:", f"    binder: {binder_id}"]
    path.write_text("\n".join(body) + "\n")


def emit_array_scripts(
    target: str,
    out_root: Path,
    manifest: Path,
    n: int,
    config,
    max_concurrent: int,
) -> list[Path]:
    slurm = config.slurm
    logs = REPO / "experiments" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    scripts: list[Path] = []
    chunk = 0
    for offset in range(0, n, MAX_ARRAY + 1):
        hi = min(offset + MAX_ARRAY, n - 1)
        span = hi - offset  # array is 0..span, real idx = task + offset
        sp = out_root / f"submit_holo_chunk{chunk:02d}.sbatch.sh"
        extra = "\n".join(f"#SBATCH {a}" for a in (slurm.extra_sbatch_args or []))
        sp.write_text(
            "#!/usr/bin/env bash\n"
            f"#SBATCH --job-name={target}_holo{chunk:02d}\n"
            f"#SBATCH --output={logs}/holo-%A_%a.out\n"
            f"#SBATCH --error={logs}/holo-%A_%a.err\n"
            f"#SBATCH --array=0-{span}%{max_concurrent}\n"
            f"#SBATCH --gres=gpu:{slurm.gpus}\n"
            "#SBATCH --cpus-per-task=8\n"
            f"#SBATCH --mem={slurm.mem}\n"
            f"#SBATCH --time={fmt_time(slurm.time)}\n"
            f"#SBATCH --partition={slurm.partition}\n"
            + (extra + "\n" if extra else "")
            + "set -euo pipefail\n"
            f"cd {REPO}\n"
            f"OFFSET={offset}\n"
            f'IDX=$((SLURM_ARRAY_TASK_ID + OFFSET))\n'
            f'mapfile -t WRAPPERS < {manifest}\n'
            'W="${WRAPPERS[$IDX]}"\n'
            'echo "[holo] task=$SLURM_ARRAY_TASK_ID idx=$IDX wrapper=$W"\n'
            'bash "$W"\n'
        )
        sp.chmod(0o755)
        scripts.append(sp)
        chunk += 1
    return scripts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["L01", "L02"])
    ap.add_argument("--config", type=Path, default=BASE / "screen_config.yaml")
    ap.add_argument("--limit", type=int, default=0, help="only first N fragments (dry-run)")
    ap.add_argument("--max-concurrent", type=int, default=8, help="SLURM array %% throttle")
    ap.add_argument("--backend", default="slurm", choices=["slurm", "local"])
    ap.add_argument("--subdir", default="holo",
                    help="output subdir under experiments/ligand_series/<T>/ (use a "
                         "different one for a reduced-stage rerun so full-pipeline "
                         "runs are not clobbered)")
    ap.add_argument("--only-missing-affinity", action="store_true",
                    help="only fragments that have no Boltz affinity json yet under --skip-existing-root")
    ap.add_argument("--skip-existing-root", default="holo",
                    help="subdir checked by --only-missing-affinity")
    ap.add_argument("--ligands-csv", type=Path, default=None,
                    help="override the fragment list (e.g. a desalted subset)")
    ap.add_argument("--msa-data-json", type=Path, default=None,
                    help="AF3 *_data.json to seed into every run (default: the apo "
                         "run's). Point this at a regenerated MSA to re-fold with it.")
    ap.add_argument("--stage2-csv", type=Path, default=None,
                    help="ligands_stage2.csv (per-binder components + stoichiometry + "
                         "cofactor). Switches to stage-2 complexes: N fragment copies "
                         "plus the receptor cofactor, instead of one fragment.")
    ap.add_argument("--modifications", default=None,
                    help="receptor modifications as POS:CCD[,POS:CCD] (default: the "
                         "target's declared ones, see TARGET_MODIFICATIONS; 'none' to drop)")
    args = ap.parse_args()

    target = args.target
    seq = read_receptor(target)
    stage2 = read_stage2(args.stage2_csv) if args.stage2_csv else None
    if stage2:
        ligs = [(r["id"], r["components"][0]["smiles"]) for r in stage2]
        by_id = {r["id"]: r["components"] for r in stage2}
        multi = sum(1 for r in stage2 if r["components"][0]["n"] > 1)
        cof = sorted({c["code"] for r in stage2 for c in r["components"][1:]})
        print(f"  stage 2: {len(stage2)} binders, {multi} with >1 fragment copy, "
              f"cofactor(s) {cof}")
    else:
        by_id = None
        ligs = read_ligands(target, args.ligands_csv)
    if args.only_missing_affinity:
        # A fragment counts as covered once its Boltz affinity json exists — that
        # is the signal stage 1 needs, so anything already holding one is skipped.
        prev = EXP / target / args.skip_existing_root
        have = {p.name for p in prev.glob("L0*")
                if any(p.glob("outputs/boltz*/seed_*/**/affinity_*.json"))}
        before = len(ligs)
        ligs = [(i, s) for i, s in ligs if i not in have]
        print(f"  --only-missing-affinity: {before} -> {len(ligs)} (skipped {len(have)} covered)")
    if args.limit:
        ligs = ligs[: args.limit]
    data_json = args.msa_data_json.resolve() if args.msa_data_json else apo_data_json(target)
    if not data_json or not data_json.exists():
        sys.exit(f"missing MSA *_data.json for {target} ({data_json}); run apo cofold first")
    mods = parse_modifications(args.modifications, target)
    if mods:
        for pos, ccd in mods:
            if not 1 <= pos <= len(seq):
                sys.exit(f"modification position {pos} outside chain A (len {len(seq)})")
            print(f"  modification: {seq[pos - 1]}{pos} -> {ccd}")

    config = load_runner_config(args.config, validate=False)
    seed = config.cofolding_seeds[0] if config.cofolding_seeds else 101

    out_root = (EXP / target / args.subdir).resolve()
    yaml_dir = out_root / "_yamls"
    yaml_dir.mkdir(parents=True, exist_ok=True)

    wrappers: list[str] = []
    for i, (fid, smi) in enumerate(ligs):
        y = yaml_dir / f"{fid}.yaml"
        if by_id is not None:
            write_yaml_stage2(y, fid, seq, by_id[fid], seed, mods)
        else:
            write_yaml(y, fid, seq, smi, seed, mods)
        common = load_common_input(y)
        prepared = prepare_stage_wrapper_run(common, config, out_root, args.backend, None)
        rd = prepared.run_dir
        # Seed the reused MSA so the bridge guard skips jackhmmer.
        seed_dir = rd / "outputs" / "msa_pipeline" / f"{fid}_reused"
        seed_dir.mkdir(parents=True, exist_ok=True)
        link = seed_dir / f"{fid}_data.json"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(data_json)
        wrappers.append(str(prepared.shell_script))
        if (i + 1) % 100 == 0:
            print(f"  prepared {i + 1}/{len(ligs)}")

    manifest = out_root / "_manifest.txt"
    manifest.write_text("\n".join(wrappers) + "\n")
    scripts = emit_array_scripts(
        target, out_root, manifest, len(wrappers), config, args.max_concurrent
    )

    print(f"\n{target}: {len(wrappers)} fragment runs prepared")
    print(f"  manifest: {manifest}")
    print(f"  reused MSA: {data_json}")
    print("  submit:")
    for s in scripts:
        print(f"    sbatch {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
