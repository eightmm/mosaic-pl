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


def read_ligands(target: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for line in (BASE / target / "ligands.csv").read_text().splitlines():
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


def write_yaml(path: Path, fid: str, seq: str, smi: str, seed: int) -> None:
    # Matches the holo target format (see inputs/T2409.yaml): protein chain A +
    # ligand chain L + affinity binder. No `msa:` field on the protein so the
    # bridge distributes the reused MSA into it.
    # Single-quote SMILES: many start with '[' (YAML flow-seq) or contain
    # special chars. Backslash is literal inside single quotes (double any ').
    smi_q = "'" + smi.replace("'", "''") + "'"
    path.write_text(
        "version: 1\n"
        f"name: {fid}\n"
        f"seed: {seed}\n"
        "sequences:\n"
        "- protein:\n"
        "    id: A\n"
        f"    sequence: {seq}\n"
        "- ligand:\n"
        "    id: L\n"
        f"    smiles: {smi_q}\n"
        "properties:\n"
        "- affinity:\n"
        "    binder: L\n"
    )


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
    args = ap.parse_args()

    target = args.target
    seq = read_receptor(target)
    ligs = read_ligands(target)
    if args.limit:
        ligs = ligs[: args.limit]
    data_json = apo_data_json(target)
    if not data_json:
        sys.exit(f"missing apo *_data.json for {target}; run apo cofold first")

    config = load_runner_config(args.config, validate=False)
    seed = config.cofolding_seeds[0] if config.cofolding_seeds else 101

    out_root = (EXP / target / "holo").resolve()
    yaml_dir = out_root / "_yamls"
    yaml_dir.mkdir(parents=True, exist_ok=True)

    wrappers: list[str] = []
    for i, (fid, smi) in enumerate(ligs):
        y = yaml_dir / f"{fid}.yaml"
        write_yaml(y, fid, seq, smi, seed)
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
