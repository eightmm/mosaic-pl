#!/usr/bin/env python3
"""Build a CASP17 LG-format submission for RNA-ligand cofolding output.

**Authoritative format reference: ``docs/casp17_lg_format.md``** —
mirror of <https://predictioncenter.org/casp17/index.cgi?page=format>.
Always consult that file (and its RNA addendum
``docs/casp17_rna_ligand_recipe.md``) before changing the LG output
shape; the builder and the linter both rely on it as the single source
of truth.

Works on cofolding-only runs (Boltz-2, Boltz-2x, Protenix, AlphaFold3) for
RNA receptor + small-molecule ligand targets such as R2314 / R2317 / R2318.
There is no docking stage to draw poses from, so every MODEL is a
"cofolded snapshot" — receptor + ligand are extracted from the same CIF.

Usage:
    uv run python scripts/build_rna_ligand_lg_submission.py \
        --run-dir experiments/CASP17/R2314 \
        --target-id R2314 \
        --ligand-name TRP \
        --author 0000-0000-0000 \
        --method "Boltz-2 + Boltz-2x + Protenix + AF3 cofolding (5 seeds × 5 samples; AF3 unified RNA MSA)"
        # default --output: experiments/CASP17/submissions/R2314_LCDD.lg
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Alignment is a separate pipeline stage (scripts/align_cofolding_outputs.py)
# that writes ``*_aligned.cif`` next to every cofolding cif. This builder
# *only* consumes the aligned variants so MODEL 1..5 are guaranteed to live
# in a single coordinate frame. Run the alignment stage before this script.

# Reuse the LG assembler + cif/PDB helpers from the protein-ligand builder.
# We do *not* reuse extract_cofolded_ligand_mdl because Boltz cif tags the
# ligand residue with its CCD code (e.g. "TRP") which gemmi classifies as
# an amino acid, so the generic extractor (which skips amino/nucleic acids)
# drops the ligand. Here we extract by chain id instead, which is unambiguous
# for our single-ligand RNA targets.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_casp_submission import (  # type: ignore  # noqa: E402
    build_lg_submission,
    cif_to_pdb_with_plddt,
    extract_pdb_atom_lines,
)


@dataclass
class CofoldSample:
    tool: str           # "boltz2" | "boltz2x" | "protenix" | "alphafold3"
    seed: int
    sample: int         # diffusion sample index (0..N-1)
    cif: Path
    conf: float | None  # normalized confidence in [0, 1]; higher = better

    @property
    def name(self) -> str:
        return f"{self.tool}_seed_{self.seed}_sample_{self.sample}"


# ---------- score readers ----------------------------------------------------

def _orig_stem(cif: Path) -> str:
    """Strip a trailing ``_aligned`` from a cif stem so the confidence JSON
    can be located next to the original cofolding output."""
    return cif.stem[:-len("_aligned")] if cif.stem.endswith("_aligned") else cif.stem


def _read_boltz_conf(cif: Path) -> float | None:
    """Boltz writes confidence_<stem>.json next to the cif."""
    json_path = cif.parent / f"confidence_{_orig_stem(cif)}.json"
    if not json_path.exists():
        return None
    try:
        d = json.loads(json_path.read_text())
    except Exception:
        return None
    return float(d.get("confidence_score") or d.get("complex_plddt") or 0.0) or None


def _read_protenix_conf(cif: Path) -> float | None:
    """Protenix writes ``<base>_summary_confidence_sample_<k>.json`` next to
    each ``<base>_sample_<k>.cif``. We resolve the matching json by suffix."""
    stem = _orig_stem(cif)  # e.g. "R2314_sample_3"
    if "_sample_" not in stem:
        return None
    base, sample_idx = stem.rsplit("_sample_", 1)
    json_path = cif.parent / f"{base}_summary_confidence_sample_{sample_idx}.json"
    if not json_path.exists():
        return None
    try:
        d = json.loads(json_path.read_text())
    except Exception:
        return None
    # Protenix doesn't emit ``confidence_score`` directly. We synthesise one
    # by combining iptm + plddt — same idea Boltz uses internally
    # (confidence ≈ 0.8·iptm + 0.2·plddt). plddt is on a 0-100 scale.
    iptm = float(d.get("iptm") or 0.0)
    plddt_100 = float(d.get("plddt") or 0.0)
    return 0.8 * iptm + 0.2 * (plddt_100 / 100.0) if (iptm or plddt_100) else None


def _read_af3_conf(cif: Path) -> float | None:
    """AF3 writes ``<base>_summary_confidences.json`` next to each
    ``<base>_model.cif`` under ``<sanitised>_<timestamp>/seed-<S>_sample-<K>/``."""
    orig_name = f"{_orig_stem(cif)}.cif"
    if orig_name.endswith("_model.cif"):
        json_path = cif.parent / orig_name.replace("_model.cif", "_summary_confidences.json")
    else:
        json_path = cif.parent / "summary_confidences.json"
    if not json_path.exists():
        return None
    try:
        d = json.loads(json_path.read_text())
    except Exception:
        return None
    # AF3 ranking_score is a [0,1] scalar combining iptm + chain confidence.
    if isinstance(d.get("ranking_score"), (int, float)):
        return float(d["ranking_score"])
    iptm = float(d.get("iptm") or 0.0)
    ptm = float(d.get("ptm") or 0.0)
    return 0.8 * iptm + 0.2 * ptm if (iptm or ptm) else None


# ---------- discovery --------------------------------------------------------

def _discover(run_dir: Path) -> list[CofoldSample]:
    """Locate ``*_aligned.cif`` cofolding outputs across all tools.

    Alignment is a separate pipeline stage; if no ``*_aligned.cif`` files
    exist for a tool, we skip it and the caller will see a low sample
    count (the LG file will still be valid but smaller).
    """
    out: list[CofoldSample] = []
    for tool in ("boltz2", "boltz2x"):
        base = run_dir / "outputs" / tool
        for cif in base.rglob("*model_*_aligned.cif"):
            seed = _seed_from_path(cif, prefix="seed_")
            sample = _int_after(_orig_stem(cif), "model_")
            out.append(CofoldSample(tool, seed, sample, cif, _read_boltz_conf(cif)))
    base = run_dir / "outputs" / "protenix"
    for cif in base.rglob("*_sample_*_aligned.cif"):
        seed = _seed_from_path(cif, prefix="seed_")
        sample = _int_after(_orig_stem(cif), "sample_")
        out.append(CofoldSample("protenix", seed, sample, cif, _read_protenix_conf(cif)))
    base = run_dir / "outputs" / "alphafold3"
    for cif in base.rglob("*_seed-*_sample-*_model_aligned.cif"):
        parent = cif.parent.name
        seed = _int_after(parent, "seed-")
        sample = _int_after(parent, "sample-")
        out.append(CofoldSample("alphafold3", seed, sample, cif, _read_af3_conf(cif)))
    return out


def _seed_from_path(p: Path, prefix: str) -> int:
    for part in p.parts:
        if part.startswith(prefix):
            try:
                return int(part[len(prefix):])
            except ValueError:
                continue
    return 0


def _int_after(text: str, marker: str) -> int:
    idx = text.find(marker)
    if idx < 0:
        return 0
    tail = text[idx + len(marker):]
    digits = []
    for c in tail:
        if c.isdigit():
            digits.append(c)
        else:
            break
    return int("".join(digits)) if digits else 0


# ---------- ligand pose clustering (post-alignment) --------------------------

def _heavy_atom_coords(mol_block: str):
    from rdkit import Chem
    mol = Chem.MolFromMolBlock(mol_block, removeHs=True, sanitize=False)
    if mol is None:
        return None
    conf = mol.GetConformer()
    coords = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "H":
            continue
        p = conf.GetAtomPosition(atom.GetIdx())
        coords.append((p.x, p.y, p.z))
    return coords


def _rmsd(a, b) -> float:
    if not a or not b or len(a) != len(b):
        return float("inf")
    return math.sqrt(sum((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2
                         for (ax, ay, az), (bx, by, bz) in zip(a, b)) / len(a))


@dataclass
class Cluster:
    seed_name: str             # sample.name of the cluster seed (highest-conf member)
    seed_coords: list          # heavy-atom coords of the seed pose (in ref frame)
    members: list[CofoldSample] = field(default_factory=list)

    @property
    def best_conf(self) -> float:
        return max((m.conf or 0.0) for m in self.members)


def _cluster_and_pick_top_k(
    samples: list[CofoldSample],
    ligand_blocks: dict[str, str],
    k: int,
    rmsd_threshold: float,
) -> tuple[list[CofoldSample], list[Cluster]]:
    """Greedy single-link clustering on aligned heavy-atom coords.

    The pool is sorted by confidence (highest first). The first sample
    seeds cluster 1. Each subsequent sample is attached to the *first*
    existing cluster whose seed-vs-sample heavy-atom RMSD is below
    ``rmsd_threshold``; otherwise it spawns a new cluster. Within each
    cluster the seed is the highest-confidence member by construction
    (we walk in confidence order).

    Returns:
        (representatives, clusters) — the top-k cluster seeds (each is the
        highest-conf member of its cluster) and the full cluster list (for
        diagnostics). The order of ``representatives`` is by ``best_conf``
        descending so MODEL 1 is the strongest cluster's seed.
    """
    sorted_samples = sorted(
        (s for s in samples if s.conf is not None and s.name in ligand_blocks),
        key=lambda s: s.conf or 0.0,
        reverse=True,
    )

    clusters: list[Cluster] = []
    for s in sorted_samples:
        coords = _heavy_atom_coords(ligand_blocks[s.name])
        if coords is None:
            continue
        attached = False
        for c in clusters:
            if _rmsd(coords, c.seed_coords) < rmsd_threshold:
                c.members.append(s)
                attached = True
                break
        if not attached:
            new_c = Cluster(seed_name=s.name, seed_coords=coords, members=[s])
            clusters.append(new_c)

    clusters.sort(key=lambda c: c.best_conf, reverse=True)
    representatives = [c.members[0] for c in clusters[:k]]
    return representatives, clusters


# ---------- per-sample MDL extraction ----------------------------------------

def _extract_ligand_mdl(sample: CofoldSample, workdir: Path,
                        ligand_chain_id: str, ligand_name: str,
                        ligand_smiles: str | None = None) -> str | None:
    """Extract atoms from ``ligand_chain_id`` only and emit an MDL V2000
    block matching the verified CASP17 LG spec (see
    ``docs/casp17_lg_format.md`` §8 + ``memory/casp17_lg_format_spec.md``).

    Bond perception via ``MolFromPDBFile`` is unreliable on aromatic /
    multi-bond ligands. When the CASP-issued ligand SMILES is supplied
    (``ligand_smiles``) we treat the SMILES topology as authoritative and
    copy only the predicted 3D coordinates from the cofolding output. If
    the atom counts do not match, we fall back to template bond-order
    assignment, then finally to the bond-perception path (degraded — the
    server validator may reject the resulting topology).

    Other invariants (verified 2026-05-06 against the live LG validator):
    1. Per-atom chirality + per-bond direction reset — 3D coords already
       encode stereochemistry; emitting stereo=1 in V2000 crashes the
       validator on chiral atoms.
    2. Explicit hydrogens are omitted. The CASP17 format page says
       hydrogen atoms are optional, and the live validator rejected
       explicit-H SAM/TPP blocks as invalid ligand topology.
    3. MDL header line 1 is overwritten with ``ligand_name`` (e.g. ``TRP``)
       — RDKit's ``_Name`` property is occasionally elided by
       ``MolToMolBlock``, so we splice the name onto the first line
       directly. The server matches this string against the LIGAND
       record verbatim.
    4. ``MolToMolBlock(mol, kekulize=True)`` — matches the form
       CASP17_own's already-accepted submissions emitted.
    """
    import gemmi
    from rdkit import Chem
    from rdkit.Chem import AllChem

    structure = gemmi.read_structure(str(sample.cif))
    tmp_pdb = workdir / f"lig_{sample.name}.lig.pdb"
    out_mol = workdir / f"lig_{sample.name}.mol"
    lig_lines: list[str] = []
    atom_idx = 0
    for model in structure:
        for chain in model:
            if chain.name != ligand_chain_id:
                continue
            for residue in chain:
                if residue.name in {"HOH", "WAT", "DOD"}:
                    continue
                for atom in residue:
                    atom_idx += 1
                    el = (atom.element.name or atom.name.strip()[:1]).upper()
                    lig_lines.append(
                        f"HETATM{atom_idx:5d} {atom.name:<4.4s} {residue.name:<3.3s} "
                        f"{ligand_chain_id}{residue.seqid.num:4d}    "
                        f"{atom.pos.x:8.3f}{atom.pos.y:8.3f}{atom.pos.z:8.3f}"
                        f"  1.00{atom.b_iso:6.2f}          {el:>2s}"
                    )
        break
    if not lig_lines:
        return None
    tmp_pdb.write_text("\n".join(lig_lines) + "\nEND\n")
    raw = Chem.MolFromPDBFile(str(tmp_pdb), removeHs=True, sanitize=False)
    tmp_pdb.unlink(missing_ok=True)
    if raw is None or raw.GetNumAtoms() == 0:
        return None
    mol = raw
    if ligand_smiles:
        try:
            ref = Chem.MolFromSmiles(ligand_smiles)
            if ref is not None:
                # AssignBondOrdersFromTemplate keeps raw's atom order + 3D
                # coords and transfers only the bond orders from the template.
                # NEVER copy coords ref<-raw by atom index: PDB/CCD atom order
                # != SMILES parse order, so an index copy scrambles the
                # connectivity (phantom multi-Angstrom edges across the
                # ligand). This bug shipped earlier on R2390 (NMN).
                mol = AllChem.AssignBondOrdersFromTemplate(ref, raw)
        except Exception:
            mol = raw
    try:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE ^ Chem.SANITIZE_PROPERTIES,
        )
    except Exception:
        pass
    for atom in mol.GetAtoms():
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in mol.GetBonds():
        bond.SetBondDir(Chem.BondDir.NONE)
    block = Chem.MolToMolBlock(mol, kekulize=True)
    lines = block.splitlines()
    if lines:
        lines[0] = ligand_name
    body = "\n".join(lines).rstrip()
    if not body.endswith("M  END"):
        body = body + "\nM  END"
    body = body + "\n"
    out_mol.write_text(body)
    return body


# ---------- main -------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--ligand-name", default="LIG",
                        help="Ligand name for the LIGAND record + MDL title "
                             "(e.g. 'TRP', 'GLY'). Must match the Name column "
                             "of the CASP-issued SMILES file for the target "
                             "(``target.cgi?target=<TID>&view=smiles``).")
    parser.add_argument("--ligand-id", type=int, default=0,
                        help="Integer ID for the LIGAND record, copied verbatim "
                             "from the SMILES file's ID column. **Do not "
                             "zero-pad** — the server's lookup against the "
                             "SMILES file is exact-match. Default 0 matches "
                             "every CASP17 RNA-ligand target observed so far.")
    parser.add_argument("--ligand-chain-id", default="L",
                        help="Chain id of the ligand in cofolding cif output. "
                             "Default 'L' matches our input yaml convention.")
    parser.add_argument("--ligand-smiles", default=None,
                        help="SMILES for the ligand (matching the CASP "
                             "SMILES file). When supplied, the builder runs "
                             "AssignBondOrdersFromTemplate so the MDL bond "
                             "block has correct heavy-atom bond orders. "
                             "Auto-loaded from "
                             "``inputs/casp17_rna_lig/smiles/<TARGET>.smiles`` "
                             "when present.")
    parser.add_argument("--receptor-chain-id", default="0",
                        help="Chain id to write into receptor PDB ATOM lines. "
                             "Default '0' matches the CASP-issued RNA template "
                             "convention (single zero character at PDB col 22). "
                             "Pass an empty string to keep cofolding's native id.")
    parser.add_argument("--reference-template", type=Path, default=None,
                        help="Optional CASP-issued receptor template "
                             "(PDB-style ATOM coordinates). When provided, the "
                             "builder reads chain id, residue numbering, and "
                             "per-residue atom names from the template and "
                             "auto-matches the LG output to it (overrides "
                             "--receptor-chain-id). Also reports any atom-set "
                             "diffs vs the cofolding output.")
    parser.add_argument("--author", default="6095-5696-9732",
                        help="CASP 12-digit registration code (default: group "
                             "LCDD code 6095-5696-9732).")
    parser.add_argument("--method", required=True,
                        help="Method description line. The fixed footer "
                             "'Pipeline configured and executed via Claude "
                             "Code agentic decision-making' is appended "
                             "automatically.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output LG path. CASP17 group convention forces "
                             "the basename to '{target}_LCDD.lg' regardless of "
                             "what is passed (only the directory is honoured). "
                             "Default: experiments/CASP17/submissions/{target}_LCDD.lg.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--cluster-rmsd", type=float, default=3.0,
                        help="Single-link clustering threshold on heavy-atom "
                             "RMSD (Å) between aligned ligand poses. Two poses "
                             "with RMSD below this threshold join the same "
                             "cluster; we emit the highest-confidence member "
                             "of the top-K clusters as MODEL 1..K.")
    parser.add_argument("--parent", default="N/A")
    parser.add_argument("--no-slack", action="store_true",
                        help="skip the post-build viewer HTML + Slack summary "
                             "(default: build viz/standalone + notify when a "
                             "Slack webhook is configured)")
    args = parser.parse_args()

    # CASP17 group convention (LCDD): the submission basename MUST be
    # '{target}_LCDD.lg'. We honour only the output *directory*; the
    # filename is forced. See docs/casp17_lg_format.md and the
    # casp_author_code memory.
    forced_name = f"{args.target_id}_LCDD.lg"
    out_dir = args.output.parent if args.output is not None else Path("experiments/CASP17/submissions")
    if args.output is not None and args.output.name != forced_name:
        print(f"NOTE: overriding output name '{args.output.name}' -> "
              f"'{forced_name}' (CASP17 LCDD naming convention)")
    args.output = out_dir / forced_name

    run_dir = args.run_dir.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    workdir = args.output.parent / f".{args.target_id}_workdir"
    workdir.mkdir(exist_ok=True)

    template_atom_set: dict[int, set[str]] | None = None
    if args.reference_template and args.reference_template.exists():
        tpl_lines = [ln for ln in args.reference_template.read_text().splitlines()
                     if ln.startswith("ATOM")]
        if tpl_lines:
            tpl_chain_ids = sorted({ln[21] for ln in tpl_lines})
            if tpl_chain_ids:
                args.receptor_chain_id = tpl_chain_ids[0]
                print(f"Reference template {args.reference_template}: "
                      f"chain id(s) {tpl_chain_ids}, using '{args.receptor_chain_id}'")
            template_atom_set = {}
            for ln in tpl_lines:
                resnum = int(ln[22:26])
                template_atom_set.setdefault(resnum, set()).add(ln[12:16].strip())
            print(f"  template residues: {len(template_atom_set)}, "
                  f"total atoms: {sum(len(s) for s in template_atom_set.values())}")

    samples = _discover(run_dir)
    print(f"Discovered {len(samples)} cofolding samples across "
          f"{len({s.tool for s in samples})} tools")
    samples = [s for s in samples if s.conf is not None]
    print(f"  {len(samples)} samples carry a confidence score")

    # Pre-extract every ligand MDL block; samples whose ligand can't be
    # parsed (broken cif, missing ligand) drop out of the pool.
    # Auto-load CASP-issued SMILES if a file exists at the conventional path.
    ligand_smiles = args.ligand_smiles
    if ligand_smiles is None:
        smi_path = (
            _REPO_ROOT
            / "inputs"
            / "casp17_rna_lig"
            / "smiles"
            / f"{args.target_id}.smiles"
        )
        if smi_path.exists():
            for line in smi_path.read_text().splitlines():
                parts = line.split("\t") if "\t" in line else line.split()
                if len(parts) >= 3 and parts[0].isdigit():
                    ligand_smiles = parts[2]
                    break
            if ligand_smiles:
                print(f"Loaded ligand SMILES from {smi_path}: {ligand_smiles}")

    ligand_blocks: dict[str, str] = {}
    for s in samples:
        mdl = _extract_ligand_mdl(
            s, workdir,
            ligand_chain_id=args.ligand_chain_id,
            ligand_name=args.ligand_name,
            ligand_smiles=ligand_smiles,
        )
        if mdl:
            ligand_blocks[s.name] = mdl

    print(f"  {len(ligand_blocks)} samples produced a parseable MDL ligand block")

    picked, clusters = _cluster_and_pick_top_k(
        samples, ligand_blocks, k=args.top_k, rmsd_threshold=args.cluster_rmsd,
    )
    if not picked:
        print("ERROR: no usable cofolding samples found "
              "(did the alignment stage run? expected '*_aligned.cif').",
              file=sys.stderr)
        return 1
    print(f"\nFormed {len(clusters)} cluster(s) at RMSD threshold "
          f"{args.cluster_rmsd} Å; emitting top {len(picked)} as MODEL blocks:")
    for i, (s, c) in enumerate(zip(picked, clusters[:len(picked)]), start=1):
        print(f"  MODEL {i}: {s.name}  conf={s.conf:.3f}  "
              f"(cluster size={len(c.members)})")

    models: list[dict] = []
    for i, s in enumerate(picked, start=1):
        receptor_pdb = workdir / f"receptor_{s.name}.pdb"
        cif_to_pdb_with_plddt(s.cif, receptor_pdb, args.target_id)
        receptor_lines = extract_pdb_atom_lines(receptor_pdb)
        if args.receptor_chain_id:
            cid = args.receptor_chain_id[0]
            receptor_lines = [
                ln[:21] + cid + ln[22:] if ln.startswith(("ATOM", "HETATM", "TER")) else ln
                for ln in receptor_lines
            ]
        if template_atom_set is not None and i == 1:
            our_atoms: dict[int, set[str]] = {}
            for ln in receptor_lines:
                if ln.startswith("ATOM"):
                    our_atoms.setdefault(int(ln[22:26]), set()).add(ln[12:16].strip())
            missing = sum(len(template_atom_set[r] - our_atoms.get(r, set()))
                          for r in template_atom_set)
            extra = sum(len(our_atoms.get(r, set()) - template_atom_set[r])
                        for r in template_atom_set)
            if missing or extra:
                print(f"  template diff (MODEL 1): missing {missing} atom(s), "
                      f"extra {extra} atom(s) — atom names that differ from "
                      f"the CASP template; the assessor matches by atom name "
                      f"so missing atoms are skipped from RMSD scoring.")
        models.append({
            "protein_pdb_lines": receptor_lines,
            "parent": args.parent,
            "remark": f"{s.tool}_seed_{s.seed}_sample_{s.sample} (conf={s.conf:.3f})",
            "ligands": [{
                "ligand_number": args.ligand_id,
                "ligand_name": args.ligand_name,
                "ligand_mdl": ligand_blocks[s.name],
                "lscore": round(min(max(s.conf, 0.0), 1.0), 3),
            }],
        })

    # METHOD footer is mandated by ``memory/casp17_lg_format_spec.md``.
    # ``build_lg_submission`` now splits the method string on newlines and
    # prefixes each non-empty line with ``METHOD ``, so we just append the
    # footer with a plain newline separator.
    method_with_footer = (
        args.method.rstrip()
        + "\nPipeline configured and executed via Claude Code agentic decision-making"
    )

    submission_text = build_lg_submission(
        target_id=args.target_id,
        author=args.author,
        method=method_with_footer,
        models=models,
        parent=args.parent,
    )
    args.output.write_text(submission_text)
    print(f"\nWrote {args.output} ({len(submission_text)} chars, "
          f"{len(models)} MODEL block(s))")

    if not args.no_slack:
        # viewer HTML + standalone + Slack summary (no-op if no webhook).
        # Isolated in a subprocess so a notify failure never fails the build.
        import subprocess
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent / "notify_lg.py"),
             "--lg", str(args.output)],
            check=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
