"""Generate unified input YAMLs for novel2025 targets.

Design:
  - Cofolding input includes every chain (multi-chain supported) and every
    non-crystallization-aid ligand (metals, cofactors, drug-like candidates).
    Metals / ions / metal clusters use CCD codes; drug-like molecules use
    SMILES. Crystallization aids (EDO, GOL, PEG, sulfate, ...) are skipped.
  - The first drug-like candidate is placed as ligand id ``L`` so that
    ``src/casp17/adapters.py`` auto-wires it as the affinity binder and
    ``prepare_docking_inputs.py`` auto-picks it as the docking target
    (``prep["ligands"][0]``). All other ligands follow as X2, X3, ...
  - YAML is produced via ``yaml.safe_dump`` so weird SMILES characters
    (backslash E/Z, apostrophes in names, etc.) are escaped automatically.

Selection criteria:
  - deposition_date >= 2025-01-01
  - num_candidate_ligands >= 1
  - 100 <= sum(chain lengths) <= 900 (GPU budget; scales ~O(n^2))
  - candidate_smiles[0] is non-empty

Writes ``pipeline/{pdb_id}_input.yaml``.
"""
from pathlib import Path
import csv
import yaml

ROOT = Path(__file__).parent
TSV = Path("/home/jaemin/DB/RCSB/processed/seqid_zones_2025_nonredundant.tsv")
OUT = ROOT / "pipeline"
OUT.mkdir(exist_ok=True)

# Ligand types that should be ignored entirely (pure crystallization aids).
SKIP_TYPES = {"crystallization_aid"}

# Ligand types that should be passed to cofolding as CCD codes rather than
# SMILES. Boltz/Protenix/AF3 handle metals and small inorganic clusters more
# reliably via their CCD lookup than via free-form SMILES.
CCD_TYPES = {"metal", "ion", "metal_cluster"}


def split_field(row, key):
    return (row.get(key) or "").split("|")


# The upstream RCSB index uses ``|`` both as a multi-ligand separator AND as
# an intra-SMILES segment separator. The latter is triggered when a single
# SMILES contains consecutive ring-closure digits that the source tool emits
# as 1-char slices (e.g. ``[Fe]56`` in HEM becomes ``[Fe]5|6|``). Naive
# ``str.split("|")`` therefore over-splits HEM-like ligands into garbage
# fragments (``CC1=...[Fe]5`` / ``6`` / ``N2=...`` / ``[nH]1...``), which
# propagates into the YAML and kills AF3's RDKit parser. Fix: split, then
# greedy-concat consecutive tokens until each chunk is a valid SMILES. CCD
# count is the ground truth for ligand count; if reassembly produces a
# different count, we fall back to naive split so the target isn't dropped.
def _smart_split_smiles(raw, expected_count=None):
    tokens = (raw or "").split("|")
    try:
        from rdkit import Chem
        from rdkit.RDLogger import DisableLog
        DisableLog("rdApp.*")
    except Exception:
        return tokens  # no RDKit → fall back to naive split

    def valid(s):
        return bool(s) and Chem.MolFromSmiles(s) is not None

    out, buf = [], ""
    for tok in tokens:
        if not tok:
            if buf:
                out.append(buf); buf = ""
            continue
        cand = (buf + tok) if buf else tok
        if valid(cand):
            out.append(cand); buf = ""
        else:
            buf = cand
    if buf:
        out.append(buf)

    # If CCD count known, prefer output that matches it (trust CCD list as ground truth)
    if expected_count is not None and expected_count > 0 and len(out) != expected_count:
        # Fall back only if the naive split matched the count
        if len(tokens) == expected_count:
            return tokens
    return out


def total_length(row) -> int:
    return sum(len(s) for s in split_field(row, "sequences") if s)


def build_sequences_block(row):
    """Return a list of entity dicts for the ``sequences:`` YAML field, plus
    a set of ligand ids that were actually added (for validation).

    Ligand ordering: first drug-like candidate is id ``L`` (docking target
    + affinity binder), remaining candidates are L2, L3, ..., then
    non-candidate non-aid ligands as X2, X3, ... with CCD or SMILES.
    """
    entries = []

    # Proteins: one entity per chain, ids A, B, C, ...
    chain_ids = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    seqs = [s for s in split_field(row, "sequences") if s]
    for i, seq in enumerate(seqs):
        entries.append({
            "protein": {
                "id": chain_ids[i],
                "sequence": seq,
                # msa field omitted → Boltz fetches MSA from ColabFold server
                # when use_msa_server: true in config. Setting "msa: empty"
                # forces single-sequence mode even with the server enabled.
            }
        })

    # Candidate ligands: these come first, drug-like, use SMILES.
    cand_ccds = [c for c in split_field(row, "candidate_ccd_codes") if c]
    cand_smis = [s for s in _smart_split_smiles(row.get("candidate_smiles") or "",
                                                expected_count=len(cand_ccds)) if s]
    binder_ccd = None
    added_ccds: set[str] = set()
    lig_counter = 0
    for ccd, smi in zip(cand_ccds, cand_smis):
        lig_counter += 1
        lid = "L" if lig_counter == 1 else f"L{lig_counter}"
        entries.append({"ligand": {"id": lid, "smiles": smi}})
        added_ccds.add(ccd)
        if binder_ccd is None:
            binder_ccd = ccd

    # Non-candidate ligands: metals, metal clusters, cofactors not flagged
    # as candidates, etc. Skip crystallization aids entirely.
    all_ccds = split_field(row, "ligand_ccd_codes")
    all_types = split_field(row, "ligand_types")
    all_smis = _smart_split_smiles(row.get("ligand_smiles") or "",
                                   expected_count=len(all_ccds))
    x_counter = 1
    for ccd, smi, typ in zip(all_ccds, all_smis, all_types):
        if not ccd:
            continue
        if typ in SKIP_TYPES:
            continue
        if ccd in added_ccds:
            continue  # already included as a candidate
        x_counter += 1
        xid = f"X{x_counter}"
        if typ in CCD_TYPES:
            entries.append({"ligand": {"id": xid, "ccd": ccd}})
        else:
            entries.append({"ligand": {"id": xid, "smiles": smi}})
        added_ccds.add(ccd)

    return entries


def build_yaml(row) -> dict:
    return {
        "version": 1,
        "seed": 42,
        "sequences": build_sequences_block(row),
        "properties": [{"affinity": {"binder": "L"}}],
    }


def eligible(row) -> bool:
    if (row.get("deposition_date") or "") < "2025-01-01":
        return False
    try:
        if int(row.get("num_candidate_ligands") or 0) < 1:
            return False
    except ValueError:
        return False
    L = total_length(row)
    if not (100 <= L <= 900):
        return False
    first_cand = split_field(row, "candidate_smiles")[0] if split_field(row, "candidate_smiles") else ""
    if not first_cand:
        return False
    return True


def main():
    written = 0
    skipped = 0
    multi_chain = 0
    multi_lig = 0
    with TSV.open() as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if not eligible(row):
                skipped += 1
                continue
            tid = row["pdb_id"].strip()
            n_ch = sum(1 for s in split_field(row, "sequences") if s)
            data = build_yaml(row)
            n_ligs = sum(1 for e in data["sequences"] if "ligand" in e)
            if n_ch > 1:
                multi_chain += 1
            if n_ligs > 1:
                multi_lig += 1
            dep = row.get("deposition_date", "")
            org = (row.get("organism_name") or "")[:60]
            zone = row.get("seq_zone", "")
            header = (
                f"# Source: RCSB {tid} (deposited {dep}, {zone}, {org})\n"
                f"# chains={n_ch} ligands={n_ligs} "
                f"(candidates={row.get('num_candidate_ligands')}, "
                f"total_len={total_length(row)})\n"
            )
            out = header + yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
            (OUT / f"{tid}_input.yaml").write_text(out)
            written += 1
    print(f"wrote {written} YAMLs → {OUT}")
    print(f"  multi-chain: {multi_chain}")
    print(f"  multi-ligand (inc. cofold extras): {multi_lig}")
    print(f"skipped {skipped} rows")


if __name__ == "__main__":
    main()
