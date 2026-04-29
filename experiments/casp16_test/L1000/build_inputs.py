"""Generate per-ligand input YAMLs for L1000 (CASP16 chymase) targets."""
from pathlib import Path

CHYMASE = (
    "MLLLPLPLLLFLLCSRAEAGEIIGGTESKPHSRPYMAYLEIVTSNGPSKFCGGFLIRRNFVLTAAHC"
    "AGRSITVTLGAHNITEEEDTWQKLEVIKQFRHPKYNTSTLHHDIMLLKLKEKASLTLAVGTLPFPSQ"
    "KNFVPPGRMCRVAGWGRTGVLKPGSDTLQEVKLRLMDPQACSHFRDFDHNLQLCVGNPRKTKSAFKG"
    "DSGGPLLCAGVAQGIVSYGRSDAKPPAVFTRISHYRPWINQILQAN"
)

ROOT = Path(__file__).parent
SMILES_DIR = ROOT / "L1000"
OUT_DIR = ROOT / "pipeline"
OUT_DIR.mkdir(exist_ok=True)

TEMPLATE = """version: 1
seed: 42
sequences:
  - protein:
      id: A
      sequence: {seq}
      msa: empty
  - ligand:
      id: L
      smiles: "{smiles}"
properties:
  - affinity:
      binder: L
"""

for tsv in sorted(SMILES_DIR.glob("L*.tsv")):
    target = tsv.stem
    rows = tsv.read_text().strip().splitlines()
    header = rows[0].split("\t")
    data = rows[1].split("\t")
    smiles = data[header.index("SMILES")]
    yaml_text = TEMPLATE.format(seq=CHYMASE, smiles=smiles)
    out = OUT_DIR / f"{target}_input.yaml"
    out.write_text(yaml_text)
    print(f"wrote {out.name}: {smiles}")
