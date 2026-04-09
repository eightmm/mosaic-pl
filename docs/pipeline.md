# CASP17 Protein-Ligand Pipeline

```
                          ┌──────────────────────┐
                          │      INPUT           │
                          │  Protein Sequence    │
                          │  Ligand SMILES       │
                          │  (unified YAML)      │
                          └──────────┬───────────┘
                                     │
                ┌────────────────────┼────────────────────┐
                │                    │                    │
                ▼                    ▼                    │
  ┌──────────────────────┐ ┌─────────────────────────┐   │
  │  SEQUENCE SEARCH     │ │  CO-FOLDING             │   │
  │                      │ │                         │   │
  │  MMseqs2             │ │  ┌───────┐  ┌────────┐  │   │
  │  488k seqs DB        │ │  │Boltz-2│  │Boltz-2x│  │   │
  │  ~3s                 │ │  │       │  │(potent)│  │   │
  │         │            │ │  │  +aff │  │  +aff  │  │   │
  │         ▼            │ │  └───┬───┘  └───┬────┘  │   │
  │  ┌──────────────┐   │ │      │  MSA ──────┼──┐   │   │
  │  │rcsb_index.db │   │ │      │           │  │   │   │
  │  │ligand filter │   │ │  ┌───┴────┐  ┌──┴──▼┐  │   │
  │  │(CCD classify)│   │ │  │Protenix│  │ AF3  │  │   │
  │  └──────────────┘   │ │  │ v1/v2  │  │(JAX) │  │   │
  │         │            │ │  └───┬────┘  └──┬───┘  │   │
  │         ▼            │ │      │          │      │   │
  │  hits + ligand info  │ │      ▼          ▼      │   │
  └──────────────────────┘ │  4 structures + scores │   │
                           └────────────┬────────────┘   │
                                        │                │
                ┌───────────────────────┼────────────────┘
                │                       │
                ▼                       ▼
  ┌──────────────────────┐ ┌─────────────────────────────┐
  │  STRUCTURE SEARCH    │ │  DOCKING PREP (auto)        │
  │                      │ │                             │
  │  Foldseek × 4 models │ │  1. Best model (pLDDT)     │
  │  251k struct DB      │ │     AF3 > Boltz > Protenix │
  │                      │ │                             │
  │  Cross-model         │ │  2. Binding site            │
  │  consensus           │ │     SwinSite (ML, 6s)      │
  │  + ligand filter     │ │     P2Rank (surface, 8s)   │
  │                      │ │                             │
  │  "PDB X found by     │ │  3. Receptor prep           │
  │   all 4 models with  │ │     CIF → PDB (gemmi)      │
  │   SAH cofactor"      │ │     PDB → PDBQT (pdb2pqr)  │
  └──────────────────────┘ │     + protonated PDB        │
                           │                             │
                           │  4. Ligand prep              │
                           │     SMILES → SDF (RDKit)    │
                           │     SDF → PDBQT (meeko)     │
                           │                             │
                           │  5. Box: 22.5Å, sp=0.375Å  │
                           └──────────────┬──────────────┘
                                          │
                          ┌───────────────┼───────────────┐
                          │               │               │
                          ▼               ▼               ▼
                   ┌────────────┐ ┌──────────────┐ ┌────────────┐
                   │    Vina    │ │ AutoDock-GPU │ │ Protenix-  │
                   │  (Py API) │ │   (CUDA)     │ │   Dock     │
                   │           │ │              │ │ (AmberFF)  │
                   │   ~3s     │ │    ~10s      │ │  ~5-30min  │
                   │           │ │              │ │            │
                   │ 9 poses   │ │  18 poses    │ │  N poses   │
                   │ best:     │ │  best:       │ │            │
                   │ -6.0      │ │ -7.2 kcal    │ │            │
                   └─────┬─────┘ └──────┬───────┘ └─────┬──────┘
                         │              │               │
                         ▼              ▼               ▼
                   ┌──────────────────────────────────────────┐
                   │  POST-ANALYSIS                           │
                   │                                          │
                   │  PDBQT/DLG → SDF (meeko mk_export.py)   │
                   │                                          │
                   │  ┌────────────┐    ┌─────────────┐      │
                   │  │  BA-Pred   │    │ RMSD-Pred   │      │
                   │  │  (GNN)     │    │  (GNN)      │      │
                   │  │            │    │             │      │
                   │  │ pKd pred   │    │ pRMSD pred  │      │
                   │  │ kcal/mol   │    │ >2Å prob    │      │
                   │  └────────────┘    └─────────────┘      │
                   └──────────────────────┬───────────────────┘
                                          │
                          ┌───────────────▼───────────────┐
                          │          RESULTS              │
                          │                               │
                          │  experiments/runs/<target>/    │
                          │  ├── boltz2/    (CIF+affinity)│
                          │  ├── boltz2x/   (CIF+affinity)│
                          │  ├── protenix/  (CIF)         │
                          │  ├── alphafold3/(CIF)         │
                          │  ├── vina/      (poses+score) │
                          │  ├── autodock_gpu/(poses)     │
                          │  ├── protenix_dock/(poses)    │
                          │  ├── structure_search/        │
                          │  │   (consensus JSON)         │
                          │  └── analysis/                │
                          │      (BA-Pred + RMSD-Pred)    │
                          └───────────────────────────────┘
```

## Timing (RTX 6000 Ada, single GPU)

```
Stage                          Time
─────────────────────────────────────
MMseqs2 sequence search         3s
Boltz-2 + affinity            143s
Boltz-2x + affinity            88s
Protenix v1                   133s
Boltz MSA → AF3 bridge         <1s
AlphaFold3                    129s
─────────────────────────────────────
Cofolding total               ~8min
─────────────────────────────────────
Foldseek × 4 models           ~30s
Docking prep (P2Rank+SwinSite) ~10s
Vina                            3s
AutoDock-GPU                   10s
Protenix-Dock                5-30min
─────────────────────────────────────
BA-Pred + RMSD-Pred           ~10s
─────────────────────────────────────
TOTAL                       ~15-40min
```

## Tool Ecosystem

```
┌─────────────────────────────────────────────────────────────────┐
│                    .venvs/ (isolated environments)              │
│                                                                 │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  ┌───────────────┐  │
│  │  boltz   │  │ protenix │  │ alphafold3│  │ protenix-dock │  │
│  │ Py 3.12  │  │ Py 3.12  │  │  Py 3.12  │  │  micromamba   │  │
│  │torch 2.11│  │torch 2.7 │  │   JAX     │  │ torch 2.1    │  │
│  │CUDA 13.0 │  │CUDA 12.6 │  │          │  │ ambertools   │  │
│  │cuequiv.  │  │cuequiv.  │  │          │  │ tleap, vina  │  │
│  └─────────┘  └──────────┘  └───────────┘  └───────────────┘  │
│                                                                 │
│  ┌─────────────────────────────────────────┐                   │
│  │               pred                      │                   │
│  │  Py 3.12, torch 2.4, dgl 2.4           │                   │
│  │  BA-Pred, RMSD-Pred, SwinSite          │                   │
│  │  openbabel, meeko, gemmi, timm          │                   │
│  └─────────────────────────────────────────┘                   │
│                                                                 │
│  ┌──────────────────────────────────────┐                      │
│  │            .local/bin/               │                      │
│  │  mmseqs, foldseek, autogrid4        │                      │
│  │  autodock_gpu_128wi, prank (JDK21)  │                      │
│  └──────────────────────────────────────┘                      │
│                                                                 │
│  ┌──────────────────────────────────────┐                      │
│  │         data/search_dbs/             │                      │
│  │  sequence/rcsb_seqDB  (2.0GB)       │                      │
│  │  structure/rcsb_structDB (7.7GB)     │                      │
│  └──────────────────────────────────────┘                      │
│                                                                 │
│  ┌──────────────────────────────────────┐                      │
│  │    ~/DB/RCSB/processed/              │                      │
│  │  rcsb_index.db (SQLite, 251k PDBs)  │                      │
│  │  Updated weekly by mmcif-parser      │                      │
│  └──────────────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```
