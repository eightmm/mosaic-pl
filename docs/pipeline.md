# CASP17 Protein-Ligand Pipeline

## Pipeline Overview

```mermaid
flowchart TB
    subgraph INPUT["📥 Input"]
        A["Protein Sequence + Ligand SMILES<br/>(unified YAML)"]
    end

    subgraph S1["Stage 1: Template Search"]
        B1["MMseqs2<br/>488k seqs DB<br/>~3s"]
        B2["rcsb_index.db<br/>Ligand Filter<br/>(CCD classify)"]
        B1 --> B2
    end

    subgraph S2["Stage 2: Co-folding"]
        direction LR
        C1["Boltz-2<br/>+ affinity"]
        C2["Boltz-2x<br/>+ affinity<br/>(potentials)"]
        C3["Protenix<br/>v1/v2"]
        C4["AlphaFold3<br/>(JAX)"]
        C1 -.->|MSA bridge| C4
    end

    subgraph S3["Stage 3: Structure Search"]
        D1["Foldseek × 4 models<br/>251k struct DB"]
        D2["Cross-model Consensus<br/>+ Ligand Filter"]
        D1 --> D2
    end

    subgraph S4["Stage 4: Docking Prep"]
        E1["Auto-select Best Model<br/>(pLDDT score)"]
        E2["Binding Site<br/>SwinSite > P2Rank"]
        E3["Receptor Prep<br/>CIF→PDB→PDBQT"]
        E4["Ligand Prep<br/>SMILES→SDF→PDBQT"]
        E1 --> E2 --> E3 & E4
    end

    subgraph S5["Stage 5: Docking"]
        direction LR
        F1["Vina<br/>~3s"]
        F2["AutoDock-GPU<br/>~10s"]
        F3["Protenix-Dock<br/>~5-30min"]
    end

    subgraph S6["Stage 6: Post-analysis"]
        direction LR
        G1["BA-Pred<br/>(pKd)"]
        G2["RMSD-Pred<br/>(pRMSD)"]
    end

    subgraph OUTPUT["📦 Results"]
        H["experiments/runs/&lt;target&gt;/<br/>structures + affinities + poses + scores"]
    end

    A --> S1 & S2
    S2 --> S3
    S2 --> S4
    S4 --> S5
    S5 --> S6
    S1 & S3 & S6 --> OUTPUT
```

## Stage Details

### Stage 1: Sequence-based Template Search

```mermaid
flowchart LR
    A["Protein\nSequence"] --> B["MMseqs2\neasy-search"]
    B --> C["Hit List\n(PDB IDs)"]
    C --> D["rcsb_index.db\nSQLite Lookup"]
    D --> E["Filtered Hits\n+ ligand CCD\n+ SMILES\n+ category"]

    style D fill:#f9f,stroke:#333
```

**CCD Classification** (48,965 entries → 10 categories):

| Category | Candidate | Examples |
|----------|:---------:|---------|
| `small_molecule` | ✅ | Drug-like inhibitors |
| `cofactor` | ✅ | ATP, NAD, FAD, HEM |
| `metabolite` | ✅ | Sterols, bile acids |
| `peptide_like` | ✅ | Short peptide inhibitors |
| `nucleotide_like` | ✅ | Nucleoside analogs |
| `ion` | ❌ | ZN, MG, FE, CA |
| `crystallization_aid` | ❌ | GOL, EDO, PEG, SO4 |
| `glycan` | ❌ | NAG, MAN, GAL |
| `membrane_lipid` | ❌ | Phospholipids, detergents |
| `pigment` | ❌ | Carotenoids, chlorophylls |

### Stage 2: Co-folding

```mermaid
flowchart TB
    INPUT["Unified YAML\n+ MSA Server"] --> B2["Boltz-2\n(no potentials)"]
    INPUT --> B2X["Boltz-2x\n(use_potentials)"]
    INPUT --> PX["Protenix v1/v2"]
    B2 -->|MSA CSV→A3M| AF3["AlphaFold3"]
    INPUT --> AF3

    B2 --> OUT1["outputs/boltz2/\nCIF + confidence\n+ affinity JSON"]
    B2X --> OUT2["outputs/boltz2x/\nCIF + confidence\n+ affinity JSON"]
    PX --> OUT3["outputs/protenix/\nCIF + confidence"]
    AF3 --> OUT4["outputs/alphafold3/\nCIF + confidence\n+ ranking"]

    style OUT1 fill:#e1f5fe
    style OUT2 fill:#e1f5fe
    style OUT3 fill:#e1f5fe
    style OUT4 fill:#e1f5fe
```

**Boltz Affinity Output:**
```json
{
  "affinity_pred_value": 2.62,        // log10(IC50) μM — lower = stronger
  "affinity_probability_binary": 0.41  // binder probability [0-1]
}
```

### Stage 3: Structure Search (Foldseek Consensus)

```mermaid
flowchart TB
    B2["Boltz-2\nCIF"] --> FS1["Foldseek"]
    B2X["Boltz-2x\nCIF"] --> FS2["Foldseek"]
    PX["Protenix\nCIF"] --> FS3["Foldseek"]
    AF3["AF3\nCIF"] --> FS4["Foldseek"]

    FS1 & FS2 & FS3 & FS4 --> MERGE["Merge &\nConsensus"]
    MERGE --> FILTER["Ligand Filter\n(rcsb_index.db)"]
    FILTER --> RESULT["Ranked PDBs\n• num_models found\n• avg TM-score\n• ligand info"]

    style RESULT fill:#fff9c4
```

### Stage 4: Docking Preparation

```mermaid
flowchart TB
    MODELS["4 Model\nOutputs"] --> SELECT["Auto-select\nBest (pLDDT)"]
    SELECT --> CIF["Best CIF"]

    CIF --> GEMMI["gemmi\nCIF→PDB"]
    GEMMI --> PDB2PQR["pdb2pqr\n+H, charges\nHIS→HID/HIE"]
    PDB2PQR --> PDBQT_R["receptor.pdbqt\n(AD4 types)"]
    PDB2PQR --> PDB_P["receptor_protonated.pdb\n(for Protenix-Dock)"]

    SMILES["Ligand\nSMILES"] --> RDKIT["RDKit\nETKDG+MMFF"]
    RDKIT --> SDF["ligand.sdf"]
    SDF --> MEEKO["meeko"]
    MEEKO --> PDBQT_L["ligand.pdbqt"]

    CIF --> SWIN["SwinSite\n(Swin-Unet)"]
    CIF --> P2R["P2Rank\n(surface)"]
    SWIN & P2R --> BOX["Docking Box\n22.5Å × 22.5Å × 22.5Å\nspacing 0.375Å"]

    style BOX fill:#c8e6c9
    style PDBQT_R fill:#e1f5fe
    style PDBQT_L fill:#e1f5fe
```

### Stage 5: Docking

```mermaid
flowchart LR
    subgraph VINA["Vina (Python API)"]
        V1["compute_vina_maps"] --> V2["dock()"] --> V3["write_poses()"]
    end

    subgraph ADG["AutoDock-GPU"]
        A1["autogrid4\n(grid maps)"] --> A2["autodock_gpu\n(CUDA, 100 runs)"]
    end

    subgraph PXDOCK["Protenix-Dock"]
        P1["prepare_receptor\n(tleap)"] --> P2["generate_cache_maps"] --> P3["run_docking"]
    end

    style VINA fill:#e8f5e9
    style ADG fill:#fff3e0
    style PXDOCK fill:#fce4ec
```

### Stage 6: Post-analysis

```mermaid
flowchart LR
    VINA_OUT["Vina\ndocked.pdbqt"] --> MK["mk_export.py\n→ SDF"]
    ADG_OUT["ADG\ndocking.dlg"] --> MK
    MK --> BA["BA-Pred\n(GNN)"]
    MK --> RMSD["RMSD-Pred\n(GNN)"]
    SDF_IN["Input\nligand.sdf"] --> BA & RMSD

    BA --> BA_OUT["pKd\nkcal/mol"]
    RMSD --> RMSD_OUT["pRMSD\n>2Å prob"]

    style BA_OUT fill:#fff9c4
    style RMSD_OUT fill:#fff9c4
```

## Timing (RTX 6000 Ada)

```mermaid
gantt
    title Pipeline Execution Timeline
    dateFormat X
    axisFormat %s

    section Search
    MMseqs2 sequence search     :0, 3

    section Co-folding
    Boltz-2 + affinity          :3, 146
    Boltz-2x + affinity         :146, 234
    Protenix v1                 :234, 367
    Boltz MSA → AF3 bridge      :367, 368
    AlphaFold3                  :368, 497

    section Docking Prep
    Auto-select + P2Rank + SwinSite :497, 507

    section Docking
    Vina                        :507, 510
    AutoDock-GPU                :510, 520
    Protenix-Dock               :520, 2240

    section Post-analysis
    BA-Pred + RMSD-Pred         :2240, 2250
```

| Stage | Time | % of Total |
|-------|-----:|:----------:|
| MMseqs2 | 3s | <1% |
| Boltz-2 + affinity | 143s | 6% |
| Boltz-2x + affinity | 88s | 4% |
| Protenix v1 | 133s | 6% |
| AlphaFold3 | 129s | 6% |
| Foldseek × 4 | ~30s | 1% |
| Docking prep | ~10s | <1% |
| Vina | 3s | <1% |
| AutoDock-GPU | 10s | <1% |
| **Protenix-Dock** | **~29min** | **77%** |
| Post-analysis | ~10s | <1% |
| **Total** | **~37min** | |

> Protenix-Dock is the bottleneck (77% of total time). All other stages complete in ~8 min.

## Tool Ecosystem

```mermaid
graph TB
    subgraph VENVS[".venvs/ (isolated environments)"]
        BOLTZ["boltz<br/>Py3.12 + torch 2.11<br/>CUDA 13.0"]
        PROTENIX["protenix<br/>Py3.12 + torch 2.7<br/>CUDA 12.6"]
        AF3["alphafold3<br/>Py3.12 + JAX"]
        PXDOCK["protenix-dock<br/>micromamba<br/>ambertools + tleap"]
        PRED["pred<br/>Py3.12 + torch 2.4<br/>dgl 2.4 + openbabel"]
    end

    subgraph BINS[".local/bin/"]
        MMSEQS["mmseqs"]
        FOLDSEEK["foldseek"]
        AUTOGRID["autogrid4"]
        ADGPU["autodock_gpu_128wi"]
        PRANK["prank (JDK 21)"]
    end

    subgraph DBS["Search Databases"]
        SEQDB["sequence/rcsb_seqDB<br/>488k seqs, 2.0GB"]
        STRUCTDB["structure/rcsb_structDB<br/>251k structs, 7.7GB"]
        RCSBDB["rcsb_index.db<br/>251k PDBs, 2.5M ligands"]
    end

    BOLTZ --- |"Boltz-2/2x"| COFOLDING["Co-folding"]
    PROTENIX --- COFOLDING
    AF3 --- COFOLDING
    PXDOCK --- |"Protenix-Dock + Vina"| DOCKING["Docking"]
    PRED --- |"BA-Pred + RMSD-Pred + SwinSite"| ANALYSIS["Analysis"]
    MMSEQS --- SEARCH["Search"]
    FOLDSEEK --- SEARCH
    ADGPU --- DOCKING
    PRANK --- BINDING["Binding Site"]

    style VENVS fill:#e3f2fd
    style BINS fill:#f3e5f5
    style DBS fill:#e8f5e9
```
