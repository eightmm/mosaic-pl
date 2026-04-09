from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .ccd_categories import (
    COFACTORS,
    CRYSTALLIZATION_AIDS,
    GLYCANS,
    IONS,
    MEMBRANE_LIPIDS,
    METABOLITES,
    METAL_CLUSTERS,
    PIGMENTS,
    STEROIDS,
)

if TYPE_CHECKING:
    from .ccd_lookup import CCDEntry

# name 키워드 기반 분류 (소문자 매칭)
_COFACTOR_KEYWORDS = frozenset({
    "coenzyme", "cofactor", "flavin", "nicotinamide", "pyridoxal",
    "thiamine", "biotin", "lipoic", "tetrahydrofolate", "porphyrin",
    "cytochrome", "ubiquinone", "menaquinone", "plastoquinone",
    "adenosyl", "methylcobalamin", "cobalamin",
})
_PIGMENT_KEYWORDS = frozenset({
    "carotenoid", "retinal", "retinol", "chlorophyll", "bacteriochlorophyll",
    "chromophore", "biliverdin", "bilirubin", "phycocyanin", "phycoerythrin",
})
_METABOLITE_KEYWORDS = frozenset({
    "sterol", "steroid", "cholesterol", "bile acid", "bile salt",
    "prostaglandin", "leukotriene", "sphingosine", "ceramide",
    "glucosylceramide", "ganglioside",
})
_MEMBRANE_LIPID_KEYWORDS = frozenset({
    "phospholipid", "phosphatidyl", "lysophosphatidyl", "sphingomyelin",
    "diacylglycerol", "triacylglycerol", "triglyceride", "cardiolipin",
    "lysophospholipid",
})
# pdbx_type → ligand_type 매핑
_PDBX_TYPE_MAP: dict[str, str] = {
    "ATOMN": "nucleotide_like",    # 뉴클레오타이드/뉴클레오시드 스캐폴드
    "HETAC": "cofactor",           # CoA 유도체 (진짜 조효소)
    "HETAI": "ion",                # 단순 이온
    "HETIC": "ion",                # 배위 이온
    "HETAS": "crystallization_aid",
}

# ligand_type 값 상수
LIGAND_TYPE_SMALL_MOLECULE = "small_molecule"
LIGAND_TYPE_ION = "ion"
LIGAND_TYPE_GLYCAN = "glycan"
LIGAND_TYPE_CRYSTALLIZATION_AID = "crystallization_aid"
LIGAND_TYPE_MEMBRANE_LIPID = "membrane_lipid"
LIGAND_TYPE_COFACTOR = "cofactor"
LIGAND_TYPE_PIGMENT = "pigment"
LIGAND_TYPE_METABOLITE = "metabolite"
LIGAND_TYPE_PEPTIDE_LIKE = "peptide_like"
LIGAND_TYPE_SACCHARIDE = "saccharide"
LIGAND_TYPE_NUCLEIC_ACID_LIKE = "nucleic_acid_like"
LIGAND_TYPE_NUCLEOTIDE_LIKE = "nucleotide_like"
LIGAND_TYPE_METAL_CLUSTER = "metal_cluster"
LIGAND_TYPE_STEROID = "steroid"

# is_candidate=False by default for these types
_NON_CANDIDATE_TYPES = frozenset({
    LIGAND_TYPE_ION,
    LIGAND_TYPE_CRYSTALLIZATION_AID,
    LIGAND_TYPE_METAL_CLUSTER,
    LIGAND_TYPE_MEMBRANE_LIPID,
    LIGAND_TYPE_STEROID,
    LIGAND_TYPE_GLYCAN,
    LIGAND_TYPE_PIGMENT,
    LIGAND_TYPE_SACCHARIDE,
})


@dataclass(frozen=True, slots=True)
class LigandClassification:
    ligand_type: str
    is_candidate: bool
    exclude_reason: str | None = None


class LigandRules:
    """
    Classify any CCD code into a typed ligand category.

    Priority order:
      1. crystallization_aid  (buffers, cryoprotectants, solvents)
      2. ion                  (metal & inorganic ions)
      3. glycan               (mono/oligo/polysaccharides)
      4. membrane_lipid       (phospholipids, detergents)
      5. cofactor             (nucleotides, redox/vitamin cofactors, porphyrins)
      6. pigment              (carotenoids, retinal, chlorophylls)
      7. metabolite           (endogenous metabolites, sterols)
      8. chem_comp_type       (peptide_like / saccharide / nucleic_acid_like)
      9. small_molecule       (default, drug-like)

    is_candidate=True for: small_molecule, cofactor, metabolite,
    peptide_like, nucleic_acid_like, nucleotide_like.
    is_candidate=False for: ion, crystallization_aid, metal_cluster,
    membrane_lipid, steroid, glycan, pigment, saccharide.
    """

    def __init__(self, min_heavy_atoms: int = 6) -> None:
        self._min_heavy_atoms = min_heavy_atoms

    def classify(
        self,
        ccd_code: str,
        chem_comp_type: str | None,
        heavy_atom_count: int,
        ccd_entry: "CCDEntry | None" = None,
    ) -> LigandClassification:
        if heavy_atom_count < self._min_heavy_atoms:
            return LigandClassification(
                LIGAND_TYPE_ION if heavy_atom_count <= 1 else LIGAND_TYPE_CRYSTALLIZATION_AID,
                False,
                "too_few_heavy_atoms",
            )

        code = ccd_code.upper()

        # ── Priority 1: crystallization aids / solvents / buffers ── #
        if code in CRYSTALLIZATION_AIDS:
            return LigandClassification(LIGAND_TYPE_CRYSTALLIZATION_AID, False, None)

        # ── Priority 2: ions ──────────────────────────────────────── #
        if code in IONS:
            return LigandClassification(LIGAND_TYPE_ION, False, None)

        # ── Priority 3: glycans ───────────────────────────────────── #
        if code in GLYCANS:
            return LigandClassification(LIGAND_TYPE_GLYCAN, False, None)

        # ── Priority 4: membrane lipids / detergents ──────────────── #
        if code in MEMBRANE_LIPIDS:
            return LigandClassification(LIGAND_TYPE_MEMBRANE_LIPID, False, None)

        # ── Priority 5: metal clusters (inorganic, is_candidate=False) ── #
        if code in METAL_CLUSTERS:
            return LigandClassification(LIGAND_TYPE_METAL_CLUSTER, False, None)

        # ── Priority 6: cofactors ─────────────────────────────────── #
        if code in COFACTORS:
            return LigandClassification(LIGAND_TYPE_COFACTOR, True, None)

        # ── Priority 7: pigments / chromophores ───────────────────── #
        if code in PIGMENTS:
            return LigandClassification(LIGAND_TYPE_PIGMENT, False, None)

        # ── Priority 8: metabolites ───────────────────────────────── #
        if code in METABOLITES:
            return LigandClassification(LIGAND_TYPE_METABOLITE, True, None)

        # ── Priority 8.1: steroids ────────────────────────────────── #
        if code in STEROIDS:
            return LigandClassification(LIGAND_TYPE_STEROID, False, None)

        # ── Priority 8: chem_comp_type fallback ───────────────────── #
        normalized = (chem_comp_type or "").strip().lower()
        if "peptide" in normalized:
            return LigandClassification(LIGAND_TYPE_PEPTIDE_LIKE, True, None)
        if "saccharide" in normalized:
            return LigandClassification(LIGAND_TYPE_SACCHARIDE, False, None)
        if "rna" in normalized or "dna" in normalized:
            return LigandClassification(LIGAND_TYPE_NUCLEIC_ACID_LIKE, True, None)

        # ── Priority 8.5: CCD pdbx_type + name keyword fallback ───── #
        if ccd_entry is not None:
            pdbx = (ccd_entry.pdbx_type or "").upper()
            mapped = _PDBX_TYPE_MAP.get(pdbx)
            if mapped == "nucleotide_like":
                return LigandClassification(LIGAND_TYPE_NUCLEOTIDE_LIKE, True, None)
            if mapped == "cofactor":
                return LigandClassification(LIGAND_TYPE_COFACTOR, True, None)
            if mapped == "ion":
                return LigandClassification(LIGAND_TYPE_ION, False, None)
            if mapped == "crystallization_aid":
                return LigandClassification(LIGAND_TYPE_CRYSTALLIZATION_AID, False, None)

            name_lower = (ccd_entry.name or "").lower()
            if any(kw in name_lower for kw in _COFACTOR_KEYWORDS):
                return LigandClassification(LIGAND_TYPE_COFACTOR, True, None)
            if any(kw in name_lower for kw in _PIGMENT_KEYWORDS):
                return LigandClassification(LIGAND_TYPE_PIGMENT, False, None)
            if any(kw in name_lower for kw in _MEMBRANE_LIPID_KEYWORDS):
                return LigandClassification(LIGAND_TYPE_MEMBRANE_LIPID, False, None)
            if any(kw in name_lower for kw in _METABOLITE_KEYWORDS):
                return LigandClassification(LIGAND_TYPE_METABOLITE, True, None)

        # ── Priority 9: default small molecule ────────────────────── #
        return LigandClassification(LIGAND_TYPE_SMALL_MOLECULE, True, None)
