"""CCD (Chemical Component Dictionary) classification for ligand filtering."""

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
from .ccd_lookup import CCDEntry, load_ccd_lookup
from .ligands import LigandClassification, LigandRules
