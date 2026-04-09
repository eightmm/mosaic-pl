"""
CCD code category sets for ligand classification.

Priority order in classify():
  1. crystallization_aid  — buffers, solvents, cryoprotectants
  2. ion                  — metal ions, inorganic ions
  3. glycan               — mono/oligo/polysaccharides (AF3 Table 11)
  4. membrane_lipid       — phospholipids, detergents
  5. cofactor             — nucleotide/redox/vitamin cofactors, porphyrins
  6. pigment              — carotenoids, retinal, chlorophylls
  7. metabolite           — endogenous metabolites, sterols, bile acids
  8. (chem_comp_type)     — peptide_like, saccharide, nucleic_acid_like
  9. small_molecule       — default drug-like

is_candidate defaults:
  True  — small_molecule, cofactor, metabolite, peptide_like,
           nucleic_acid_like, nucleotide_like
  False — ion, crystallization_aid, metal_cluster, membrane_lipid,
           steroid, glycan, pigment, saccharide, too_few_heavy_atoms,
           no_protein_contact
"""
from __future__ import annotations

# ── 1. Crystallization aids, buffers, solvents ──────────────────────────── #
# AF3 Table 9 + Boltz1/AF3 Table 10 + NON_KEEPWORTHY solvents
CRYSTALLIZATION_AIDS: frozenset[str] = frozenset({
    # AF3 Table 9
    "SO4","GOL","EDO","PO4","ACT","PEG","DMS","TRS","PGE","PG4",
    "FMT","EPE","MPD","MES","CD","IOD",
    # AF3 Table 10 / Boltz1 LIGAND_EXCLUSION
    "144","15P","1PE","2F2","2JC","3HR","3SY","7N5","7PE","9JE",
    "AAE","ABA","ACE","ACN","ACT","ACY","AZI","BAM","BCN","BCT",
    "BDN","BEN","BME","BO3","BTB","BTC","BU1","C8E","CAD","CAQ",
    "CBM","CCN","CIT","CL","CLR","CM","CMO","CO3","CPT","CXS",
    "D10","DEP","DIO","DMS","DN","DOD","DOX","EDO","EEE","EGL",
    "EOH","EOX","EPE","ETF","FCY","FJO","FLC","FMT","FW5","GOL",
    "GSH","GTT","GYF","HED","IHP","IHS","IMD","IOD","IPA","IPH",
    "LDA","MB3","MEG","MES","MLA","MLI","MOH","MPD","MRD","MSE",
    "MYR","N","NA","NH2","NH4","NHE","NO3","O4B","OHE","OLA",
    "OLC","OMB","OME","OXA","P6G","PE3","PE4","PEG","PEO","PEP",
    "PG0","PG4","PGE","PGR","PLM","PO4","POL","POP","PVO","SAR",
    "SCN","SEO","SEP","SIN","SO4","SPD","SPM","SR","STE","STO",
    "STU","TAR","TBU","TME","TPO","TRS","UNK","UNL","UNX","UPL","URE",
    # NON_KEEPWORTHY solvents/buffers/alkanes
    "2PE","8K6","AKG","B3P","BU3","C14","D12","DPO","DTT","GLU",
    "HEZ","HEX","KZF","LYS","MET","MLT","MPO","OCT","OXE","P33",
    "P4G","PE8","PG5","PG6","PRO","R16","TEW","TFA","TLA","TOE",
    # Heavy-atom phasing reagents (osmium, tantalum, etc.)
    "OHX","CPS","2AN",
    # Coordination complexes / hexammine phasing reagents
    "NCO","IRI","NRU","RHD","CON","PTN","TCN","CUA","CUZ",
    # Polyatomic inorganic anions
    "BO4","NO2","SO3","VN3","VO4","MOO","FPO","SE4","PO3","PI","OXL","LCO","LCP",
    "KO4","WO5","BF4","BS3","ZCM","ZO3","SMO","PER",
    # Organic quaternary ions / phase-transfer reagents
    "TMA","TEA","TBA","E4N","NET","DME","HAI","CHT","T1A","DMI","DTI","3MT",
    # Heavy-atom phasing compounds (multi-atom mercury, lead, gold, osmium)
    "EMC","HGC","MAC","MMC","PBM","AUC","OS4","CSB",
    # Metal-water / octahedral calcium clusters
    "OC1","OC2","OC3","OC4","OC5","OC6","OC7","OC8","OCL","OCM","OCN","OCO",
    # Molybdenum / tungsten oxo-clusters
    "MO1","MO2","MO3","MO4","MO5","MO6","MOW","MW1","MW2","MW3",
    # Sodium clusters / sodium oxide species
    "NA2","NA5","NA6","NAO","NAW",
    # Aluminum / beryllium fluoride complexes
    "ALF","BEF",
    # Hydroxide / simple inorganic anions (AF3 Table 12)
    "OH",
    # Miscellaneous multi-atom species from old IONS
    "ATH","BSY","CAC","CO5","CYN","DSC","EDR","GEP",
    "MH2","MH3","MN5","MN6","O4M","OF1","OF2","OF3","THE","TRA",
    "ZN2","ZN3","ZNO",
    # Cryo/buffer/detergent additives (previously small_molecule)
    "TAM","TRD","HP6","1PG","12P","HTO","PE5","DD9","MYS","UND","SRT","NPO",
    # Synthetic phasing/crystallization porphyrin reagent
    "SFP",
    # PEG polymers (CCD full-scan)
    "P4K","P2K","JEF",
    # Vanadate transition-state analog (zone scan)
    "AD9",
    # Heavy-atom phasing reagent (hexatantalum dodecabromide)
    "TBR",
    # PEG polymer (zone scan)
    "PEU",
    # Maleic acid (pH buffer/crystallization agent)
    "MAE",
    # Hexanetriol cryoprotectant
    "1JW",
})

# ── 2. Ions ─────────────────────────────────────────────────────────────── #
# Monoatomic single-element ions only (no coordination complexes, no polyatomic species)
IONS: frozenset[str] = frozenset({
    # ── Alkali metals ──────────────────────────────────────────────────── #
    "LI","K","RB","CS",
    # ── Alkaline earth metals ──────────────────────────────────────────── #
    "MG","CA","SR","BA",
    # ── 3d transition metals ───────────────────────────────────────────── #
    "V","CR","MN","MN3","FE","FE2","CO","NI","NI1","NI2","NI3",
    "CU","CU1","CU2","CU3","ZN",
    # ── 4d transition metals ───────────────────────────────────────────── #
    "Y","Y1","ZR","MO","RU","RH3","PD","AG","CD","CD1","CD3","CD5",
    # ── 5d transition metals ───────────────────────────────────────────── #
    "W","OS","IR","IR3","PT","PT4","AU","AU3","HG",
    # ── Post-transition metals ─────────────────────────────────────────── #
    "AL","GA","IN","SB","TL","PB",
    # ── Lanthanides ────────────────────────────────────────────────────── #
    "LA","CE","PR","SM","EU","EU3","GD3","TB","DY","HO3","ER3","YB","YB2","LU",
    # ── Actinides ──────────────────────────────────────────────────────── #
    "TH","AM","CF",
    # ── Halogens (anions) ──────────────────────────────────────────────── #
    "F","BR",
    # ── Non-standard CCD codes for single-element ions ─────────────────── #
    "1CU","IUM","SEK","YH","YT3",
    "118","119",                          # synthetic superheavy elements
    "1AL","2FK","2HP","2OF","3CO","3NI","3OF","4MO","4PU","4TI","543","6MO",
})

# ── 3. Glycans ──────────────────────────────────────────────────────────── #
# AF3 Table 11: mono/oligo/polysaccharides (992 entries)
GLYCANS: frozenset[str] = frozenset({
    "045","05L","07E","07Y","08U","09X","0BD","0H0","0HX","0LP",
    "0MK","0NZ","0UB","0V4","0WK","0XY","0YT","10M","12E","145",
    "147","149","14T","15L","16F","16G","16O","17T","18D","18O",
    "1CF","1FT","1GL","1GN","1LL","1S3","1S4","1SD","1X4","20S",
    "20X","22O","22S","23V","24S","25E","26O","27C","289","291",
    "293","2DG","2DR","2F8","2FG","2FL","2GL","2GS","2H5","2HA",
    "2M4","2M5","2M8","2OS","2WP","2WS","32O","34V","38J","3BU",
    "3DO","3DY","3FM","3GR","3HD","3J3","3J4","3LJ","3LR","3MG",
    "3MK","3R3","3S6","3SA","3YW","40J","42D","445","44S","46D",
    "46Z","475","48Z","491","49A","49S","49T","49V","4AM","4CQ",
    "4GC","4GL","4GP","4JA","4N2","4NN","4QY","4R1","4RS","4SG",
    "4UZ","4V5","50A","51N","56N","57S","5GF","5GO","5II","5KQ",
    "5KS","5KT","5KV","5L3","5LS","5LT","5MM","5N6","5QP","5SP",
    "5TH","5TJ","5TK","5TM","61J","62I","64K","66O","6BG","6C2",
    "6DM","6GB","6GP","6GR","6K3","6KH","6KL","6KS","6KU","6KW",
    "6LA","6LS","6LW","6MJ","6MN","6PZ","6S2","6UD","6YR","6ZC",
    "73E","79J","7CV","7D1","7GP","7JZ","7K2","7K3","7NU","83Y",
    "89Y","8B7","8B9","8EX","8GA","8GG","8GP","8I4","8LR","8OQ",
    "8PK","8S0","8YV","95Z","96O","98U","9AM","9C1","9CD","9GP",
    "9KJ","9MR","9OK","9PG","9QG","9S7","9SG","9SJ","9SM","9SP",
    "9T1","9T7","9VP","9WJ","9WN","9WZ","9YW","A0K","A1Q","A2G",
    "A5C","A6P","AAL","ABD","ABE","ABF","ABL","AC1","ACR","ACX",
    "ADA","AF1","AFD","AFO","AFP","AGL","AH2","AH8","AHG","AHM",
    "AHR","AIG","ALL","ALX","AMG","AMN","AMU","AMV","ANA","AOG",
    "AQA","ARA","ARB","ARI","ARW","ASC","ASG","ASO","AXP","AXR",
    "AY9","AZC","B0D","B16","B1H","B1N","B2G","B4G","B6D","B7G",
    "B8D","B9D","BBK","BBV","BCD","BDF","BDG","BDP","BDR","BEM",
    "BFN","BG6","BG8","BGC","BGL","BGN","BGP","BGS","BHG","BM3",
    "BM7","BMA","BMX","BND","BNG","BNX","BO1","BOG","BQY","BS7",
    "BTG","BTU","BW3","BWG","BXF","BXP","BXX","BXY","BZD","C3B",
    "C3G","C3X","C4B","C4W","C5X","CBF","CBI","CBK","CDR","CE5",
    "CE6","CE8","CEG","CEZ","CGF","CJB","CKB","CKP","CNP","CR1",
    "CR6","CRA","CT3","CTO","CTR","CTT","D1M","D5E","D6G","DAF",
    "DAG","DAN","DDA","DDL","DEG","DEL","DFR","DFX","DG0","DGO",
    "DGS","DGU","DJB","DJE","DK4","DKX","DKZ","DL6","DLD","DLF",
    "DLG","DNO","DO8","DOM","DPC","DQR","DR2","DR3","DR5","DRI",
    "DSR","DT6","DVC","DYM","E3M","E5G","EAG","EBG","EBQ","EEN",
    "EEQ","EGA","EMP","EMZ","EPG","EQP","EQV","ERE","ERI","ETT",
    "EUS","F1P","F1X","F55","F58","F6P","F8X","FBP","FCA","FCB",
    "FCT","FDP","FDQ","FFC","FFX","FIF","FK9","FKD","FMF","FMO",
    "FNG","FNY","FRU","FSA","FSI","FSM","FSW","FUB","FUC","FUD",
    "FUF","FUL","FUY","FVQ","FX1","FYJ","G0S","G16","G1P","G20",
    "G28","G2F","G3F","G3I","G4D","G4S","G6D","G6P","G6S","G7P",
    "G8Z","GAA","GAC","GAD","GAF","GAL","GAT","GBH","GC1","GC4",
    "GC9","GCB","GCD","GCN","GCO","GCS","GCT","GCU","GCV","GCW",
    "GDA","GDL","GE1","GE3","GFP","GIV","GL0","GL1","GL2","GL4",
    "GL5","GL6","GL7","GL9","GLA","GLC","GLD","GLF","GLG","GLO",
    "GLP","GLS","GLT","GM0","GMB","GMH","GMT","GMZ","GN1","GN4",
    "GNS","GNX","GP0","GP1","GP4","GPH","GPK","GPM","GPO","GPQ",
    "GPU","GPV","GPW","GQ1","GRF","GRX","GS1","GS9","GTK","GTM",
    "GTR","GU0","GU1","GU2","GU3","GU4","GU5","GU6","GU8","GU9",
    "GUF","GUL","GUP","GUZ","GXL","GXV","GYE","GYG","GYP","GYU",
    "GYV","GZL","H1M","H1S","H2P","H3S","H53","H6Q","H6Z","HBZ",
    "HD4","HNV","HNW","HSG","HSH","HSJ","HSQ","HSX","HSY","HTG",
    "HTM","HVC","IAB","IDC","IDF","IDG","IDR","IDS","IDU","IDX",
    "IDY","IEM","IN1","IPT","ISD","ISL","ISX","IXD","J5B","JFZ",
    "JHM","JLT","JRV","JSV","JV4","JVA","JVS","JZR","K5B","K99",
    "KBA","KBG","KD5","KDA","KDB","KDD","KDE","KDF","KDM","KDN",
    "KDO","KDR","KFN","KG1","KGM","KHP","KME","KO1","KO2","KOT",
    "KTU","L0W","L1L","L6S","L6T","LAG","LAH","LAI","LAK","LAO",
    "LAT","LB2","LBS","LBT","LCN","LDY","LEC","LER","LFC","LFR",
    "LGC","LGU","LKA","LKS","LM2","LMO","LNV","LOG","LOX","LRH",
    "LTG","LVO","LVZ","LXB","LXC","LXZ","LZ0","M1F","M1P","M2F",
    "M3M","M3N","M55","M6D","M6P","M7B","M7P","M8C","MA1","MA2",
    "MA3","MA8","MAB","MAF","MAG","MAL","MAN","MAT","MAV","MAW",
    "MBE","MBF","MBG","MCU","MDA","MDP","MFB","MFU","MG5","MGC",
    "MGL","MGS","MJJ","MLB","MLR","MMA","MN0","MNA","MQG","MQT",
    "MRH","MRP","MSX","MTT","MUB","MUR","MVP","MXY","MXZ","MYG",
    "N1L","N3U","N9S","NA1","NAA","NAG","NBG","NBX","NBY","NDG",
    "NFG","NG1","NG6","NGA","NGC","NGE","NGK","NGR","NGS","NGY",
    "NGZ","NHF","NLC","NM6","NM9","NNG","NPF","NSQ","NT1","NTF",
    "NTO","NTP","NXD","NYT","OAK","OI7","OPM","OSU","OTG","OTN",
    "OTU","OX2","P53","P6P","P8E","PA1","PAV","PDX","PH5","PKM",
    "PNA","PNG","PNJ","PNW","PPC","PRP","PSG","PSV","PTQ","PUF",
    "PZU","QDK","QIF","QKH","QPS","QV4","R1P","R1X","R2B","R2G",
    "RAE","RAF","RAM","RAO","RB5","RBL","RCD","RER","RF5","RG1",
    "RGG","RHA","RHC","RI2","RIB","RIP","RM4","RP3","RP5","RP6",
    "RR7","RRJ","RRY","RST","RTG","RTV","RUG","RUU","RV7","RVG",
    "RVM","RWI","RY7","RZM","S7P","S81","SA0","SCG","SCR","SDY",
    "SEJ","SF6","SF9","SFU","SG4","SG5","SG6","SG7","SGA","SGC",
    "SGD","SGN","SHB","SHD","SHG","SIA","SID","SIO","SIZ","SLB",
    "SLM","SLT","SMD","SN5","SNG","SOE","SOG","SOL","SOR","SR1",
    "SSG","SSH","STW","STZ","SUC","SUP","SUS","SWE","SZZ","T68",
    "T6D","T6P","T6T","TA6","TAG","TCB","TDG","TEU","TF0","TFU",
    "TGA","TGK","TGR","TGY","TH1","TM5","TM6","TMR","TMX","TNX",
    "TOA","TOC","TQY","TRE","TRV","TS8","TT7","TTV","TU4","TUG",
    "TUJ","TUP","TUR","TVD","TVG","TVM","TVS","TVV","TVY","TW7",
    "TWA","TWD","TWG","TWJ","TWY","TXB","TYV","U1Y","U2A","U2D",
    "U63","U8V","U97","U9A","U9D","U9G","U9J","U9M","UAP","UBH",
    "UBO","UDC","UEA","V3M","V3P","V71","VG1","VJ1","VJ4","VKN",
    "VTB","W9T","WIA","WOO","WUN","WZ1","WZ2","X0X","X1P","X1X",
    "X2F","X2Y","X34","X6X","X6Y","XDX","XGP","XIL","XKJ","XLF",
    "XLS","XMM","XS2","XXM","XXR","XXX","XYF","XYL","XYP","XYS",
    "XYT","XYZ","YDR","YIO","YJM","YKR","YO5","YX0","YX1","YYB",
    "YYH","YYJ","YYK","YYM","YYQ","YZ0","Z0F","Z15","Z16","Z2D",
    "Z2T","Z3K","Z3L","Z3Q","Z3U","Z4K","Z4R","Z4S","Z4U","Z4V",
    "Z4W","Z4Y","Z57","Z5J","Z5L","Z61","Z6H","Z6J","Z6W","Z8H",
    "Z8T","Z9D","Z9E","Z9H","Z9K","Z9L","Z9M","Z9N","Z9W","ZB0",
    "ZB1","ZB2","ZB3","ZCD","ZCZ","ZD0","ZDC","ZDO","ZEE","ZEL",
    "ZGE","ZMR",
})

# ── 4. Membrane lipids and detergents ────────────────────────────────────── #
MEMBRANE_LIPIDS: frozenset[str] = frozenset({
    # Locally curated (MEMBRANE_LIPIDS_AND_DETERGENTS)
    "2CV","3PE","6PL","9Y0","AV0","C15","CDL","CPL","DDQ","DMU",
    "FO4","LBN","LHG","LMG","LMN","LMT","LMU","LPE","MC3","P1O",
    "PC1","PCW","PEE","PEF","PEK","PGT","PGV","PLC","PSC","PTY",
    "PX4","T7X","TRT","UMQ","Y01",
    # NON_KEEPWORTHY lipid subset
    "1VU","6OU","DGD","LFA","LOP","LPP","MW9","P5S","PIO","POV","PT5","SQD",
    # Glycerolipids / glycerophospholipids (previously falling to small_molecule)
    "TGL","DGA","3PH","PCF","PGW","PLX",
    # Additional lipids/detergents (previously small_molecule)
    "DPV","78M","MPG","17F","OLB","PEV","PA8","78N","A1A0P","UJO","46E",
    "7Q9","6V6","L2P","LI1","1N7","CVM","DAO","OCA","EIC","DKA","AJP",
    # Monoacylglycerols (CCD full-scan)
    "A1H2K","A1H52",
    # Additional phospholipids / lysophospholipids (zone scan)
    "LPC","PC8","D21","ZP7","4AG","LJQ","ACD",
    # Fatty acids / phospholipids (zone scan round 2)
    "PG8","8Z9","TDA","11A","PC7","A1EW5",
    # Diacylglycerol / phospholipids (zone scan round 3)
    "Z41","A1D7S","A1D7T","A1EZI","A1EZJ","A1H2V",
    "A1INB","A1INC","A1IVO","A1D9Z",
    # Fatty acid (petroselinic acid) / fatty alcohol (octanol)
    "4I1","OC9",
    # Fluorinated phosphocholine detergent
    "A1H8K",
    # Ceramide / NBD-labeled sphingolipid (zone scan round 4)
    "A1D90","A1EC8",
})

# ── 5. Metal clusters ────────────────────────────────────────────────────── #
# Purely inorganic metal-cluster cofactors (Fe-S, Mo-Fe, etc.) — is_candidate=False
# Note: porphyrins/heme (HEM, HEC) stay in COFACTORS — they are organic metalloporphyrins
METAL_CLUSTERS: frozenset[str] = frozenset({
    # Iron-sulfur clusters
    "FES","SF4","F3S","3FE","FEO","FE4","CLF","OEX",
    # Mo-Fe / nitrogenase clusters
    "MOS","MFN",
    # Additional metal clusters (previously small_molecule)
    "ICS",
    # Fe-S / FeFe hydrogenase clusters (CCD zone scan)
    "9S8","S5Q",
})

# ── 6. Cofactors ─────────────────────────────────────────────────────────── #
# Nucleotide/redox/vitamin cofactors (organic, reversibly bound)
COFACTORS: frozenset[str] = frozenset({
    "5GP","ACP","ACO","ADP","AGS","AMP","ANP","AR6","ATP","B12",
    "BTN","COA","DTP","FAD","FDA","FMN","FNR","GDP","GNP","GTP",
    "H4B","IMP","MGD","MTA","NAI","NAD","NAP","NDP","PLP","RBF",
    "SAH","SAM","SFG","TYM","UDP","UPG","APR","TPP","UTP",
    # Porphyrins / heme (organic metalloporphyrins)
    "HEM","HEC",
    # Nucleotide mono/di/triphosphates (substrates/cofactors)
    "TTP","CTP","DGT","DCP","D5M","UMP","U5P","08T","DTP","GMP","CMP",
    # Vitamin B6, pantothenate, CoA-related
    "PMP","PNS","EHZ","8Q1","ZMP","PPC",
    # Additional cofactors (previously small_molecule)
    "GSP","G2P","C5P","UD1","TYD","FOL","PLG","CDP","DUP","PCG",
    "IPE","MYA","FCO","HDD","PPV","BTI",
    # CoA fragments / pantetheine (CCD full-scan)
    "4PS",
    # Metalloporphyrins / heme variants (cofactors, not pigments)
    "CCH","DDH","FEC","HDE","HEV","MHM","F0L","F0X","SIR",
    # Acyl-CoA substrates/cofactors (CCD full-scan)
    "PKZ","A1IZB","ZOZ","HSC","MFK","NBC","ZKK","3H9",
    # Molybdopterin cofactors (CCD full-scan)
    "NWS","PCD","MTV","MTQ","MSS",
    # Thiamine diphosphate variant (TDP; TPP already present)
    "TDP",
    # Inosine di/triphosphate (IMP already present)
    "IDP","ITT",
    # Methane metabolism cofactors
    "F43","COM","TP7",
    # Heme variants
    "VOV",
    # Acyl-CoA additional (zone scan)
    "4KX","MLC","SCA","CO8",
    # Nucleotide cofactors / second messengers
    "B4P","A1AEP",
    # Isoprenyl pyrophosphates (farnesyl-PP, geranylgeranyl-PP)
    "FPS","GGS",
    # CoA derivatives (zone scan round 2)
    "3VV","A1CZD","A1IJB","A1JLN",
    # cGAMP second messenger
    "1SY",
    # Cobalamin / cobamide (zone scan round 3)
    "B13","DMD",
    # AMP phosphoramidate / adenosine polyphosphate analogs
    "AN2","ZAN","HQG","DQV","A1JDM","A1II9","LRM",
    # Pyridoxal phosphate (PLP) Schiff-base adducts
    "7VO","A1A8V","A1BC7","A1BD0","A1BD1","A1BDC","ILP",
    # Nucleotide sugar cofactors (GDP-/UDP-sugars)
    "A1CCU","UD2","A1EJE","A1EJJ",
    # NAD/ADP-ribose derivatives
    "A1EL0","A1EL1",
    # Nucleoside / nucleotide analogs
    "A1EMQ","A1EOD","OJC","3D1","FGR",
    # Flavin nucleotide derivative
    "4LU",
    # CoA-thiol conjugate (phosphopantetheine-sulfoethyl)
    "SHT",
    # Menaquinone-10 (respiratory quinone cofactor)
    "MQE",
    # Bis-purine nucleoside analogs / polyphosphate dinucleotides (zone scan round 4)
    "4BW","2BA","6YY","Y65",
})

# ── 6. Pigments ──────────────────────────────────────────────────────────── #
# Carotenoids, retinal, chlorophylls, quinones, other chromophores
PIGMENTS: frozenset[str] = frozenset({
    "5X6","8CT","A1EL2","A1ELD","A1L1G","A1L6D","BCL","BCR","BPH",
    "CHL","CLA","CL0","CRT","CYC","DD6","II0","IHT","KC2","LUT",
    "NEX","PHO","PL9","PQN","RET","SPO","SPN","U10","U4Z","WVN","XAT",
    "DVT","HEA","UQ8",
    # Phycobilins and light-harvesting pigments (previously in small_molecule)
    "PEB","PUB","A86","V7N","C2E","PVB","PCB","DBV",
    # Additional chlorophylls/bacteriochlorophylls
    "CL7","BCB","KC1","07D","MBR","BCQ","BCC",
    # Carotenoids
    "CAR","LYC","ZEA","VIO","ANT","DIA","FUC","PXN",
    # Additional pigments (previously small_molecule)
    "ZEX","PID","LBV",
    # Additional carotenoids (CCD full-scan)
    "AXT","K3I","A1LXP","LUX","45D","45H",
    # Phytochromobilin bound form
    "2VO",
    # Ubiquinone / menaquinone variants (zone scan)
    "UQ5","1L3",
    # Carotenoid precursors
    "A1MBA","A1L0S",
    # Carotenoid / polyene pigments (zone scan round 2)
    "H4X","A1EYK","A1L1F","A1EFU",
    # Bilin / bile-pigment type (zone scan round 4)
    "A1L6M",
})

# ── 7. Metabolites ───────────────────────────────────────────────────────── #
# Endogenous metabolites, sterols, bile acids, nucleoside metabolites
METABOLITES: frozenset[str] = frozenset({
    "3PG","5AD","6NA","A3P","ABU","ADN","APC","6V0","B81","CHD",
    "CHO","CMP","ERG","FPP","G3P","GCP","GSU","OG6","ORO","OXM",
    "OAA","PA5","PAM","PYR","TMP",
    # Additional metabolites (previously small_molecule)
    "PUT","OGA","ADE","LMR","URA","PGA","13P","GUN","SRO","FUM","HCA","I3P","HC4",
    # Bile acids / bile salts (CCD full-scan)
    "TCH","TUD","GCH","DHO","JN3","CHC","IU5","IU6","A1EPX",
    # Retinoic acids / vitamin A metabolites
    "3KV","9CR","REA",
    # Sphinganine precursor
    "VSD",
    # Vitamin K3 (menadione, provitamin)
    "VK3",
    # Flavonoids / plant polyphenols (metabolite subclass)
    "QUE","FSE","HW2",
    # Sphingolipid backbone
    "SPH",
    # Nicotinamide (vitamin B3 precursor)
    "NCA",
    # Oncometabolite / TCA cycle intermediates (zone scan round 2)
    "2HG","COI",
    # Riboflavin precursors / lumazine
    "LMZ","DLZ",
    # Inositol polyphosphate / phosphorylated metabolites (zone scan round 3)
    "I8P","HPV","S2G","3OH","TTN","P7I",
    # Quinic acid / flavonoids / phenolic metabolites
    "QIC","6JP","H9R","I75","YMR","A1JKI",
    # Pyrrolopyrimidine (folate/purine pathway)
    "2KA",
    # Rare sugars / amino acids
    "A1EGE","78U","A1H6S",
    # N-acyl amino acid / siderophore
    "A1BIH","FCE",
    # Phenolic / aromatic metabolites (zone scan round 4)
    "BZS","HCI","A1CAY",
    # Phosphinothricin (natural glutamate analog from Streptomyces)
    "PPQ",
})

# ── 8. Steroids ──────────────────────────────────────────────────────────── #
# Steroid hormones, corticosteroids, sex hormones, vitamin D, anabolic steroids
# is_candidate=True (many are drug targets / drug molecules)
STEROIDS: frozenset[str] = frozenset({
    # Glucocorticoids / corticosteroids
    "DEX","TUA","UQC","ZK5","1TA",
    # Mineralocorticoids
    "AS4",
    # Sex hormones (estrogens)
    "EST","ESL","ECO","EEU","E3G","STG","U38","FY5","ESO","3WF",
    # Sex hormones (androgens / progestogens)
    "TES","BDT","DHT","Q6J","STR","SIH","TH2","A1IOS",
    # DHEA / adrenal androgens
    "ZQK",
    # 25-hydroxycholesterol analog (zone scan round 2)
    "A1IF2",
    # Steroid drug/hormone (zone scan round 3)
    "A1CJG",
})

# ── Convenience: all non-candidate sets ─────────────────────────────────── #
# is_candidate=False by default (trivial artifacts)
NON_CANDIDATE_TYPES: frozenset[str] = frozenset({
    "ion", "crystallization_aid", "metal_cluster",
    "membrane_lipid", "steroid", "glycan", "pigment", "saccharide",
})

# Full exclusion set for backward compat / config defaults
ALL_EXCLUDED: frozenset[str] = (
    CRYSTALLIZATION_AIDS | IONS | GLYCANS | MEMBRANE_LIPIDS
)
