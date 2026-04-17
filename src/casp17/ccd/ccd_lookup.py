from __future__ import annotations

import contextlib
import gzip
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import gemmi

# RCSB 공식 CCD 배포 URL (매주 갱신)
CCD_DOWNLOAD_URL = "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz"


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    v = value.strip()
    if v in ("?", ".", ""):
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _safe_str(value: str | None) -> str | None:
    if value is None:
        return None
    v = value.strip().strip("'\"")
    return v if v not in ("?", ".") else None


@dataclass(frozen=True, slots=True)
class CCDEntry:
    chem_comp_type: str | None
    formula_weight: float | None
    formula: str | None
    smiles: str | None
    pdbx_type: str | None = None
    name: str | None = None


def download_components_cif(
    dest_dir: Path | str,
    url: str = CCD_DOWNLOAD_URL,
    force: bool = False,
) -> Path:
    """
    RCSB에서 components.cif.gz를 받아 dest_dir에 압축 해제.

    - 이미 존재하면 스킵 (force=True 시 재다운로드).
    - 다운로드 중 .tmp 파일로 받고 완료 후 rename (원자적 교체).
    - 반환값: 압축 해제된 components.cif 경로.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "components.cif"

    if dest.exists() and not force:
        print(f"[ccd] {dest} already exists (use force=True to re-download)")
        return dest

    gz_tmp = dest_dir / "components.cif.gz.tmp"
    cif_tmp = dest_dir / "components.cif.tmp"

    print(f"[ccd] downloading {url} ...")
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        with gz_tmp.open("wb") as f:
            while chunk := resp.read(1 << 20):  # 1MB chunks
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {downloaded/1e6:.0f}/{total/1e6:.0f} MB ({pct:.0f}%)", end="", flush=True)
    print()

    print("[ccd] decompressing ...")
    with gzip.open(gz_tmp, "rb") as gz_in, cif_tmp.open("wb") as out:
        shutil.copyfileobj(gz_in, out)
    gz_tmp.unlink()
    cif_tmp.replace(dest)

    print(f"[ccd] saved to {dest}")
    return dest


def load_ccd_lookup(path: Path | str) -> dict[str, CCDEntry]:
    """
    components.cif 전체를 파싱해 {CCD코드 → CCDEntry} dict 반환.
    gemmi의 C++ 파서로 처리하므로 474MB 파일도 수초 내 완료.
    """
    doc = gemmi.cif.read(str(path))
    result: dict[str, CCDEntry] = {}

    for block in doc:
        ccd_code = _safe_str(block.find_value("_chem_comp.id")) or block.name
        chem_comp_type = _safe_str(block.find_value("_chem_comp.type"))
        formula_weight = _safe_float(block.find_value("_chem_comp.formula_weight"))
        formula = _safe_str(block.find_value("_chem_comp.formula"))

        # SMILES_CANONICAL from CACTVS (가장 표준적인 소스)
        smiles: str | None = None
        with contextlib.suppress(Exception):
            table = block.find(
                "_pdbx_chem_comp_descriptor.",
                ["type", "program", "descriptor"],
            )
            for row in table:
                if row[0].strip() == "SMILES_CANONICAL" and "CACTVS" in row[1]:
                    smiles = _safe_str(row[2])
                    break

        pdbx_type = _safe_str(block.find_value("_chem_comp.pdbx_type"))
        name = _safe_str(block.find_value("_chem_comp.name"))

        result[ccd_code.upper()] = CCDEntry(
            chem_comp_type=chem_comp_type,
            formula_weight=formula_weight,
            formula=formula,
            smiles=smiles,
            pdbx_type=pdbx_type,
            name=name,
        )

    return result
