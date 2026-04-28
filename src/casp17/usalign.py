"""USalign wrapper — structure-based protein superposition.

Used wherever we need a transform aligning one protein structure onto another
without relying on sequence identity. Pocket extraction is the primary
caller: foldseek finds templates by 3Di + structural similarity, so the
matching alignment must also be structure-based; gemmi's sequence-anchored
``calculate_superposition`` collapses on distant homologs and produces
unreliable transforms for those hits.

The binary lives under ``.local/bin/USalign`` (installed via the external
models bootstrap script). USalign auto-detects mmCIF vs PDB from file
extension, so callers can pass either.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
USALIGN_BIN = REPO_ROOT / ".local" / "bin" / "USalign"


def run_usalign(
    pred_path: Path,
    ref_path: Path,
    *,
    timeout: int = 120,
    binary: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """Run USalign and return ``(R, t, tm_score, rmsd_after_align)``.

    The matrix transforms ``pred`` coordinates onto ``ref``:
    ``ref ≈ R @ pred + t``. TM-score is the reference-normalized value
    (USalign's canonical output, normalized by Structure_2). RMSD is
    reported on the aligned positions only (not structure-wide).

    Args:
        pred_path: structure to be moved (mmCIF or PDB; USalign auto-detects).
        ref_path: reference structure (mmCIF or PDB).
        timeout: subprocess timeout in seconds.
        binary: override the USalign binary path. Defaults to ``USALIGN_BIN``.

    Returns:
        ``(R, t, tm, rmsd)`` on success, ``None`` on USalign failure or
        unparseable output. Both R and t are numpy arrays sized 3x3 / 3.
    """
    bin_path = binary or USALIGN_BIN
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mat", delete=False) as matf:
        mat_path = Path(matf.name)
    try:
        res = subprocess.run(
            [str(bin_path), str(pred_path), str(ref_path),
             "-m", str(mat_path), "-ter", "1"],
            capture_output=True, text=True, timeout=timeout,
        )
        if res.returncode != 0:
            return None
        text = mat_path.read_text() if mat_path.exists() else ""
        R = np.zeros((3, 3))
        t = np.zeros(3)
        parsed = False
        for line in text.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[0] in ("0", "1", "2"):
                i = int(parts[0])
                t[i] = float(parts[1])
                R[i] = [float(parts[2]), float(parts[3]), float(parts[4])]
                parsed = True
        if not parsed:
            return None
        tm: float | None = None
        rmsd: float | None = None
        # USalign emits two TM-score lines; the reference-normalized one
        # (Structure_2) is the canonical value. Fall back to any TM-score
        # line in case the upstream wording shifts.
        for line in res.stdout.splitlines():
            if tm is None and "TM-score=" in line and "Structure_2" in line:
                m = re.search(r"TM-score=\s*([\d.]+)", line)
                if m:
                    tm = float(m.group(1))
            if rmsd is None and "RMSD=" in line:
                m = re.search(r"RMSD=\s*([\d.]+)", line)
                if m:
                    rmsd = float(m.group(1))
        if tm is None:
            for line in res.stdout.splitlines():
                if "TM-score=" in line:
                    m = re.search(r"TM-score=\s*([\d.]+)", line)
                    if m:
                        tm = float(m.group(1))
                        break
        return R, t, (tm or 0.0), (rmsd or 0.0)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    finally:
        try:
            mat_path.unlink()
        except OSError:
            pass


def transform_point(R: np.ndarray, t: np.ndarray, x: float, y: float, z: float) -> tuple[float, float, float]:
    """Apply USalign's ``ref ≈ R @ pred + t`` to a single 3D point."""
    p = np.asarray([x, y, z], dtype=float)
    out = R @ p + t
    return float(out[0]), float(out[1]), float(out[2])
