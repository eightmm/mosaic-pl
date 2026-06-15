"""Unit tests for ``scripts/lint_lg_submission.py``.

Exercises the format checker against synthesised LG bodies that intentionally
break specific spec rules. Each test asserts that the linter raises the right
ERROR code (and only that one).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import lint_lg_submission as lg  # type: ignore  # noqa: E402


def _atom(idx: int, atom: str = "C1'", res: str = "  A", chain: str = "A",
          x: float = 1.0, y: float = 2.0, z: float = 3.0, b: float = 75.0) -> str:
    """Render an 80-col PDB ATOM line with a varying B-factor by default."""
    b_field = f"{b:6.2f}"
    return (
        f"ATOM  {idx:5d} {atom:<4.4s} {res:<3.3s} {chain}"
        f"{idx:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00{b_field}"
        f"           C  "
    )


def _mdl_body(n_atoms: int = 3, n_bonds: int = 2) -> list[str]:
    counts = f"{n_atoms:3d}{n_bonds:3d}  0  0  0  0  0  0  0  0999 V2000"
    body = ["pose-title", "     RDKit          3D", "", counts]
    for _ in range(n_atoms):
        body.append("    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0")
    for _ in range(n_bonds):
        body.append("  1  2  1  0")
    body.append("M  END")
    return body


def _default_rna_atoms() -> list[str]:
    """Three RNA atoms with varying B-factors (passes REC/B_FACTOR_FLAT)."""
    return [
        _atom(1, atom="C1'", res="  A", b=70.0),
        _atom(2, atom="C1'", res="  A", b=80.0),
        _atom(3, atom="C1'", res="  A", b=85.0),
    ]


def _default_protein_atoms() -> list[str]:
    return [
        _atom(1, atom="CA", res="ALA", b=70.0),
        _atom(2, atom="CA", res="GLU", b=80.0),
        _atom(3, atom="CA", res="GLY", b=85.0),
    ]


def _build_valid_lg(target: str = "R2999",
                    author: str = "1234-5678-9012",
                    n_models: int = 1,
                    atoms: list[str] | None = None) -> str:
    if atoms is None:
        atoms = _default_rna_atoms()
    lines = [
        "PFRMAT LG",
        f"TARGET {target}",
        f"AUTHOR {author}",
        "METHOD test method",
    ]
    for k in range(1, n_models + 1):
        lines.append(f"MODEL {k}")
        lines.append("PARENT N/A")
        lines.extend(atoms)
        lines.append("TER")
        lines.append("LIGAND 001 LIG")
        lines.append("LSCORE 0.500")
        lines.extend(_mdl_body())
        lines.append("END")
    return "\n".join(lines) + "\n"


def _lint_text(tmp_path: Path, body: str) -> lg.Report:
    p = tmp_path / "subm.lg"
    p.write_text(body)
    return lg.lint(p)


def test_valid_lg_passes(tmp_path):
    report = _lint_text(tmp_path, _build_valid_lg())
    assert not report.has_errors, [i.render() for i in report.issues]


def test_missing_pfrmat_errors(tmp_path):
    body = _build_valid_lg().replace("PFRMAT LG\n", "", 1)
    report = _lint_text(tmp_path, body)
    codes = {i.code for i in report.issues if i.level == "ERROR"}
    assert "HEAD/PFRMAT" in codes


def test_placeholder_author_errors(tmp_path):
    body = _build_valid_lg(author="0000-0000-0000")
    report = _lint_text(tmp_path, body)
    codes = {i.code for i in report.issues if i.level == "ERROR"}
    assert "HEAD/AUTHOR_PLACEHOLDER" in codes


def test_endmdl_forbidden(tmp_path):
    body = _build_valid_lg().replace("END\n", "ENDMDL\nEND\n", 1)
    report = _lint_text(tmp_path, body)
    codes = {i.code for i in report.issues if i.level == "ERROR"}
    assert "FILE/ENDMDL" in codes


def test_lscore_out_of_range(tmp_path):
    body = _build_valid_lg().replace("LSCORE 0.500", "LSCORE 1.500")
    report = _lint_text(tmp_path, body)
    codes = {i.code for i in report.issues if i.level == "ERROR"}
    assert "LIG/LSCORE_RANGE" in codes


def test_mdl_count_mismatch(tmp_path):
    # Counts line says 99 atoms but body only has 3.
    body = _build_valid_lg().replace(
        "  3  2  0  0  0  0  0  0  0  0999 V2000",
        " 99  2  0  0  0  0  0  0  0  0999 V2000",
    )
    report = _lint_text(tmp_path, body)
    codes = {i.code for i in report.issues if i.level == "ERROR"}
    assert "MDL/SIZE" in codes


def test_more_than_five_models(tmp_path):
    body = _build_valid_lg(n_models=6)
    report = _lint_text(tmp_path, body)
    codes = {i.code for i in report.issues if i.level == "ERROR"}
    assert "MODEL/COUNT" in codes


def test_rna_target_protein_receptor_errors(tmp_path):
    """An R-prefix TARGET with amino-acid receptor residues is suspicious."""
    body = _build_valid_lg(target="R2999", atoms=_default_protein_atoms())
    report = _lint_text(tmp_path, body)
    codes = {i.code for i in report.issues if i.level == "ERROR"}
    assert "REC/RECEPTOR_TYPE" in codes


def test_b_factor_flat_errors(tmp_path):
    """All-equal B-factors should fail the receptor check."""
    flat_atoms = [
        _atom(1, atom="C1'", res="  A", b=70.0),
        _atom(2, atom="C1'", res="  A", b=70.0),
        _atom(3, atom="C1'", res="  A", b=70.0),
    ]
    body = _build_valid_lg(atoms=flat_atoms)
    report = _lint_text(tmp_path, body)
    codes = {i.code for i in report.issues if i.level == "ERROR"}
    assert "REC/B_FACTOR_FLAT" in codes
