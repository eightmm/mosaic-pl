#!/usr/bin/env python3
"""Lint a CASP17 LG-format submission file against ``docs/casp17_lg_format.md``.

**Authoritative format reference: ``docs/casp17_lg_format.md``** —
mirror of <https://predictioncenter.org/casp17/index.cgi?page=format>.
Every check below maps to a rule in that file; if the upstream spec
changes, update the doc first and then propagate the new check here.

Prints one diagnostic per problem and exits non-zero if any ERROR-level check
fails. WARNING-level issues are reported but do not fail the run.

Usage:
    uv run python scripts/lint_lg_submission.py experiments/CASP17/submissions/R2314_LCDD.lg
    uv run python scripts/lint_lg_submission.py experiments/CASP17/submissions/*.lg
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Issue:
    level: str          # "ERROR" or "WARN"
    code: str
    message: str
    where: str = ""     # e.g. "MODEL 2 / LIGAND 001"

    def render(self) -> str:
        head = f"[{self.level}] {self.code}"
        if self.where:
            head += f" ({self.where})"
        return f"{head}: {self.message}"


@dataclass
class Report:
    path: Path
    issues: list[Issue] = field(default_factory=list)
    n_models: int = 0
    n_ligands_per_model: list[int] = field(default_factory=list)

    def err(self, code: str, msg: str, where: str = "") -> None:
        self.issues.append(Issue("ERROR", code, msg, where))

    def warn(self, code: str, msg: str, where: str = "") -> None:
        self.issues.append(Issue("WARN", code, msg, where))

    @property
    def has_errors(self) -> bool:
        return any(i.level == "ERROR" for i in self.issues)


# ---------- header ----------------------------------------------------------

_AUTHOR_RE = re.compile(r"^AUTHOR\s+(\d{4}-\d{4}-\d{4})\s*$")


def _check_header(report: Report, lines: list[str]) -> tuple[str | None, str | None]:
    """Run header checks and return (target_id, author_code) for downstream
    cross-checks. Either may be None if missing/malformed."""
    target_id: str | None = None
    author_code: str | None = None
    if not lines or not lines[0].startswith("PFRMAT LG"):
        report.err("HEAD/PFRMAT", "first line must be 'PFRMAT LG'")
    if len(lines) < 2 or not lines[1].startswith("TARGET "):
        report.err("HEAD/TARGET", "second line must start with 'TARGET '")
    else:
        target_id = lines[1][len("TARGET "):].strip()
    if len(lines) < 3 or not lines[2].startswith("AUTHOR "):
        report.err("HEAD/AUTHOR", "third line must start with 'AUTHOR '")
    else:
        m = _AUTHOR_RE.match(lines[2])
        if not m:
            report.warn(
                "HEAD/AUTHOR_FORMAT",
                "AUTHOR should be a 12-digit registration code formatted as XXXX-XXXX-XXXX",
            )
        else:
            author_code = m.group(1)
        if "0000-0000-0000" in lines[2]:
            report.err(
                "HEAD/AUTHOR_PLACEHOLDER",
                "AUTHOR is the placeholder '0000-0000-0000'; set your real CASP "
                "registration code before submitting",
            )
    method_lines = [ln for ln in lines[:50] if ln.startswith("METHOD ")]
    if not method_lines:
        report.err("HEAD/METHOD", "at least one METHOD line is required before MODEL 1")
    return target_id, author_code


# ---------- MODEL blocks ----------------------------------------------------

_LIGAND_RE = re.compile(r"^LIGAND\s+(\d{1,3})\s+(\S+)\s*$")
_LSCORE_RE = re.compile(r"^LSCORE\s+([+-]?\d+(?:\.\d+)?)\s*$")
_AFFNTY_RE = re.compile(r"^AFFNTY\s+([+-]?\d+(?:\.\d+)?)\s+(aa|ra|lr)\s*$")
_COUNTS_V2000_RE = re.compile(r"^.{33}\s*V2000\s*$")
_COUNTS_V3000_RE = re.compile(r"^.{33}\s*V3000\s*$")


def _parse_models(lines: list[str]) -> list[tuple[int, int, list[str]]]:
    """Return (model_idx, model_number, body_lines) for each MODEL block."""
    blocks: list[tuple[int, int, list[str]]] = []
    cur_num: int | None = None
    cur_body: list[str] = []
    for i, line in enumerate(lines):
        if line.startswith("MODEL"):
            parts = line.split()
            if len(parts) < 2 or not parts[1].isdigit():
                continue
            if cur_num is not None:
                blocks.append((len(blocks) + 1, cur_num, cur_body))
            cur_num = int(parts[1])
            cur_body = []
        elif line.strip() == "END" and cur_num is not None:
            blocks.append((len(blocks) + 1, cur_num, cur_body))
            cur_num = None
            cur_body = []
        elif cur_num is not None:
            cur_body.append(line)
    return blocks


def _check_model(report: Report, midx: int, mnum: int, body: list[str]) -> int:
    """Validate one MODEL body. Returns ligand count for cross-MODEL audit."""
    where_model = f"MODEL {mnum}"

    # PARENT must appear once.
    parents = [ln for ln in body if ln.startswith("PARENT ")]
    if not parents:
        report.err("MODEL/PARENT_MISSING", "PARENT line is required", where_model)
    elif len(parents) > 1:
        report.warn("MODEL/PARENT_MULTI", f"{len(parents)} PARENT lines (expected 1)", where_model)

    # No ENDMDL anywhere; we already terminate per MODEL with END.
    if any(ln.strip() == "ENDMDL" for ln in body):
        report.err("MODEL/ENDMDL", "ENDMDL is not allowed (use END only)", where_model)

    # Ligand sub-blocks: walk through, picking up LIGAND / LSCORE / MDL body / M END.
    ligand_numbers: list[int] = []
    in_ligand = False
    in_mdl = False
    mdl_buffer: list[str] = []
    saw_lscore_in_current_ligand = False
    cur_lig_n: int | None = None
    cur_lig_name: str | None = None

    def _flush_ligand() -> None:
        nonlocal in_ligand, in_mdl, mdl_buffer, saw_lscore_in_current_ligand
        nonlocal cur_lig_n, cur_lig_name
        where_lig = f"{where_model} / LIGAND {cur_lig_n:03d} {cur_lig_name}"
        if not mdl_buffer:
            report.err("LIG/MDL_EMPTY", "ligand has no MDL body", where_lig)
        else:
            _check_mdl_body(report, mdl_buffer, where_lig)
        if not saw_lscore_in_current_ligand:
            report.warn(
                "LIG/LSCORE_MISSING",
                "LSCORE not provided (allowed but recommended)",
                where_lig,
            )
        in_ligand = False
        in_mdl = False
        mdl_buffer = []
        saw_lscore_in_current_ligand = False
        cur_lig_n = None
        cur_lig_name = None

    for ln in body:
        if ln.startswith("LIGAND"):
            if in_ligand:
                _flush_ligand()
            m = _LIGAND_RE.match(ln)
            if not m:
                report.err(
                    "LIG/HEADER",
                    f"malformed LIGAND header {ln!r} "
                    f"(expected 'LIGAND <NNN> <NAME>')",
                    where_model,
                )
                continue
            cur_lig_n = int(m.group(1))
            cur_lig_name = m.group(2)
            ligand_numbers.append(cur_lig_n)
            in_ligand = True
            in_mdl = False
            continue

        if not in_ligand:
            continue

        if ln.startswith("LSCORE"):
            m = _LSCORE_RE.match(ln)
            if not m:
                report.err("LIG/LSCORE_FORMAT", f"malformed LSCORE: {ln!r}",
                           f"{where_model} / LIGAND {cur_lig_n:03d}")
            else:
                v = float(m.group(1))
                if not (0.0 <= v <= 1.0):
                    report.err(
                        "LIG/LSCORE_RANGE",
                        f"LSCORE {v} outside [0, 1]",
                        f"{where_model} / LIGAND {cur_lig_n:03d}",
                    )
                saw_lscore_in_current_ligand = True
            continue

        if ln.strip() == "M  END":
            mdl_buffer.append(ln)
            _flush_ligand()
            continue

        if ln.startswith("M END"):
            report.err(
                "LIG/M_END_SPACING",
                "MDL terminator must be 'M  END' (two spaces between M and END)",
                f"{where_model} / LIGAND {cur_lig_n:03d}",
            )
            mdl_buffer.append(ln)
            _flush_ligand()
            continue

        # Anything else inside a ligand block is the MDL body
        mdl_buffer.append(ln)
        in_mdl = True

    if in_ligand:
        report.err(
            "LIG/UNTERMINATED",
            "LIGAND block did not terminate with 'M  END'",
            where_model,
        )

    # AFFNTY: at most one per MODEL, after last LIGAND, before END.
    affnty_lines = [ln for ln in body if ln.startswith("AFFNTY")]
    if len(affnty_lines) > 1:
        report.err("MODEL/AFFNTY_MULTI", f"{len(affnty_lines)} AFFNTY lines (max 1)", where_model)
    for ln in affnty_lines:
        if not _AFFNTY_RE.match(ln):
            report.err("MODEL/AFFNTY_FORMAT", f"malformed AFFNTY: {ln!r}", where_model)

    return len(ligand_numbers)


def _check_mdl_body(report: Report, mdl_lines: list[str], where: str) -> None:
    """Validate an MDL V2000 body. The buffer should include the trailing
    'M  END' (we trim before counting).

    The V2000 counts line is fixed-width — atoms and bonds occupy three
    characters each with no separator (Symyx CTfile spec):

        AAABBBLLLFFFCCCSSSxxxRRRPPPIIIMMMVVVVVV

    so a 142-atom / 141-bond ligand renders as ``142141  0  0…`` (no
    space between AAA and BBB). Earlier versions of the linter required
    whitespace between the two fields and produced false positives on
    every ligand with 100+ atoms or bonds. We now slice by column.

    V3000 counts lines are detected separately and rejected — the CASP
    LG validator only accepts V2000.
    """
    body = [ln for ln in mdl_lines if ln.strip() != "M  END"]
    # Header block: 3 lines (title, program/timestamp, blank), then counts line.
    if len(body) < 4:
        report.err("MDL/TRUNCATED", "MDL body shorter than 4 lines", where)
        return
    counts_line = body[3]
    if _COUNTS_V3000_RE.match(counts_line):
        report.err(
            "MDL/V3000",
            "MDL counts line is V3000 — CASP LG accepts V2000 only "
            "(this happens when RDKit auto-falls-back to V3000 because "
            "the ligand has more than 999 atoms or bonds; the builder "
            "needs a V2000-fitting workaround for that ligand)",
            where,
        )
        return
    if not _COUNTS_V2000_RE.match(counts_line):
        report.err(
            "MDL/COUNTS_FORMAT",
            f"counts line does not match V2000 grammar: {counts_line!r}",
            where,
        )
        return
    # Fixed-width parse: atoms in cols 1-3, bonds in cols 4-6 (1-indexed).
    try:
        n_atoms = int(counts_line[0:3].strip() or "0")
        n_bonds = int(counts_line[3:6].strip() or "0")
    except ValueError:
        report.err(
            "MDL/COUNTS_FORMAT",
            f"could not parse atom/bond counts in {counts_line!r}",
            where,
        )
        return
    expected = 4 + n_atoms + n_bonds
    if len(body) < expected:
        report.err(
            "MDL/SIZE",
            f"counts say {n_atoms} atoms + {n_bonds} bonds "
            f"(expected {expected} non-M-END lines), got {len(body)}",
            where,
        )


# ---------- cross-MODEL checks ----------------------------------------------

_SINGLE_MODEL_FILENAME_RE = re.compile(r"_([1-5])\.lg$", re.I)

#: MODEL label groups for a two-conformation target (T2451). Group 2 ends in 0,
#: so the sequence is not monotonic and cannot be checked with ``range()``.
_CONFORMATION_LABELS = ((1, 2, 3, 4, 5), (6, 7, 8, 9, 0))


def _numbering_ok(nums: list[int], allow_conformations: bool = False) -> bool:
    """True when the MODEL numbers are a layout CASP accepts.

    Default is the plain ``1..n`` with n <= 5. ``allow_conformations`` also
    accepts conformation 1 as a prefix of ``1,2,3,4,5`` followed by
    conformation 2 as a prefix of ``6,7,8,9,0``.

    The second layout is opt-in on purpose: ``1,2,3,4,5,6`` is ambiguous — it
    is either a one-MODEL conformation-2 group or a plain overflow past the
    5-MODEL cap, and only the target page settles which. Defaulting to strict
    keeps every ordinary target's overflow an error.
    """
    if len(nums) <= 5 and nums == list(range(1, len(nums) + 1)):
        return True
    if not allow_conformations:
        return False
    g1, g2 = _CONFORMATION_LABELS
    for split in range(1, min(len(nums), len(g1)) + 1):
        head, tail = nums[:split], nums[split:]
        if head == list(g1[:split]) and tail == list(g2[:len(tail)]):
            return True
    return False


def _check_cross_models(
    report: Report,
    blocks: list[tuple[int, int, list[str]]],
    file_path: Path | None = None,
    allow_conformations: bool = False,
) -> None:
    if not blocks:
        report.err("MODEL/NONE", "no MODEL blocks found")
        return

    # Submission files: when the filename ends with ``_<n>.lg`` (n∈1..5)
    # the file is the per-rank submission file the CASP server validator
    # accepts. Those must contain exactly one MODEL block, matching the
    # rank in the filename. Multi-MODEL files crash the validator (verified
    # 2026-05-06; see ``docs/casp17_lg_format.md`` §1).
    expect_single = False
    expected_rank: int | None = None
    if file_path is not None:
        m = _SINGLE_MODEL_FILENAME_RE.search(file_path.name)
        if m:
            expect_single = True
            expected_rank = int(m.group(1))

    if expect_single:
        if len(blocks) != 1:
            report.err(
                "MODEL/SINGLE_FILE",
                f"submission file {file_path.name} must contain exactly one "
                f"MODEL block (got {len(blocks)})",
            )
        elif expected_rank is not None and blocks[0][1] != expected_rank:
            report.err(
                "MODEL/RANK_MISMATCH",
                f"filename rank {expected_rank} does not match "
                f"MODEL number {blocks[0][1]}",
            )
    else:
        # Review file: normally 1..n with n <= 5. A target whose CASP page asks
        # for two crystal conformations gets a second group — T2451: "submit
        # models for conformation 1 as models 1-5, and those for conformation 2
        # as 6,7,8,9,0". That layout is the only accepted way past 5 MODELs.
        nums = [n for _, n, _ in blocks]
        if not _numbering_ok(nums, allow_conformations):
            if len(blocks) > 5:
                report.err(
                    "MODEL/COUNT",
                    f"{len(blocks)} MODEL blocks numbered {nums} — over the "
                    "5-MODEL cap and not the conformation layout "
                    "(1..5 then 6,7,8,9,0)",
                )
            else:
                report.warn("MODEL/NUMBERING",
                            f"MODEL numbers are {nums} (expected 1..{len(nums)})")

    # Each MODEL must contain the same set of ligand_number values.
    ligand_sets = []
    for midx, mnum, body in blocks:
        ligand_sets.append(
            tuple(sorted({int(m.group(1)) for ln in body
                          for m in [_LIGAND_RE.match(ln)] if m}))
        )
    if len(set(ligand_sets)) > 1:
        report.err(
            "MODEL/LIGANDS_INCONSISTENT",
            f"each MODEL must list every ligand of the target; "
            f"got {ligand_sets}",
        )


# ---------- receptor-specific checks ----------------------------------------

_RNA_RES = {"A", "U", "G", "C", "DA", "DT", "DG", "DC", "DU"}


def _check_receptor(
    report: Report,
    blocks: list[tuple[int, int, list[str]]],
    target_id: str | None,
) -> None:
    """Look at MODEL 1's receptor block. Verify B-factor variation and check
    that residue type matches the TARGET prefix convention (R*=RNA, T/H/M=
    protein/hybrid)."""
    if not blocks:
        return
    _, _, body = blocks[0]
    atoms = [ln for ln in body if ln.startswith(("ATOM", "HETATM"))]
    if not atoms:
        report.err("REC/EMPTY", "MODEL 1 has no receptor ATOM/HETATM lines")
        return

    bs: list[float] = []
    res_names: set[str] = set()
    for ln in atoms[:1000]:  # cap for speed; PDB B-factor at columns 61-66
        try:
            bs.append(float(ln[60:66]))
        except (ValueError, IndexError):
            pass
        rn = ln[17:20].strip()
        if rn:
            res_names.add(rn)
    if bs and len({round(b, 1) for b in bs}) <= 2:
        report.err(
            "REC/B_FACTOR_FLAT",
            f"B-factor distribution is nearly uniform "
            f"({sorted({round(b, 1) for b in bs})}); "
            "CASP rejects flat B-factors. The cofolding pLDDT should populate "
            "this column.",
        )
    is_rna_only = bool(res_names) and all(rn in _RNA_RES for rn in res_names)
    target_is_rna = bool(target_id) and target_id.startswith("R")
    if target_is_rna and not is_rna_only:
        report.err(
            "REC/RECEPTOR_TYPE",
            f"TARGET {target_id} is RNA (R-prefix) but receptor residues "
            f"include non-RNA names {sorted(res_names - _RNA_RES)}",
        )
    elif not target_is_rna and is_rna_only:
        report.err(
            "REC/RECEPTOR_TYPE",
            f"receptor is RNA-only ({sorted(res_names)}) but TARGET id "
            f"{target_id!r} does not look like an RNA target",
        )


# ---------- driver ----------------------------------------------------------

def lint(path: Path, conformations: int = 1) -> Report:
    report = Report(path=path)
    text = path.read_text()
    if "ENDMDL" in text:
        report.err("FILE/ENDMDL", "file contains 'ENDMDL' (LG format uses END only)")
    lines = text.splitlines()
    target_id, _author = _check_header(report, lines)
    blocks = _parse_models(lines)
    report.n_models = len(blocks)
    for midx, mnum, body in blocks:
        n_lig = _check_model(report, midx, mnum, body)
        report.n_ligands_per_model.append(n_lig)
    _check_cross_models(report, blocks, file_path=path,
                        allow_conformations=conformations >= 2)
    _check_receptor(report, blocks, target_id)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="LG file(s) to lint")
    parser.add_argument("--strict", action="store_true",
                        help="Treat WARN as ERROR (exit non-zero on any issue)")
    parser.add_argument("--conformations", type=int, default=1, choices=[1, 2],
                        help="2 accepts the multi-conformation MODEL layout "
                             "(1-5 then 6,7,8,9,0) that a target page can request, "
                             "e.g. T2451. Default 1 keeps the strict 5-MODEL cap.")
    args = parser.parse_args()

    n_errors = 0
    n_warns = 0
    for p in args.paths:
        if not p.exists():
            print(f"[ERROR] {p}: file not found", file=sys.stderr)
            n_errors += 1
            continue
        report = lint(p, conformations=args.conformations)
        print(f"\n=== {p} ===")
        print(f"  MODELs: {report.n_models}")
        print(f"  ligands per MODEL: {report.n_ligands_per_model}")
        for issue in report.issues:
            print(f"  {issue.render()}")
            if issue.level == "ERROR":
                n_errors += 1
            elif issue.level == "WARN":
                n_warns += 1
        if not report.issues:
            print("  (no issues)")
    print(f"\nSummary: {n_errors} ERROR(s), {n_warns} WARN(s) across {len(args.paths)} file(s).")
    if args.strict:
        return 1 if (n_errors + n_warns) else 0
    return 1 if n_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
