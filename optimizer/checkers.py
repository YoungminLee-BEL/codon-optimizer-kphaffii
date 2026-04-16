"""
Sequence quality checks for K. phaffii codon-optimized sequences.
Each check returns: {"name": str, "passed": bool, "details": str, "positions": list[int]}
"""

import re
import math
from .codon_table import CODON_USAGE, GENETIC_CODE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reverse_complement(seq: str) -> str:
    comp = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(comp.get(b, b) for b in reversed(seq.upper()))


def _gc_content(seq: str) -> float:
    if not seq:
        return 0.0
    gc = sum(1 for b in seq.upper() if b in 'GC')
    return gc / len(seq)


# ---------------------------------------------------------------------------
# Group 1 — ORF integrity & expression signals
# ---------------------------------------------------------------------------

def orf_integrity(sequence: str) -> dict:
    seq = sequence.upper()
    issues = []
    positions = []

    if not seq.startswith('ATG'):
        issues.append("Sequence does not start with ATG")
        positions.append(0)

    if len(seq) % 3 != 0:
        issues.append(f"Sequence length {len(seq)} is not divisible by 3")

    # Check internal stops (all codons except the last)
    codons = [seq[i:i+3] for i in range(0, len(seq) - 2, 3)]
    stop_codons = {'TAA', 'TAG', 'TGA'}
    for i, codon in enumerate(codons[:-1]):
        if codon in stop_codons:
            pos = i * 3
            issues.append(f"Internal stop codon {codon} at position {pos}")
            positions.append(pos)

    passed = len(issues) == 0
    details = "ORF is valid" if passed else "; ".join(issues)
    return {"name": "orf_integrity", "passed": passed, "details": details, "positions": positions}


def internal_initiation(sequence: str) -> dict:
    seq = sequence.upper()
    rc = _reverse_complement(seq)
    rbs_patterns = ['GGAGG', 'TAAGGAG']
    positions = []

    for strand_seq, strand_label in [(seq, '+'), (rc, '-')]:
        atg_positions = [m.start() for m in re.finditer('ATG', strand_seq)]
        for atg_pos in atg_positions:
            if atg_pos == 0 and strand_label == '+':
                continue  # Skip the start codon on forward strand
            # Check 5–15 bp upstream
            window_start = max(0, atg_pos - 15)
            window_end = max(0, atg_pos - 5)
            upstream = strand_seq[window_start:window_end]
            for rbs in rbs_patterns:
                if rbs in upstream:
                    real_pos = atg_pos if strand_label == '+' else len(seq) - atg_pos - 3
                    positions.append(real_pos)
                    break

    passed = len(positions) == 0
    details = (
        "No internal RBS-ATG pairs found"
        if passed
        else f"Potential internal initiation sites at positions: {positions}"
    )
    return {"name": "internal_initiation", "passed": passed, "details": details, "positions": positions}


def cryptic_promoter(sequence: str) -> dict:
    seq = sequence.upper()
    rc = _reverse_complement(seq)
    positions = []

    minus35 = 'TTGACA'
    minus10 = 'TATAAT'

    for strand_seq in [seq, rc]:
        m35_positions = [m.start() for m in re.finditer(minus35, strand_seq)]
        m10_positions = [m.start() for m in re.finditer(minus10, strand_seq)]
        for p35 in m35_positions:
            for p10 in m10_positions:
                gap = p10 - (p35 + len(minus35))
                if 15 <= gap <= 20:
                    positions.append(p35)

    passed = len(positions) == 0
    details = (
        "No cryptic prokaryotic promoter elements found"
        if passed
        else f"Potential −35/−10 promoter pairs at positions: {positions}"
    )
    return {"name": "cryptic_promoter", "passed": passed, "details": details, "positions": positions}


def strong_rbs(sequence: str) -> dict:
    seq = sequence.upper()
    rc = _reverse_complement(seq)
    rbs_patterns = ['GGAGG', 'TAAGGAG']
    positions = []

    for strand_seq, strand_label in [(seq, '+'), (rc, '-')]:
        for rbs in rbs_patterns:
            for m in re.finditer(rbs, strand_seq):
                real_pos = m.start() if strand_label == '+' else len(seq) - m.start() - len(rbs)
                positions.append(real_pos)

    passed = len(positions) == 0
    details = (
        "No strong RBS sequences found"
        if passed
        else f"Potential RBS sequences at positions: {sorted(set(positions))}"
    )
    return {"name": "strong_rbs", "passed": passed, "details": details,
            "positions": sorted(set(positions))}


def terminator_sequences(sequence: str) -> dict:
    seq = sequence.upper()
    positions = []

    for base in 'TA':
        run = base * 5
        for m in re.finditer(run + '+', seq):
            positions.append(m.start())

    passed = len(positions) == 0
    details = (
        "No terminator-like poly-A/T runs found"
        if passed
        else f"Poly-A/T runs (≥5) at positions: {positions}"
    )
    return {"name": "terminator_sequences", "passed": passed, "details": details,
            "positions": positions}


# ---------------------------------------------------------------------------
# Group 2 — Twist Bioscience manufacturability
# ---------------------------------------------------------------------------

def repeat_sequences(sequence: str) -> dict:
    seq = sequence.upper()
    n = len(seq)
    min_repeat = 21
    max_repeat = 50  # cap to keep O(n) per length, avoid O(n²) blow-up
    found: set[int] = set()

    for length in range(min_repeat, min(max_repeat + 1, n // 2 + 1)):
        seen: dict[str, int] = {}
        for i in range(n - length + 1):
            subseq = seq[i:i + length]
            if subseq in seen:
                found.add(seen[subseq])
                found.add(i)
            else:
                seen[subseq] = i

    positions = sorted(found)
    passed = len(positions) == 0
    details = (
        "No repeated subsequences (21–50 bp) found"
        if passed
        else f"Repeated sequences (21–50 bp) at positions: {positions[:10]}"
              + (" ..." if len(positions) > 10 else "")
    )
    return {"name": "repeat_sequences", "passed": passed, "details": details,
            "positions": positions}


def homopolymer_runs(sequence: str) -> dict:
    seq = sequence.upper()
    positions = []

    for m in re.finditer(r'([ACGT])\1{9,}', seq):
        positions.append(m.start())

    passed = len(positions) == 0
    details = (
        "No homopolymer runs ≥ 10 bp found"
        if passed
        else f"Homopolymer runs (≥10) at positions: {positions}"
    )
    return {"name": "homopolymer_runs", "passed": passed, "details": details,
            "positions": positions}


def global_gc(sequence: str) -> dict:
    seq = sequence.upper()
    gc = _gc_content(seq)
    pct = gc * 100
    passed = 25.0 <= pct <= 65.0
    details = f"Global GC content: {pct:.1f}% ({'OK' if passed else 'out of 25–65% range'})"
    return {"name": "global_gc", "passed": passed, "details": details, "positions": []}


def local_gc_windows(sequence: str) -> dict:
    seq = sequence.upper()
    window = 50
    positions = []

    for i in range(len(seq) - window + 1):
        w = seq[i:i + window]
        gc = _gc_content(w) * 100
        if gc < 35.0 or gc > 65.0:
            positions.append(i)

    passed = len(positions) == 0
    details = (
        "All 50 bp windows have GC% between 35–65%"
        if passed
        else f"{len(positions)} windows with GC% outside 35–65% range"
    )
    return {"name": "local_gc_windows", "passed": passed, "details": details,
            "positions": positions[:50]}


# ---------------------------------------------------------------------------
# Group 3 — Expression quality
# ---------------------------------------------------------------------------

def rare_codons(sequence: str) -> dict:
    seq = sequence.upper()
    threshold = 0.08
    positions = []
    rare = []

    codons = [seq[i:i+3] for i in range(0, len(seq) - 2, 3)]
    for i, codon in enumerate(codons):
        freq = CODON_USAGE.get(codon, 0.0)
        if freq < threshold and GENETIC_CODE.get(codon, '*') != '*':
            pos = i * 3
            aa = GENETIC_CODE.get(codon, '?')
            positions.append(pos)
            rare.append(f"{codon}({aa}) at pos {pos} [{freq:.2f}]")

    passed = len(positions) == 0
    details = (
        f"No rare codons (< {threshold} frequency) found"
        if passed
        else f"Rare codons: {', '.join(rare[:10])}" + (" ..." if len(rare) > 10 else "")
    )
    return {"name": "rare_codons", "passed": passed, "details": details, "positions": positions}


# ---------------------------------------------------------------------------
# 5' Structure — fast heuristic (GC% + inverted repeat detection, no ΔG)
# ---------------------------------------------------------------------------

_RC_MAP = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}


def _has_inverted_repeat(
    seq: str, min_arm: int = 8, min_loop: int = 3, max_loop: int = 10
) -> tuple[bool, str]:
    """
    Scan for a perfect inverted repeat (potential stem-loop) in seq.
    Returns (found, description_string).
    """
    n = len(seq)
    for arm_len in range(min_arm, n // 2 + 1):
        for loop_len in range(min_loop, max_loop + 1):
            total = 2 * arm_len + loop_len
            if total > n:
                break
            for i in range(n - total + 1):
                arm5 = seq[i:i + arm_len]
                j = i + arm_len + loop_len
                arm3 = seq[j:j + arm_len]
                rc_arm5 = ''.join(_RC_MAP.get(b, 'N') for b in reversed(arm5))
                if arm3 == rc_arm5:
                    loop = seq[i + arm_len:j]
                    return True, (
                        f"Inverted repeat: 5' arm={arm5}, 3' arm={arm3} "
                        f"(len={arm_len}), loop={loop} (len={loop_len}) at pos {i}"
                    )
    return False, ""


def fiveprime_structure(sequence: str) -> dict:
    """
    Heuristic 5' structure check on the first 48 bp:
      1. GC% must be in 30–70%.
      2. No perfect inverted repeat with arm ≥ 8 bp and loop 3–10 nt.
    """
    seq = sequence.upper()
    region = seq[:48]
    issues = []
    positions = []

    gc = _gc_content(region) * 100
    if gc < 30.0 or gc > 70.0:
        issues.append(f"5' GC% = {gc:.1f}% (outside 30–70%)")
        positions.append(0)

    found, desc = _has_inverted_repeat(region)
    if found:
        issues.append(desc)
        positions.append(0)

    passed = len(issues) == 0
    details = (
        "5' region (48 bp): GC% in range and no inverted repeats detected"
        if passed
        else "; ".join(issues)
    )
    return {"name": "fiveprime_structure", "passed": passed, "details": details,
            "positions": positions}


# ---------------------------------------------------------------------------
# Run all checks
# ---------------------------------------------------------------------------

ALL_CHECKS = [
    orf_integrity,
    internal_initiation,
    cryptic_promoter,
    strong_rbs,
    terminator_sequences,
    repeat_sequences,
    homopolymer_runs,
    global_gc,
    local_gc_windows,
    rare_codons,
    fiveprime_structure,
]

CHECK_GROUPS = {
    "ORF & Expression Signals": [
        "orf_integrity",
        "internal_initiation",
        "cryptic_promoter",
        "strong_rbs",
        "terminator_sequences",
    ],
    "Manufacturability": [
        "repeat_sequences",
        "homopolymer_runs",
        "global_gc",
        "local_gc_windows",
    ],
    "Expression Quality": [
        "rare_codons",
        "fiveprime_structure",
    ],
}


def run_all_checks(sequence: str, options: dict | None = None) -> list[dict]:
    """Run all quality checks on a sequence. Returns list of check result dicts."""
    if options is None:
        options = {}

    results = []
    for check_fn in ALL_CHECKS:
        name = check_fn.__name__
        if name == "repeat_sequences" and not options.get("check_repeats", True):
            continue
        try:
            results.append(check_fn(sequence))
        except Exception as e:
            results.append({
                "name": name,
                "passed": False,
                "details": f"Check failed with error: {e}",
                "positions": [],
            })
    return results
