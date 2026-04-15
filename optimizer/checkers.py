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
    positions = []
    min_repeat = 20

    seen = {}
    for length in range(min_repeat, n // 2 + 1):
        for i in range(n - length + 1):
            subseq = seq[i:i + length]
            if subseq in seen:
                first_pos = seen[subseq]
                if i not in positions:
                    positions.append(first_pos)
                    positions.append(i)
            else:
                seen[subseq] = i

    positions = sorted(set(positions))
    passed = len(positions) == 0
    details = (
        "No repeated subsequences > 20 bp found"
        if passed
        else f"Repeated sequences (>20 bp) at positions: {positions[:10]}"
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
# 5' Hairpin — pure-Python nearest-neighbor ΔG estimation
# ---------------------------------------------------------------------------

# Simplified RNA Turner 1999 nearest-neighbor stacking energies (kcal/mol)
# Key: 5'XY3' / 3'X'Y'5' → stacking energy
# Format: "XY" where X and Y are 5'→3' on the sense strand, complement implicit
_RNA_NN_DG: dict[str, float] = {
    'AA': -0.9, 'AU': -0.9, 'UA': -1.1, 'UU': -0.9,
    'GA': -1.3, 'GU': -2.1, 'CA': -1.8, 'CU': -1.7,
    'AG': -1.3, 'UG': -2.1, 'AC': -1.8, 'UC': -1.7,
    'GG': -3.4, 'GC': -2.9, 'CG': -3.4, 'CC': -3.3,
}

_WC_PAIRS = {('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G'), ('G', 'U'), ('U', 'G')}


def _dna_to_rna(seq: str) -> str:
    return seq.upper().replace('T', 'U')


def _is_complement(a: str, b: str) -> bool:
    return (a, b) in _WC_PAIRS


def _stem_dg(stem_seq: str, complement_seq: str) -> float:
    """Estimate stacking ΔG for a stem using nearest-neighbor model."""
    dg = 0.0
    n = len(stem_seq)
    for i in range(n - 1):
        pair = stem_seq[i] + stem_seq[i + 1]
        dg += _RNA_NN_DG.get(pair, -1.0)
    # Initiation penalty
    first_pair = (stem_seq[0], complement_seq[-1])
    last_pair = (stem_seq[-1], complement_seq[0])
    for pair in [first_pair, last_pair]:
        if pair[0] in 'AU' or pair[1] in 'AU':
            dg += 0.45  # AU-end penalty
    return dg


def fiveprime_hairpin(sequence: str) -> dict:
    """
    Check the first 48 bp for hairpin structures using a pure-Python stem-loop detector.
    Note: ΔG estimated with simplified nearest-neighbor model. For precise values, use ViennaRNA.
    """
    region = _dna_to_rna(sequence[:48])
    n = len(region)

    best_dg = 0.0
    best_structure = None
    threshold_dg = -8.0

    for stem_len in range(4, 16):
        for loop_len in range(3, 11):
            for i in range(n - 2 * stem_len - loop_len + 1):
                stem5 = region[i:i + stem_len]
                j = i + stem_len + loop_len
                if j + stem_len > n:
                    break
                stem3 = region[j:j + stem_len]
                # Check Watson-Crick complementarity for all base pairs
                complement = stem3[::-1]
                valid = all(_is_complement(stem5[k], complement[k]) for k in range(stem_len))
                if not valid:
                    continue
                dg = _stem_dg(stem5, complement)
                if dg < best_dg:
                    best_dg = dg
                    loop = region[i + stem_len:j]
                    best_structure = (
                        f"Stem: {stem5} / {stem3[::-1]} "
                        f"(len={stem_len}), Loop: {loop} "
                        f"(len={loop_len}), ΔG≈{dg:.1f} kcal/mol"
                    )

    passed = best_dg >= threshold_dg
    if best_structure:
        details = (
            f"{'No problematic' if passed else 'Potential'} 5' hairpin detected. "
            f"{best_structure}. "
            "Note: ΔG estimated with simplified nearest-neighbor model. "
            "For precise values, use ViennaRNA."
        )
    else:
        details = (
            "No hairpin structures detected in first 48 bp. "
            "Note: ΔG estimated with simplified nearest-neighbor model. "
            "For precise values, use ViennaRNA."
        )

    positions = [0] if not passed else []
    return {"name": "fiveprime_hairpin", "passed": passed, "details": details,
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
    fiveprime_hairpin,
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
        "fiveprime_hairpin",
    ],
}


def run_all_checks(sequence: str, options: dict | None = None) -> list[dict]:
    """Run all quality checks on a sequence. Returns list of check result dicts."""
    if options is None:
        options = {}

    results = []
    for check_fn in ALL_CHECKS:
        name = check_fn.__name__
        if name == "fiveprime_hairpin" and not options.get("check_hairpin", True):
            continue
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
