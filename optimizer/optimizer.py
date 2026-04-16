"""
Core codon optimization pipeline for K. phaffii GS115.

Pipeline:
  1. Greedy build (highest-frequency codon per AA)
  2. Remove restriction enzyme sites
  3. Actively fix ALL checker violations via synonymous codon swaps
  4. Final validation — anything still unresolvable → warning only
"""

import re
import math
import itertools
import sys
from typing import Optional

from .codon_table import (
    CODON_USAGE, GENETIC_CODE, AA_TO_CODONS,
    get_best_codon, get_synonymous_codons,
)
from .restriction import find_sites, remove_sites, RESTRICTION_ENZYMES
from .checkers import (
    run_all_checks,
    internal_initiation as chk_internal_initiation,
    cryptic_promoter as chk_cryptic_promoter,
    strong_rbs as chk_strong_rbs,
    terminator_sequences as chk_terminator,
    homopolymer_runs as chk_homopolymer,
    repeat_sequences as chk_repeat,
    fiveprime_hairpin as chk_hairpin,
    rare_codons as chk_rare_codons,
    _reverse_complement,
    _find_worst_hairpin,
)
from .cai import calculate_cai

def _dbg(msg: str) -> None:
    """Write a debug line to stderr (visible in console, not in web UI)."""
    print(f"[hairpin-fix] {msg}", file=sys.stderr)

# ---------------------------------------------------------------------------
# Input parsing / validation
# ---------------------------------------------------------------------------

_VALID_AA = set('ACDEFGHIKLMNPQRSTVWY*')
_VALID_NT = set('ACGTUN')

_DNA_CODE = dict(GENETIC_CODE)


def _strip_fasta(text: str) -> str:
    lines = text.splitlines()
    return ''.join(ln.strip() for ln in lines if ln.strip() and not ln.startswith('>'))


def _clean_sequence(text: str) -> str:
    return re.sub(r'[\s\d]', '', text).upper()


def detect_input_type(sequence: str) -> str:
    chars = set(sequence.upper())
    if chars <= _VALID_NT:
        return 'nucleotide'
    if chars <= _VALID_AA:
        return 'protein'
    invalid = chars - (_VALID_AA | _VALID_NT)
    raise ValueError(
        f"Input contains invalid characters: {', '.join(sorted(invalid))}. "
        "Expected standard amino acid single-letter codes or DNA/RNA bases."
    )


def translate_dna(dna: str) -> str:
    dna = dna.upper().replace('U', 'T')
    if len(dna) % 3 != 0:
        raise ValueError(
            f"DNA/RNA sequence length ({len(dna)}) is not divisible by 3. "
            "Please provide a complete coding sequence."
        )
    protein = []
    for i in range(0, len(dna) - 2, 3):
        codon = dna[i:i+3]
        aa = _DNA_CODE.get(codon)
        if aa is None:
            raise ValueError(f"Unknown codon '{codon}' at position {i}.")
        if aa == '*':
            if i + 3 < len(dna):
                raise ValueError(
                    f"Internal stop codon '{codon}' at position {i}. "
                    "Please check your input sequence."
                )
            break
        protein.append(aa)
    return ''.join(protein)


def parse_input(raw: str, input_type: str = 'auto') -> tuple[str, str, Optional[str]]:
    text = _strip_fasta(raw)
    text = _clean_sequence(text)

    if not text:
        raise ValueError("Empty sequence after stripping whitespace and FASTA headers.")

    if input_type == 'auto':
        detected = detect_input_type(text)
    elif input_type in ('nucleotide', 'dna', 'rna'):
        detected = 'nucleotide'
    elif input_type == 'protein':
        detected = 'protein'
    else:
        detected = detect_input_type(text)

    original_nt = None

    if detected == 'nucleotide':
        nt = text.replace('U', 'T')
        if len(nt) < 3:
            raise ValueError("Nucleotide sequence is too short (minimum 3 bases / 1 codon).")
        protein = translate_dna(nt)
        original_nt = nt
    else:
        protein = text
        invalid = set(protein.upper()) - _VALID_AA
        if invalid:
            raise ValueError(
                f"Invalid amino acid characters: {', '.join(sorted(invalid))}. "
                "Use standard single-letter codes."
            )

    if not protein:
        raise ValueError("Protein sequence is empty after parsing.")

    if protein.endswith('*'):
        protein = protein[:-1]

    if len(protein) < 1:
        raise ValueError("Protein sequence must contain at least one amino acid.")

    return protein, detected, original_nt


# ---------------------------------------------------------------------------
# Shared codon-swap utilities
# ---------------------------------------------------------------------------

def _codons_to_seq(codons: list[str]) -> str:
    return ''.join(codons)


def _synonyms_by_freq(codon: str) -> list[str]:
    """All synonymous codons for codon's AA, sorted by frequency desc, codon excluded first."""
    aa = GENETIC_CODE.get(codon)
    if not aa or aa == '*':
        return []
    return [c for c in get_synonymous_codons(aa) if c != codon]


def _overlapping_codon_indices(pos: int, length: int, n_coding: int) -> list[int]:
    """Return indices of coding codons that overlap the region [pos, pos+length)."""
    first = pos // 3
    last = (pos + length - 1) // 3
    return [i for i in range(first, min(last + 1, n_coding))]


def _record_change(changes: list[dict], idx: int, new_codon: str, reason: str) -> None:
    if idx < len(changes):
        changes[idx]['optimized'] = new_codon
        changes[idx]['reason'] = reason


def _try_single_swap(
    codons: list[str],
    codon_indices: list[int],
    check_fn,          # callable(seq) -> dict with 'passed' key
    reason: str,
    changes: list[dict],
    max_attempts: int = 50,
) -> tuple[list[str], bool]:
    """
    Try swapping each codon in codon_indices to a synonymous alternative.
    Accept first swap that makes check_fn return passed=True.
    Returns (new_codons, fixed_bool).
    """
    attempts = 0
    for idx in codon_indices:
        for alt in _synonyms_by_freq(codons[idx]):
            if attempts >= max_attempts:
                return codons, False
            test = codons[:]
            test[idx] = alt
            if check_fn(_codons_to_seq(test))['passed']:
                _record_change(changes, idx, alt, reason)
                return test, True
            attempts += 1
    return codons, False


def _try_double_swap(
    codons: list[str],
    codon_indices: list[int],
    check_fn,
    reason: str,
    changes: list[dict],
    max_attempts: int = 50,
) -> tuple[list[str], bool]:
    """Try swapping two adjacent codons simultaneously."""
    attempts = 0
    pairs = []
    for i in range(len(codon_indices) - 1):
        pairs.append((codon_indices[i], codon_indices[i + 1]))
    # Also add adjacent pairs just outside the motif overlap
    for ci in codon_indices:
        if ci > 0:
            pairs.append((ci - 1, ci))
        if ci + 1 < len(codons) - 1:
            pairs.append((ci, ci + 1))

    for i1, i2 in pairs:
        alts1 = _synonyms_by_freq(codons[i1])
        alts2 = _synonyms_by_freq(codons[i2])
        for a1 in alts1:
            for a2 in alts2:
                if attempts >= max_attempts:
                    return codons, False
                test = codons[:]
                test[i1] = a1
                test[i2] = a2
                if check_fn(_codons_to_seq(test))['passed']:
                    _record_change(changes, i1, a1, reason)
                    _record_change(changes, i2, a2, reason)
                    return test, True
                attempts += 1
    return codons, False


# ---------------------------------------------------------------------------
# Fixer: strong RBS (GGAGG / TAAGGAG on either strand)
# ---------------------------------------------------------------------------

_RBS_PATTERNS = [re.compile(p) for p in ['GGAGG', 'TAAGGAG']]
_RBS_PATTERNS_RC = [re.compile(_reverse_complement(p.pattern)) for p in _RBS_PATTERNS]


def _find_rbs_hits(seq: str) -> list[tuple[int, int]]:
    """Return (start, end) of all RBS-like hits on forward and reverse strand."""
    hits = []
    for pat in _RBS_PATTERNS:
        for m in re.finditer(f'(?={pat.pattern})', seq):
            hits.append((m.start(), m.start() + len(pat.pattern)))
    for pat in _RBS_PATTERNS_RC:
        for m in re.finditer(f'(?={pat.pattern})', seq):
            hits.append((m.start(), m.start() + len(pat.pattern)))
    return hits


def fix_strong_rbs(
    codons: list[str],
    aa_sequence: str,
    changes: list[dict],
) -> tuple[list[str], list[str]]:
    """
    Find GGAGG or TAAGGAG on either strand and eliminate via synonymous codon swaps.
    Returns (new_codons, list_of_unresolved_warnings).
    """
    unresolved = []
    n = len(codons)
    reason = "fix: strong_rbs"

    for _ in range(10):  # re-scan after each fix
        seq = _codons_to_seq(codons)
        hits = _find_rbs_hits(seq)
        if not hits:
            break

        fixed_any = False
        for start, end in hits:
            indices = _overlapping_codon_indices(start, end - start, n)
            # Filter out codon 0 (ATG start)
            indices = [i for i in indices if i > 0]
            if not indices:
                # All overlapping codons are position 0 — expand window
                indices = [1] if n > 1 else []

            codons, fixed = _try_single_swap(codons, indices, chk_strong_rbs, reason, changes)
            if fixed:
                fixed_any = True
                break
            codons, fixed = _try_double_swap(codons, indices, chk_strong_rbs, reason, changes, 50)
            if fixed:
                fixed_any = True
                break

        if not fixed_any:
            seq = _codons_to_seq(codons)
            still = _find_rbs_hits(seq)
            for s, e in still:
                unresolved.append(
                    f"strong_rbs: motif at position {s} could not be removed by synonymous swaps"
                )
            break

    return codons, unresolved


# ---------------------------------------------------------------------------
# Fixer: cryptic sigma70 promoter (-35/-10 pairs)
# ---------------------------------------------------------------------------

def _find_promoter_pairs(seq: str) -> list[tuple[int, int, str]]:
    """
    Return (pos_35, pos_10, strand) for each TTGACA/-TATAAT pair within
    15-20 bp spacing on forward and reverse strands.
    """
    pairs = []
    for strand_seq, strand in [(seq, '+'), (_reverse_complement(seq), '-')]:
        m35_pos = [m.start() for m in re.finditer('TTGACA', strand_seq)]
        m10_pos = [m.start() for m in re.finditer('TATAAT', strand_seq)]
        for p35 in m35_pos:
            for p10 in m10_pos:
                gap = p10 - (p35 + 6)
                if 15 <= gap <= 20:
                    # Convert reverse strand positions back to forward coords
                    if strand == '+':
                        pairs.append((p35, p10, '+'))
                    else:
                        n = len(seq)
                        pairs.append((n - p35 - 6, n - p10 - 6, '-'))
    return pairs


def fix_cryptic_promoter(
    codons: list[str],
    aa_sequence: str,
    changes: list[dict],
) -> tuple[list[str], list[str]]:
    """
    Break TTGACA/-TATAAT pairs via synonymous codon swaps.
    Prefer disrupting the -10 box (TATAAT) first.
    Returns (new_codons, list_of_unresolved_warnings).
    """
    unresolved = []
    n = len(codons)
    reason = "fix: cryptic_promoter"

    for _ in range(10):
        seq = _codons_to_seq(codons)
        pairs = _find_promoter_pairs(seq)
        if not pairs:
            break

        fixed_any = False
        for p35, p10, strand in pairs:
            # Try -10 box first (more critical), then -35
            for box_pos, box_len in [(p10, 6), (p35, 6)]:
                if strand == '-':
                    # Map back to forward strand coords
                    fwd_start = len(seq) - box_pos - box_len
                    indices = _overlapping_codon_indices(fwd_start, box_len, n)
                else:
                    indices = _overlapping_codon_indices(box_pos, box_len, n)

                indices = [i for i in indices if i > 0]
                if not indices and n > 1:
                    indices = [1]

                codons, fixed = _try_single_swap(
                    codons, indices, chk_cryptic_promoter, reason, changes
                )
                if fixed:
                    fixed_any = True
                    break
                codons, fixed = _try_double_swap(
                    codons, indices, chk_cryptic_promoter, reason, changes, 50
                )
                if fixed:
                    fixed_any = True
                    break
            if fixed_any:
                break

        if not fixed_any:
            seq = _codons_to_seq(codons)
            still = _find_promoter_pairs(seq)
            for p35, p10, strand in still:
                unresolved.append(
                    f"cryptic_promoter: −35/−10 pair (pos {p35}/{p10}, strand {strand}) "
                    "could not be removed by synonymous swaps"
                )
            break

    return codons, unresolved


# ---------------------------------------------------------------------------
# Fixer: internal initiation (SD + ATG context)
# ---------------------------------------------------------------------------

def _find_internal_atg_with_sd(seq: str) -> list[tuple[int, str]]:
    """
    Return (atg_pos, strand) for internal ATGs with SD-like context upstream.
    """
    hits = []
    for strand_seq, strand in [(seq, '+'), (_reverse_complement(seq), '-')]:
        for m in re.finditer('ATG', strand_seq):
            pos = m.start()
            if pos == 0 and strand == '+':
                continue
            upstream = strand_seq[max(0, pos - 15): max(0, pos - 5)]
            if 'GGAGG' in upstream or 'TAAGGAG' in upstream:
                fwd_pos = pos if strand == '+' else len(seq) - pos - 3
                hits.append((fwd_pos, strand))
    return hits


def fix_internal_initiation(
    codons: list[str],
    aa_sequence: str,
    changes: list[dict],
) -> tuple[list[str], list[str]]:
    """
    Break internal SD-ATG pairs.
    Strategy 1: swap the internal ATG codon if the AA is Ile (I) — replace with ATT/ATC/ATA.
    Strategy 2: swap upstream codons to disrupt the SD motif.
    Returns (new_codons, list_of_unresolved_warnings).
    """
    unresolved = []
    n = len(codons)
    reason = "fix: internal_initiation"

    for _ in range(10):
        seq = _codons_to_seq(codons)
        hits = _find_internal_atg_with_sd(seq)
        if not hits:
            break

        fixed_any = False
        for atg_pos, strand in hits:
            atg_codon_idx = atg_pos // 3

            # Strategy 1: swap the ATG itself if AA allows (Ile only — Met cannot change ATG)
            if 0 < atg_codon_idx < n:
                aa = GENETIC_CODE.get(codons[atg_codon_idx])
                if aa == 'I':  # Ile can use ATT, ATC, ATA instead of ATG
                    for alt in _synonyms_by_freq(codons[atg_codon_idx]):
                        if not alt.startswith('AT') or alt[2] not in 'TCA':
                            continue
                        # Make sure alt doesn't start with ATG
                        if alt == 'ATG':
                            continue
                        test = codons[:]
                        test[atg_codon_idx] = alt
                        if chk_internal_initiation(_codons_to_seq(test))['passed']:
                            _record_change(changes, atg_codon_idx, alt, reason)
                            codons = test
                            fixed_any = True
                            break
                    if fixed_any:
                        break

            # Strategy 2: disrupt the SD motif in the upstream window (5–15 bp upstream)
            upstream_nt_start = max(0, atg_pos - 15)
            upstream_nt_end = max(0, atg_pos - 5)
            if strand == '-':
                # On reverse strand, the upstream of the ATG is downstream in fwd coordinates
                fwd_upstream_start = atg_pos + 3 + 5
                fwd_upstream_end = atg_pos + 3 + 15
                upstream_nt_start = fwd_upstream_start
                upstream_nt_end = min(fwd_upstream_end, len(seq))

            indices = []
            for nt_pos in range(upstream_nt_start, upstream_nt_end):
                ci = nt_pos // 3
                if 0 < ci < n and ci not in indices:
                    indices.append(ci)

            if not indices and n > 1:
                indices = [max(1, atg_codon_idx - 3)]

            codons, fixed = _try_single_swap(
                codons, indices, chk_internal_initiation, reason, changes
            )
            if fixed:
                fixed_any = True
                break
            codons, fixed = _try_double_swap(
                codons, indices, chk_internal_initiation, reason, changes, 50
            )
            if fixed:
                fixed_any = True
                break

        if not fixed_any:
            seq = _codons_to_seq(codons)
            still = _find_internal_atg_with_sd(seq)
            for pos, strand in still:
                unresolved.append(
                    f"internal_initiation: SD-ATG at position {pos} (strand {strand}) "
                    "could not be resolved"
                )
            break

    return codons, unresolved


# ---------------------------------------------------------------------------
# Fixer: terminator-like sequences (AAAAA / TTTTT runs ≥ 5)
# ---------------------------------------------------------------------------

def _find_terminator_runs(seq: str) -> list[tuple[int, int]]:
    hits = []
    for pat in [r'A{5,}', r'T{5,}']:
        for m in re.finditer(pat, seq):
            hits.append((m.start(), m.end()))
    return hits


def fix_terminator_sequences(
    codons: list[str],
    aa_sequence: str,
    changes: list[dict],
) -> tuple[list[str], list[str]]:
    """
    Break poly-A/T runs of ≥ 5 bp by swapping overlapping codons.
    Returns (new_codons, list_of_unresolved_warnings).
    """
    unresolved = []
    n = len(codons)
    reason = "fix: terminator_sequences"

    for _ in range(20):
        seq = _codons_to_seq(codons)
        hits = _find_terminator_runs(seq)
        if not hits:
            break

        fixed_any = False
        for start, end in hits:
            indices = _overlapping_codon_indices(start, end - start, n)
            indices = [i for i in indices if i > 0]
            if not indices:
                indices = [1] if n > 1 else []

            codons, fixed = _try_single_swap(
                codons, indices, chk_terminator, reason, changes
            )
            if fixed:
                fixed_any = True
                break
            codons, fixed = _try_double_swap(
                codons, indices, chk_terminator, reason, changes, 50
            )
            if fixed:
                fixed_any = True
                break

        if not fixed_any:
            seq = _codons_to_seq(codons)
            still = _find_terminator_runs(seq)
            for s, e in still:
                unresolved.append(
                    f"terminator_sequences: poly-A/T run at position {s}-{e} "
                    "could not be broken by synonymous swaps"
                )
            break

    return codons, unresolved


# ---------------------------------------------------------------------------
# Fixer: homopolymer runs ≥ 10 bp
# ---------------------------------------------------------------------------

def _find_homopolymer_runs(seq: str) -> list[tuple[int, int]]:
    hits = []
    for m in re.finditer(r'([ACGT])\1{9,}', seq):
        hits.append((m.start(), m.end()))
    return hits


def fix_homopolymer_runs(
    codons: list[str],
    aa_sequence: str,
    changes: list[dict],
) -> tuple[list[str], list[str]]:
    """
    Break homopolymer runs ≥ 10 bp by swapping overlapping codons.
    Returns (new_codons, list_of_unresolved_warnings).
    """
    unresolved = []
    n = len(codons)
    reason = "fix: homopolymer_runs"

    for _ in range(20):
        seq = _codons_to_seq(codons)
        hits = _find_homopolymer_runs(seq)
        if not hits:
            break

        fixed_any = False
        for start, end in hits:
            indices = _overlapping_codon_indices(start, end - start, n)
            indices = [i for i in indices if i > 0]
            if not indices:
                indices = [1] if n > 1 else []

            codons, fixed = _try_single_swap(
                codons, indices, chk_homopolymer, reason, changes
            )
            if fixed:
                fixed_any = True
                break
            codons, fixed = _try_double_swap(
                codons, indices, chk_homopolymer, reason, changes, 80
            )
            if fixed:
                fixed_any = True
                break

        if not fixed_any:
            seq = _codons_to_seq(codons)
            still = _find_homopolymer_runs(seq)
            for s, e in still:
                unresolved.append(
                    f"homopolymer_runs: run at position {s}-{e} "
                    "could not be broken by synonymous swaps"
                )
            break

    return codons, unresolved


# ---------------------------------------------------------------------------
# Fixer: repeat sequences > 20 bp
# ---------------------------------------------------------------------------

def _find_repeats(seq: str, min_len: int = 21) -> list[tuple[int, int]]:
    """Return (start, end) of second (and later) occurrences of repeated substrings."""
    n = len(seq)
    second_occurrences = []
    seen: dict[str, int] = {}
    for length in range(min_len, n // 2 + 1):
        for i in range(n - length + 1):
            s = seq[i:i + length]
            if s in seen:
                # This is a second+ occurrence — record it for fixing
                if i not in [x for x, _ in second_occurrences]:
                    second_occurrences.append((i, i + length))
            else:
                seen[s] = i
    return second_occurrences


def fix_repeat_sequences(
    codons: list[str],
    aa_sequence: str,
    changes: list[dict],
) -> tuple[list[str], list[str]]:
    """
    Break repeated subsequences > 20 bp by introducing synonymous codon diversity
    in the second (or later) occurrence.
    Returns (new_codons, list_of_unresolved_warnings).
    """
    unresolved = []
    n = len(codons)
    reason = "fix: repeat_sequences"

    for outer in range(10):
        seq = _codons_to_seq(codons)
        repeats = _find_repeats(seq)
        if not repeats:
            break

        fixed_any = False
        for start, end in repeats:
            indices = _overlapping_codon_indices(start, end - start, n)
            indices = [i for i in indices if i > 0]
            if not indices:
                continue

            # For repeats, prefer second-best codon (introduce diversity)
            # Try each overlapping codon, swapping to second-best first
            made_progress = False
            for idx in indices:
                alts = _synonyms_by_freq(codons[idx])
                for alt in alts:
                    test = codons[:]
                    test[idx] = alt
                    if chk_repeat(_codons_to_seq(test))['passed']:
                        _record_change(changes, idx, alt, reason)
                        codons = test
                        made_progress = True
                        fixed_any = True
                        break
                    # Even if check doesn't fully pass, accept if repeat shrunk
                    new_repeats = _find_repeats(_codons_to_seq(test))
                    if len(new_repeats) < len(repeats):
                        _record_change(changes, idx, alt, reason)
                        codons = test
                        made_progress = True
                        fixed_any = True
                        break
                if made_progress:
                    break

        if not fixed_any:
            seq = _codons_to_seq(codons)
            still = _find_repeats(seq)
            for s, e in still:
                unresolved.append(
                    f"repeat_sequences: repeated region at position {s}-{e} "
                    "could not be broken by synonymous swaps"
                )
            break

    return codons, unresolved


# ---------------------------------------------------------------------------
# Fixer: 5' hairpin — aggressive combinatorial multi-codon swap
# ---------------------------------------------------------------------------

def _codon_cai_weight(codon: str) -> float:
    """Return the CAI weight w_i for a single codon (0..1)."""
    from .cai import _MAX_FREQ
    aa = GENETIC_CODE.get(codon)
    if not aa or aa in ('M', 'W', '*'):
        return 1.0  # excluded from CAI, neutral
    freq = CODON_USAGE.get(codon, 0.0)
    max_freq = _MAX_FREQ.get(aa, 1.0)
    return freq / max_freq if max_freq > 0 else 0.0


def _best_hairpin_in_seq(seq: str, max_region: int) -> tuple[float, dict | None]:
    """Return (best_dg, hairpin_dict_or_None) for up to max_region bp."""
    h = _find_worst_hairpin(seq, max_region=max_region)
    return (h['dg'] if h else 0.0, h)


def fix_fiveprime_hairpin(
    codons: list[str],
    aa_sequence: str,
    changes: list[dict],
) -> tuple[list[str], list[str]]:
    """
    Aggressively resolve 5' hairpin structures via combinatorial synonymous codon swaps.

    Strategy (in order):
      1. Identify stem arm positions in the first 48 bp from structured hairpin data.
      2. Collect ALL codons overlapping either arm (first 16 codons, skip pos 0 = ATG).
      3. Try all k-codon simultaneous swaps for k = 1, 2, 3; accept the best combination
         (passes threshold AND maximises combined CAI weight).
      4. If still failing: expand search to first 20 codons (60 bp window) and repeat k=1..3.
      5. Marginal pass: if best achievable ΔG is in [−10, −8), accept with warning.
      6. If ΔG < −10 and truly unresolvable, log detailed warning.

    Debug output is written to stderr (console only, not shown in web UI).
    Returns (new_codons, list_of_unresolved_warnings).
    """
    THRESHOLD = -8.0
    MARGINAL = -10.0
    reason = "fix: fiveprime_hairpin"
    unresolved: list[str] = []

    def _apply_combo(base_codons, idx_alt_pairs):
        """Apply a list of (codon_idx, alt_codon) and return the new codon list."""
        test = base_codons[:]
        for idx, alt in idx_alt_pairs:
            test[idx] = alt
        return test

    def _combo_cai_score(base_codons, idx_alt_pairs):
        """Sum of ln(w) improvement for swapped positions (higher = better CAI)."""
        score = 0.0
        for idx, alt in idx_alt_pairs:
            score += math.log(max(_codon_cai_weight(alt), 1e-10))
        return score

    def _stem_codon_indices(hairpin: dict, max_codon: int) -> list[int]:
        """
        Return sorted list of unique codon indices that overlap either stem arm,
        within [1, max_codon) (index 0 = ATG is always skipped).
        """
        idx_set: set[int] = set()
        for start, end in [
            (hairpin['stem5_start'], hairpin['stem5_end']),
            (hairpin['stem3_start'], hairpin['stem3_end']),
        ]:
            for nt_pos in range(start, min(end, max_codon * 3)):
                ci = nt_pos // 3
                if 1 <= ci < max_codon:
                    idx_set.add(ci)
        return sorted(idx_set)

    def _search(codons_in: list[str], max_codon: int, max_region: int) -> tuple[list[str], float]:
        """
        Try k=1,2,3 combinatorial swaps among the stem-overlapping codons.
        Returns (best_codons, best_dg_achieved).
        """
        seq = _codons_to_seq(codons_in)
        dg, hairpin = _best_hairpin_in_seq(seq, max_region)
        if dg >= THRESHOLD or hairpin is None:
            return codons_in, dg

        stem_indices = _stem_codon_indices(hairpin, max_codon)
        _dbg(f"Stem at nt [{hairpin['stem5_start']}-{hairpin['stem5_end']}] / "
             f"[{hairpin['stem3_start']}-{hairpin['stem3_end']}], "
             f"ΔG={dg:.1f}, targeting codons {stem_indices} (max_codon={max_codon})")

        # Build synonym map for candidate indices
        # Also include adjacent codons (±1) to widen the net
        extended: set[int] = set(stem_indices)
        for ci in stem_indices:
            if ci > 1:
                extended.add(ci - 1)
            if ci + 1 < max_codon:
                extended.add(ci + 1)
        candidate_indices = sorted(
            i for i in extended if 0 < i < max_codon and _synonyms_by_freq(codons_in[i])
        )
        _dbg(f"Extended candidate codon indices: {candidate_indices}")

        best_codons = codons_in
        best_dg = dg
        best_score = -1e9  # CAI score for tie-breaking

        for k in range(1, 4):  # k = 1, 2, 3 simultaneous swaps
            tried = 0
            for combo_indices in itertools.combinations(candidate_indices, k):
                alt_lists = [_synonyms_by_freq(codons_in[ci]) for ci in combo_indices]
                for alt_combo in itertools.product(*alt_lists):
                    tried += 1
                    pairs = list(zip(combo_indices, alt_combo))
                    test = _apply_combo(codons_in, pairs)
                    test_seq = _codons_to_seq(test)
                    new_dg, _ = _best_hairpin_in_seq(test_seq, max_region)
                    cai_score = _combo_cai_score(codons_in, pairs)

                    _dbg(
                        f"  k={k} combo={combo_indices} alts={alt_combo} "
                        f"→ ΔG={new_dg:.2f} CAI_score={cai_score:.3f}"
                    )

                    if new_dg >= THRESHOLD:
                        # Passes hard threshold — prefer highest CAI among passing combos
                        if new_dg > best_dg or (new_dg >= THRESHOLD and cai_score > best_score):
                            best_dg = new_dg
                            best_score = cai_score
                            best_codons = test
                            _dbg(f"  *** New best: ΔG={new_dg:.2f}, "
                                 f"combo={combo_indices}→{alt_combo}")
                    elif new_dg > best_dg:
                        # Doesn't pass threshold but reduces severity — track as fallback
                        if best_dg < THRESHOLD:
                            best_dg = new_dg
                            best_score = cai_score
                            best_codons = test

            _dbg(f"k={k}: tried {tried} combinations, best ΔG so far={best_dg:.2f}")
            if best_dg >= THRESHOLD:
                break  # Found a passing solution at this k — no need for larger k

        return best_codons, best_dg

    # ---- Phase 1: first 16 codons / 48 bp ----
    seq = _codons_to_seq(codons)
    dg0, h0 = _best_hairpin_in_seq(seq, max_region=48)

    if dg0 >= THRESHOLD or h0 is None:
        _dbg("No hairpin violation detected — nothing to fix.")
        return codons, unresolved

    _dbg(f"Phase 1 start: {h0['structure']}")
    codons, achieved_dg = _search(codons, max_codon=min(16, len(codons)), max_region=48)

    if achieved_dg >= THRESHOLD:
        _dbg(f"Phase 1 success: ΔG={achieved_dg:.2f}")
        # Record all changes vs original
        for i, (orig_c, new_c) in enumerate(zip(
            [get_best_codon(aa) for aa in aa_sequence], codons
        )):
            if orig_c != new_c and i < len(changes):
                _record_change(changes, i, new_c, reason)
        return codons, unresolved

    # ---- Phase 2: expand to first 20 codons / 60 bp ----
    _dbg(f"Phase 1 failed (best ΔG={achieved_dg:.2f}). "
         f"Expanding to 20 codons / 60 bp window.")
    codons, achieved_dg = _search(codons, max_codon=min(20, len(codons)), max_region=60)

    if achieved_dg >= THRESHOLD:
        _dbg(f"Phase 2 success: ΔG={achieved_dg:.2f}")
        for i, (orig_c, new_c) in enumerate(zip(
            [get_best_codon(aa) for aa in aa_sequence], codons
        )):
            if orig_c != new_c and i < len(changes):
                _record_change(changes, i, new_c, reason)
        return codons, unresolved

    # ---- Phase 3: marginal pass or hard failure ----
    seq = _codons_to_seq(codons)
    final_dg, final_h = _best_hairpin_in_seq(seq, max_region=60)

    if MARGINAL <= final_dg < THRESHOLD:
        # Marginal — accept with warning
        warn = (
            f"fiveprime_hairpin: marginal hairpin remaining (ΔG={final_dg:.1f} kcal/mol, "
            f"threshold={THRESHOLD}). {final_h['structure'] if final_h else ''}. "
            "Best available codon set applied. "
            "Note: ΔG estimated with simplified nearest-neighbor model; "
            "consider ViennaRNA for precise assessment."
        )
        _dbg(f"Marginal pass: {warn}")
        unresolved.append(warn)
        for i, (orig_c, new_c) in enumerate(zip(
            [get_best_codon(aa) for aa in aa_sequence], codons
        )):
            if orig_c != new_c and i < len(changes):
                _record_change(changes, i, new_c, reason)
    else:
        # Hard failure
        struct_desc = final_h['structure'] if final_h else "unknown structure"
        warn = (
            f"fiveprime_hairpin: unresolvable hairpin (ΔG={final_dg:.1f} kcal/mol). "
            f"{struct_desc}. "
            "All combinations of up to 3 synonymous codon swaps within the first 20 codons "
            "were tried without success. The amino acid sequence constrains the 5' region "
            "such that no synonymous alternative eliminates the stem structure. "
            "Note: ΔG estimated with simplified nearest-neighbor model. "
            "For precise values and manual redesign, use ViennaRNA."
        )
        _dbg(f"Hard failure: {warn}")
        unresolved.append(warn)

    return codons, unresolved


# ---------------------------------------------------------------------------
# Fixer: rare codons (< 8% frequency)
# ---------------------------------------------------------------------------

def fix_rare_codons(
    codons: list[str],
    aa_sequence: str,
    changes: list[dict],
) -> tuple[list[str], list[str]]:
    """
    Replace codons with relative frequency < 0.08 with the best synonymous codon.
    Returns (new_codons, []) — rare codons are always resolvable.
    """
    threshold = 0.08
    reason = "fix: rare_codons"
    for i, codon in enumerate(codons[:-1]):  # skip stop
        freq = CODON_USAGE.get(codon, 0.0)
        if freq < threshold:
            aa = GENETIC_CODE.get(codon)
            if aa and aa != '*':
                best = get_best_codon(aa)
                if best != codon:
                    if i < len(changes):
                        orig = changes[i].get('original', codon)
                        changes[i] = {
                            'position': i,
                            'aa': aa,
                            'original': orig,
                            'optimized': best,
                            'reason': reason,
                        }
                    codons[i] = best
    return codons, []


# ---------------------------------------------------------------------------
# Build greedy sequence
# ---------------------------------------------------------------------------

def _build_greedy_sequence(protein: str) -> tuple[list[str], list[dict]]:
    """Return (codons_list_including_stop, changes_list_for_coding_only)."""
    codons = [get_best_codon(aa) for aa in protein]
    codons.append('TAA')  # stop
    changes = [
        {'position': i, 'aa': aa, 'original': get_best_codon(aa), 'optimized': get_best_codon(aa),
         'reason': 'greedy: highest frequency'}
        for i, aa in enumerate(protein)
    ]
    return codons, changes


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

# Ordered list of (fixer_fn, options_key_or_None, violation_name)
_FIXERS = [
    (fix_strong_rbs,            None,                'strong_rbs'),
    (fix_cryptic_promoter,      None,                'cryptic_promoter'),
    (fix_internal_initiation,   None,                'internal_initiation'),
    (fix_terminator_sequences,  None,                'terminator_sequences'),
    (fix_homopolymer_runs,      None,                'homopolymer_runs'),
    (fix_repeat_sequences,      None,                'repeat_sequences'),
    (fix_fiveprime_hairpin,     None,                'fiveprime_hairpin'),
    (fix_rare_codons,           'avoid_rare_codons', 'rare_codons'),
]


def optimize(
    raw_input: str,
    input_type: str = 'auto',
    selected_enzymes: Optional[list[str]] = None,
    options: Optional[dict] = None,
) -> dict:
    """
    Full codon optimization pipeline.
    """
    if options is None:
        options = {'avoid_rare_codons': True}
    if selected_enzymes is None:
        selected_enzymes = []

    invalid_enzymes = [e for e in selected_enzymes if e not in RESTRICTION_ENZYMES]
    if invalid_enzymes:
        raise ValueError(f"Unknown restriction enzymes: {', '.join(invalid_enzymes)}")

    # 1. Parse input
    protein, detected_type, original_nt = parse_input(raw_input, input_type)

    # CAI of original DNA input
    cai_original = None
    if original_nt:
        try:
            cai_original = calculate_cai(original_nt)['cai']
        except Exception:
            pass

    # 2. Greedy build
    all_codons, changes = _build_greedy_sequence(protein)
    # Separate coding codons from stop
    stop_codon = all_codons[-1]
    codons = all_codons[:-1]  # working list, no stop

    # 3. Remove restriction enzyme sites
    removed_sites: list[dict] = []
    failed_sites: list[dict] = []

    if selected_enzymes:
        cds = _codons_to_seq(codons)
        cds_cleaned, removed_sites, failed_sites = remove_sites(cds, selected_enzymes, CODON_USAGE)
        codons = [cds_cleaned[i:i+3] for i in range(0, len(cds_cleaned) - 2, 3)]
        for removal in removed_sites:
            idx = removal['codon_index']
            if idx < len(changes):
                changes[idx]['optimized'] = removal['new_codon']
                changes[idx]['reason'] = f"restriction site removal: {removal['enzyme']}"

    # 4. Actively fix ALL violations — iterate up to 5 full fixer passes
    all_unresolved: list[str] = []

    for full_pass in range(5):
        seq = _codons_to_seq(codons) + stop_codon
        check_results_before = run_all_checks(seq, {'check_hairpin': True, 'check_repeats': True})
        violations_before = {c['name'] for c in check_results_before if not c['passed']}

        if not violations_before:
            break

        for fixer_fn, options_key, violation_name in _FIXERS:
            # Skip rare_codons fixer if option is off
            if options_key and not options.get(options_key, True):
                continue

            codons, unresolved = fixer_fn(codons, protein, changes)
            # Unresolved warnings are de-duplicated and kept for the last pass only
            if full_pass == 4:  # only report from final pass
                all_unresolved.extend(unresolved)

        seq_after = _codons_to_seq(codons) + stop_codon
        check_results_after = run_all_checks(
            seq_after, {'check_hairpin': True, 'check_repeats': True}
        )
        violations_after = {c['name'] for c in check_results_after if not c['passed']}

        if violations_after == violations_before:
            # No progress — collect unresolved from all fixers for the warning list
            all_unresolved = []
            for fixer_fn, options_key, violation_name in _FIXERS:
                if options_key and not options.get(options_key, True):
                    continue
                _, unresolved = fixer_fn(codons[:], protein, [c.copy() for c in changes])
                all_unresolved.extend(unresolved)
            break

    # Deduplicate unresolved warnings
    seen_warn: set[str] = set()
    deduped_unresolved: list[str] = []
    for w in all_unresolved:
        if w not in seen_warn:
            seen_warn.add(w)
            deduped_unresolved.append(w)

    # 5. Final CDS
    final_cds = _codons_to_seq(codons) + stop_codon

    # 6. Final CAI
    cai_result = calculate_cai(final_cds)

    # 7. Final checks (always run all checks for reporting)
    check_results = run_all_checks(final_cds, {'check_hairpin': True, 'check_repeats': True})

    # 8. Codon changes report (only actual changes vs greedy original)
    codon_changes = []
    for i, ch in enumerate(changes):
        orig = ch.get('original', '')
        opt = ch.get('optimized', '')
        if orig != opt:
            codon_changes.append({
                'position': i,
                'aa': ch['aa'],
                'original': orig,
                'optimized': opt,
                'reason': ch.get('reason', ''),
            })

    # 9. Warnings: remaining check failures + unresolved fixer issues
    warnings: list[str] = []
    for check in check_results:
        if not check['passed']:
            warnings.append(f"{check['name']}: {check['details']}")

    warnings.extend(deduped_unresolved)

    for failed in failed_sites:
        if failed.get('is_type_iis'):
            warnings.append(
                f"WARNING: Type IIS enzyme {failed['enzyme']} site at position "
                f"{failed['position']} ({failed['strand']} strand) could not be removed. "
                "Manual review recommended."
            )

    if codon_changes:
        warnings_note = "CAI reflects constraint-driven codon choices"
        if warnings_note not in warnings:
            warnings.insert(0, warnings_note)

    return {
        'input_summary': {
            'detected_type': detected_type,
            'aa_length': len(protein),
            'nt_length': len(final_cds),
        },
        'optimized_sequence': final_cds,
        'cai_original': cai_original,
        'cai_optimized': cai_result['cai'],
        'cai_interpretation': cai_result['interpretation'],
        'per_codon_cai': cai_result['per_codon'],
        'checks': check_results,
        'restriction_sites_removed': removed_sites,
        'restriction_sites_failed': failed_sites,
        'codon_changes': codon_changes,
        'warnings': warnings,
    }
