"""
Core codon optimization pipeline for K. phaffii GS115.
"""

import re
from typing import Optional

from .codon_table import (
    CODON_USAGE, GENETIC_CODE, AA_TO_CODONS,
    get_best_codon, get_synonymous_codons,
)
from .restriction import find_sites, remove_sites, RESTRICTION_ENZYMES
from .checkers import run_all_checks, rare_codons as check_rare_codons
from .cai import calculate_cai

# ---------------------------------------------------------------------------
# Input parsing / validation
# ---------------------------------------------------------------------------

_VALID_AA = set('ACDEFGHIKLMNPQRSTVWY*')
_VALID_NT = set('ACGTUN')

_STANDARD_CODE = GENETIC_CODE  # codon -> AA

# Reverse map for translation: DNA codons
_DNA_CODE = {
    k: v for k, v in GENETIC_CODE.items()
}


def _strip_fasta(text: str) -> str:
    """Remove FASTA header lines (starting with '>') and return joined sequence."""
    lines = text.splitlines()
    seq_lines = [ln.strip() for ln in lines if ln.strip() and not ln.startswith('>')]
    return ''.join(seq_lines)


def _clean_sequence(text: str) -> str:
    """Strip whitespace, digits, and newlines."""
    return re.sub(r'[\s\d]', '', text).upper()


def detect_input_type(sequence: str) -> str:
    """
    Return 'nucleotide' if all characters are in ACGTUN, otherwise 'protein'.
    Raises ValueError for completely invalid input.
    """
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
    """Translate a DNA sequence to a protein sequence."""
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
            # Stop codon — stop translation (include if at end, skip if internal)
            if i + 3 < len(dna):
                raise ValueError(
                    f"Internal stop codon '{codon}' at position {i}. "
                    "Please check your input sequence."
                )
            break
        protein.append(aa)
    return ''.join(protein)


def parse_input(raw: str, input_type: str = 'auto') -> tuple[str, str, Optional[str]]:
    """
    Parse and validate raw sequence input.

    Returns: (protein_sequence, detected_type, original_nt_sequence_or_None)
    Raises ValueError with descriptive messages on invalid input.
    """
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
        # Validate amino acids
        invalid = set(protein.upper()) - _VALID_AA
        if invalid:
            raise ValueError(
                f"Invalid amino acid characters: {', '.join(sorted(invalid))}. "
                "Use standard single-letter codes."
            )

    if not protein:
        raise ValueError("Protein sequence is empty after parsing.")

    # Remove trailing stop if present
    if protein.endswith('*'):
        protein = protein[:-1]

    if len(protein) < 1:
        raise ValueError("Protein sequence must contain at least one amino acid.")

    return protein, detected, original_nt


# ---------------------------------------------------------------------------
# Core optimization
# ---------------------------------------------------------------------------

def _build_greedy_sequence(protein: str) -> tuple[str, list[dict]]:
    """Build initial CDS using highest-frequency codon per amino acid."""
    codons = []
    changes = []
    for i, aa in enumerate(protein):
        best = get_best_codon(aa)
        codons.append(best)
        changes.append({
            'position': i,
            'aa': aa,
            'original': best,
            'optimized': best,
            'reason': 'greedy: highest frequency',
        })
    # Add stop codon (TAA is most frequent in K. phaffii)
    codons.append('TAA')
    return ''.join(codons), changes


def _fix_rare_codons(codons: list[str], changes: list[dict]) -> list[str]:
    """Replace rare codons (< 0.08 frequency) with the best synonymous codon."""
    threshold = 0.08
    for i, codon in enumerate(codons[:-1]):  # Skip stop codon
        freq = CODON_USAGE.get(codon, 0.0)
        if freq < threshold:
            aa = GENETIC_CODE.get(codon)
            if aa and aa != '*':
                best = get_best_codon(aa)
                if best != codon:
                    codons[i] = best
                    changes[i] = {
                        'position': i,
                        'aa': aa,
                        'original': changes[i].get('original', codon),
                        'optimized': best,
                        'reason': f'replaced rare codon {codon} ({freq:.2f}) with {best}',
                    }
    return codons


def _fix_checker_violations(
    codons: list[str],
    changes: list[dict],
    options: dict,
    max_iter: int = 1000,
) -> list[str]:
    """
    Iteratively fix checker violations by trying synonymous codon swaps.
    Focuses on rare codons and GC content issues.
    """
    for iteration in range(max_iter):
        seq = ''.join(codons)
        check_results = run_all_checks(seq, options)
        violations = [c for c in check_results if not c['passed']]

        if not violations:
            break

        fixed_any = False

        for violation in violations:
            name = violation['name']

            # Fix rare codons
            if name == 'rare_codons' and options.get('avoid_rare_codons', True):
                codons = _fix_rare_codons(codons, changes)
                fixed_any = True
                continue

            # For GC issues: try swapping codons at failing positions
            if name in ('local_gc_windows', 'global_gc'):
                positions = violation.get('positions', [])
                for pos in positions[:5]:  # Limit to avoid infinite work
                    codon_idx = pos // 3
                    if codon_idx >= len(codons) - 1:
                        continue
                    codon = codons[codon_idx]
                    aa = GENETIC_CODE.get(codon)
                    if not aa or aa == '*':
                        continue
                    synonyms = get_synonymous_codons(aa)
                    for alt in synonyms:
                        if alt != codon:
                            codons[codon_idx] = alt
                            if codon_idx < len(changes):
                                changes[codon_idx]['optimized'] = alt
                                changes[codon_idx]['reason'] = f'GC content fix'
                            fixed_any = True
                            break

        if not fixed_any:
            break

    return codons


def optimize(
    raw_input: str,
    input_type: str = 'auto',
    selected_enzymes: Optional[list[str]] = None,
    options: Optional[dict] = None,
) -> dict:
    """
    Full codon optimization pipeline.

    Returns a result dict with all relevant information.
    """
    if options is None:
        options = {
            'avoid_rare_codons': True,
            'check_hairpin': True,
            'check_repeats': True,
        }
    if selected_enzymes is None:
        selected_enzymes = []

    # Validate enzymes
    invalid_enzymes = [e for e in selected_enzymes if e not in RESTRICTION_ENZYMES]
    if invalid_enzymes:
        raise ValueError(f"Unknown restriction enzymes: {', '.join(invalid_enzymes)}")

    # 1. Parse input
    protein, detected_type, original_nt = parse_input(raw_input, input_type)

    # 2. Calculate CAI of original sequence (if DNA input)
    cai_original = None
    if original_nt:
        try:
            cai_original = calculate_cai(original_nt)['cai']
        except Exception:
            cai_original = None

    # 3. Build initial greedy sequence
    codons, changes = _build_greedy_sequence(protein)
    # Remove stop codon for processing (re-add at end)
    stop_codon = codons[-3:]
    codons_list = [codons[i:i+3] for i in range(0, len(codons) - 2, 3)]
    # Last element is the stop codon
    stop = codons_list[-1]
    working_codons = codons_list[:-1]  # coding codons only

    # Initialize changes list properly (without stop codon)
    changes = []
    for i, aa in enumerate(protein):
        best = get_best_codon(aa)
        changes.append({
            'position': i,
            'aa': aa,
            'original': best,
            'optimized': best,
            'reason': 'greedy: highest frequency',
        })

    # 4. Remove restriction enzyme sites
    cds_without_stop = ''.join(working_codons)
    removed_sites = []
    failed_sites = []

    if selected_enzymes:
        cds_cleaned, removed_sites, failed_sites = remove_sites(
            cds_without_stop, selected_enzymes, CODON_USAGE
        )
        working_codons = [cds_cleaned[i:i+3] for i in range(0, len(cds_cleaned) - 2, 3)]

        # Update changes for swapped codons
        for removal in removed_sites:
            idx = removal['codon_index']
            if idx < len(changes):
                original = changes[idx]['original']
                changes[idx]['optimized'] = removal['new_codon']
                changes[idx]['reason'] = f"restriction site removal: {removal['enzyme']}"

    # 5. Fix checker violations
    working_codons = _fix_checker_violations(working_codons, changes, options)

    # 6. Final CDS
    final_cds = ''.join(working_codons) + stop

    # 7. Final CAI
    cai_result = calculate_cai(final_cds)

    # 8. Final checks
    check_results = run_all_checks(final_cds, options)

    # 9. Build codon changes report (only actual changes)
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

    # 10. Warnings
    warnings = []
    for check in check_results:
        if not check['passed']:
            warnings.append(f"{check['name']}: {check['details']}")

    for failed in failed_sites:
        if failed.get('is_type_iis'):
            warnings.append(
                f"WARNING: Type IIS enzyme {failed['enzyme']} site at position "
                f"{failed['position']} ({failed['strand']} strand) could not be removed. "
                "Manual review recommended."
            )

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
