"""
Restriction enzyme site detection and removal.
Hahn Lab enzyme set for K. phaffii cloning.
"""

import re
from typing import Optional
from .codon_table import GENETIC_CODE, AA_TO_CODONS, CODON_USAGE

RESTRICTION_ENZYMES = {
    'ApaI':    'GGGCCC',
    'BamHI':   'GGATCC',
    'BcuI':    'ACTAGT',
    'BglII':   'AGATCT',
    'BpiI':    'GAAGACNN',
    'BsaI':    'GGTCTCN',
    'BshTI':   'ACCGGT',
    'Bsp120I': 'GGGCCC',
    'BssHII':  'GCGCGC',
    'ClaI':    'ATCGAT',
    'Eco32I':  'GATATC',
    'EcoRI':   'GAATTC',
    'EcoRV':   'GATATC',
    'HindIII': 'AAGCTT',
    'MauBI':   'CGCGCGCG',
    'MluI':    'ACGCGT',
    'MunI':    'CAATTG',
    'NcoI':    'CCATGG',
    'NdeI':    'CATATG',
    'NheI':    'GCTAGC',
    'NotI':    'GCGGCCGC',
    'NsiI':    'ATGCAT',
    'PacI':    'TTAATTAA',
    'PfoI':    'TCCNGGA',
    'PstI':    'CTGCAG',
    'PvuI':    'CGATCG',
    'PvuII':   'CAGCTG',
    'SacI':    'GAGCTC',
    'SalI':    'GTCGAC',
    'SdaI':    'CCTGCAGG',
    'SgsI':    'GGCGCGCC',
    'SmaI':    'CCCGGG',
    'SmiI':    'ATTTAAAT',
    'SpeI':    'ACTAGT',
    'StuI':    'AGGCCT',
    'XbaI':    'TCTAGA',
    'XhoI':    'CTCGAG',
    'XmaJI':   'CCTAGG',
}

TYPE_IIS_ENZYMES = {'BpiI', 'BsaI'}

# Isoschizomers (same recognition sequence)
ISOSCHIZOMERS = {
    'BcuI': 'SpeI',
    'SpeI': 'BcuI',
    'Eco32I': 'EcoRV',
    'EcoRV': 'Eco32I',
    'Bsp120I': 'ApaI',
    'ApaI': 'Bsp120I',
}

IAMBIGUOUS = {'N': '[ACGT]', 'R': '[AG]', 'Y': '[CT]', 'S': '[GC]',
              'W': '[AT]', 'K': '[GT]', 'M': '[AC]', 'B': '[CGT]',
              'D': '[AGT]', 'H': '[ACT]', 'V': '[ACG]'}


def _site_to_regex(site: str) -> str:
    """Convert an IUPAC recognition site to a regex pattern."""
    pattern = ''
    for ch in site.upper():
        if ch in IAMBIGUOUS:
            pattern += IAMBIGUOUS[ch]
        else:
            pattern += ch
    return pattern


def _reverse_complement(seq: str) -> str:
    comp = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(comp.get(b, b) for b in reversed(seq.upper()))


def find_sites(sequence: str, enzymes: Optional[list[str]] = None) -> list[tuple]:
    """
    Find restriction enzyme sites in sequence (both strands).

    Returns list of (enzyme_name, position, strand, matched_site) tuples.
    Position is 0-based on the forward strand.
    """
    sequence = sequence.upper()
    if enzymes is None:
        enzymes = list(RESTRICTION_ENZYMES.keys())

    results = []
    seen = set()

    for enzyme in enzymes:
        site = RESTRICTION_ENZYMES.get(enzyme)
        if site is None:
            continue
        pattern = _site_to_regex(site)

        # Forward strand
        for m in re.finditer(f'(?={pattern})', sequence):
            pos = m.start()
            matched = sequence[pos:pos + len(site)]
            key = (enzyme, pos, '+')
            if key not in seen:
                seen.add(key)
                results.append((enzyme, pos, '+', matched))

        # Reverse strand
        rc_site = _site_to_regex(_reverse_complement(site))
        for m in re.finditer(f'(?={rc_site})', sequence):
            pos = m.start()
            matched = sequence[pos:pos + len(site)]
            key = (enzyme, pos, '-')
            if key not in seen:
                seen.add(key)
                results.append((enzyme, pos, '-', matched))

    results.sort(key=lambda x: (x[1], x[0]))
    return results


def _codons_from_sequence(sequence: str) -> list[str]:
    return [sequence[i:i+3] for i in range(0, len(sequence) - 2, 3)]


def _sequence_from_codons(codons: list[str]) -> str:
    return ''.join(codons)


def remove_sites(
    sequence: str,
    selected_enzymes: list[str],
    codon_table: dict
) -> tuple[str, list[dict], list[dict]]:
    """
    Remove restriction enzyme sites from a coding sequence via synonymous codon swaps.

    Args:
        sequence: CDS (must be divisible by 3)
        selected_enzymes: list of enzyme names to remove
        codon_table: CODON_USAGE dict (not used directly but kept for API consistency)

    Returns:
        (cleaned_sequence, removed_list, failed_list)
        removed_list: list of dicts with enzyme/position/strand/original_codon/new_codon
        failed_list: list of dicts with enzyme/position/strand/reason
    """
    codons = list(_codons_from_sequence(sequence))
    removed_list = []
    failed_list = []

    max_iterations = 50
    changed = True
    iteration = 0

    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        current_seq = _sequence_from_codons(codons)
        sites = find_sites(current_seq, selected_enzymes)
        if not sites:
            break

        for enzyme, pos, strand, matched in sites:
            site_len = len(RESTRICTION_ENZYMES[enzyme].replace('N', 'X'))
            site_len = len(RESTRICTION_ENZYMES[enzyme])
            resolved = False

            # Identify which codons overlap the site
            # pos is 0-based start of the site in the sequence
            first_codon_idx = pos // 3
            # Site can span up to ceil(site_len / 3) + 1 codons
            last_codon_idx = (pos + site_len - 1) // 3

            # Try swapping each overlapping codon
            for codon_idx in range(first_codon_idx, min(last_codon_idx + 1, len(codons))):
                original_codon = codons[codon_idx]
                aa = GENETIC_CODE.get(original_codon)
                if aa is None or aa == '*':
                    continue

                synonyms = AA_TO_CODONS.get(aa, [])
                # Sort by frequency descending (prefer higher-frequency alternatives)
                synonyms_sorted = sorted(
                    synonyms,
                    key=lambda c: CODON_USAGE.get(c, 0.0),
                    reverse=True
                )

                for alt_codon in synonyms_sorted:
                    if alt_codon == original_codon:
                        continue
                    # Try the swap
                    test_codons = codons[:]
                    test_codons[codon_idx] = alt_codon
                    test_seq = _sequence_from_codons(test_codons)
                    remaining = find_sites(test_seq, [enzyme])
                    # Check if this specific site at this position is gone
                    site_still_present = any(
                        s[1] == pos and s[0] == enzyme and s[2] == strand
                        for s in remaining
                    )
                    if not site_still_present:
                        removed_list.append({
                            'enzyme': enzyme,
                            'position': pos,
                            'strand': strand,
                            'codon_index': codon_idx,
                            'original_codon': original_codon,
                            'new_codon': alt_codon,
                            'is_type_iis': enzyme in TYPE_IIS_ENZYMES,
                        })
                        codons = test_codons
                        changed = True
                        resolved = True
                        break

                if resolved:
                    break

            if not resolved:
                already_failed = any(
                    f['enzyme'] == enzyme and f['position'] == pos and f['strand'] == strand
                    for f in failed_list
                )
                if not already_failed:
                    reason = (
                        "Type IIS enzyme — removal may not always be possible via synonymous swaps"
                        if enzyme in TYPE_IIS_ENZYMES
                        else "No synonymous codon swap eliminates this site"
                    )
                    failed_list.append({
                        'enzyme': enzyme,
                        'position': pos,
                        'strand': strand,
                        'matched_site': matched,
                        'reason': reason,
                        'is_type_iis': enzyme in TYPE_IIS_ENZYMES,
                    })

    return _sequence_from_codons(codons), removed_list, failed_list
