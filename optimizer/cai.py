"""
Codon Adaptation Index (CAI) calculation.
Method: Sharp & Li (1987).
"""

import math
from .codon_table import CODON_USAGE, GENETIC_CODE, AA_TO_CODONS

# Codons excluded from CAI calculation (Met, Trp, Stop)
_EXCLUDED_AAS = {'M', 'W', '*'}

# Pre-compute max frequency per amino acid
_MAX_FREQ: dict[str, float] = {}
for _aa, _codons in AA_TO_CODONS.items():
    if _aa not in _EXCLUDED_AAS:
        _MAX_FREQ[_aa] = max(CODON_USAGE.get(c, 0.0) for c in _codons)


def calculate_cai(sequence: str) -> dict:
    """
    Calculate CAI for a coding DNA sequence.

    Returns:
        {
            "cai": float,
            "interpretation": str,
            "per_codon": [{"position": int, "codon": str, "aa": str, "w": float}]
        }
    """
    seq = sequence.upper()
    codons = [seq[i:i+3] for i in range(0, len(seq) - 2, 3)]

    per_codon = []
    log_w_sum = 0.0
    count = 0

    for i, codon in enumerate(codons):
        aa = GENETIC_CODE.get(codon)
        if aa is None or aa in _EXCLUDED_AAS:
            per_codon.append({
                "position": i * 3,
                "codon": codon,
                "aa": aa or "?",
                "w": None,
            })
            continue

        freq = CODON_USAGE.get(codon, 0.0)
        max_freq = _MAX_FREQ.get(aa, 1.0)
        w = freq / max_freq if max_freq > 0 else 0.0

        # Avoid log(0)
        w_safe = max(w, 1e-10)
        log_w_sum += math.log(w_safe)
        count += 1

        per_codon.append({
            "position": i * 3,
            "codon": codon,
            "aa": aa,
            "w": round(w, 4),
        })

    if count == 0:
        cai_value = 0.0
    else:
        cai_value = math.exp(log_w_sum / count)

    cai_value = round(cai_value, 4)

    if cai_value >= 0.8:
        interpretation = "Excellent (>0.8)"
    elif cai_value >= 0.6:
        interpretation = "Good (0.6–0.8)"
    else:
        interpretation = "Poor (<0.6)"

    return {
        "cai": cai_value,
        "interpretation": interpretation,
        "per_codon": per_codon,
    }
