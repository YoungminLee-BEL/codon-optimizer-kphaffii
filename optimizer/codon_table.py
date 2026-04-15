"""
K. phaffii GS115 codon usage table.
Source: Kazusa Codon Usage Database (https://www.kazusa.or.jp/codon/)
Frequencies are relative (normalized to sum to 1.0 per synonymous codon group).
"""

CODON_USAGE = {
    'TTT': 0.46, 'TTC': 0.54,
    'TTA': 0.10, 'TTG': 0.29, 'CTT': 0.13, 'CTC': 0.12, 'CTA': 0.08, 'CTG': 0.28,
    'ATT': 0.34, 'ATC': 0.44, 'ATA': 0.22,
    'ATG': 1.00,
    'GTT': 0.29, 'GTC': 0.22, 'GTA': 0.15, 'GTG': 0.34,
    'TCT': 0.26, 'TCC': 0.22, 'TCA': 0.19, 'TCG': 0.09, 'AGT': 0.12, 'AGC': 0.13,
    'CCT': 0.30, 'CCC': 0.17, 'CCA': 0.38, 'CCG': 0.15,
    'ACT': 0.30, 'ACC': 0.28, 'ACA': 0.28, 'ACG': 0.14,
    'GCT': 0.35, 'GCC': 0.26, 'GCA': 0.24, 'GCG': 0.15,
    'TAT': 0.44, 'TAC': 0.56,
    'TAA': 0.50, 'TAG': 0.20, 'TGA': 0.30,
    'CAT': 0.40, 'CAC': 0.60,
    'CAA': 0.55, 'CAG': 0.45,
    'AAT': 0.40, 'AAC': 0.60,
    'AAA': 0.46, 'AAG': 0.54,
    'GAT': 0.46, 'GAC': 0.54,
    'GAA': 0.53, 'GAG': 0.47,
    'TGT': 0.44, 'TGC': 0.56,
    'TGG': 1.00,
    'CGT': 0.13, 'CGC': 0.12, 'CGA': 0.07, 'CGG': 0.10, 'AGA': 0.35, 'AGG': 0.23,
    'GGT': 0.37, 'GGC': 0.22, 'GGA': 0.25, 'GGG': 0.16,
}

# Standard genetic code: codon -> amino acid (single letter)
GENETIC_CODE = {
    'TTT': 'F', 'TTC': 'F',
    'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I',
    'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S', 'AGT': 'S', 'AGC': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y',
    'TAA': '*', 'TAG': '*', 'TGA': '*',
    'CAT': 'H', 'CAC': 'H',
    'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N',
    'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D',
    'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C',
    'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

# Build reverse map: amino acid -> list of codons
AA_TO_CODONS: dict[str, list[str]] = {}
for codon, aa in GENETIC_CODE.items():
    AA_TO_CODONS.setdefault(aa, []).append(codon)


def get_best_codon(aa: str) -> str:
    """Return the highest-frequency codon for the given amino acid."""
    codons = AA_TO_CODONS.get(aa.upper(), [])
    if not codons:
        raise ValueError(f"Unknown amino acid: {aa!r}")
    return max(codons, key=lambda c: CODON_USAGE.get(c, 0.0))


def get_synonymous_codons(aa: str) -> list[str]:
    """Return all synonymous codons for the given amino acid, sorted by frequency descending."""
    codons = AA_TO_CODONS.get(aa.upper(), [])
    if not codons:
        raise ValueError(f"Unknown amino acid: {aa!r}")
    return sorted(codons, key=lambda c: CODON_USAGE.get(c, 0.0), reverse=True)


def get_codon_table() -> dict[str, float]:
    """Return the full CODON_USAGE dictionary."""
    return dict(CODON_USAGE)
