"""K. phaffii GS115 Codon Optimizer package."""

from .optimizer import optimize, parse_input, detect_input_type
from .codon_table import get_best_codon, get_synonymous_codons, get_codon_table
from .cai import calculate_cai
from .restriction import find_sites, remove_sites, RESTRICTION_ENZYMES
from .checkers import run_all_checks

__all__ = [
    'optimize',
    'parse_input',
    'detect_input_type',
    'get_best_codon',
    'get_synonymous_codons',
    'get_codon_table',
    'calculate_cai',
    'find_sites',
    'remove_sites',
    'RESTRICTION_ENZYMES',
    'run_all_checks',
]
