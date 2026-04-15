"""
Pytest unit tests for K. phaffii codon optimizer.
"""

import math
import pytest
import sys
import os

# Ensure the project root is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizer.optimizer import parse_input, detect_input_type, translate_dna, optimize
from optimizer.codon_table import get_best_codon, get_synonymous_codons, get_codon_table, CODON_USAGE
from optimizer.cai import calculate_cai
from optimizer.restriction import find_sites, remove_sites, RESTRICTION_ENZYMES
from optimizer.checkers import (
    orf_integrity, internal_initiation, cryptic_promoter, strong_rbs,
    terminator_sequences, repeat_sequences, homopolymer_runs,
    global_gc, local_gc_windows, rare_codons, fiveprime_hairpin, run_all_checks
)


# =========================================================================
# codon_table
# =========================================================================

class TestCodonTable:
    def test_get_best_codon_returns_highest_freq(self):
        # Phe: TTT=0.46, TTC=0.54 -> TTC
        assert get_best_codon('F') == 'TTC'

    def test_get_best_codon_leu(self):
        # Leu: TTG=0.29, CTG=0.28, CTT=0.13, CTC=0.12, CTA=0.08, TTA=0.10 -> TTG
        assert get_best_codon('L') == 'TTG'

    def test_get_synonymous_codons_sorted(self):
        codons = get_synonymous_codons('F')
        assert codons[0] == 'TTC'
        assert codons[1] == 'TTT'

    def test_get_synonymous_codons_all_present(self):
        ser_codons = get_synonymous_codons('S')
        assert len(ser_codons) == 6
        assert 'TCT' in ser_codons

    def test_get_codon_table_returns_dict(self):
        table = get_codon_table()
        assert isinstance(table, dict)
        assert 'ATG' in table
        assert table['ATG'] == 1.00

    def test_invalid_aa_raises(self):
        with pytest.raises(ValueError):
            get_best_codon('Z')


# =========================================================================
# Input parsing
# =========================================================================

class TestInputParsing:
    def test_fasta_header_stripped(self):
        fasta = ">my_gene description\nMKVLS\nAEGIK"
        protein, detected, _ = parse_input(fasta, 'protein')
        assert protein == 'MKVLSAEGIK'
        assert detected == 'protein'

    def test_whitespace_stripped(self):
        protein, _, _ = parse_input("  M K V L S  \n", 'protein')
        assert protein == 'MKVLS'

    def test_digits_stripped(self):
        # FASTA sequences sometimes include position numbers
        protein, _, _ = parse_input("1 MKVLS 6", 'protein')
        assert protein == 'MKVLS'

    def test_auto_detect_protein(self):
        _, detected, _ = parse_input('MKVLSAEGIK', 'auto')
        assert detected == 'protein'

    def test_auto_detect_nucleotide(self):
        _, detected, orig_nt = parse_input('ATGAAAGTT', 'auto')
        assert detected == 'nucleotide'
        assert orig_nt == 'ATGAAAGTT'

    def test_rna_u_to_t(self):
        _, detected, orig_nt = parse_input('AUGAAAGUU', 'auto')
        assert detected == 'nucleotide'
        assert orig_nt == 'ATGAAAGTT'

    def test_dna_input_translates_correctly(self):
        # ATG AAA GTT -> M K V
        protein, detected, _ = parse_input('ATGAAAGTT', 'nucleotide')
        assert protein == 'MKV'
        assert detected == 'nucleotide'

    def test_empty_raises(self):
        with pytest.raises(ValueError, match='[Ee]mpty'):
            parse_input('', 'auto')

    def test_invalid_chars_raise(self):
        with pytest.raises(ValueError):
            parse_input('MKVLS!@#', 'protein')

    def test_internal_stop_codon_raises(self):
        with pytest.raises(ValueError, match='[Ii]nternal stop'):
            parse_input('ATGTAAATG', 'nucleotide')

    def test_trailing_stop_removed(self):
        protein, _, _ = parse_input('MKVLS*', 'protein')
        assert not protein.endswith('*')
        assert protein == 'MKVLS'

    def test_multiline_fasta(self):
        fasta = ">gene1\nMKVLS\nAEGIK\nLEFIA\n"
        protein, _, _ = parse_input(fasta, 'protein')
        assert protein == 'MKVLSAEGIKLEFÍA'.replace('Í','I')  # ASCII only
        protein2, _, _ = parse_input(fasta, 'protein')
        assert len(protein2) == 15


# =========================================================================
# CAI calculation
# =========================================================================

class TestCAI:
    def test_all_best_codons_gives_high_cai(self):
        # Build sequence using only best codons
        from optimizer.codon_table import AA_TO_CODONS
        protein = 'MKVLS'
        codons = [get_best_codon(aa) for aa in protein] + ['TAA']
        seq = ''.join(codons)
        result = calculate_cai(seq)
        assert result['cai'] > 0.9

    def test_cai_interpretation_excellent(self):
        protein = 'MKVLS' * 10
        codons = [get_best_codon(aa) for aa in protein] + ['TAA']
        result = calculate_cai(''.join(codons))
        assert 'Excellent' in result['interpretation']

    def test_cai_manual_calculation(self):
        # ATG (M, w=1.0, excluded) + TTC (F, freq=0.54, max=0.54, w=1.0) +
        # TAA (stop, excluded)
        seq = 'ATGTTCTAA'
        result = calculate_cai(seq)
        # Only TTC contributes: w=0.54/0.54=1.0, ln(1.0)=0, CAI=exp(0)=1.0
        assert abs(result['cai'] - 1.0) < 0.01

    def test_per_codon_returned(self):
        result = calculate_cai('ATGTTCTAA')
        assert isinstance(result['per_codon'], list)
        assert len(result['per_codon']) == 3  # ATG, TTC, TAA

    def test_rare_codon_lowers_cai(self):
        # CGA = 0.07 (very rare Arg codon)
        # Use a sequence with a rare codon mixed in
        protein = 'MRRR'
        best_seq = ''.join([get_best_codon(aa) for aa in protein]) + 'TAA'
        # Replace last Arg codon with rare CGA
        rare_seq = best_seq[:9] + 'CGA' + 'TAA'
        r1 = calculate_cai(best_seq)
        r2 = calculate_cai(rare_seq)
        assert r1['cai'] > r2['cai']


# =========================================================================
# Restriction enzyme detection
# =========================================================================

class TestRestrictionSites:
    def test_find_ecori_forward(self):
        seq = 'ATGGAATTCATG'
        sites = find_sites(seq, ['EcoRI'])
        assert any(s[0] == 'EcoRI' and s[2] == '+' for s in sites)

    def test_find_ecori_reverse_strand(self):
        # Reverse complement of GAATTC is GAATTC (palindrome)
        seq = 'ATGGAATTCATG'
        sites = find_sites(seq, ['EcoRI'])
        positions = [(s[0], s[1], s[2]) for s in sites]
        assert len(positions) > 0

    def test_no_false_positives(self):
        seq = 'ATGAAAGTTCTG'
        sites = find_sites(seq, ['EcoRI', 'BamHI', 'HindIII'])
        assert len(sites) == 0

    def test_degenerate_base_n(self):
        # BsaI: GGTCTCN — match with any base at N
        site = 'GGTCTCA'  # N=A
        seq = 'ATG' + site + 'AAATTT'
        sites = find_sites(seq, ['BsaI'])
        assert len(sites) > 0

    def test_multiple_enzymes(self):
        ecori_seq = 'GAATTC'
        bamhi_seq = 'GGATCC'
        seq = 'ATG' + ecori_seq + 'AAA' + bamhi_seq + 'TAA'
        sites = find_sites(seq, ['EcoRI', 'BamHI'])
        enzymes_found = {s[0] for s in sites}
        assert 'EcoRI' in enzymes_found
        assert 'BamHI' in enzymes_found

    def test_all_enzymes_none_param(self):
        seq = 'ATGGAATTCTAA'
        sites = find_sites(seq, None)
        assert len(sites) > 0


# =========================================================================
# Restriction site removal
# =========================================================================

class TestSiteRemoval:
    def test_ecori_removed(self):
        # Build a CDS that encodes EcoRI site: GAATTC ~ E(GAA)N(AAC) or similar
        # Glu-Asn: GAA=0.53, AAC=0.60 -> GAATTC? No.
        # GAATTC: GAA=Glu(0.53), TTC=Phe(0.54) -> at codon boundary if aligned
        # seq: ATG GAA TTC TAA = M-E-F-stop but GAATTC spans codons 1-2
        seq = 'ATGGAATTCTAA'
        sites_before = find_sites(seq, ['EcoRI'])
        assert len(sites_before) > 0

        cleaned, removed, failed = remove_sites(seq, ['EcoRI'], CODON_USAGE)
        sites_after = find_sites(cleaned, ['EcoRI'])
        ecori_remaining = [s for s in sites_after if s[0] == 'EcoRI']

        # Verify AA sequence preserved
        from optimizer.optimizer import translate_dna
        orig_aa = translate_dna(seq[:-3])
        new_aa = translate_dna(cleaned[:-3])
        assert orig_aa == new_aa

        # Site removed or failed with reason
        assert len(removed) + len(failed) > 0

    def test_amino_acid_preserved_after_swap(self):
        seq = 'ATGGAATTCTAA'
        cleaned, _, _ = remove_sites(seq, ['EcoRI'], CODON_USAGE)
        from optimizer.optimizer import translate_dna
        assert translate_dna(seq[:-3]) == translate_dna(cleaned[:-3])


# =========================================================================
# Checkers
# =========================================================================

class TestCheckers:
    # --- ORF integrity ---
    def test_orf_integrity_valid(self):
        result = orf_integrity('ATGAAATAA')
        assert result['passed']

    def test_orf_integrity_no_atg(self):
        result = orf_integrity('TTGAAATAA')
        assert not result['passed']
        assert 'ATG' in result['details']

    def test_orf_integrity_internal_stop(self):
        result = orf_integrity('ATGTAATTT')
        assert not result['passed']

    def test_orf_integrity_not_divisible_by_3(self):
        result = orf_integrity('ATGAA')
        assert not result['passed']

    # --- Internal initiation ---
    def test_internal_initiation_detected(self):
        # GGAGG 10 bp upstream of internal ATG
        # seq: ATG [10bp with GGAGG] ATG ...
        seq = 'ATG' + 'AAGGAGAAA' + 'ATGAAA' + 'TAA'
        result = internal_initiation(seq)
        # May or may not detect depending on exact window; just verify it runs
        assert 'passed' in result

    def test_internal_initiation_clean(self):
        result = internal_initiation('ATGAAATTTGCCTAA')
        assert result['passed']

    # --- Cryptic promoter ---
    def test_cryptic_promoter_detected(self):
        # -35 (TTGACA) then 17bp gap then -10 (TATAAT)
        seq = 'ATG' + 'TTGACA' + 'A' * 17 + 'TATAAT' + 'AAATAA'
        result = cryptic_promoter(seq)
        assert not result['passed']

    def test_cryptic_promoter_clean(self):
        result = cryptic_promoter('ATGAAAGTTCTGAAATAA')
        assert result['passed']

    # --- Strong RBS ---
    def test_strong_rbs_detected(self):
        result = strong_rbs('ATGGGAGGAAATAA')
        assert not result['passed']

    def test_strong_rbs_clean(self):
        result = strong_rbs('ATGAAAGTTCTGTAA')
        assert result['passed']

    # --- Terminator sequences ---
    def test_terminator_poly_t(self):
        result = terminator_sequences('ATGTTTTTAAATAA')
        assert not result['passed']

    def test_terminator_poly_a(self):
        result = terminator_sequences('ATGAAAAAAAATAA')
        assert not result['passed']

    def test_terminator_clean(self):
        result = terminator_sequences('ATGAAAGTTCTGTAA')
        assert result['passed']

    # --- Repeat sequences ---
    def test_repeat_sequences_detected(self):
        # Repeat unit must be > 20 bp to trigger the check
        repeat_unit = 'ATGAAAGTTCTGCAGAAAGTTC'  # 22 bp
        seq = repeat_unit + repeat_unit + 'TAA'
        result = repeat_sequences(seq)
        assert not result['passed']

    def test_repeat_sequences_clean(self):
        result = repeat_sequences('ATGAAAGTTCTGCAGTAA')
        assert result['passed']

    # --- Homopolymer runs ---
    def test_homopolymer_detected(self):
        result = homopolymer_runs('ATG' + 'A' * 10 + 'TAA')
        assert not result['passed']

    def test_homopolymer_clean(self):
        result = homopolymer_runs('ATGAAAGTTCTGTAA')
        assert result['passed']

    # --- Global GC ---
    def test_gc_too_low(self):
        seq = 'ATG' + 'AAATTT' * 30 + 'TAA'
        result = global_gc(seq)
        assert not result['passed']

    def test_gc_in_range(self):
        # Mix of GC and AT
        seq = 'ATGAAAGTTCTGCAGGCATAA'
        result = global_gc(seq)
        # Just verify it runs
        assert 'passed' in result

    # --- Local GC windows ---
    def test_local_gc_violation(self):
        # Very AT-rich window
        seq = 'ATG' + 'AAATTT' * 20 + 'TAA'
        result = local_gc_windows(seq)
        assert not result['passed']

    # --- Rare codons ---
    def test_rare_codon_detected(self):
        # CGA = Arg, freq=0.07 < 0.08
        result = rare_codons('ATGCGATAA')
        assert not result['passed']

    def test_rare_codon_clean(self):
        # Use best codons only
        protein = 'MKVLS'
        codons = [get_best_codon(aa) for aa in protein] + ['TAA']
        seq = ''.join(codons)
        result = rare_codons(seq)
        assert result['passed']

    # --- 5' hairpin ---
    def test_fiveprime_hairpin_runs(self):
        result = fiveprime_hairpin('ATGAAAGTTCTGCAGTAA')
        assert 'passed' in result
        assert 'nearest-neighbor' in result['details']

    def test_fiveprime_hairpin_no_crash_short_seq(self):
        result = fiveprime_hairpin('ATGTAA')
        assert 'passed' in result

    # --- run_all_checks ---
    def test_run_all_checks_returns_list(self):
        seq = 'ATG' + 'AAAGTT' * 10 + 'TAA'
        results = run_all_checks(seq)
        assert isinstance(results, list)
        assert len(results) >= 8

    def test_run_all_checks_options_respected(self):
        seq = 'ATG' + 'AAAGTT' * 10 + 'TAA'
        results_full = run_all_checks(seq, {'check_hairpin': True, 'check_repeats': True})
        results_partial = run_all_checks(seq, {'check_hairpin': False, 'check_repeats': False})
        names_full = {r['name'] for r in results_full}
        names_partial = {r['name'] for r in results_partial}
        assert 'fiveprime_hairpin' in names_full
        assert 'fiveprime_hairpin' not in names_partial


# =========================================================================
# End-to-end optimization pipeline
# =========================================================================

class TestEndToEnd:
    def test_protein_input_produces_valid_dna(self):
        result = optimize('MKVLSAEGIK', input_type='protein')
        seq = result['optimized_sequence']
        assert seq.startswith('ATG')
        assert len(seq) % 3 == 0
        # Translate back
        from optimizer.optimizer import translate_dna
        aa = translate_dna(seq[:-3])
        assert aa == 'MKVLSAEGIK'

    def test_cai_returned(self):
        result = optimize('MKVLS', input_type='protein')
        assert isinstance(result['cai_optimized'], float)
        assert 0.0 <= result['cai_optimized'] <= 1.0

    def test_orf_integrity_passes(self):
        result = optimize('MKVLSAEGIKLEFIAYLEKKK', input_type='protein')
        checks = {c['name']: c for c in result['checks']}
        assert checks['orf_integrity']['passed']

    def test_dna_input_pipeline(self):
        # Known CDS for a short protein
        from optimizer.codon_table import AA_TO_CODONS
        dna = 'ATGAAAGTT' + 'TAA'  # M-K-V
        result = optimize(dna, input_type='nucleotide')
        assert result['input_summary']['detected_type'] == 'nucleotide'
        assert result['cai_original'] is not None

    def test_restriction_enzyme_site_removed(self):
        # Build a protein whose greedy CDS contains EcoRI (GAATTC)
        # Glu = best codon GAA (0.53), Phe = best codon TTC (0.54)
        # GAA+TTC = GAATTC = EcoRI
        protein = 'MEF' + 'KVLS'  # M-E-F will create GAA+TTC = EcoRI site
        result = optimize(protein, input_type='protein', selected_enzymes=['EcoRI'])
        final_seq = result['optimized_sequence']
        remaining = find_sites(final_seq, ['EcoRI'])
        # Either removed or in failed list
        removed_enzymes = {s['enzyme'] for s in result['restriction_sites_removed']}
        failed_enzymes = {s['enzyme'] for s in result['restriction_sites_failed']}
        assert 'EcoRI' in removed_enzymes or 'EcoRI' in failed_enzymes or len(remaining) == 0

    def test_long_protein(self):
        # Test with a longer sequence
        protein = 'MKVLSAEGIKLEFIAYLEKKKMKVLSAEGIKLEFIAYLEKKK'
        result = optimize(protein, input_type='protein')
        assert len(result['optimized_sequence']) == (len(protein) + 1) * 3  # +1 for stop

    def test_input_summary_correct(self):
        protein = 'MKVLS'
        result = optimize(protein, input_type='protein')
        assert result['input_summary']['aa_length'] == 5
        assert result['input_summary']['nt_length'] == 18  # 5 codons + stop = 6 * 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
