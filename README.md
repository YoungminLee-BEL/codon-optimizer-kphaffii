# K. phaffii Codon Optimizer

A Python-based codon optimization tool for *Komagataella phaffii* (Pichia pastoris) GS115 with a Flask web interface.

## Features

- **Codon optimization** using K. phaffii GS115 codon usage frequencies (Kazusa DB)
- **CAI calculation** using the Sharp & Li (1987) method
- **Restriction enzyme site avoidance** — synonymous codon swaps to eliminate sites (Hahn Lab set, 38 enzymes)
- **Comprehensive quality checks:**

### ORF & Expression Signals
| Check | Description |
|---|---|
| ORF Integrity | No internal stops, starts with ATG, divisible by 3 |
| Internal Initiation | Detects RBS-ATG pairs that could cause internal translation initiation |
| Cryptic Promoter | Finds −35/−10 box pairs resembling prokaryotic promoters |
| Strong RBS | Detects Shine-Dalgarno-like sequences (GGAGG, TAAGGAG) |
| Terminator Sequences | Flags poly-A/T runs ≥5 bp |

### Manufacturability (Twist Bioscience guidelines)
| Check | Description |
|---|---|
| Repeat Sequences | Detects repeated subsequences > 20 bp |
| Homopolymer Runs | Flags runs of identical bases ≥ 10 bp |
| Global GC Content | Must be 25–65% |
| Local GC Windows | Sliding 50 bp window; each window must be 35–65% GC |

### Expression Quality
| Check | Description |
|---|---|
| Rare Codons | Flags codons with relative usage < 8% in K. phaffii |
| 5′ Hairpin | Pure-Python nearest-neighbor ΔG estimation for first 48 bp |

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10+ required.

## Usage

```bash
python web/app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

### API

**POST `/optimize`**

```json
{
  "sequence": "MKVLSAEGIK...",
  "input_type": "auto",
  "selected_enzymes": ["EcoRI", "BamHI"],
  "options": {
    "avoid_rare_codons": true,
    "check_hairpin": true,
    "check_repeats": true
  }
}
```

## Notes

- **5′ hairpin check** uses a built-in pure-Python nearest-neighbor ΔG approximation (no ViennaRNA required). For precise thermodynamic values, use [ViennaRNA](https://www.tbi.univie.ac.at/RNA/).
- **Type IIS enzymes** (BpiI, BsaI) are flagged with special warnings — removal via synonymous codon swaps may not always be possible.
- Input can be protein (single-letter AA codes), DNA, or RNA. FASTA format is accepted; the header line is automatically stripped.

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
codon-optimizer-kphaffii/
├── optimizer/
│   ├── codon_table.py   # K. phaffii GS115 codon usage + genetic code
│   ├── optimizer.py     # Core optimization pipeline
│   ├── cai.py           # CAI calculation (Sharp & Li 1987)
│   ├── checkers.py      # All 11 quality checks
│   └── restriction.py   # Restriction enzyme detection & removal
├── web/
│   ├── app.py           # Flask web server
│   └── templates/
│       └── index.html   # Single-page UI (self-contained HTML/CSS/JS)
└── tests/
    └── test_optimizer.py
```

## References

- Sharp PM, Li WH (1987). "The codon adaptation index — a measure of directional synonymous codon usage bias, and its potential applications." *Nucleic Acids Research* 15(3):1281–95. doi:10.1093/nar/15.3.1281
- Lorenz R et al. (2011). "ViennaRNA Package 2.0." *Algorithms for Molecular Biology* 6:26. doi:10.1186/1748-7188-6-26
- Twist Bioscience Gene Synthesis Guidelines (https://www.twistbioscience.com)
- Kazusa Codon Usage Database — *K. phaffii* GS115: https://www.kazusa.or.jp/codon/

## License

MIT License
