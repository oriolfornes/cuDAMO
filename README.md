cuDAMO
=====
A PyTorch-based (*i.e.* cuda) optimization of DAMO (https://github.com/sx-ruan/DAMO).

Dependencies
-----
cuDAMO has the following Python dependencies.

| Package | Version |
|-|-|
| [Biopython](https://biopython.org/) | 1.79 |
| [Click](https://click.palletsprojects.com/en/8.0.x/) | 7.1.2 |
| [NumPy](https://numpy.org/) | 1.21.2 |
| [scikit-learn](https://scikit-learn.org/) | 0.24.2 |
| [Python](https://www.python.org/) | 3.9.6 |
| [PyTorch](https://pytorch.org/) | 1.9.0 |
| [tqdm](https://github.com/tqdm/tqdm) | 4.62.1 |

Fidelity to DAMO
-----
Checked against `DAMO.py` (level 1). Fixed:

- **Pseudocounts.** DAMO turns counts into frequencies, then adds 1e-4 and renormalizes. cuDAMO added 1e-4 to the raw counts, so an absent base got log(1e-4/n) (e.g. -15.4 instead of -9.2 at n = 500), which made the steps much larger.
- **`--iterations`.** The option was passed positionally into `threads`, so every run used the 500-iteration default.
- **Perfect separation.** When every positive outscores every negative, the overlap zone is empty. cuDAMO built PWMs from no sites (NaN) and crashed; it now stops, as there is nothing left to learn.

Known deviations (not fixed):

- **No level 2.** DAMO's adjacent-dinucleotide terms (`-i` in DAMO) are not implemented.
- **Fixed sequence length.** The max pool assumes that all sequences have the same length. Variable-length input is not supported, and windows that overlap an `N` are scored (an `N` contributes 0) rather than skipped.
- **Seed PWM.** The JASPAR seed gets Biopython's JASPAR pseudocounts, not DAMO's.

Tests: `python -m pytest tests/ -q`.
