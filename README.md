# cuDAMO

A PyTorch version of [DAMO](https://github.com/sx-ruan/DAMO). It refines a JASPAR motif so that its best sites separate positive sequences from negative ones. It runs on the GPU when one is available.

## Install

With [pixi](https://pixi.sh):

```bash
pixi install            # CUDA build of PyTorch
pixi install -e cpu     # CPU only
```

Or with pip:

```bash
pip install -e .
```

## Usage

```bash
pixi run cudamo positive_seqs.fa.gz negative_seqs.fa.gz motif.jaspar -o motif.pwm.txt
```

Inputs are a FASTA file of positives, a FASTA file of negatives and a motif to start from, as a JASPAR matrix or a MEME file (detected from the content). Plain or gzipped files both work.

Options:

- `-i, --iterations`: maximum number of iterations (default 500).
- `-o, --output-file`: where to write the PWM. Without it, the PWM goes to stdout.
- `-d, --debugging`: use only the first 1,000 sequences of each file.
- `-w, --optimize-width`: also optimize the motif width, as in DiMO (see below).
- `-m, --motif`: which motif to use when a MEME file has several, by name or by ID (e.g. `MA1419.1`). Not needed for single-motif files. MEME matrices are probabilities, so 1e-4 is added and the columns renormalized before taking the log, as DAMO does for the PFMs it builds.
- `--pfm`: also write a PFM (JASPAR format, frequencies) to this file. The PWM weights are read as energies in kT and converted with a Boltzmann distribution (a softmax over A, C, G and T at each position), as in [DiMO](https://pubmed.ncbi.nlm.nih.gov/24369152/).
- `--scale`: factor applied to the PWM before that conversion (default 1). Above 1 sharpens the columns, below 1 flattens them. The weights are discriminative scores, not probabilities, so the PFM at scale 1 is not guaranteed to look like a typical binding-site PFM.

The output is the log PWM, one line per motif position with four columns (A, C, G, T). The `AUC optimized from ... to ...` line is printed to stdout too, so use `-o` to keep the matrix on its own.

To try it on the bundled example:

```bash
pixi run example        # writes results/MA1419.1.pwm.txt
```

## Width optimization

`-w` follows section 2.6 of the [DiMO paper](https://pubmed.ncbi.nlm.nih.gov/24369152/):

1. 20% of the sequences of each class are held out (fixed seed).
2. The seed is optimized on the other 80%, padded with uniform columns on both sides to 19 or 20 columns (fewer if the sequences are short), and optimized again.
3. The result is cut to its central 6 columns. Columns are then restored one at a time, 5' side first and then the 3' side, as long as each one raises the AUC on the held-out sequences by at least 0.003. Restoring stops on a side at the first column that does not.

The output is the chosen columns of the extended motif; they are not optimized again. The AUC message reports the held-out AUC of the seed, of the motif optimized at the seed width, and of the final motif, so you can see what the width choice cost or gained.

This works best for motifs whose information sits in a compact core. It can pick a narrower motif that scores worse than the seed width when the signal is spread over many columns or split in half-sites (the IRF4 example loses AUC with `-w`). The paper's penalty is deliberately strict, to avoid wide motifs on small data. Compare the held-out AUCs it reports before using the result.

Differences from the paper: the first optimization (step 2) also uses only the 80%, and "at least 0.3%" is read as 0.003 of AUC.

## Python API

```python
from cudamo.damo import damo

pwm = damo("positive_seqs.fa.gz", "negative_seqs.fa.gz", "motif.jaspar", iterations=500)
```

`pwm` is a NumPy array of shape (4, motif length).

## Differences from DAMO

- Only level 1, which is DiMO. The adjacent-dinucleotide terms (`-i` in DAMO.py) are not implemented.
- All sequences in an input file must have the same length. Windows that overlap an `N` are scored with the `N` as zeros, not skipped.
- The seed PWM gets Biopython's JASPAR pseudocounts, not DAMO's.

## Tests

```bash
pixi run test
```

The one-hot encoding in `cudamo/sequence.py` is copied from ExplaiNN, so both packages encode sequences the same way.
