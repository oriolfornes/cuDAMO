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

Inputs are a FASTA file of positives, a FASTA file of negatives and a JASPAR matrix to start from. Plain or gzipped files both work.

Options:

- `-i, --iterations`: maximum number of iterations (default 500).
- `-o, --output-file`: where to write the PWM. Without it, the PWM goes to stdout.
- `-d, --debugging`: use only the first 1,000 sequences of each file.

The output is the log PWM, one line per motif position with four columns (A, C, G, T). The `AUC optimized from ... to ...` line is printed to stdout too, so use `-o` to keep the matrix on its own.

To try it on the bundled example:

```bash
pixi run example        # writes results/MA1419.1.pwm.txt
```

## Python API

```python
from cudamo.damo import damo

pwm = damo("positive_seqs.fa.gz", "negative_seqs.fa.gz", "motif.jaspar", iterations=500)
```

`pwm` is a NumPy array of shape (4, motif length).

## Differences from DAMO

- Only level 1. The adjacent-dinucleotide terms (`-i` in DAMO.py) are not implemented.
- All sequences in an input file must have the same length. Windows that overlap an `N` are scored with the `N` as zeros, not skipped.
- The seed PWM gets Biopython's JASPAR pseudocounts, not DAMO's.

## Tests

```bash
pixi run test
```

The one-hot encoding in `cudamo/sequence.py` is copied from ExplaiNN, so both packages encode sequences the same way.
