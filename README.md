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
- `-w, --optimize-width`: also optimize the motif width: try several widths and keep the best on held-out sequences (see below).
- `--widths`: comma-separated widths to try; implies `-w`.
- `-m, --motif`: which motif to use when a MEME file has several, by name or by ID (e.g. `MA1419.1`). Not needed for single-motif files. MEME matrices are probabilities, so 1e-4 is added and the columns renormalized before taking the log, as DAMO does for the PFMs it builds.
- `--pfm`: also write a PFM (JASPAR format, frequencies) to this file. The PWM weights are read as energies in kT and converted with a Boltzmann distribution (a softmax over A, C, G and T at each position), as in [DiMO](https://pubmed.ncbi.nlm.nih.gov/24369152/).
- `--scale`: factor applied to the PWM before that conversion (default 1). Above 1 sharpens the columns, below 1 flattens them. The weights are discriminative scores, not probabilities, so the PFM at scale 1 is not guaranteed to look like a typical binding-site PFM.

The output is the log PWM, one line per motif position with four columns (A, C, G, T). The `AUC optimized from ... to ...` line is printed to stdout too, so use `-o` to keep the matrix on its own.

To try it on the bundled example:

```bash
pixi run example        # writes results/MA1419.1.pwm.txt
```

## Motif width

By default the motif keeps the width of the seed. `-w` tries several widths and keeps the best one:

```bash
pixi run cudamo pos.fa.gz neg.fa.gz motif.jaspar -w -o motif.pwm.txt
pixi run cudamo pos.fa.gz neg.fa.gz motif.jaspar --widths 6,9,12 -o motif.pwm.txt
```

- 20% of the sequences of each class are held out (fixed seed).
- For each width the seed is cut to its most informative run of columns, or padded with uniform columns (half on each side), and optimized on the other 80%. The seed's own width is always tried too. Widths longer than the sequences are skipped.
- Each result is scored on the held-out sequences, and the smallest width within 0.002 AUC of the best one wins, so extra columns have to pay for themselves.
- The widths and their held-out AUCs are printed, with the chosen one marked. `--widths` sets the list (default 8, 10, 12, 15, 17, 19) and implies `-w`.

This is not DiMO's own width procedure (section 2.6 of the [paper](https://pubmed.ncbi.nlm.nih.gov/24369152/)), which pads the motif to about 20 columns, cuts it to the central 6 and restores columns one at a time. On the IRF4 example that procedure picked 8 columns and lost held-out AUC (0.668 against 0.712 at the seed width), because the signal is spread over many columns and it stops at the first column that does not pay.

The returned motif is optimized on the 80% only, and its held-out AUC was also used to pick the width, so that number is slightly optimistic. Held-out gains are modest: on 154 IBIS motifs with 19 bp windows, the chosen width averaged 13.6 columns and scored about the same as the full 19 on a separate test set (+0.002, within noise), while 8 columns lost 0.02.

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
