"""One-hot encoding of DNA sequences.

Copied from ExplaiNN (explainn/utils/tools.py) so both packages encode
sequences the same way.
"""

import numpy as np


def dna_one_hot(seq, seq_len=None, flatten=False):
    """
    One-hot encode a DNA sequence.

    Alphabet: A=0, C=1, G=2, T=3. Non-ACGT characters are encoded as
    all zeros.

    Args:
        seq: DNA string.
        seq_len: Target length (trims or pads if different from len(seq)).
        flatten: If True, return a 1D vector.

    Returns:
        Array of shape (4, seq_len).
    """
    if seq_len is None:
        seq_len = len(seq)
        seq_start = 0
    else:
        if seq_len <= len(seq):
            seq_trim = (len(seq) - seq_len) // 2
            seq = seq[seq_trim:seq_trim + seq_len]
            seq_start = 0
        else:
            seq_start = (seq_len - len(seq)) // 2

    seq = seq.upper().replace("A", "0").replace("C", "1").replace("G", "2").replace("T", "3")

    seq_code = np.zeros((4, seq_len), dtype="float16")
    for i in range(seq_len):
        if i >= seq_start:
            try:
                seq_code[int(seq[i - seq_start]), i] = 1.0
            except (ValueError, IndexError):
                pass  # non-standard nucleotide → all zeros

    if flatten:
        seq_code = seq_code.flatten()[None, :]

    return seq_code


def dna_one_hot_many(seqs):
    """One-hot encode a list of DNA sequences."""
    return np.array([dna_one_hot(str(seq)) for seq in seqs])
