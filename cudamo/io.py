"""Reading and writing the files cuDAMO uses."""

import gzip

import numpy as np
from Bio import SeqIO, motifs


def open_file(file_name, mode="rt"):
    """Open a file, decompressing it on the fly if it ends in .gz."""
    if file_name.endswith(".gz"):
        return gzip.open(file_name, mode)
    return open(file_name, mode)


def read_sequences(fasta_file, debugging=False):
    """
    Read the records of a FASTA file.

    Args:
        fasta_file: Path to a FASTA file (plain or gzipped).
        debugging: If True, keep only the first 1,000 records.

    Returns:
        List of Bio.SeqRecord.
    """
    handle = open_file(fasta_file)
    sequences = list(SeqIO.parse(handle, "fasta"))
    handle.close()

    if debugging:
        return sequences[:1000]

    return sequences


def read_pwm(jaspar_profile):
    """
    Read a JASPAR matrix as a log PWM.

    Pseudocounts are the ones Biopython derives from the matrix.

    Args:
        jaspar_profile: Path to a JASPAR matrix file (plain or gzipped).

    Returns:
        Array of shape (4, motif length), rows A, C, G and T.
    """
    handle = open_file(jaspar_profile)
    m = motifs.read(handle, "jaspar")
    handle.close()
    m.pseudocounts = motifs.jaspar.calculate_pseudocounts(m)

    pwm = []
    for nucleotide in "ACGT":
        pwm.append([i for i in m.pwm[nucleotide]])

    return np.log(pwm)


def write_pwm(pwm, handle):
    """Write a (4, motif length) PWM, one position per line."""
    for row in pwm.T:
        s = " ".join([str(round(r, 9)).rjust(15) for r in row.tolist()]) + "\n"
        handle.write(s)
