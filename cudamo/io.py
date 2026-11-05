"""Reading and writing the files cuDAMO uses."""

import gzip
import os
import shutil
import tempfile

import numpy as np
from Bio import SeqIO, motifs


def open_file(file_name, mode="rt"):
    """Open a file, decompressing it on the fly if it ends in .gz."""
    if file_name.endswith(".gz"):
        return gzip.open(file_name, mode if "t" in mode or "b" in mode
                         else mode + "t")
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


def read_pwm(jaspar_profile, motif_name=None):
    """
    Read a JASPAR or MEME motif as a log PWM.

    The format is detected from the content: a file that starts with
    "MEME" is read as MEME (as ExplaiNN-2 does, with memelite), anything
    else as JASPAR.

    JASPAR matrices are counts, and get the pseudocounts Biopython derives
    from them. MEME matrices are probabilities, and get 1e-4 added before
    renormalizing, as DAMO does for the PFMs it builds.

    Args:
        jaspar_profile: Path to a JASPAR or MEME file (plain or gzipped).
        motif_name: Which motif to read from a MEME file with several. Not
            needed when the file has only one.

    Returns:
        Array of shape (4, motif length), rows A, C, G and T.
    """
    if _is_meme(jaspar_profile):
        return _read_meme_pwm(jaspar_profile, motif_name)

    handle = open_file(jaspar_profile)
    m = motifs.read(handle, "jaspar")
    handle.close()
    m.pseudocounts = motifs.jaspar.calculate_pseudocounts(m)

    pwm = []
    for nucleotide in "ACGT":
        pwm.append([i for i in m.pwm[nucleotide]])

    return np.log(pwm)


def _is_meme(file_name):
    """True if the first non-empty line of the file starts with MEME."""
    with open_file(file_name) as handle:
        for line in handle:
            if line.strip():
                return line.startswith("MEME")
    return False


def _read_meme_pwm(meme_file, motif_name=None):
    """Read one motif of a MEME file as a log PWM (see read_pwm)."""
    from memelite.io import read_meme

    # memelite opens plain text only
    with tempfile.TemporaryDirectory() as tmp:
        path = meme_file
        if meme_file.endswith(".gz"):
            path = os.path.join(tmp, "motifs.meme")
            with open_file(meme_file) as src, open(path, "w") as dst:
                shutil.copyfileobj(src, dst)
        ppms = read_meme(path)

    if not ppms:
        raise ValueError(f"No motifs found in {meme_file}")
    if motif_name is None:
        if len(ppms) > 1:
            raise ValueError(
                f"{meme_file} has {len(ppms)} motifs; choose one with "
                f"--motif. Names: {', '.join(ppms)}"
            )
        motif_name = next(iter(ppms))
    elif motif_name not in ppms:
        # Accept the first word of the name too, e.g. MA1419.1 for
        # "MA1419.1 IRF4"
        ids = [n for n in ppms if n.split()[0] == motif_name]
        if len(ids) != 1:
            raise ValueError(
                f"Motif {motif_name!r} not found in {meme_file}. "
                f"Names: {', '.join(ppms)}"
            )
        motif_name = ids[0]

    ppm = np.asarray(ppms[motif_name], float) + 1e-4

    return np.log(ppm / ppm.sum(axis=0, keepdims=True))


def write_pfm(pfm, handle, name="cudamo"):
    """Write a (4, motif length) PFM of frequencies in JASPAR format."""
    handle.write(f">{name}\n")
    for nucleotide, row in zip("ACGT", pfm):
        values = " ".join(f"{r:.6f}" for r in row.tolist())
        handle.write(f"{nucleotide} [ {values} ]\n")


def write_pwm(pwm, handle):
    """Write a (4, motif length) PWM, one position per line."""
    for row in pwm.T:
        s = " ".join([str(round(r, 9)).rjust(15) for r in row.tolist()]) + "\n"
        handle.write(s)
