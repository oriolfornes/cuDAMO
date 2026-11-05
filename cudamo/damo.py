"""DAMO (Discriminative Additive Model Optimization) in PyTorch.

Level 1 of DAMO.py (https://github.com/sx-ruan/DAMO), which is DiMO
(Patel and Stormo, 2014): a single PWM is refined so that the best site in
each sequence separates positives from negatives. The motif can also be
optimized at several widths, keeping the one that does best on held-out
sequences.
"""

import copy
from collections import Counter

import numpy as np
import torch
from sklearn import metrics
from tqdm import tqdm

from cudamo.io import read_pwm, read_sequences
from cudamo.sequence import dna_one_hot_many

bar_format = "{percentage:3.0f}%|{bar:20}{r_bar}"


# Motif width search: widths tried when none are given, how much held-out AUC
# a wider motif may lose against the best one before it is passed over, and
# the fraction of sequences held out
DEFAULT_WIDTHS = (8, 10, 12, 15, 17, 19)
WIDTH_TOLERANCE = 0.002
VALIDATION_FRACTION = 0.2
LEARNING_RATES = [1.0, 0.55, 0.1]


def damo(positive_seqs_file, negative_seqs_file, jaspar_profile_file,
         debugging=False, iterations=500, motif_name=None,
         widths=None, seed=1714):
    """
    Optimize a PWM to separate positive from negative sequences.

    Every sequence is scanned on both strands. Each iteration moves the
    PWM towards the sites of positives in the overlap zone and away from
    the sites of negatives. It stops early when the AUC stops improving,
    or when the classes are perfectly separated.

    With widths, the motif is optimized at each width and the best one is
    kept. 20% of the sequences of each class are held out. For each width
    the seed is cut to its most informative columns, or padded with
    uniform ones, optimized on the other 80%, and scored on the held-out
    sequences. The seed's own width is always tried. The smallest width
    within 0.002 of the best held-out AUC wins.

    Args:
        positive_seqs_file: FASTA file of positive sequences.
        negative_seqs_file: FASTA file of negative sequences.
        jaspar_profile_file: JASPAR or MEME motif used as the seed PWM.
        debugging: If True, use only the first 1,000 sequences per file.
        iterations: Maximum number of optimization iterations.
        motif_name: Motif to use when a MEME file has several.
        widths: Motif widths to try (None: only the seed's width, with no
            held-out set). Widths longer than the sequences are skipped.
        seed: Random seed of the held-out split (widths only).

    Returns:
        Log PWM of shape (4, motif length), rows A, C, G and T. The
        motif length is the seed's unless widths are given.
    """

    device = _get_device()

    # Get sequences, and the seed PWM
    positive_records = read_sequences(positive_seqs_file, debugging)
    negative_records = read_sequences(negative_seqs_file, debugging)
    pwm = read_pwm(jaspar_profile_file, motif_name)

    if widths is None:
        positives = _encode(positive_records, device)
        negatives = _encode(negative_records, device)
        pwm, original_auc, auc = _optimize(pwm, positives, negatives,
            iterations, device)
        print("\n" + f"AUC optimized from {original_auc} to {auc}." + "\n")

        return pwm

    # Hold out sequences to compare the widths
    positive_records, positive_val = _split(positive_records, seed)
    negative_records, negative_val = _split(negative_records, seed)
    positives = _encode(positive_records, device)
    negatives = _encode(negative_records, device)
    positives_val = _encode(positive_val, device)
    negatives_val = _encode(negative_val, device)

    # Widths to try, as far as the sequences allow
    sequence_length = positives[1].shape[-1]
    candidates = sorted({int(k) for k in widths} | {pwm.shape[1]})
    candidates = [k for k in candidates if 1 <= k <= sequence_length]
    if not candidates:
        raise ValueError(f"No width fits sequences of length {sequence_length}")

    # Optimize at each width
    pwms, aucs = {}, {}
    for k in candidates:
        pwms[k], _, _ = _optimize(_resize(pwm, k), positives, negatives,
            iterations, device, desc=f"Width {k}")
        aucs[k] = _auc(pwms[k], positives_val, negatives_val, device)
    best = _choose_width(aucs)

    seed_auc = _auc(pwm, positives_val, negatives_val, device)
    print("\n" + f"AUC on held-out sequences: seed {seed_auc}")
    for k in candidates:
        print(f"  width {k:>3}: {aucs[k]}" + ("  <- chosen" if k == best else ""))
    print()

    return pwms[best]


def _get_device():
    """The best available torch device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _encode(records, device):
    """
    Both strands of the records, one-hot encoded.

    Returns:
        Tuple (sequences, tensor): the forward strands first and the
        reverse complements after, as strings and as a one-hot tensor.
    """
    sequences = [str(s.seq).upper() for s in records] + \
        [str(s.reverse_complement().seq).upper() for s in records]
    one_hot = dna_one_hot_many(sequences)

    return sequences, torch.Tensor(one_hot.tolist()).to(device)


def _optimize(pwm, positives, negatives, iterations, device, desc=None):
    """
    Run the optimization loop on a seed PWM.

    Args:
        pwm: Seed log PWM of shape (4, motif length).
        positives: Tuple (sequences, tensor) from _encode.
        negatives: Tuple (sequences, tensor) from _encode.
        iterations: Maximum number of iterations.
        device: Torch device.
        desc: Label for the progress bar.

    Returns:
        Tuple (pwm, original_auc, auc): the optimized PWM, and the AUC
        of the seed and of the PWM returned.
    """
    positive_seqs, positive_seqs_tensor = positives
    negative_seqs, negative_seqs_tensor = negatives
    sequence_length = positive_seqs_tensor.shape[-1]

    # Get a PyTorch model of the PWM
    model = _get_pwm_model(pwm, sequence_length, device)

    # Score sequences
    positive_scores, positive_sites = _score_sequences(
        model, positive_seqs_tensor, positive_seqs, pwm.shape[1]
    )
    negative_scores, negative_sites = _score_sequences(
        model, negative_seqs_tensor, negative_seqs, pwm.shape[1],
    )

    # Compute AUC
    original_auc, y_true, y_score = _compute_auc(positive_scores,
        negative_scores)
    auc = copy.copy(original_auc)

    # For each iteration...
    for _ in tqdm(range(iterations), total=iterations, desc=desc,
                  bar_format=bar_format):

        # Initialize
        pwm_0 = copy.copy(pwm)
        auc_0 = copy.copy(auc)
        y_true_0 = copy.copy(y_true)
        y_score_0 = copy.copy(y_score)
        sites_0 = copy.copy(positive_sites) + copy.copy(negative_sites)

        # For each learning rate...
        for lr in LEARNING_RATES:

            # Update PWM
            pwm_1 = _update_pwm(y_true_0, y_score_0, sites_0, lr, pwm_0)

            # Perfect separation (i.e. no overlap zone): nothing to learn
            if pwm_1 is None:
                break

            # Get a PyTorch model of the updated PWM
            model = _get_pwm_model(pwm_1, sequence_length, device)

            # Score sequences
            positive_scores, positive_sites = _score_sequences(
                model, positive_seqs_tensor, positive_seqs, pwm_1.shape[1]
            )
            negative_scores, negative_sites = _score_sequences(
                model, negative_seqs_tensor, negative_seqs, pwm_1.shape[1],
            )

            # Compute AUC of the candidate
            auc_1, y_true_1, y_score_1 = _compute_auc(positive_scores,
                negative_scores)

            # Accept the candidate only if it improves the AUC. A rejected
            # candidate must not leak into auc, which is reported at the end.
            if auc_1 > auc_0:
                pwm = copy.copy(pwm_1)
                auc, y_true, y_score = auc_1, y_true_1, y_score_1
                break

        # Early stop
        if np.array_equal(pwm_0, pwm):
            break

    return pwm, original_auc, auc


def _auc(pwm, positives, negatives, device):
    """AUC of a PWM on encoded sequences (best site on either strand)."""
    sequence_length = positives[1].shape[-1]
    model = _get_pwm_model(pwm, sequence_length, device)
    positive_scores, _ = _score_sequences(model, positives[1], positives[0],
        pwm.shape[1])
    negative_scores, _ = _score_sequences(model, negatives[1], negatives[0],
        pwm.shape[1])

    return _compute_auc(positive_scores, negative_scores)[0]


def _split(records, seed, fraction=VALIDATION_FRACTION):
    """Randomly hold out a fraction of the records; returns (train, held out)."""
    n_val = max(1, int(round(len(records) * fraction)))
    if len(records) - n_val < 1:
        raise ValueError("Too few sequences to hold out a validation set")
    held_out = set(np.random.RandomState(seed).permutation(len(records))[:n_val].tolist())

    return ([r for i, r in enumerate(records) if i not in held_out],
            [records[i] for i in sorted(held_out)])


def _resize(pwm, width):
    """
    Cut a PWM down to its most informative columns, or pad it.

    A narrower width keeps the run of columns with the most information
    (ties go to the most central run). A wider one adds uniform columns,
    half on each side (the odd one on the right).

    Args:
        pwm: Log PWM of shape (4, motif length).
        width: Wanted number of columns.

    Returns:
        Log PWM of shape (4, width).
    """
    motif_length = pwm.shape[1]
    if width == motif_length:
        return pwm

    if width > motif_length:
        left = (width - motif_length) // 2
        right = width - motif_length - left
        return np.hstack([np.full((4, left), np.log(0.25)), pwm,
                          np.full((4, right), np.log(0.25))])

    # Information content of each column, in bits
    pfm = pwm_to_pfm(pwm)
    information = (pfm * np.log2(pfm / 0.25)).sum(axis=0)
    totals = np.convolve(information, np.ones(width), mode="valid")
    centre = (motif_length - width) / 2
    start = min(range(len(totals)), key=lambda i: (-round(totals[i], 9),
                                                    abs(i - centre)))

    return pwm[:, start:start + width]


def _choose_width(aucs, tolerance=WIDTH_TOLERANCE):
    """
    Pick a width from the held-out AUC of each.

    The smallest width within the tolerance of the best AUC wins, so extra
    columns have to pay for themselves.

    Args:
        aucs: Dict mapping width to held-out AUC.
        tolerance: Held-out AUC a narrower motif may lose against the best.

    Returns:
        The chosen width.
    """
    best = max(aucs.values())

    return min(k for k, auc in aucs.items() if auc >= best - tolerance - 1e-12)


def pwm_to_pfm(pwm, scale=1.0):
    """
    Convert a log PWM to a PFM with a Boltzmann distribution.

    As in DiMO (Patel and Stormo, 2014), the weights are read as energies
    in units of kT: each position is a softmax over A, C, G and T. A scale
    above 1 sharpens the columns and one below 1 flattens them.

    Args:
        pwm: Log PWM of shape (4, motif length), rows A, C, G and T.
        scale: Factor applied to the weights before the softmax.

    Returns:
        Frequencies of shape (4, motif length); each column sums to 1.
    """
    scaled = scale * np.asarray(pwm, float)
    z = np.exp(scaled - scaled.max(axis=0, keepdims=True))

    return z / z.sum(axis=0, keepdims=True)


def _get_pwm_model(pwm, sequence_length, device="cpu"):
    """
    Build a fixed PyTorch model that scores a PWM.

    A convolution with the PWM as its only filter, followed by a max pool
    over the whole sequence. The max pool returns the position of the best
    site too.

    Args:
        pwm: Log PWM of shape (4, motif length).
        sequence_length: Length of the (one-hot) sequences to score.
        device: Torch device to place the model on.

    Returns:
        torch.nn.Sequential returning (scores, site positions).
    """
    conv1d = torch.nn.Conv1d(4, 1, kernel_size=pwm.shape[1])
    conv1d.bias.data = torch.Tensor([0.])
    conv1d.weight.data = torch.tensor(np.array([pwm]), dtype=torch.float32)
    for p in conv1d.parameters():
        p.requires_grad = False
    maxpool1d = torch.nn.MaxPool1d(sequence_length - pwm.shape[1] + 1, 1,
        return_indices=True)

    return torch.nn.Sequential(conv1d, maxpool1d).to(device)


def _score_sequences(model, sequences_tensor, sequences, motif_length):
    """
    Score each sequence by its best site on either strand.

    Args:
        model: Model from _get_pwm_model.
        sequences_tensor: One-hot sequences, forward strands first and
            reverse complements after (as built in damo).
        sequences: Sequence strings in the same order as the tensor.
        motif_length: Length of the PWM.

    Returns:
        Tuple (scores, sites): best score per sequence, and the matching
        site string.
    """
    # Get predictions
    with torch.no_grad():
        scores_tmp, sites_tmp = model(sequences_tensor)
    scores_tmp = scores_tmp.flatten()
    sites_tmp = sites_tmp.flatten()

    # Forward strands come first, reverse complements after. Ties go to the
    # reverse strand, as in DAMO.
    half = len(sequences) // 2
    use_fwd = scores_tmp[:half] > scores_tmp[half:]
    scores = torch.where(use_fwd, scores_tmp[:half], scores_tmp[half:]).tolist()
    positions = torch.where(use_fwd, sites_tmp[:half], sites_tmp[half:]).tolist()
    use_fwd = use_fwd.tolist()

    # Get the site strings
    sites = []
    for i in range(half):
        seq = sequences[i] if use_fwd[i] else sequences[half + i]
        sites.append(seq[positions[i]:positions[i] + motif_length])

    return scores, sites


def _compute_auc(positive_scores, negative_scores):
    """
    ROC AUC of the scores, with positives labelled 1.

    Returns:
        Tuple (auc, labels, scores), where labels and scores are the
        concatenated inputs.
    """
    y_true = [1] * len(positive_scores) + [0] * len(negative_scores)
    y_score = positive_scores + negative_scores

    return metrics.roc_auc_score(y_true, y_score), y_true, y_score


def _update_pwm(y_true, y_score, sites, lr, pwm):
    """
    One update step, as DAMO's makeAMove.

    Sites that rank between the first negative and the last positive (the
    overlap zone) are turned into a positive and a negative PFM. The PWM
    moves by lr times their log-odds difference.

    Args:
        y_true: Labels, 1 for positive and 0 for negative.
        y_score: Scores, same order as y_true.
        sites: Best site string per sequence, same order.
        lr: Learning rate.
        pwm: Current log PWM of shape (4, motif length).

    Returns:
        Updated log PWM, or None if the classes are perfectly separated.
    """
    positive_mix = []
    negative_mix = []

    # Sort and zip sequences and labels
    idxs = np.argsort(-np.asarray(y_score), kind="mergesort")
    sequences = [sites[ix] for ix in idxs]
    labels = [y_true[ix] for ix in idxs]
    z = list(zip(sequences, labels))

    # Get index of first negative and last positive
    ix_first_negative = labels.index(0)
    ix_last_positive = len(labels) - labels[::-1].index(1)

    # Perfect separation: no positive ranks below a negative
    if ix_first_negative >= ix_last_positive:
        return None

    # For each label, sequence...
    for s, l in z[ix_first_negative:ix_last_positive]:
        positive_mix.append(s) if l == 1 else negative_mix.append(s)

    positive_pwm = _generate_pwm(positive_mix, pwm.shape[1])
    negative_pwm = _generate_pwm(negative_mix, pwm.shape[1])

    return pwm + lr * (positive_pwm - negative_pwm)


def _generate_pwm(sequences, motif_length):
    """
    Log PWM from a set of aligned sites, as DAMO's gen_pwm.

    Counts are turned into frequencies, then 1e-4 is added and the result
    is renormalized. Without that, an absent base would get
    log(1e-4 / n) and make the steps too large.

    Args:
        sequences: Sites of equal length.
        motif_length: Length of the sites.

    Returns:
        Array of shape (4, motif length), rows A, C, G and T.
    """
    pfm = np.zeros((motif_length, 4))
    for i in range(motif_length):
        c = Counter(s[i:i+1] for s in sequences)
        pfm[i, :] = [c[key] for key in "ACGT"]
    pfm = _normalize_pfm(_normalize_pfm(pfm), 1e-4)

    return np.log(pfm).T


def _normalize_pfm(pfm, pseudocounts=0.):
    """Add pseudocounts to each row and divide it by its sum."""
    pfm = np.array(pfm, float) + pseudocounts

    return pfm / np.sum(pfm, axis=1, keepdims=True)
