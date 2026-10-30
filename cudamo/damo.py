"""DAMO (Discriminative Additive Model Optimization) in PyTorch.

Level 1 of DAMO.py (https://github.com/sx-ruan/DAMO): a single PWM is
refined so that the best site in each sequence separates positives
from negatives.
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


def damo(positive_seqs_file, negative_seqs_file, jaspar_profile_file,
         debugging=False, iterations=500):
    """
    Optimize a PWM to separate positive from negative sequences.

    Every sequence is scanned on both strands. Each iteration moves the
    PWM towards the sites of positives in the overlap zone and away from
    the sites of negatives. It stops early when the AUC stops improving,
    or when the classes are perfectly separated.

    Args:
        positive_seqs_file: FASTA file of positive sequences.
        negative_seqs_file: FASTA file of negative sequences.
        jaspar_profile_file: JASPAR matrix used as the seed PWM.
        debugging: If True, use only the first 1,000 sequences per file.
        iterations: Maximum number of optimization iterations.

    Returns:
        Log PWM of shape (4, motif length), rows A, C, G and T.
    """

    # Initialize
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    LEARNING_RATES = [1.0, 0.55, 0.1]

    # Get sequences, both strands
    positive_seqs = read_sequences(positive_seqs_file, debugging)
    negative_seqs = read_sequences(negative_seqs_file, debugging)
    positive_seqs = [str(s.seq).upper() for s in positive_seqs] + \
        [str(s.reverse_complement().seq).upper() for s in positive_seqs]
    negative_seqs = [str(s.seq).upper() for s in negative_seqs] + \
        [str(s.reverse_complement().seq).upper() for s in negative_seqs]

    # One-hot encode sequences
    positive_1he_seqs = dna_one_hot_many(positive_seqs)
    negative_1he_seqs = dna_one_hot_many(negative_seqs)

    # Sequences to Tensor
    positive_seqs_tensor = torch.Tensor(positive_1he_seqs.tolist()).to(device)
    negative_seqs_tensor = torch.Tensor(negative_1he_seqs.tolist()).to(device)

    # Get PWM
    pwm = read_pwm(jaspar_profile_file)

    # Get a PyTorch model of the PWM
    model = _get_pwm_model(pwm, positive_1he_seqs.shape[-1], device)

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
    for _ in tqdm(range(iterations), total=iterations, bar_format=bar_format):

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
            model = _get_pwm_model(pwm_1, positive_1he_seqs.shape[-1], device)

            # Score sequences
            positive_scores, positive_sites = _score_sequences(
                model, positive_seqs_tensor, positive_seqs, pwm_1.shape[1]
            )
            negative_scores, negative_sites = _score_sequences(
                model, negative_seqs_tensor, negative_seqs, pwm_1.shape[1],
            )

            # Compute AUC
            auc, y_true, y_score = _compute_auc(positive_scores,
                negative_scores)

            # Early stop
            if auc > auc_0:
                pwm = copy.copy(pwm_1)
                break

        # Early stop
        if np.array_equal(pwm_0, pwm):
            break

    print("\n" + f"AUC optimized from {original_auc} to {auc}." + "\n")

    return pwm


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
