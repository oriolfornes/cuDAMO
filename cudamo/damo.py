"""DAMO (Discriminative Additive Model Optimization) in PyTorch.

Level 1 of DAMO.py (https://github.com/sx-ruan/DAMO), which is DiMO
(Patel and Stormo, 2014): a single PWM is refined so that the best site in
each sequence separates positives from negatives. DiMO's width optimization
is optional.
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


# Width optimization, as in DiMO (Patel and Stormo, 2014, section 2.6)
EXTENDED_WIDTH = 20
MIN_WIDTH = 6
WIDTH_PENALTY = 0.003
VALIDATION_FRACTION = 0.2
LEARNING_RATES = [1.0, 0.55, 0.1]


def damo(positive_seqs_file, negative_seqs_file, jaspar_profile_file,
         debugging=False, iterations=500, motif_name=None,
         optimize_width=False, seed=1714):
    """
    Optimize a PWM to separate positive from negative sequences.

    Every sequence is scanned on both strands. Each iteration moves the
    PWM towards the sites of positives in the overlap zone and away from
    the sites of negatives. It stops early when the AUC stops improving,
    or when the classes are perfectly separated.

    With optimize_width, the width of the motif is optimized too, as in
    DiMO. 20% of the sequences of each class are held out. The seed is
    optimized on the other 80% and padded with uniform positions to about
    20 columns, equally on both sides, and optimized again. The result
    is cut to its central 6 columns, which are then restored one by one,
    first on the 5' side and then on the 3' side, as long as each one
    raises the AUC on the held-out sequences by at least 0.003.

    Args:
        positive_seqs_file: FASTA file of positive sequences.
        negative_seqs_file: FASTA file of negative sequences.
        jaspar_profile_file: JASPAR or MEME motif used as the seed PWM.
        debugging: If True, use only the first 1,000 sequences per file.
        iterations: Maximum number of optimization iterations.
        motif_name: Motif to use when a MEME file has several.
        optimize_width: If True, optimize the motif width too.
        seed: Random seed of the held-out split (optimize_width only).

    Returns:
        Log PWM of shape (4, motif length), rows A, C, G and T. The
        motif length is the seed's, unless optimize_width is set.
    """

    device = _get_device()

    # Get sequences, and the seed PWM
    positive_records = read_sequences(positive_seqs_file, debugging)
    negative_records = read_sequences(negative_seqs_file, debugging)
    pwm = read_pwm(jaspar_profile_file, motif_name)

    if not optimize_width:
        positives = _encode(positive_records, device)
        negatives = _encode(negative_records, device)
        pwm, original_auc, auc = _optimize(pwm, positives, negatives,
            iterations, device)
        print("\n" + f"AUC optimized from {original_auc} to {auc}." + "\n")

        return pwm

    # Hold out sequences to choose the width
    positive_records, positive_val = _split(positive_records, seed)
    negative_records, negative_val = _split(negative_records, seed)
    positives = _encode(positive_records, device)
    negatives = _encode(negative_records, device)
    positives_val = _encode(positive_val, device)
    negatives_val = _encode(negative_val, device)
    seed_width = pwm.shape[1]

    # Optimize at the width of the seed, then pad and optimize again
    optimized, _, _ = _optimize(pwm, positives, negatives, iterations,
        device, desc="Seed")
    extended, pad = _extend(optimized, positives[1].shape[-1])
    if pad:
        extended, _, _ = _optimize(extended, positives, negatives,
            iterations, device, desc="Extended")

    # Cut to the core, and restore columns while the held-out AUC allows it
    def auc_of(start, end):
        return _auc(extended[:, start:end], positives_val, negatives_val,
            device)

    start, end, val_auc = _select_width(auc_of, extended.shape[1])
    final = extended[:, start:end]

    print("\n" + f"Width optimized from {seed_width} to {end - start}. "
          f"AUC on held-out sequences: seed "
          f"{_auc(pwm, positives_val, negatives_val, device)}, optimized at "
          f"the seed width "
          f"{_auc(optimized, positives_val, negatives_val, device)}, "
          f"final {val_auc}." + "\n")

    return final


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


def _extend(pwm, sequence_length, target=EXTENDED_WIDTH):
    """
    Pad a PWM with uniform positions, equally on both sides.

    The PWM grows to about the target width, as far as the sequences allow.
    A PWM that is already as wide is returned unchanged.

    Returns:
        Tuple (pwm, pad): the padded PWM and the positions added per side.
    """
    pad = (min(target, sequence_length) - pwm.shape[1]) // 2
    if pad <= 0:
        return pwm, 0
    uniform = np.full((4, pad), np.log(0.25))

    return np.hstack([uniform, pwm, uniform]), pad


def _select_width(auc_of, width, min_width=MIN_WIDTH, penalty=WIDTH_PENALTY):
    """
    Choose which columns of a wide PWM to keep.

    Start from the central min_width columns and restore columns one at a
    time, on the 5' side until one fails and then on the 3' side. A column
    is kept only if the AUC rises by at least the penalty, which keeps the
    motif from growing without a real gain.

    Args:
        auc_of: Function (start, end) returning the held-out AUC of the
            columns start to end - 1.
        width: Width of the PWM.
        min_width: Width of the starting core.
        penalty: Minimum AUC gain to keep a restored column.

    Returns:
        Tuple (start, end, auc) of the columns kept and their AUC.
    """
    core = min(min_width, width)
    start = (width - core) // 2
    end = start + core
    best = auc_of(start, end)

    while start > 0:
        candidate = auc_of(start - 1, end)
        if candidate - best < penalty - 1e-12:
            break
        start, best = start - 1, candidate

    while end < width:
        candidate = auc_of(start, end + 1)
        if candidate - best < penalty - 1e-12:
            break
        end, best = end + 1, candidate

    return start, end, best


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
