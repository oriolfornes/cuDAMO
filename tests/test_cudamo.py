"""Checks that cuDAMO follows DAMO.py (https://github.com/sx-ruan/DAMO).

    pixi run test
"""

import gzip
import os
import random

import numpy as np
import torch
from Bio import motifs
from Bio.Seq import Seq
from click.testing import CliRunner

import pytest

from cudamo import cli
from cudamo.io import read_pwm
from cudamo.damo import (_compute_auc, _generate_pwm, _get_pwm_model,
                         _score_sequences, _update_pwm, damo, pwm_to_pfm)
from cudamo.damo import _extend, _select_width, _split
from cudamo.io import read_sequences
from cudamo.sequence import dna_one_hot, dna_one_hot_many

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JASPAR = os.path.join(ROOT, "examples", "MA1419.1.jaspar")


def test_pseudocounts_follow_damo():
    # DAMO gen_pwm: PFM.Normalize(PFM.Normalize(counts), 1e-4), i.e. counts to
    # frequencies first, then add 1e-4 and renormalize
    pwm = _generate_pwm(["A"] * 500, 1)
    freq = np.array([1., 0., 0., 0.]) + 1e-4
    np.testing.assert_allclose(pwm[:, 0], np.log(freq / freq.sum()))
    assert abs(pwm[1, 0] - np.log(1e-4 / 1.0004)) < 1e-9     # -9.2, not log(1e-4 / 500.0004)


def test_one_hot_encodes_bases_and_zeroes_n():
    encoded = dna_one_hot("ACGTN")
    assert encoded.shape == (4, 5)
    np.testing.assert_array_equal(encoded.argmax(axis=0)[:4], [0, 1, 2, 3])
    assert encoded[:, 4].sum() == 0


def test_iterations_option_reaches_damo(monkeypatch, tmp_path):
    seen = {}

    def fake_damo(*args, **kwargs):
        seen.update(kwargs)
        return np.zeros((4, 2))

    monkeypatch.setattr(cli, "damo", fake_damo)
    fa = os.path.join(ROOT, "examples", "positive_seqs.fa.gz")
    res = CliRunner().invoke(cli.main, [fa, fa, JASPAR, "-i", "3", "-o", str(tmp_path / "out.txt")])
    assert res.exit_code == 0, res.output
    assert seen["iterations"] == 3


def test_pfm_is_boltzmann_distribution():
    pwm = np.log(np.array([[.7, .1], [.1, .2], [.1, .3], [.1, .4]]))
    # exp(W) normalized per column; a log PWM that is already a
    # distribution comes back unchanged
    np.testing.assert_allclose(pwm_to_pfm(pwm), np.exp(pwm))
    np.testing.assert_allclose(pwm_to_pfm(pwm).sum(axis=0), 1.)
    # no overflow on large weights
    assert np.isfinite(pwm_to_pfm(pwm * 1e4)).all()


def test_pfm_scale_sharpens_and_flattens():
    pwm = np.log(np.array([[.7], [.1], [.1], [.1]]))
    assert pwm_to_pfm(pwm, 2.)[0, 0] > pwm_to_pfm(pwm)[0, 0] > pwm_to_pfm(pwm, .5)[0, 0]


def test_pfm_option_writes_readable_jaspar(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "damo", lambda *a, **k: np.log(np.full((4, 3), .25)))
    fa = os.path.join(ROOT, "examples", "positive_seqs.fa.gz")
    out = tmp_path / "motif.pfm.gz"
    res = CliRunner().invoke(cli.main, [fa, fa, JASPAR, "--pfm", str(out), "--scale", "2"])
    assert res.exit_code == 0, res.output
    with gzip.open(out, "rt") as f:
        m = motifs.read(f, "jaspar")
    np.testing.assert_allclose([m.counts[b] for b in "ACGT"], .25)


MEME = """MEME version 4

ALPHABET= ACGT

strands: + -

Background letter frequencies
A 0.25 C 0.25 G 0.25 T 0.25

MOTIF MA0001.1 first
letter-probability matrix: alength= 4 w= 2 nsites= 20 E= 0
  0.700000  0.100000  0.100000  0.100000
  0.100000  0.100000  0.100000  0.700000

MOTIF MA0002.1 second
letter-probability matrix: alength= 4 w= 3 nsites= 20 E= 0
  0.250000  0.250000  0.250000  0.250000
  0.250000  0.250000  0.250000  0.250000
  0.250000  0.250000  0.250000  0.250000
"""


def test_read_meme_motif_by_name_or_id(tmp_path):
    meme = tmp_path / "motifs.meme"
    meme.write_text(MEME)
    for name in ("MA0001.1 first", "MA0001.1"):
        pwm = read_pwm(str(meme), name)
        assert pwm.shape == (4, 2)
        # probabilities, with 1e-4 added and renormalized
        expected = np.array([[.7, .1], [.1, .1], [.1, .1], [.1, .7]]) + 1e-4
        np.testing.assert_allclose(pwm, np.log(expected / expected.sum(axis=0)))
    assert read_pwm(str(meme), "MA0002.1").shape == (4, 3)


def test_read_meme_needs_a_name_when_there_are_several(tmp_path):
    meme = tmp_path / "motifs.meme"
    meme.write_text(MEME)
    with pytest.raises(ValueError, match="--motif"):
        read_pwm(str(meme))
    with pytest.raises(ValueError, match="not found"):
        read_pwm(str(meme), "nope")


def test_read_meme_single_motif_gzipped(tmp_path):
    single = MEME.split("MOTIF MA0002.1")[0]
    meme = tmp_path / "motif.meme.gz"
    with gzip.open(meme, "wt") as f:
        f.write(single)
    assert read_pwm(str(meme)).shape == (4, 2)


def test_motif_option_reaches_damo(monkeypatch, tmp_path):
    seen = {}

    def fake_damo(*args, **kwargs):
        seen.update(kwargs)
        return np.zeros((4, 2))

    monkeypatch.setattr(cli, "damo", fake_damo)
    fa = os.path.join(ROOT, "examples", "positive_seqs.fa.gz")
    res = CliRunner().invoke(cli.main, [fa, fa, JASPAR, "-m", "MA0001.1", "-o", str(tmp_path / "o.txt")])
    assert res.exit_code == 0, res.output
    assert seen["motif_name"] == "MA0001.1"


def test_update_stops_on_perfect_separation():
    sites = ["AC", "AG", "CA", "TT"]
    pwm = np.zeros((4, 2))
    assert _update_pwm([1, 1, 0, 0], [3., 2., 1., 0.], sites, 1., pwm) is None
    assert _update_pwm([1, 0, 1, 0], [3., 2., 1., 0.], sites, 1., pwm) is not None


def test_damo_returns_on_separable_data(tmp_path):
    with open(JASPAR) as f:
        site = str(motifs.read(f, "jaspar").consensus)
    rng = random.Random(1714)
    rand = lambda n: "".join(rng.choice("ACGT") for _ in range(n))  # noqa: E731
    pos, neg = tmp_path / "pos.fa", tmp_path / "neg.fa"
    pos.write_text("".join(f">p{i}\n{rand(20)}{site}{rand(20)}\n" for i in range(50)))
    neg.write_text("".join(f">n{i}\n{rand(40 + len(site))}\n" for i in range(50)))
    pwm = damo(str(pos), str(neg), JASPAR, iterations=20)
    assert np.isfinite(pwm).all()


def test_reported_auc_is_the_accepted_one(tmp_path, capsys):
    # Weak signal: the run ends when every learning rate is rejected. The
    # reported final AUC must be that of the returned PWM, not of the last
    # rejected candidate.
    with open(JASPAR) as f:
        site = str(motifs.read(f, "jaspar").consensus)
    rng = random.Random(7)
    rand = lambda n: "".join(rng.choice("ACGT") for _ in range(n))  # noqa: E731
    pos, neg = tmp_path / "pos.fa", tmp_path / "neg.fa"
    pos.write_text("".join(
        f">p{i}\n{rand(30)}{site if i % 3 == 0 else rand(len(site))}{rand(30)}\n"
        for i in range(300)))
    neg.write_text("".join(f">n{i}\n{rand(60 + len(site))}\n" for i in range(300)))
    pwm = damo(str(pos), str(neg), JASPAR, iterations=100)

    # Recompute the AUC of the returned PWM, forward and reverse strands
    def best(path):
        seqs = [str(r.seq).upper() for r in read_sequences(path)]
        both = seqs + [str(Seq(x).reverse_complement()) for x in seqs]
        model = _get_pwm_model(pwm, len(seqs[0]))
        scores, _ = _score_sequences(model, torch.Tensor(dna_one_hot_many(both).tolist()),
                                     both, pwm.shape[1])
        return scores

    expected = _compute_auc(best(str(pos)), best(str(neg)))[0]
    reported = float(capsys.readouterr().out.split(" to ")[-1].rstrip(".\n "))
    assert abs(reported - expected) < 1e-9


def test_extend_pads_equally_with_uniform_columns():
    pwm = np.log(np.full((4, 15), .1) + np.eye(4)[:, [0] * 15] * .6)
    big, pad = _extend(pwm, 200)
    assert pad == 2 and big.shape == (4, 19)
    np.testing.assert_array_equal(big[:, 2:-2], pwm)
    np.testing.assert_allclose(big[:, :2], np.log(.25))
    np.testing.assert_allclose(big[:, -2:], np.log(.25))


def test_extend_respects_sequence_length_and_wide_motifs():
    pwm = np.zeros((4, 8))
    # 19 bp windows leave room for 5 columns on each side at most
    assert _extend(pwm, 19)[0].shape == (4, 18)
    # no room, or already wide enough: unchanged
    assert _extend(pwm, 9)[1] == 0
    assert _extend(np.zeros((4, 21)), 200)[1] == 0


def test_select_width_keeps_columns_that_pay_for_themselves():
    # Columns 2-9 are informative, except column 1 (gain below the 0.003
    # penalty) and column 10 (none)
    gain = {1: .001, 2: .02, 3: .02, 4: .02, 5: .02, 6: .02, 7: .02,
            8: .02, 9: .02, 10: 0.}

    def auc_of(start, end):
        return .5 + sum(gain.get(i, 0.) for i in range(start, end))

    start, end, auc = _select_width(auc_of, 12)
    assert (start, end) == (2, 10)
    assert abs(auc - auc_of(2, 10)) < 1e-12


def test_select_width_stops_at_the_first_column_that_does_not_pay():
    # Column 1 does not pay, so column 0 is never tried, even if it would
    gain = {0: .5, 1: .0, 2: .02}
    auc_of = lambda start, end: .5 + sum(gain.get(i, 0.) for i in range(start, end))  # noqa: E731
    start, _, _ = _select_width(auc_of, 12)
    assert start == 2


def test_split_holds_out_a_fraction():
    train, val = _split(list(range(100)), 1714)
    assert len(val) == 20 and len(train) == 80
    assert sorted(train + val) == list(range(100))
    assert _split(list(range(100)), 1714) == (train, val)


def test_optimize_width_returns_a_valid_pwm(tmp_path):
    with open(JASPAR) as f:
        site = str(motifs.read(f, "jaspar").consensus)[3:12]   # 9 of 15 columns
    rng = random.Random(1714)
    rand = lambda n: "".join(rng.choice("ACGT") for _ in range(n))  # noqa: E731
    pos, neg = tmp_path / "pos.fa", tmp_path / "neg.fa"
    pos.write_text("".join(f">p{i}\n{rand(40)}{site}{rand(40)}\n" for i in range(200)))
    neg.write_text("".join(f">n{i}\n{rand(89)}\n" for i in range(200)))
    pwm = damo(str(pos), str(neg), JASPAR, iterations=30, optimize_width=True)
    assert pwm.shape[0] == 4 and 6 <= pwm.shape[1] <= 20
    assert np.isfinite(pwm).all()


def test_optimize_width_option_reaches_damo(monkeypatch, tmp_path):
    seen = {}

    def fake_damo(*args, **kwargs):
        seen.update(kwargs)
        return np.zeros((4, 2))

    monkeypatch.setattr(cli, "damo", fake_damo)
    fa = os.path.join(ROOT, "examples", "positive_seqs.fa.gz")
    out = str(tmp_path / "o.txt")
    CliRunner().invoke(cli.main, [fa, fa, JASPAR, "-w", "-o", out])
    assert seen["optimize_width"] is True
    CliRunner().invoke(cli.main, [fa, fa, JASPAR, "-o", out])
    assert seen["optimize_width"] is False
