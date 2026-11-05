"""Checks that daimo follows DAMO.py (https://github.com/sx-ruan/DAMO).

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

from daimo import cli
from daimo.io import read_pwm
from daimo.damo import (_compute_auc, _generate_pwm, _get_pwm_model,
                         _score_sequences, _update_pwm, damo, pwm_to_pfm)
from daimo.damo import _choose_width, _resize, _split, parse_widths
from daimo.io import read_sequences
from daimo.sequence import dna_one_hot, dna_one_hot_many

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


def _run(monkeypatch, tmp_path, args, pwm=None):
    pwm = np.log(np.full((4, 3), .25)) if pwm is None else pwm
    monkeypatch.setattr(cli, "damo", lambda *a, **k: pwm)
    fa = os.path.join(ROOT, "examples", "positive_seqs.fa.gz")
    return CliRunner().invoke(cli.main, [fa, fa, JASPAR] + args)


def test_oformat_jaspar_is_readable_and_keeps_the_name(monkeypatch, tmp_path):
    out = tmp_path / "motif.jaspar.gz"
    res = _run(monkeypatch, tmp_path, ["--oformat", "jaspar", "--scale", "2", "-o", str(out)])
    assert res.exit_code == 0, res.output
    with gzip.open(out, "rt") as f:
        m = motifs.read(f, "jaspar")
    np.testing.assert_allclose([m.counts[b] for b in "ACGT"], .25)
    assert (m.matrix_id, m.name) == ("MA1419.1", "IRF4")


def test_oformat_meme_round_trips(monkeypatch, tmp_path):
    pwm = np.log(np.array([[.7, .1], [.1, .2], [.1, .3], [.1, .4]]))
    out = tmp_path / "motif.meme"
    res = _run(monkeypatch, tmp_path, ["--oformat", "meme", "--scale", "1", "-o", str(out)], pwm)
    assert res.exit_code == 0, res.output
    assert "MOTIF MA1419.1 IRF4" in out.read_text()
    back = read_pwm(str(out))
    # read as probabilities: 1e-4 added and renormalized
    expected = np.exp(pwm) + 1e-4
    np.testing.assert_allclose(np.exp(back), expected / expected.sum(axis=0), atol=1e-6)


def test_oformat_defaults_to_the_log_pwm(monkeypatch, tmp_path):
    pwm = np.log(np.array([[.7, .1], [.1, .2], [.1, .3], [.1, .4]]))
    out = tmp_path / "motif.txt"
    assert _run(monkeypatch, tmp_path, ["-o", str(out)], pwm).exit_code == 0
    np.testing.assert_allclose(np.loadtxt(out).T, pwm, atol=1e-8)


def test_frequency_formats_default_to_scale_0_1(monkeypatch, tmp_path):
    pwm = np.log(np.array([[.7, .1], [.1, .2], [.1, .3], [.1, .4]])) * 20
    out = tmp_path / "motif.meme"
    assert _run(monkeypatch, tmp_path, ["--oformat", "meme", "-o", str(out)], pwm).exit_code == 0
    written = np.exp(read_pwm(str(out)))
    written /= written.sum(axis=0)
    np.testing.assert_allclose(written, pwm_to_pfm(pwm, .1), atol=5e-4)
    # an explicit scale of 1 still gives DiMO's unscaled conversion
    out1 = tmp_path / "motif1.meme"
    assert _run(monkeypatch, tmp_path, ["--oformat", "meme", "--scale", "1", "-o", str(out1)], pwm).exit_code == 0
    unscaled = np.exp(read_pwm(str(out1)))
    unscaled /= unscaled.sum(axis=0)
    np.testing.assert_allclose(unscaled, pwm_to_pfm(pwm, 1.), atol=5e-4)


def test_oformat_writes_to_stdout(monkeypatch, tmp_path):
    res = _run(monkeypatch, tmp_path, ["--oformat", "meme"])
    assert res.exit_code == 0 and res.output.startswith("MEME version 4")


def test_scale_needs_a_frequency_format(monkeypatch, tmp_path):
    res = _run(monkeypatch, tmp_path, ["--scale", "2", "-o", str(tmp_path / "o.txt")])
    assert res.exit_code != 0 and "--oformat" in res.output


def test_meme_input_name_is_kept_in_the_output(monkeypatch, tmp_path):
    meme = tmp_path / "in.meme"
    meme.write_text(MEME)
    monkeypatch.setattr(cli, "damo", lambda *a, **k: np.log(np.full((4, 2), .25)))
    fa = os.path.join(ROOT, "examples", "positive_seqs.fa.gz")
    res = CliRunner().invoke(cli.main, [fa, fa, str(meme), "-m", "MA0002.1", "--oformat", "jaspar"])
    assert res.exit_code == 0, res.output
    assert res.output.startswith(">MA0002.1 second")


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


def test_resize_pads_with_uniform_columns_half_on_each_side():
    pwm = np.log(np.full((4, 3), .1) + np.eye(4)[:, [0, 1, 2]] * .6)
    big = _resize(pwm, 8)
    assert big.shape == (4, 8)
    np.testing.assert_array_equal(big[:, 2:5], pwm)           # 2 left, 3 right
    np.testing.assert_allclose(big[:, :2], np.log(.25))
    np.testing.assert_allclose(big[:, 5:], np.log(.25))
    assert _resize(pwm, 3) is pwm


def test_resize_keeps_the_most_informative_columns():
    uniform, sharp = [.25] * 4, [.97, .01, .01, .01]
    # columns 3 and 4 are the informative ones
    pfm = np.array([uniform, uniform, uniform, sharp, sharp, uniform, uniform]).T
    pwm = _resize(np.log(pfm), 2)
    np.testing.assert_allclose(pwm, np.log(pfm[:, 3:5]))
    # ties go to the most central run
    flat = np.log(np.full((4, 7), .25))
    np.testing.assert_array_equal(_resize(flat, 3), flat[:, 2:5])


def test_choose_width_prefers_the_smallest_within_tolerance():
    assert _choose_width({8: .700, 10: .7015, 12: .702}) == 8
    assert _choose_width({8: .700, 10: .7030, 12: .704}) == 10
    assert _choose_width({8: .700, 10: .700, 12: .710}) == 12


def test_split_holds_out_a_fraction():
    train, val = _split(list(range(100)), 1714)
    assert len(val) == 20 and len(train) == 80
    assert sorted(train + val) == list(range(100))
    assert _split(list(range(100)), 1714) == (train, val)


def test_widths_search_returns_a_valid_pwm(tmp_path, capsys):
    with open(JASPAR) as f:
        site = str(motifs.read(f, "jaspar").consensus)[3:12]   # 9 of 15 columns
    rng = random.Random(1714)
    rand = lambda n: "".join(rng.choice("ACGT") for _ in range(n))  # noqa: E731
    pos, neg = tmp_path / "pos.fa", tmp_path / "neg.fa"
    pos.write_text("".join(f">p{i}\n{rand(40)}{site}{rand(40)}\n" for i in range(200)))
    neg.write_text("".join(f">n{i}\n{rand(89)}\n" for i in range(200)))
    pwm = damo(str(pos), str(neg), JASPAR, iterations=30, widths=[8, 12, 200])
    out = capsys.readouterr().out
    assert pwm.shape[0] == 4 and pwm.shape[1] in (8, 12, 15)   # 200 is too long
    assert np.isfinite(pwm).all()
    assert "width   8" in out and "width  15" in out and "width 200" not in out
    assert out.count("<- chosen") == 1


def test_width_options_reach_damo(monkeypatch, tmp_path):
    seen = {}

    def fake_damo(*args, **kwargs):
        seen.update(kwargs)
        return np.zeros((4, 2))

    monkeypatch.setattr(cli, "damo", fake_damo)
    fa = os.path.join(ROOT, "examples", "positive_seqs.fa.gz")
    out = str(tmp_path / "o.txt")
    CliRunner().invoke(cli.main, [fa, fa, JASPAR, "-o", out])
    assert seen["widths"] is None
    CliRunner().invoke(cli.main, [fa, fa, JASPAR, "-w", "6,9", "-o", out])
    assert seen["widths"] == "6,9"
    CliRunner().invoke(cli.main, [fa, fa, JASPAR, "-w", "5-19", "-o", out])
    assert seen["widths"] == "5-19"


def test_widths_syntax_errors_are_usage_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "damo", lambda *a, **k: np.zeros((4, 2)))
    fa = os.path.join(ROOT, "examples", "positive_seqs.fa.gz")
    for bad in ("19-5", "x", "0", "-", "8,,10"):
        res = CliRunner().invoke(cli.main, [fa, fa, JASPAR, "-w", bad])
        assert res.exit_code == 2 and "Invalid width" in res.output, bad


def test_parse_widths_like_cut():
    assert parse_widths("5-19") == list(range(5, 20))
    assert parse_widths("8,10,12") == [8, 10, 12]
    assert parse_widths("12,8,10,8") == [8, 10, 12]
    assert parse_widths(" 8 , 10-12 ") == [8, 10, 11, 12]
    assert parse_widths("-4") == [1, 2, 3, 4]
    assert parse_widths("5-8,12,15-", maximum=17) == [5, 6, 7, 8, 12, 15, 16, 17]
    assert parse_widths("5-8,6-10") == [5, 6, 7, 8, 9, 10]
    assert parse_widths("7") == [7]


def test_parse_widths_rejects_bad_input():
    for bad in ("", "a", "19-5", "0", "-", "5--7", "8,,10", "1.5", "0-3"):
        with pytest.raises(ValueError, match="Invalid width"):
            parse_widths(bad)
    with pytest.raises(ValueError, match="open-ended"):
        parse_widths("5-")


def test_widths_string_reaches_the_search(tmp_path, capsys):
    with open(JASPAR) as f:
        site = str(motifs.read(f, "jaspar").consensus)[3:12]
    rng = random.Random(1714)
    rand = lambda n: "".join(rng.choice("ACGT") for _ in range(n))  # noqa: E731
    pos, neg = tmp_path / "pos.fa", tmp_path / "neg.fa"
    pos.write_text("".join(f">p{i}\n{rand(40)}{site}{rand(40)}\n" for i in range(100)))
    neg.write_text("".join(f">n{i}\n{rand(89)}\n" for i in range(100)))
    # open-ended: 14 up to the sequence length; here 14 and the seed's 15, up to 89,
    # so cap it with a range to keep the run short
    pwm = damo(str(pos), str(neg), JASPAR, iterations=5, widths="5-7,15")
    out = capsys.readouterr().out
    for k in (5, 6, 7, 15):
        assert f"width {k:>3}:" in out
    assert "width   8" not in out and pwm.shape[1] in (5, 6, 7, 15)
