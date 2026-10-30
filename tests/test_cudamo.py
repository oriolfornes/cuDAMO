"""Checks that cuDAMO follows DAMO.py (https://github.com/sx-ruan/DAMO).

    pixi run test
"""

import os
import random

import numpy as np
from Bio import motifs
from click.testing import CliRunner

from cudamo import cli
from cudamo.damo import _generate_pwm, _update_pwm, damo
from cudamo.sequence import dna_one_hot

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
