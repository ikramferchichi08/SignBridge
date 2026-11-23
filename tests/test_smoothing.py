import numpy as np

from src.smoothing import WordSmoother

N = 4


def prob(i, p=0.8):
    v = np.full(N, (1 - p) / (N - 1))
    v[i] = p
    return v


def make():
    return WordSmoother(avg=3, hold=2, threshold=0.5, cooldown=1.0)


def test_stable_word_is_accepted_once():
    s = make()
    accepted = [s.update(prob(0), t)[2] for t in (0.0, 0.2, 0.4, 0.6, 0.8)]
    assert accepted == [None, 0, None, None, None]  # not repeated without a rest


def test_same_word_again_after_rest():
    s = make()
    for t in (0.0, 0.2):
        s.update(prob(0), t)
    assert s.update(None, 1.0) == (None, 0.0, None)  # hands down
    accepted = [s.update(prob(0), t)[2] for t in (2.0, 2.2)]
    assert accepted == [None, 0]


def test_low_confidence_is_never_accepted():
    s = make()
    flat = np.full(N, 1 / N)
    assert all(s.update(flat, t * 0.2)[2] is None for t in range(20))
    assert s.update(flat, 5.0)[0] is None  # reported as uncertain


def test_alternating_predictions_are_not_accepted():
    s = WordSmoother(avg=4, hold=2, threshold=0.5, cooldown=0.0)
    out = [s.update(prob(i % 2, 0.9), t * 0.2)[2] for t, i in enumerate(range(12))]
    assert all(a is None for a in out)


def test_cooldown_delays_the_next_word():
    s = make()
    for t in (0.0, 0.2):
        s.update(prob(0), t)  # word 0 accepted at t = 0.2
    times = []
    for k in range(1, 9):
        t = 0.2 + 0.2 * k
        if s.update(prob(1), t)[2] == 1:
            times.append(t)
    assert times and min(times) >= 1.2 - 1e-9
