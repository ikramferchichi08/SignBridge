"""Temporal smoothing: turn noisy per-window predictions into a clean word transcript.

The live demo predicts a sign every few frames. A single prediction is noisy, so a word is
added to the transcript only when
  - the average probabilities of the last `avg` predictions agree on one sign,
  - its confidence is at least `threshold`,
  - the same sign won `hold` predictions in a row,
  - at least `cooldown` seconds passed since the previous accepted word, and
  - it is a new word, or the signer rested (hands down) since the same word was accepted.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class WordSmoother:
    def __init__(self, avg: int = 5, hold: int = 3, threshold: float = 0.35,
                 cooldown: float = 1.5):
        self.avg, self.hold = avg, hold
        self.threshold, self.cooldown = threshold, cooldown
        self.history: deque = deque(maxlen=avg)
        self.candidate: int | None = None
        self.streak = 0
        self.armed = True  # True when the same word may be accepted again
        self.last_accepted: int | None = None
        self.last_time = float("-inf")

    def update(self, probs, now: float):
        """Process one prediction.

        probs: class probabilities, or None when nobody is signing (rest).
        now:   current time in seconds.
        returns (current class index or None, confidence, accepted class index or None)
        """
        if probs is None:  # rest: forget the recent past, allow the same word again
            self.history.clear()
            self.candidate, self.streak, self.armed = None, 0, True
            return None, 0.0, None

        self.history.append(np.asarray(probs, dtype=float))
        mean = np.mean(self.history, axis=0)
        top = int(mean.argmax())
        confidence = float(mean[top])
        if confidence < self.threshold:  # uncertain
            self.candidate, self.streak = None, 0
            return None, confidence, None

        if top == self.candidate:
            self.streak += 1
        else:
            self.candidate, self.streak = top, 1

        accepted = None
        is_new = self.armed or top != self.last_accepted
        if self.streak >= self.hold and is_new and now - self.last_time >= self.cooldown:
            accepted = top
            self.last_accepted, self.last_time, self.armed = top, now, False
        return top, confidence, accepted
