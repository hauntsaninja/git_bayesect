from __future__ import annotations

import argparse
import enum
import json
import sys
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import loggamma, logsumexp

ndarray = np.ndarray[Any, Any]


class Bisector:
    """
    There is some index B such that for all index:
        P(obs_yes | index >= B) = p_obs_new
        P(obs_yes | index < B) = p_obs_old

    We'd like to find B (and we don't know p_obs_new and p_obs_old).
    """

    def __init__(self, prior_weights: list[float] | list[int] | ndarray) -> None:
        if isinstance(prior_weights, list):
            prior_weights = np.array(prior_weights, dtype=np.float64)
        assert isinstance(prior_weights, np.ndarray)
        self.prior_weights = prior_weights

        self.obs_yes = np.zeros_like(prior_weights, dtype=np.int64)
        self.obs_total = np.zeros_like(prior_weights, dtype=np.int64)

        # p_obs_new ~ Beta(0.9, 0.1), so E[p_obs_new] = 0.9
        self.alpha_new = 0.9
        self.beta_new = 0.1

        # p_obs_old ~ Beta(0.05, 0.95), so E[p_obs_old] = 0.05
        self.alpha_old = 0.05
        self.beta_old = 0.95

        self.post_weights: ndarray | None = None

    def _maybe_update_posteriors(self) -> None:
        if self.post_weights is None:
            self._update_posteriors()

    def _update_posteriors(self) -> None:
        # fmt: off
        # left:  yes and no counts on or before index
        # right: yes and no counts after index
        total_left  = self.obs_total
        total_right = self.obs_total[-1] - total_left
        yes_left    = self.obs_yes
        yes_right   = yes_left[-1] - yes_left
        no_left     = total_left - yes_left
        no_right    = total_right - yes_right

        # At this point, if we knew p_obs_new and p_obs_old, we could just apply Bayes' theorem
        # and things would be straightforward. But we don't, so we have to integrate over our
        # priors of what p_obs_new and p_obs_old might be.

        # P(data) = ∫ P(data | p) P(p) dp for left and right observations
        # Thanks to Beta distribution magic, we can compute this analytically
        log_beta = lambda a, b: loggamma(a) + loggamma(b) - loggamma(a + b)
        log_likelihood_left = (
            log_beta(self.alpha_new + yes_left, self.beta_new + no_left)
            - log_beta(self.alpha_new, self.beta_new)
        )
        log_likelihood_right = (
            log_beta(self.alpha_old + yes_right, self.beta_old + no_right)
            - log_beta(self.alpha_old, self.beta_old)
        )
        # This gives us:
        # log P(data | index=b) = log_likelihood_left[b] + log_likelihood_right[b]

        log_prior = np.where(self.prior_weights > 0, np.log(self.prior_weights), -np.inf)
        # log_post[b] is now numerator of Bayes' theorem, so just normalise by sum(exp(log_post))
        log_post = log_prior + log_likelihood_left + log_likelihood_right
        self.post_weights = np.exp(log_post - logsumexp(log_post))
        # fmt: on

    def record(self, index: int, observation: bool | None) -> None:
        """Record an observation at index."""
        assert 0 <= index < len(self.prior_weights)
        self.post_weights = None
        if observation is None:
            # Similar to git bisect skip, let's just zero out the prior
            # Note we might want to lower the prior instead
            self.prior_weights[index] = 0
            return

        self.obs_total[index:] += 1
        if observation:
            self.obs_yes[index:] += 1

    def select(self) -> int:
        """Return the index which will most reduce entropy."""
        self._maybe_update_posteriors()
        assert self.post_weights is not None

        # fmt: off
        total_left  = self.obs_total
        total_right = self.obs_total[-1] - total_left
        yes_left    = self.obs_yes
        yes_right   = yes_left[-1] - yes_left

        # posterior means of the two Bernoulli parameters at each b
        p_obs_new = (self.alpha_new + yes_left) / (self.alpha_new + self.beta_new + total_left)
        p_obs_old = (self.alpha_old + yes_right) / (self.alpha_old + self.beta_old + total_right)
        # p_obs_new = yes_left / np.maximum(1e-10, total_left)
        # p_obs_old = yes_right / np.maximum(1e-10, total_right)

        # p_obs_yes[b]
        # = P(obs_yes | select=b)
        # = \sum_{i=0}^{b-1} p_obs_old[i] * post[i] + \sum_{i=b}^{n-1} p_obs_new[i] * post[i]
        w_new_yes = self.post_weights * p_obs_new
        w_old_yes = self.post_weights * p_obs_old
        p_obs_yes = (np.cumsum(w_old_yes) - w_old_yes) + np.cumsum(w_new_yes[::-1])[::-1]

        w_new_no  = self.post_weights * (1.0 - p_obs_new)
        w_old_no  = self.post_weights * (1.0 - p_obs_old)
        p_obs_no  = (np.cumsum(w_old_no)  - w_old_no)  + np.cumsum(w_new_no[::-1])[::-1]

        assert np.allclose(p_obs_yes + p_obs_no, 1)

        wlog = lambda w: np.where(w > 0.0, w * np.log2(w), 0.0)

        # To get entropy from unnormalised w_i, calculate S = \sum w_i
        # Then log S - (\sum w_i log w_i) / S
        w_new_yes_log = wlog(w_new_yes)
        w_old_yes_log = wlog(w_old_yes)
        p_obs_yes_log = (np.cumsum(w_old_yes_log) - w_old_yes_log) + np.cumsum(w_new_yes_log[::-1])[::-1]
        H_yes         = np.where(p_obs_yes > 0, np.log2(p_obs_yes) - p_obs_yes_log / p_obs_yes, 0.0)

        w_new_no_log  = wlog(w_new_no)
        w_old_no_log  = wlog(w_old_no)
        p_obs_no_log  = (np.cumsum(w_old_no_log)  - w_old_no_log)  + np.cumsum(w_new_no_log[::-1])[::-1]
        H_no          = np.where(p_obs_no  > 0, np.log2(p_obs_no)  - p_obs_no_log  / p_obs_no,  0.0)
        # fmt: on

        expected_H = H_yes * p_obs_yes + H_no * p_obs_no
        return int(np.argmin(expected_H))

    @property
    def distribution(self) -> ndarray:
        """Current posterior P(index=B | data)"""
        self._maybe_update_posteriors()
        assert self.post_weights is not None
        return self.post_weights

    @property
    def entropy(self) -> float:
        """Posterior entropy in bits"""
        self._maybe_update_posteriors()
        assert self.post_weights is not None
        probs = self.post_weights[self.post_weights > 0]
        return -float(np.sum(probs * np.log2(probs)))

    @property
    def empirical_p_obs(self) -> tuple[ndarray, ndarray]:
        """Return what we've observed for p_obs_new and p_obs_old are if each commit is B."""
        # fmt: off
        total_left  = self.obs_total
        total_right = self.obs_total[-1] - total_left
        yes_left    = self.obs_yes
        yes_right   = yes_left[-1] - yes_left

        # Use the following if you want to take the prior into account:
        # p_obs_new = (self.alpha_new + yes_left) / (self.alpha_new + self.beta_new + total_left)
        # p_obs_old = (self.alpha_old + yes_right) / (self.alpha_old + self.beta_old + total_right)

        p_obs_new = yes_left / np.maximum(1e-10, total_left)
        p_obs_old = yes_right / np.maximum(1e-10, total_right)
        return p_obs_new, p_obs_old
        # fmt: on

    @property
    def num_total_observations(self) -> int:
        return int(self.obs_total[-1])

    @property
    def num_yes_observations(self) -> int:
        return int(self.obs_yes[-1])

    def central_range(self, mass: float) -> tuple[int, int]:
        """Return the range of indices that contain the central mass of the posterior, inclusive."""
        self._maybe_update_posteriors()
        assert self.post_weights is not None
        assert 0 <= mass <= 1
        cumsum = np.cumsum(self.post_weights)

        tail = (1 - mass) / 2
        left = np.searchsorted(cumsum, tail, side="left")
        right = np.searchsorted(cumsum, 1 - tail, side="right")
        right = min(right, len(cumsum) - 1)  # type: ignore[call-overload]

        return int(left), int(right)
