# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "scipy",
# ]
# ///

from __future__ import annotations

import argparse
import enum
import functools
import json
import math
from dataclasses import dataclass
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ndarray = np.ndarray[Any, Any]


# ==============================
# Core pure logic
# ==============================


class Bisector:
    """
    There is some index B such that for all index:
        P(obs_yes | index <= B) = p_obs_new
        P(obs_yes | index > B) = p_obs_old

    We'd like to find B (and we don't know p_obs_new and p_obs_old).
    """

    def __init__(
        self,
        prior_weights: list[float] | list[int] | ndarray,
        # WARNING: Moving the Beta priors away from the Jeffreys prior can cause overconfidence issues
        alpha_new: float = 0.5,
        beta_new: float = 0.5,
        alpha_old: float = 0.5,
        beta_old: float = 0.5,
    ) -> None:
        if isinstance(prior_weights, list):
            prior_weights = np.array(prior_weights, dtype=np.float64)
        assert isinstance(prior_weights, np.ndarray)
        if np.any(prior_weights < 0):
            raise ValueError("prior_weights must be non-negative")
        self.prior_weights = prior_weights

        self.obs_yes = np.zeros_like(prior_weights, dtype=np.int64)
        self.obs_total = np.zeros_like(prior_weights, dtype=np.int64)

        # E.g. p_obs_new ~ Beta(0.9, 0.1), so E[p_obs_new] = 0.9
        self.alpha_new = alpha_new
        self.beta_new = beta_new

        # E.g. p_obs_old ~ Beta(0.05, 0.95), so E[p_obs_old] = 0.05
        self.alpha_old = alpha_old
        self.beta_old = beta_old

        self._post_weights: ndarray | None = None

    def _maybe_update_posteriors(self) -> None:
        if self._post_weights is None:
            self._update_posteriors()

    def _update_posteriors(self) -> None:
        from scipy.special import loggamma, logsumexp

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
        self._post_weights = np.exp(log_post - logsumexp(log_post))
        # fmt: on

    def record(self, index: int, observation: bool | None) -> None:
        """Record an observation at index."""
        assert 0 <= index < len(self.prior_weights)
        self._post_weights = None
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
        assert self._post_weights is not None

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
        w_new_yes = self._post_weights * p_obs_new
        w_old_yes = self._post_weights * p_obs_old
        p_obs_yes = (np.cumsum(w_old_yes) - w_old_yes) + np.cumsum(w_new_yes[::-1])[::-1]

        w_new_no  = self._post_weights * (1.0 - p_obs_new)
        w_old_no  = self._post_weights * (1.0 - p_obs_old)
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

        # tie break the argmin by choosing the middle
        candidate_indices = np.flatnonzero(
            np.isclose(expected_H, np.min(expected_H), rtol=1e-12, atol=1e-15)
        )
        return int(candidate_indices[np.argmin(np.abs(candidate_indices - len(expected_H) // 2))])

    @property
    def distribution(self) -> ndarray:
        """Current posterior P(index=B | data)"""
        self._maybe_update_posteriors()
        assert self._post_weights is not None
        return self._post_weights

    @property
    def entropy(self) -> float:
        """Posterior entropy in bits"""
        self._maybe_update_posteriors()
        assert self._post_weights is not None
        probs = self._post_weights[self._post_weights > 0]
        return -float(np.dot(probs, np.log2(probs)))

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
    def empirical_counts(self) -> tuple[tuple[ndarray, ndarray], tuple[ndarray, ndarray]]:
        total_left = self.obs_total
        total_right = self.obs_total[-1] - total_left
        yes_left = self.obs_yes
        yes_right = yes_left[-1] - yes_left
        return (yes_left, total_left), (yes_right, total_right)

    @property
    def num_total_observations(self) -> int:
        return int(self.obs_total[-1])

    @property
    def num_yes_observations(self) -> int:
        return int(self.obs_yes[-1])

    def central_range(self, mass: float) -> tuple[int, int]:
        """Return the range of indices that contain the central mass of the posterior, inclusive."""
        self._maybe_update_posteriors()
        assert self._post_weights is not None
        assert 0 <= mass <= 1
        cumsum = np.cumsum(self._post_weights)

        tail = (1 - mass) / 2
        left = np.searchsorted(cumsum, tail, side="left")
        right = np.searchsorted(cumsum, 1 - tail, side="right")
        right = min(right, len(cumsum) - 1)  # type: ignore[arg-type]

        return int(left), int(right)


def _log_max_likelihood(yes: ndarray, total: ndarray) -> ndarray:
    """max over p of yes * log(p) + (total - yes) * log(1 - p), with 0 * log(0) = 0."""
    no = total - yes
    with np.errstate(divide="ignore", invalid="ignore"):
        log_yes = np.where(yes > 0, yes * np.log(yes / total), 0.0)
        log_no = np.where(no > 0, no * np.log(no / total), 0.0)
    result: ndarray = log_yes + log_no
    return result


def _log_jeffreys(yes: ndarray, total: ndarray) -> ndarray:
    """log P(sequence with `yes` of `total` yes observations), p ~ Beta(1/2, 1/2)."""
    from scipy.special import betaln

    return betaln(yes + 0.5, total - yes + 0.5) - betaln(0.5, 0.5)


@dataclass(frozen=True)
class SafeExperiment:
    """The announced randomized query, saved before its outcome is observed.

    The sampling distribution puts mass on at most three commits. Saving their probabilities
    and forecasts preserves earlier bets across restarts and prior changes, in O(1) space.
    """

    index: int
    indices: list[int]
    probabilities: list[float]
    predictions: list[float]

    @classmethod
    def choose(cls, bisector: Bisector, rng: np.random.Generator | None = None) -> SafeExperiment:
        from scipy.special import expit, xlogy

        posterior = bisector.distribution
        n = len(posterior)
        (yl, nl), (yr, nr) = bisector.empirical_counts
        a = (yl + bisector.alpha_new) / (nl + bisector.alpha_new + bisector.beta_new)
        b = (yr + bisector.alpha_old) / (nr + bisector.alpha_old + bisector.beta_old)
        q = np.clip(
            np.r_[0.0, np.cumsum((posterior * b)[:-1])] + np.cumsum((posterior * a)[::-1])[::-1],
            1e-12,
            1 - 1e-12,
        )
        modes = np.flatnonzero(posterior == posterior.max())
        leader = int(modes[np.argmin(abs(modes - n // 2))])
        w = np.zeros(n)
        if leader == n - 1:
            w[0] += 0.5
            w[-1] += 0.5
        else:
            entropy = lambda x: -xlogy(x, x) - xlogy(1 - x, 1 - x)
            left, right = float(a[leader]), float(b[leader])
            fraction = 0.5
            if abs(left - right) >= 1e-10:
                pooled = expit(-(entropy(left) - entropy(right)) / (left - right))
                fraction = float(np.clip((pooled - right) / (left - right), 0, 1))
            w[leader : leader + 2] = [fraction, 1 - fraction]
        w *= posterior[leader]
        w[bisector.select()] += 1 - posterior[leader]
        # Vanishing forced exploration: square rounds cycle from the ends inward.
        query = bisector.num_total_observations + 1
        root = math.isqrt(query)
        if root * root == query:
            k = (root - 1) % n
            forced = k // 2 if k % 2 == 0 else n - 1 - k // 2
            w[:] = 0
            w[forced] = 1
        w /= w.sum()
        if rng is None:
            rng = np.random.default_rng()
        indices = np.flatnonzero(w)
        # Draw among the arms with positive probability only: when rounding leaves the total just
        # below one, a draw past it must still pick one of them.
        arm = int(np.searchsorted(np.cumsum(w[indices]), rng.random(), side="right"))
        index = int(indices[min(arm, len(indices) - 1)])
        return cls(index, indices.tolist(), w[indices].tolist(), q[indices].tolist())

    @classmethod
    def single(cls, index: int) -> SafeExperiment:
        """A query that can only pick `index`. Its evidence factor is one, whatever the outcome."""
        return cls(index, [index], [1.0], [0.5])

    def as_dict(self) -> dict[str, Any]:
        return dict(
            index=self.index,
            indices=self.indices,
            probabilities=self.probabilities,
            predictions=self.predictions,
        )

    def log_factor(self, num_indices: int, observation: bool | None) -> ndarray:
        """Log evidence factors against every boundary; None stands for a missing outcome.

        A canceled draw cannot simply be forgotten: choosing which draws to honor could
        bias the experiment. Its factor is the smaller of the two possible outcome factors.
        """
        weights = np.zeros(num_indices)
        moments = np.zeros(num_indices)
        weights[self.indices] = self.probabilities
        moments[self.indices] = np.array(self.probabilities) * self.predictions
        suffix = lambda values: np.r_[np.cumsum(values[:0:-1])[::-1], 0.0]
        wl, wr = np.cumsum(weights), suffix(weights)
        ml, mr = np.cumsum(moments), suffix(moments)
        left = np.divide(ml, wl, out=np.full(num_indices, 0.5), where=wl > 0)
        right = np.divide(mr, wr, out=np.full(num_indices, 0.5), where=wr > 0)
        rates = np.clip(
            np.where(np.arange(num_indices) < self.index, right, left), 1e-12, 1 - 1e-12
        )
        prediction = self.predictions[self.indices.index(self.index)]
        yes = np.log(prediction) - np.log(rates)
        no = np.log1p(-prediction) - np.log1p(-rates)
        result: ndarray = np.minimum(yes, no) if observation is None else yes if observation else no
        return result


class SafeStopping:
    """Randomized reverse-information-projection evidence for every boundary B.

    Forecasts and arm probabilities are fixed before each draw. Project the forecasts
    onto a common rate on each side of B, using those same arm probabilities. The ratio
    of predicted to projected outcome probability has conditional expectation one under B.
    Products are nonnegative martingales, even when the two side rates change predictably
    over time. Missing outcomes use conservative factors, giving a supermartingale instead.
    By Ville's inequality, a wrong answer or empty survivor set has probability
    at most 1 - confidence. The boundary stays fixed and the conditional rate must be
    common across commits on each side; arbitrary per-commit effects are not covered.

    The optional likelihood check has the narrower stationary single-change null.
    """

    def __init__(
        self,
        num_indices: int,
        check_assumptions: bool = False,
        check_prior: list[float] | None = None,
    ) -> None:
        self.log_evidence = np.zeros(num_indices)
        self._check_prior = None if check_prior is None else np.array(check_prior)
        self.check_assumptions = check_assumptions
        # running max over time of log E_B, the evidence against hypothesis B
        self.max_log_evidence = np.full(num_indices, -np.inf)
        # running max over time of log G, the evidence against every hypothesis at once
        self.max_log_misfit = -np.inf
        # each commit's first observation and its run-to-run transitions (for the Markov check)
        self._last = np.full(num_indices, -1, dtype=np.int64)
        self._first = np.zeros((num_indices, 2), dtype=np.int64)
        self._transitions = np.zeros((num_indices, 2, 2), dtype=np.int64)

    def update(self, experiment: SafeExperiment, observation: bool | None) -> None:
        self.log_evidence += experiment.log_factor(len(self.log_evidence), observation)
        np.maximum(self.max_log_evidence, self.log_evidence, out=self.max_log_evidence)

    def record(
        self,
        bisector: Bisector,
        index: int,
        observation: bool,
        experiment: SafeExperiment | None = None,
        check: bool = True,
    ) -> None:
        """After bisector.record, update the announced bet and optional model check.

        An externally chosen observation adds no evidence factor; its data still improve future
        predictions. Any discarded pending experiment must separately be updated with None.
        """
        if experiment is not None:
            if experiment.index != index:
                raise ValueError("Observation does not match the announced randomized query")
            self.update(experiment, observation)
        if not self.check_assumptions:
            return
        if self._check_prior is None:
            self._check_prior = bisector.prior_weights.copy()
        yes = int(observation)
        if self._last[index] < 0:
            self._first[index, yes] += 1
        else:
            self._transitions[index, self._last[index], yes] += 1
        self._last[index] = yes

        if check:
            (yes_left, total_left), (yes_right, total_right) = bisector.empirical_counts
            log_max = _log_max_likelihood(yes_left, total_left) + _log_max_likelihood(
                yes_right, total_right
            )
            log_misfit = self._log_check_likelihood(bisector) - float(log_max.max())
            self.max_log_misfit = max(self.max_log_misfit, log_misfit)

    def _log_check_likelihood(self, bisector: Bisector) -> float:
        from scipy.special import logsumexp

        # every commit has its own failure rate
        yes = np.diff(bisector.obs_yes, prepend=0)
        total = np.diff(bisector.obs_total, prepend=0)
        log_own_rates = float(_log_jeffreys(yes, total).sum())

        # reruns of a commit follow a Markov chain, with separate rates on either side of B
        with np.errstate(divide="ignore"):
            assert self._check_prior is not None
            log_markov = np.log(self._check_prior / self._check_prior.sum())
        for counts in (self._first, self._transitions[:, 0], self._transitions[:, 1]):
            total_left = np.cumsum(counts.sum(axis=1))
            yes_left = np.cumsum(counts[:, 1])
            log_markov = log_markov + _log_jeffreys(yes_left, total_left)
            log_markov = log_markov + _log_jeffreys(
                yes_left[-1] - yes_left, total_left[-1] - total_left
            )

        return float(np.logaddexp(log_own_rates, logsumexp(log_markov)) - np.log(2))

    def surviving(self, confidence: float) -> ndarray:
        """Indices B not ruled out: max over time of E_B < 1 / (1 - confidence)."""
        return np.flatnonzero(self.max_log_evidence < -np.log1p(-confidence))

    def misfit(self, confidence: float) -> bool:
        """Whether the assumption check has ruled out every single-culprit explanation."""
        return bool(self.max_log_misfit >= -np.log1p(-confidence))


# ==============================
# State logic
# ==============================


class BayesectError(Exception):
    pass


def format_percent(confidence: float) -> str:
    """e.g. 95% or 99.5%: exact decimal arithmetic on the repr never rounds up to 100%."""
    from decimal import Decimal

    return f"{(Decimal(repr(confidence)) * 100).normalize():f}%"


class Result(enum.Enum):
    FAIL = "fail"
    PASS = "pass"
    SKIP = "skip"


class BetaPriors:
    def __init__(
        self, alpha_new: float, beta_new: float, alpha_old: float, beta_old: float
    ) -> None:
        self.alpha_new = alpha_new
        self.beta_new = beta_new
        self.alpha_old = alpha_old
        self.beta_old = beta_old

    def as_dict(self) -> dict[str, float]:
        return {
            "alpha_new": self.alpha_new,
            "beta_new": self.beta_new,
            "alpha_old": self.alpha_old,
            "beta_old": self.beta_old,
        }


class Stopping:
    def __init__(
        self, safe: bool, max_range: int, check_assumptions: bool, confidence: float = 0.95
    ) -> None:
        self.safe = safe
        self.max_range = max_range
        self.check_assumptions = check_assumptions
        # saved by safe sessions, so that every command reports at the level of `run --confidence`
        self.confidence = confidence

    def as_dict(self) -> dict[str, bool | int | float]:
        return {
            "safe": self.safe,
            "max_range": self.max_range,
            "check_assumptions": self.check_assumptions,
            "confidence": self.confidence,
        }


STATE_FILENAME = "BAYESECT_STATE"
STATE_VERSION = 3


class State:
    def __init__(
        self,
        old_sha: bytes,
        new_sha: bytes,
        beta_priors: BetaPriors,
        stopping: Stopping,
        priors: dict[bytes, float],
        results: list[tuple[bytes, Result]],
        commit_indices: dict[bytes, int],
        pending: SafeExperiment | None = None,
        safe_history: list[dict[str, Any]] | None = None,
        check_prior: list[float] | None = None,
        pending_started: bool = False,
        check_cache: dict[str, Any] | None = None,
    ) -> None:
        self.old_sha = old_sha
        self.new_sha = new_sha
        self.beta_priors = beta_priors
        self.stopping = stopping
        self.priors = priors
        self.results = results
        self.commit_indices = commit_indices
        self.pending = pending
        self.safe_history = [] if safe_history is None else safe_history
        self.check_prior = check_prior
        # `run` started testing the pending query, but its outcome hasn't been recorded yet
        self.pending_started = pending_started
        # The assumption check's running maximum over the first `results` results
        self.check_cache = check_cache

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "old_sha": self.old_sha.decode(),
            "new_sha": self.new_sha.decode(),
            "beta_priors": self.beta_priors.as_dict(),
            "stopping": self.stopping.as_dict(),
            "priors": {k.decode(): v for k, v in self.priors.items()},
            "results": [(k.decode(), v.value) for k, v in self.results],
            "pending": None if self.pending is None else self.pending.as_dict(),
            "pending_started": self.pending_started,
            "safe_history": self.safe_history,
            "check_prior": self.check_prior,
            "check_cache": self.check_cache,
        }

    def dump(self, repo_path: Path) -> None:
        import os

        # Replace the file in one step, so that being stopped while saving can't corrupt it
        path = git_dir(repo_path) / STATE_FILENAME
        with open(path.with_name(f"{STATE_FILENAME}.tmp"), "w") as f:
            f.write(json.dumps(self.as_dict()))
        os.replace(f.name, path)

    def record_pending(self, observation: bool | None, result_index: int | None = None) -> None:
        if self.pending is not None:
            self.safe_history.append(
                {
                    "experiment": self.pending.as_dict(),
                    "observation": observation,
                    "result_index": result_index,
                }
            )
            self.pending = None
        self.pending_started = False

    @classmethod
    def from_git_state(cls, repo_path: Path, read_only: bool = False) -> State:
        try:
            with open(git_dir(repo_path) / STATE_FILENAME) as f:
                data = f.read()
        except FileNotFoundError:
            raise BayesectError("No state file found, run `git bayesect start` first") from None

        try:
            state_dict = json.loads(data)
        except json.JSONDecodeError:
            raise BayesectError(
                "Invalid state file, run `git bayesect reset` to start afresh"
            ) from None

        if not isinstance(state_dict, dict):
            raise BayesectError("Invalid state file, run `git bayesect reset` to start afresh")

        # Upgrade upstream posterior-only sessions with no announced experiments.
        version = state_dict.get("version")
        if version == 2:
            state_dict.setdefault("stopping", Stopping(False, 1, False).as_dict())
            state_dict.update(
                version=STATE_VERSION, pending=None, safe_history=[], check_prior=None
            )
            state_dict.update(pending_started=False, check_cache=None)
        if state_dict.get("version") != STATE_VERSION:
            raise BayesectError(
                f"State file version {state_dict.get('version')} does not match, "
                "run `git bayesect reset` to start afresh"
            )

        assert set(state_dict) == {
            "version",
            "old_sha",
            "new_sha",
            "beta_priors",
            "stopping",
            "priors",
            "results",
            "pending",
            "pending_started",
            "safe_history",
            "check_prior",
            "check_cache",
        }

        old_sha: bytes = state_dict["old_sha"].encode()
        new_sha: bytes = state_dict["new_sha"].encode()
        beta_priors: BetaPriors = BetaPriors(**state_dict["beta_priors"])
        stopping: Stopping = Stopping(**state_dict["stopping"])
        priors: dict[bytes, float] = {k.encode(): float(v) for k, v in state_dict["priors"].items()}
        results: list[tuple[bytes, Result]] = [
            (k.encode(), Result(v)) for k, v in state_dict["results"]
        ]

        commit_indices = get_commit_indices(repo_path, new_sha.decode())

        state = cls(
            old_sha=old_sha,
            new_sha=new_sha,
            beta_priors=beta_priors,
            stopping=stopping,
            priors=priors,
            results=results,
            commit_indices=commit_indices,
            pending=(
                None if state_dict["pending"] is None else SafeExperiment(**state_dict["pending"])
            ),
            safe_history=state_dict["safe_history"],
            check_prior=state_dict["check_prior"],
            pending_started=state_dict["pending_started"],
            check_cache=state_dict["check_cache"],
        )
        if state.pending_started and not read_only:
            # `run` was stopped (e.g. killed) while testing: its outcome may have been seen. Charge
            # it as missing, so restarts can't filter draws. Read-only, it may be a live run.
            print(
                "The last test run was stopped before its result was recorded: "
                "counting it as a missing result",
                file=sys.stderr,
            )
            state.record_pending(None)
        return state


# ==============================
# Git logic
# ==============================


def smolsha(commit: bytes) -> str:
    return commit.decode()[:10]


@functools.lru_cache(maxsize=None)
def git_dir(path: Path) -> Path:
    # Cached: during a safe `run`, the state is saved before every test.
    path_str = subprocess.check_output(["git", "rev-parse", "--git-dir"], cwd=path)
    return Path(path_str.strip().decode()).absolute()


def parse_commit(repo_path: Path, commit: str | bytes | None) -> bytes:
    if isinstance(commit, bytes):
        assert len(commit) == 40
        return commit

    if commit is None:
        commit = "HEAD"

    commit = subprocess.check_output(["git", "rev-parse", commit], cwd=repo_path).strip()
    assert len(commit) == 40
    return commit


def get_commit_indices(repo_path: Path, head: str | bytes) -> dict[bytes, int]:
    if isinstance(head, bytes):
        head = head.decode()

    # Oldest commit has index 0
    # TODO: think about non-linear history
    # --first-parent: When finding commits to include, follow only the first parent commit
    # upon seeing a merge commit.
    output = subprocess.check_output(
        ["git", "rev-list", "--reverse", "--first-parent", head], cwd=repo_path
    )
    return {line.strip(): i for i, line in enumerate(output.splitlines())}


def get_current_commit(repo_path: Path) -> bytes:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_path).strip()


def get_merge_base(repo_path: Path, c1: bytes, c2: bytes) -> bytes:
    return subprocess.check_output(
        ["git", "merge-base", c1.decode(), c2.decode()], cwd=repo_path
    ).strip()


def get_commit_files_mapping(repo_path: Path, commits: list[bytes]) -> dict[bytes, list[str]]:
    output = subprocess.check_output(
        [
            "git",
            "diff-tree",
            "--stdin",
            "-r",
            "--root",
            "--name-only",
            "--no-renames",
            "-z",
            "--pretty=format:%H%x00",
        ],
        cwd=repo_path,
        input=b"\n".join(commits),
    )
    sections = output.split(b"\x00\x00")
    ret = {}
    for s in sections:
        commit, section = s.split(b"\n")
        commit = commit.rstrip(b"\x00")
        files = section.rstrip(b"\x00").split(b"\x00")
        ret[commit] = [p.decode() for p in files]
    return ret


def get_commit_text_mapping(repo_path: Path, commits: list[bytes]) -> dict[bytes, str]:
    ret = {}
    for commit in commits:
        output = subprocess.check_output(
            [
                "git",
                "show",
                "--no-color",
                "--format=%B",
                "--patch",
                "--no-ext-diff",
                "--no-renames",
                commit.decode(),
            ],
            cwd=repo_path,
        )
        ret[commit] = output.decode(errors="replace")
    return ret


# ==============================
# CLI logic
# ==============================


def get_bisector(state: State, safe: SafeStopping | None = None) -> Bisector:
    old_index = state.commit_indices[state.old_sha]
    new_index = state.commit_indices[state.new_sha]
    assert new_index >= old_index

    prior = np.ones(new_index - old_index + 1)
    for commit_sha, weight in state.priors.items():
        commit_index = state.commit_indices.get(commit_sha, -1)
        if commit_index < old_index:
            continue

        relative_index = new_index - commit_index
        assert 0 <= relative_index <= new_index - old_index
        prior[relative_index] = weight

    bisector = Bisector(
        prior,
        alpha_new=state.beta_priors.alpha_new,
        beta_new=state.beta_priors.beta_new,
        alpha_old=state.beta_priors.alpha_old,
        beta_old=state.beta_priors.beta_old,
    )

    # The assumption check is a running maximum over the results. Reuse its saved value for the
    # results it covers, so that each command doesn't recompute it after every past result.
    checked = 0
    if safe is not None and safe.check_assumptions and state.check_cache is not None:
        if state.check_cache["results"] <= len(state.results):
            checked = state.check_cache["results"]
            saved = state.check_cache["max_log_misfit"]
            safe.max_log_misfit = -np.inf if saved is None else saved

    for i, (commit_sha, result) in enumerate(state.results):
        if result not in {Result.FAIL, Result.PASS}:
            # TODO: handle SKIP maybe by adjusting the prior
            continue

        commit_index = state.commit_indices.get(commit_sha, -1)
        if commit_index < old_index:
            continue

        # Our bisector is set up so that:
        # - index 0 is newest commit
        # - we're recording failures
        relative_index = new_index - commit_index
        assert 0 <= relative_index <= new_index - old_index
        bisector.record(relative_index, result == Result.FAIL)
        if safe is not None:
            safe.record(bisector, relative_index, result == Result.FAIL, check=i >= checked)

    if safe is not None:
        for event in state.safe_history:
            safe.update(SafeExperiment(**event["experiment"]), event["observation"])
        save_check(state, safe)
    return bisector


def save_check(state: State, safe: SafeStopping) -> None:
    """Save the assumption check's running maximum, for the next command to reuse."""
    if safe.check_assumptions:
        saved = None if safe.max_log_misfit == -np.inf else safe.max_log_misfit
        state.check_cache = {"results": len(state.results), "max_log_misfit": saved}


def get_safe_stopping(state: State) -> SafeStopping | None:
    if not state.stopping.safe:
        return None
    old_index = state.commit_indices[state.old_sha]
    new_index = state.commit_indices[state.new_sha]
    safe = SafeStopping(
        new_index - old_index + 1, state.stopping.check_assumptions, state.check_prior
    )
    get_bisector(state, safe)
    return safe


def safe_outcome(safe: SafeStopping, confidence: float, max_range: int = 1) -> tuple[str, int, int]:
    """One of ("misfit" | "none" | "converged" | "range" | "narrowed", left, right) where
    left..right is the range of indices not yet ruled out."""
    surviving = safe.surviving(confidence)
    if safe.misfit(confidence):
        return "misfit", 0, 0
    if len(surviving) == 0:
        return "none", 0, 0
    left, right = int(surviving[0]), int(surviving[-1])
    if left == right:
        return "converged", left, right
    if right - left + 1 <= max_range:
        return "range", left, right
    return "narrowed", left, right


def print_beta_priors(state: State) -> None:
    print(
        f"Prior of failure at new commit is "
        f"{state.beta_priors.alpha_new / (state.beta_priors.alpha_new + state.beta_priors.beta_new):.0%} "
        f"(α_new = {state.beta_priors.alpha_new}, β_new = {state.beta_priors.beta_new})"
    )
    print(
        f"Prior of failure at old commit is "
        f"{state.beta_priors.alpha_old / (state.beta_priors.alpha_old + state.beta_priors.beta_old):.0%} "
        f"(α_old = {state.beta_priors.alpha_old}, β_old = {state.beta_priors.beta_old})"
    )
    print("=" * 80)


def print_status(
    repo_path: Path,
    state: State,
    bisector: Bisector,
    confidence: float = 0.95,
    safe: SafeStopping | None = None,
) -> None:
    if state.stopping.safe:
        if safe is None:
            safe = get_safe_stopping(state)
        assert safe is not None
        print_safe_status(repo_path, state, bisector, safe, state.stopping.confidence)
        return

    new_index = state.commit_indices[state.new_sha]
    old_index = state.commit_indices[state.old_sha]

    dist = bisector.distribution
    dist_p_obs_new, dist_p_obs_old = bisector.empirical_p_obs

    p_obs_new = np.dot(dist_p_obs_new, dist)
    p_obs_old = np.dot(dist_p_obs_old, dist)

    # tie break towards the middle
    max_indices = np.flatnonzero(np.isclose(dist, np.max(dist), rtol=1e-12, atol=1e-15))
    most_likely_index = int(max_indices[np.argmin(np.abs(max_indices - len(dist) // 2))])
    most_likely_prob = dist[most_likely_index]
    most_likely_p_obs_new = dist_p_obs_new[most_likely_index]
    most_likely_p_obs_old = dist_p_obs_old[most_likely_index]

    p90_left, p90_right = bisector.central_range(0.9)
    p90_range = p90_right - p90_left + 1

    indices_commits = {i: c for c, i in state.commit_indices.items()}
    most_likely_commit = smolsha(indices_commits[new_index - most_likely_index])
    p90_left_commit = smolsha(indices_commits[new_index - p90_left])
    p90_right_commit = smolsha(indices_commits[new_index - p90_right])

    if most_likely_prob >= confidence:
        most_likely_commit = smolsha(indices_commits[new_index - most_likely_index])
        msg = (
            f"Bisection converged to {most_likely_commit} ({most_likely_prob:.1%}) "
            f"after {bisector.num_total_observations} observations\n"
            f"Observed subsequent failure rate is {most_likely_p_obs_new:.1%}, "
            f"prior failure rate is {most_likely_p_obs_old:.1%}"
        )
        msg = msg.rstrip()
        print("=" * 80)
        print(msg)
        print("=" * 80)

        print(
            subprocess.check_output(
                ["git", "show", "--color", "--no-patch", "--stat", most_likely_commit],
                cwd=repo_path,
            ).decode()
        )
        print("=" * 80)
    else:
        msg = (
            f"Bisection narrowed to `{p90_right_commit}^...{p90_left_commit}` "
            f"({p90_range} commits) with 90% confidence "
            f"after {bisector.num_total_observations} observations\n"
        )
        msg += f"New failure rate estimate: {p_obs_new:.1%}, old failure rate estimate: {p_obs_old:.1%}\n\n"
        if most_likely_prob >= max(0.1, 2 / (new_index - old_index + 1)):
            msg += f"Most likely commit: {most_likely_commit} ({most_likely_prob:.1%})\n"
            msg += f"Observed subsequent failure rate is {most_likely_p_obs_new:.1%}, "
            msg += f"prior failure rate is {most_likely_p_obs_old:.1%}\n"

        msg = msg.rstrip()
        print("=" * 80)
        print(msg)
        print("=" * 80)


def print_safe_status(
    repo_path: Path,
    state: State,
    bisector: Bisector,
    safe: SafeStopping,
    confidence: float,
) -> None:
    new_index = state.commit_indices[state.new_sha]
    indices_commits = {i: c for c, i in state.commit_indices.items()}
    commit_at = lambda index: smolsha(indices_commits[new_index - index])
    observations = bisector.num_total_observations
    level = format_percent(confidence)

    outcome, left, right = safe_outcome(safe, confidence, state.stopping.max_range)
    dist = bisector.distribution
    dist_p_obs_new, dist_p_obs_old = bisector.empirical_p_obs

    print("=" * 80)
    if outcome == "misfit":
        print(
            f"These results do not fit a single change in failure rate "
            f"({level} confidence) after {observations} observations.\n"
            f"Reruns of a commit may not be independent (e.g. failures come in bursts), "
            f"or flakiness may vary between commits.\n"
            f"No commit is reported."
        )
        print("=" * 80)
    elif outcome == "none":
        print(
            f"No single commit explains these results: every commit was ruled out "
            f"({level} confidence) after {observations} observations.\n"
            f"The failure rate may not change in a single step, "
            f"e.g. several regressions or flakiness that varies between commits."
        )
        print("=" * 80)
    elif outcome == "converged":
        print(
            f"Bisection converged to {commit_at(left)} (safe at {level} confidence) "
            f"after {observations} observations\n"
            f"Observed subsequent failure rate is {dist_p_obs_new[left]:.1%}, "
            f"prior failure rate is {dist_p_obs_old[left]:.1%}"
        )
        print("=" * 80)
        print(
            subprocess.check_output(
                ["git", "show", "--color", "--no-patch", "--stat", commit_at(left)],
                cwd=repo_path,
            ).decode()
        )
        print("=" * 80)
    else:
        verb = "Bisection converged to the range" if outcome == "range" else "Bisection narrowed to"
        msg = (
            f"{verb} `{commit_at(right)}^...{commit_at(left)}` ({right - left + 1} commits) "
            f"(safe at {level} confidence) after {observations} observations\n"
        )
        p_obs_new = np.dot(dist_p_obs_new, dist)
        p_obs_old = np.dot(dist_p_obs_old, dist)
        msg += (
            f"New failure rate estimate: {p_obs_new:.1%}, "
            f"old failure rate estimate: {p_obs_old:.1%}\n"
        )
        print(msg.rstrip())
        print("=" * 80)


def safe_stop(state: State, safe: SafeStopping, confidence: float) -> bool:
    outcome, _, _ = safe_outcome(safe, confidence, state.stopping.max_range)
    return outcome != "narrowed"


def print_stopping(state: State) -> None:
    if not state.stopping.safe:
        print("Stopping when the most likely commit reaches the confidence threshold")
    else:
        msg = (
            "Stopping safely: "
            "the reported commit is wrong with probability at most 1 - confidence"
        )
        if state.stopping.max_range > 1:
            msg += (
                f"\nA range of up to {state.stopping.max_range} commits may be reported "
                "instead of one commit"
            )
        if state.stopping.check_assumptions:
            msg += (
                "\nChecking assumptions: "
                "no commit is reported if the results do not fit a single change"
            )
        print(msg)
    print("=" * 80)


def select_and_checkout(
    repo_path: Path, state: State, bisector: Bisector, testing: bool = False
) -> bytes:
    # `testing`: mark the test as started, so that an unrecorded outcome is charged as missing
    new_index = state.commit_indices[state.new_sha]

    if state.stopping.safe:
        if state.pending is None or testing:
            if state.pending is None:
                state.pending = SafeExperiment.choose(bisector)
            state.pending_started = testing
            # Persist before checkout/test, so a restart cannot replace a draw after seeing it.
            state.dump(repo_path)
        relative_index = state.pending.index
    else:
        relative_index = bisector.select()
    commit_index = new_index - relative_index
    commit_sha = {c: i for i, c in state.commit_indices.items()}[commit_index]

    print(f"Checking out next commit to test: {smolsha(commit_sha)}")
    subprocess.run(
        ["git", "checkout", commit_sha.decode()], cwd=repo_path, check=True, capture_output=True
    )
    return commit_sha


def cli_start(
    old: str,
    new: str | None,
    safe: bool = False,
    max_range: int | None = None,
    check_assumptions: bool = False,
) -> None:
    if max_range is not None and max_range < 1:
        raise BayesectError("--max-range must be at least 1")
    repo_path = Path.cwd()
    new_sha = parse_commit(repo_path, new)
    old_sha = parse_commit(repo_path, old)
    commit_indices = get_commit_indices(repo_path, new_sha)
    if old_sha not in commit_indices:
        print(f"Old commit {smolsha(old_sha)} is not an ancestor of new commit {smolsha(new_sha)}")
        merge_sha = get_merge_base(repo_path, old_sha, new_sha)
        print(f"Using merge base {smolsha(merge_sha)} as old commit instead...")
        old_sha = merge_sha
    assert old_sha in commit_indices

    b = Bisector([])
    default_beta_priors = BetaPriors(
        alpha_new=b.alpha_new,
        beta_new=b.beta_new,
        alpha_old=b.alpha_old,
        beta_old=b.beta_old,
    )

    state = State(
        old_sha=old_sha,
        new_sha=new_sha,
        beta_priors=default_beta_priors,
        # --max-range and --check-assumptions only make sense in safe mode, so they imply it
        stopping=Stopping(
            safe=safe or max_range is not None or check_assumptions,
            max_range=1 if max_range is None else max_range,
            check_assumptions=check_assumptions,
        ),
        priors={},
        results=[],
        commit_indices=commit_indices,
    )
    state.dump(repo_path)

    bisector = get_bisector(state)
    print_status(repo_path, state, bisector)
    print_beta_priors(state)
    print_stopping(state)
    select_and_checkout(repo_path, state, bisector)


def cli_reset() -> None:
    repo_path = Path.cwd()
    (git_dir(repo_path) / STATE_FILENAME).unlink(missing_ok=True)


def append_result(state: State, commit: bytes, result: Result) -> None:
    if state.stopping.safe:
        if state.check_prior is None:
            state.check_prior = get_bisector(state).prior_weights.tolist()
        if state.pending is not None:
            index = state.commit_indices[state.new_sha] - state.commit_indices.get(commit, -1)
            honored = index == state.pending.index
            state.record_pending(
                result == Result.FAIL if honored else None,
                len(state.results) if honored else None,
            )
    state.results.append((commit, result))


def cli_fail(commit: str | bytes | None) -> None:
    repo_path = Path.cwd()
    commit = parse_commit(repo_path, commit)

    state = State.from_git_state(repo_path)
    append_result(state, commit, Result.FAIL)
    state.dump(repo_path)

    bisector = get_bisector(state)
    print_status(repo_path, state, bisector)
    select_and_checkout(repo_path, state, bisector)


def cli_pass(commit: str | bytes | None) -> None:
    repo_path = Path.cwd()
    commit = parse_commit(repo_path, commit)

    state = State.from_git_state(repo_path)
    append_result(state, commit, Result.PASS)
    state.dump(repo_path)

    bisector = get_bisector(state)
    print_status(repo_path, state, bisector)
    select_and_checkout(repo_path, state, bisector)


def cli_undo() -> None:
    repo_path = Path.cwd()

    state = State.from_git_state(repo_path)
    if state.results:
        removed_index = len(state.results) - 1
        commit, result = state.results.pop()
        for event in state.safe_history:
            if event["result_index"] == removed_index:
                event["observation"] = None
                event["result_index"] = None
        match result:
            case Result.FAIL:
                print(f"Undid last observation: git bayesect fail {smolsha(commit)}")
            case Result.PASS:
                print(f"Undid last observation: git bayesect pass {smolsha(commit)}")
            case Result.SKIP:
                print(f"Undid last observation: git bayesect skip {smolsha(commit)}")
    else:
        raise BayesectError("No observation to undo")
    if state.stopping.safe:
        state.record_pending(None)
        # The saved assumption check covers the removed result: replay all results instead.
        state.check_cache = None
        old_index = state.commit_indices[state.old_sha]
        new_index = state.commit_indices[state.new_sha]
        commit_index = state.commit_indices.get(commit, -1)
        if commit_index >= old_index:
            # Check out the same commit again for the corrected result. A query that can only pick
            # that commit carries no evidence, so the correction can't favour any commit.
            state.pending = SafeExperiment.single(new_index - commit_index)
        if state.stopping.check_assumptions:
            print(
                "WARNING: choosing observations to undo because of their outcomes can "
                "invalidate the assumption check. Use undo to correct recording mistakes.",
                file=sys.stderr,
            )
    get_safe_stopping(state)  # rebuilds the assumption check without the removed result
    state.dump(repo_path)

    bisector = get_bisector(state)
    print_status(repo_path, state, bisector)
    select_and_checkout(repo_path, state, bisector)


def cli_run(cmd: list[str], confidence: float | None = None) -> None:
    repo_path = Path.cwd()

    if not cmd:
        raise BayesectError("No command to run")
    if confidence is not None and not 0 < confidence < 1:
        raise BayesectError("Confidence threshold must be between 0 and 1") from None

    state = State.from_git_state(repo_path)
    if confidence is None:
        confidence = state.stopping.confidence
    elif state.stopping.safe and confidence != state.stopping.confidence:
        # Save the new level, so that later commands report at the same level as this run.
        state.stopping.confidence = confidence
        state.dump(repo_path)
    old_index = state.commit_indices[state.old_sha]
    new_index = state.commit_indices[state.new_sha]
    assert new_index >= old_index

    safe = None
    if state.stopping.safe:
        safe = SafeStopping(
            new_index - old_index + 1,
            state.stopping.check_assumptions,
            state.check_prior,
        )
    bisector = get_bisector(state, safe)
    if safe is not None and safe_stop(state, safe, confidence):
        # already safe to stop (e.g. a resumed bisection): don't run the command again
        print_status(repo_path, state, bisector, confidence=confidence, safe=safe)
        return

    try:
        while True:
            commit = select_and_checkout(repo_path, state, bisector, testing=True)
            try:
                proc = subprocess.run(cmd, cwd=repo_path, check=False)
            except KeyboardInterrupt:
                # A result may have been seen before the interruption: charge the missing
                # outcome conservatively instead of letting restarts filter random draws.
                state.record_pending(None)
                raise
            result = Result.PASS if proc.returncode == 0 else Result.FAIL

            experiment = state.pending
            append_result(state, commit, result)
            relative_index = new_index - state.commit_indices[commit]
            assert 0 <= relative_index <= new_index - old_index
            bisector.record(relative_index, result == Result.FAIL)
            if safe is not None:
                safe.record(bisector, relative_index, result == Result.FAIL, experiment)
                save_check(state, safe)

            print_status(repo_path, state, bisector, confidence=confidence, safe=safe)
            if safe is not None:
                if safe_stop(state, safe, confidence):
                    break
            elif bisector.distribution.max() >= confidence:
                break
    finally:
        state.dump(repo_path)


def cli_prior(commit: str | bytes, weight: float) -> None:
    if weight < 0:
        raise BayesectError("Weight for prior must be non-negative") from None
    repo_path = Path.cwd()
    commit = parse_commit(repo_path, commit)

    state = State.from_git_state(repo_path)
    state.priors[commit] = weight
    state.dump(repo_path)
    print(f"Updated prior for {smolsha(commit)} to {weight}")


def cli_priors_from_filenames(filenames_callback: str) -> None:
    if "return" not in filenames_callback:
        raise BayesectError("filenames callback must end with a return statement")
    repo_path = Path.cwd()

    state = State.from_git_state(repo_path)
    files_mapping = get_commit_files_mapping(repo_path, commits=list(state.commit_indices.keys()))

    cb_globals: dict[str, Any] = {}
    cb_locals: dict[str, Any] = {}

    import textwrap

    filenames_callback = textwrap.indent(filenames_callback, "  ")
    filenames_callback = f"def _callback(filenames: list[str]) -> float:\n{filenames_callback}"
    exec(filenames_callback, cb_globals, cb_locals)
    filenames_fn = cb_locals["_callback"]

    for commit, files in files_mapping.items():
        prior = filenames_fn(files)
        if prior is not None:
            if not isinstance(prior, (int, float)) or prior < 0:
                raise BayesectError("Weight for prior must be non-negative number")
            state.priors[commit] = prior
    state.dump(repo_path)
    print(f"Updated priors for {len(state.priors)} commits")

    bisector = get_bisector(state)
    print_status(repo_path, state, bisector)
    select_and_checkout(repo_path, state, bisector)


def cli_priors_from_text(text_callback: str) -> None:
    if "return" not in text_callback:
        raise BayesectError("text callback must end with a return statement")
    repo_path = Path.cwd()

    state = State.from_git_state(repo_path)
    text_mapping = get_commit_text_mapping(repo_path, commits=list(state.commit_indices.keys()))

    cb_globals: dict[str, Any] = {}
    cb_locals: dict[str, Any] = {}

    import textwrap

    text_callback = textwrap.indent(text_callback, "  ")
    text_callback = f"def _callback(text: str) -> float:\n{text_callback}"
    exec(text_callback, cb_globals, cb_locals)
    text_fn = cb_locals["_callback"]

    for commit, text in text_mapping.items():
        prior = text_fn(text)
        if prior is not None:
            if not isinstance(prior, (int, float)) or prior < 0:
                raise BayesectError("Weight for prior must be non-negative number")
            state.priors[commit] = prior
    state.dump(repo_path)
    print(f"Updated priors for {len(state.priors)} commits")

    bisector = get_bisector(state)
    print_status(repo_path, state, bisector)
    select_and_checkout(repo_path, state, bisector)


def cli_beta_priors(
    alpha_new: float | None, beta_new: float | None, alpha_old: float | None, beta_old: float | None
) -> None:
    repo_path = Path.cwd()

    state = State.from_git_state(repo_path)
    if alpha_new is not None:
        state.beta_priors.alpha_new = alpha_new
    if beta_new is not None:
        state.beta_priors.beta_new = beta_new
    if alpha_old is not None:
        state.beta_priors.alpha_old = alpha_old
    if beta_old is not None:
        state.beta_priors.beta_old = beta_old
    state.dump(repo_path)

    bisector = get_bisector(state)
    print_status(repo_path, state, bisector)
    print_beta_priors(state)
    select_and_checkout(repo_path, state, bisector)


def cli_checkout() -> None:
    repo_path = Path.cwd()
    state = State.from_git_state(repo_path)

    bisector = get_bisector(state)
    print_status(repo_path, state, bisector)
    select_and_checkout(repo_path, state, bisector)


def cli_status() -> None:
    repo_path = Path.cwd()
    state = State.from_git_state(repo_path, read_only=True)

    bisector = get_bisector(state)
    safe = get_safe_stopping(state)
    new_index = state.commit_indices[state.new_sha]
    old_index = state.commit_indices[state.old_sha]

    dist = bisector.distribution
    dist_p_obs_new, dist_p_obs_old = bisector.empirical_p_obs
    (yes_new, total_new), (yes_old, total_old) = bisector.empirical_counts
    likely = max(0.1, 2 / (new_index - old_index + 1))
    if safe is not None and safe_outcome(safe, state.stopping.confidence)[0] in ("misfit", "none"):
        likely = np.inf  # don't highlight a likely commit when safe mode refuses to report any

    rows = []
    for commit, i in sorted(state.commit_indices.items(), key=lambda c: c[1], reverse=True):
        relative_index = new_index - i
        if relative_index == 0:
            observations = f"{yes_new[relative_index]}/{total_new[relative_index]}"
        else:
            observations = (
                f"{yes_new[relative_index] - yes_new[relative_index - 1]}/"
                f"{total_new[relative_index] - total_new[relative_index - 1]}"
            )
        rows.append(
            (
                smolsha(commit),
                f"{dist[relative_index]:.1%}",
                observations,
                f"{dist_p_obs_new[relative_index]:.1%}",
                f"({yes_new[relative_index]}/{total_new[relative_index]})",
                f"{dist_p_obs_old[relative_index]:.1%}",
                f"({yes_old[relative_index]}/{total_old[relative_index]})",
                "yes" if dist[relative_index] > likely else "",
            )
        )
        if commit == state.old_sha:
            break

    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]

    for (
        commit_str,
        likelihood,
        observations,
        p_obs_new,
        c_obs_new,
        p_obs_old,
        c_obs_old,
        should_highlight,
    ) in rows:
        if should_highlight:
            print("\033[103m", end="")
        print(
            f"{commit_str:<{widths[0]}} "
            f"likelihood {likelihood:<{widths[1]}}, "
            f"observed {observations:<{widths[2]}} failures, "
            f"subsequent failure rate {p_obs_new:<{widths[3]}} "
            f"{c_obs_new:<{widths[4]}}, "
            f"prior failure rate {p_obs_old:<{widths[5]}} "
            f"{c_obs_old:<{widths[6]}}",
            end="",
        )
        if should_highlight:
            print("\033[0m")
        else:
            print()
    print_status(repo_path, state, bisector)
    if state.pending_started:
        print("A test was started but its result isn't recorded yet: `run` may still be running")


def print_safe_log(repo_path: Path, state: State) -> None:
    print(f"# Restores a snapshot of a safe bisection with {len(state.results)} results.")
    print("# It can't be edited: its evidence depends on the random draws recorded in it.")
    if state.pending is not None:
        pending_index = state.commit_indices[state.new_sha] - state.pending.index
        head = next(c for c, i in state.commit_indices.items() if i == pending_index)
    else:
        head = get_current_commit(repo_path)
    # Use stdin rather than an argument: ordinary histories exceed OS argument limits.
    # Only the standard library is needed, including with an isolated tool installation, and -I
    # keeps modules in the bisected working tree from shadowing it.
    print("python3 -I - <<'GIT_BAYESECT_STATE'")
    print("import pathlib, subprocess")
    print(
        "directory = pathlib.Path(subprocess.check_output("
        "['git', 'rev-parse', '--git-dir'], text=True).strip())"
    )
    print(f"(directory / {STATE_FILENAME!r}).write_text({json.dumps(state.as_dict())!r})")
    print(f"subprocess.run(['git', 'checkout', {head.decode()!r}], check=True)")
    print("GIT_BAYESECT_STATE")


def cli_log() -> None:
    repo_path = Path.cwd()
    state = State.from_git_state(repo_path, read_only=True)
    if state.stopping.safe:
        print_safe_log(repo_path, state)
        return
    print(f"git bayesect start --old {smolsha(state.old_sha)} --new {smolsha(state.new_sha)}")
    print(
        f"git bayesect beta_priors "
        f"--alpha-new {state.beta_priors.alpha_new} "
        f"--beta-new {state.beta_priors.beta_new} "
        f"--alpha-old {state.beta_priors.alpha_old} "
        f"--beta-old {state.beta_priors.beta_old}"
    )

    for commit, weight in state.priors.items():
        print(f"git bayesect prior --commit {smolsha(commit)} --weight {weight}")
    print()

    for commit, result in state.results:
        match result:
            case Result.PASS:
                print(f"git bayesect pass --commit {smolsha(commit)}")
            case Result.FAIL:
                print(f"git bayesect fail --commit {smolsha(commit)}")
            case Result.SKIP:
                print(f"git bayesect skip --commit {smolsha(commit)}")


START_DESC = """\
Start a git bayesection.

Example:
$ git bayesect start --old "HEAD~100"

By default, the bisection stops once the most likely commit reaches the confidence threshold
(95%, or `git bayesect run --confidence`). That confidence is a Bayesian average: for particular
failure rates, or when the test's behaviour doesn't match the model, the reported commit can be
wrong more often than the confidence suggests.

With --safe, the bisection instead stops once every other commit has been ruled out, and the
reported commit is wrong with probability at most 1 - confidence, whatever the failure rates,
as long as the failure rate changes at a single commit. Safe mode picks the commit to test at
random, so test the commits it checks out. It typically takes 18-35% more test runs.

$ git bayesect start --old "HEAD~100" --safe

--max-range N also allows stopping with a range of up to N adjacent commits, which contains the
culprit with the same bound.

--check-assumptions also stops, without reporting a commit, if the results don't fit a single
change in failure rate: e.g. failures that come in bursts because reruns aren't independent.
"""

FAIL_DESC = """\
Record one failure observation.

Example:
$ git bayesect fail

Recording a failure observation at a specific commit:
$ git bayesect fail --commit $SHA
"""

PASS_DESC = """\
Record one success observation.

Example:
$ git bayesect pass

Recording a success observation at a specific commit:
$ git bayesect pass --commit $SHA
"""

UNDO_DESC = """\
Remove the last observation.

Example:
$ git bayesect fail  # oops, meant to run git bayesect pass
$ git bayesect undo

With --safe, undo checks out the same commit again, and the corrected result adds no evidence.
"""

RESET_DESC = """Delete all git bayesection state."""

PRIOR_DESC = """\
Set the prior for a specific commit.

The prior represents how likely it is a given commit introduced a change (prior to making any observations).
The prior is an unnormalised weight that we default to 1.

Marking $SHA as 10x more likely to have introduced a change (relative to other commits):
$ git bayesect prior --commit $SHA --weight 10
"""

PRIOR_FROM_FILENAMES_DESC = """\
Set priors for several commits based on filenames touched by the commit.

See `git bayesect prior --help` for more details on prior.

The `--filenames-callback` is Python code that is executed for each commit given the filenames the commit touches.
If it returns a number, that weight is used as the prior.
If it returns None, any previous prior is preserved.

Mark all files touching a given folder as 10x more likely to have introduced a change:
$ git bayesect prior_from_filenames --filenames-callback 'return 10 if any("suspicious" in f for f in filenames)'
"""

PRIOR_FROM_TEXT_DESC = """\
Set priors for several commits based on commit text (message + patch diff).

See `git bayesect prior --help` for more details on prior.

The `--text-callback` is Python code that is executed for each commit given the commit text.
The text is the commit message followed by the patch produced by `git show`.
If it returns a number, that weight is used as the prior.
If it returns None, any previous prior is preserved.

Mark commits mentioning timeout-related changes as 10x more likely:
$ git bayesect priors_from_text --text-callback 'return 10 if "timeout" in text.lower() else 1'
"""

BETA_PRIORS_DESC = """\
Set the beta priors for the likelihood of failure before and after the change.

Default Beta prior is α_new = 0.5, β_new = 0.5
Default Beta prior is α_old = 0.5, β_old = 0.5

WARNING: Moving the Beta priors away from the Jeffreys prior can cause overconfidence issues.

Any unspecified value is left unchanged.

Turn git bayesect into deterministic git bisect:
$ git bayesect beta_priors --alpha-new 1 --beta-new 0 --alpha-old 0 --beta-old 1
"""

CHECKOUT_DESC = """\
Checkout the most informative commit at which to record an observation.

That is, checkout the commit you should test next.
With --safe, that is the commit drawn at random for the next test.
"""

STATUS_DESC = """Show a detailed status."""

LOG_DESC = """Show commands to reconstruct the bayesection state."""

RUN_DESC = """\
Automatically run a bayesection based on running the given command.

Stops once the most likely commit reaches a confidence threshold, defaulting to 95%.
If the bisection was started with --safe, stops once it is safe to stop at that confidence.
A safe bisection saves this confidence, and later commands report at that level.

Example:
$ git bayesect run --confidence 0.95 python -m pytest
"""


def parse_options(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="git bayesect")

    subparsers = parser.add_subparsers(required=True)

    # ===== start =====
    subparser = subparsers.add_parser(
        "start",
        help=START_DESC.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=START_DESC,
    )
    subparser.set_defaults(command=cli_start)
    subparser.add_argument("--old", help="Old revision (anything commit-ish)", required=True)
    subparser.add_argument(
        "--new",
        default=None,
        help="New revision (anything commit-ish). If not specified, defaults to HEAD",
    )
    subparser.add_argument(
        "--safe",
        action="store_true",
        help="Only stop when the reported commit is wrong with probability at most 1 - confidence",
    )
    subparser.add_argument(
        "--max-range",
        type=int,
        default=None,
        help="Allow stopping with a range of up to this many commits (implies --safe)",
    )
    subparser.add_argument(
        "--check-assumptions",
        action="store_true",
        help="Stop without an answer if results don't fit one change (implies --safe)",
    )

    # ===== fail =====
    subparser = subparsers.add_parser(
        "fail",
        aliases=["failure"],
        help=FAIL_DESC.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=FAIL_DESC,
    )
    subparser.set_defaults(command=cli_fail)
    subparser.add_argument(
        "--commit",
        default=None,
        help="Revision at which to record a failure observation (anything commit-ish). If not specified, defaults to HEAD",
    )

    # ===== pass =====
    subparser = subparsers.add_parser(
        "pass",
        aliases=["success"],
        help=PASS_DESC.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=PASS_DESC,
    )
    subparser.set_defaults(command=cli_pass)
    subparser.add_argument(
        "--commit",
        default=None,
        help="Revision at which to record a success observation (anything commit-ish). If not specified, defaults to HEAD",
    )

    # ===== undo =====
    subparser = subparsers.add_parser(
        "undo",
        help=UNDO_DESC.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=UNDO_DESC,
    )
    subparser.set_defaults(command=cli_undo)

    # ===== reset =====
    subparser = subparsers.add_parser("reset", help=RESET_DESC, description=RESET_DESC)
    subparser.set_defaults(command=cli_reset)

    # ===== prior =====
    subparser = subparsers.add_parser(
        "prior",
        help=PRIOR_DESC.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=PRIOR_DESC,
    )
    subparser.set_defaults(command=cli_prior)
    subparser.add_argument("--commit", required=True)
    subparser.add_argument("--weight", type=float, required=True)

    # ===== priors_from_filenames =====
    subparser = subparsers.add_parser(
        "priors_from_filenames",
        aliases=["priors-from-filenames"],
        help=PRIOR_FROM_FILENAMES_DESC.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=PRIOR_FROM_FILENAMES_DESC,
    )
    subparser.set_defaults(command=cli_priors_from_filenames)
    subparser.add_argument(
        "--filenames-callback", help="Python code returning a float given filenames", required=True
    )

    # ===== priors_from_text =====
    subparser = subparsers.add_parser(
        "priors_from_text",
        aliases=["priors-from-text"],
        help=PRIOR_FROM_TEXT_DESC.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=PRIOR_FROM_TEXT_DESC,
    )
    subparser.set_defaults(command=cli_priors_from_text)
    subparser.add_argument(
        "--text-callback", help="Python code returning a float given commit text", required=True
    )

    # ===== beta_priors =====
    subparser = subparsers.add_parser(
        "beta_priors",
        aliases=["beta-priors"],
        help=BETA_PRIORS_DESC.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=BETA_PRIORS_DESC,
    )
    subparser.add_argument("--alpha-new", type=float)
    subparser.add_argument("--beta-new", type=float)
    subparser.add_argument("--alpha-old", type=float)
    subparser.add_argument("--beta-old", type=float)
    subparser.set_defaults(command=cli_beta_priors)

    # ===== checkout =====
    subparser = subparsers.add_parser(
        "checkout",
        help=CHECKOUT_DESC.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=CHECKOUT_DESC,
    )
    subparser.set_defaults(command=cli_checkout)

    # ===== status =====
    subparser = subparsers.add_parser("status", help=STATUS_DESC, description=STATUS_DESC)
    subparser.set_defaults(command=cli_status)

    # ===== log =====
    subparser = subparsers.add_parser("log", help=LOG_DESC, description=LOG_DESC)
    subparser.set_defaults(command=cli_log)

    # ===== run =====
    subparser = subparsers.add_parser(
        "run",
        help=RUN_DESC.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=RUN_DESC,
    )
    subparser.set_defaults(command=cli_run)
    subparser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Confidence threshold (default: 0.95, or the level a safe bisection saved)",
    )
    subparser.add_argument("cmd", nargs=argparse.REMAINDER)

    # ===== help =====
    subparser = subparsers.add_parser("help", help="Show help")
    subparser.set_defaults(command=lambda: parser.print_help())

    return parser.parse_args(argv)


def main() -> None:
    args = parse_options(sys.argv[1:])
    command = args.__dict__.pop("command")
    try:
        command(**args.__dict__)
    except BayesectError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
