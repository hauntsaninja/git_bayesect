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
        assert self.post_weights is not None
        assert 0 <= mass <= 1
        cumsum = np.cumsum(self.post_weights)

        tail = (1 - mass) / 2
        left = np.searchsorted(cumsum, tail, side="left")
        right = np.searchsorted(cumsum, 1 - tail, side="right")
        right = min(right, len(cumsum) - 1)  # type: ignore[call-overload]

        return int(left), int(right)


class BayesectError(Exception):
    pass


class Result(enum.Enum):
    FAIL = "fail"
    PASS = "pass"
    SKIP = "skip"


STATE_FILENAME = "BAYESECT_STATE"


class State:
    def __init__(
        self,
        old_sha: bytes,
        new_sha: bytes,
        priors: dict[bytes, float],
        results: list[tuple[bytes, Result]],
        commit_indices: dict[bytes, int],
    ) -> None:
        self.old_sha = old_sha
        self.new_sha = new_sha
        self.priors = priors
        self.results = results
        self.commit_indices = commit_indices

    def dump(self, repo_path: Path) -> None:
        state_dict = {
            "old_sha": self.old_sha.decode(),
            "new_sha": self.new_sha.decode(),
            "priors": {k.decode(): v for k, v in self.priors.items()},
            "results": [(k.decode(), v.value) for k, v in self.results],
        }
        with open(repo_path / ".git" / STATE_FILENAME, "w") as f:
            json.dump(state_dict, f)

    @classmethod
    def from_git_state(cls, repo_path: Path) -> State:
        try:
            with open(repo_path / ".git" / STATE_FILENAME) as f:
                state_dict = json.load(f)
        except FileNotFoundError:
            raise BayesectError("No state file found, run `git bayesect start` first")

        assert set(state_dict) == {"old_sha", "new_sha", "priors", "results"}

        old_sha: bytes = state_dict["old_sha"].encode()
        new_sha: bytes = state_dict["new_sha"].encode()
        priors: dict[bytes, float] = {k.encode(): float(v) for k, v in state_dict["priors"].items()}
        results: list[tuple[bytes, Result]] = [
            (k.encode(), Result(v)) for k, v in state_dict["results"]
        ]

        commit_indices = get_commit_indices(repo_path, new_sha.decode())

        return cls(
            old_sha=old_sha,
            new_sha=new_sha,
            priors=priors,
            results=results,
            commit_indices=commit_indices,
        )


def resolve_commit(commit_indices: dict[bytes, int], commit: str | bytes) -> bytes:
    if isinstance(commit, bytes):
        assert len(commit) == 40
        return commit

    candidates = [c for c in commit_indices if c.startswith(commit.encode())]
    if len(candidates) > 1:
        raise BayesectError(
            f"Commit {commit} is ambiguous, {len(candidates)} potential matches found"
        )
    if not candidates:
        range_min = min(commit_indices, key=lambda c: commit_indices[c]).decode()[:8]
        range_max = max(commit_indices, key=lambda c: commit_indices[c]).decode()[:8]
        raise BayesectError(f"No commits found for {commit} in range {range_min}...{range_max}")
    return candidates[0]


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


def get_bisector(state: State) -> Bisector:
    old_index = state.commit_indices[state.old_sha]
    new_index = state.commit_indices[state.new_sha]

    prior = np.ones(new_index - old_index + 1)
    for commit_sha, weight in state.priors.items():
        commit_index = state.commit_indices.get(commit_sha, -1)
        if commit_index < old_index:
            continue

        relative_index = new_index - commit_index
        assert 0 <= relative_index <= new_index - old_index
        prior[relative_index] = weight

    bisector = Bisector(prior)

    for commit_sha, result in state.results:
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

    return bisector


def print_status(state: State, bisector: Bisector) -> None:
    new_index = state.commit_indices[state.new_sha]
    old_index = state.commit_indices[state.old_sha]

    dist = bisector.distribution
    dist_p_obs_new, dist_p_obs_old = bisector.empirical_p_obs

    p_obs_new = (dist_p_obs_new * dist).sum()
    p_obs_old = (dist_p_obs_old * dist).sum()

    # TODO: maybe tie break argmax with most central?
    most_likely_index = int(np.argmax(dist))
    most_likely_prob = dist[most_likely_index]
    most_likely_p_obs_new = dist_p_obs_new[most_likely_index]
    most_likely_p_obs_old = dist_p_obs_old[most_likely_index]

    p90_left, p90_right = bisector.central_range(0.9)
    p90_range = p90_right - p90_left + 1

    indices_commits = {i: c for c, i in state.commit_indices.items()}
    most_likely_commit = indices_commits[new_index - most_likely_index].decode()[:8]
    p90_left_commit = indices_commits[new_index - p90_left].decode()[:8]
    p90_right_commit = indices_commits[new_index - p90_right].decode()[:8]

    print("=" * 80)

    if most_likely_prob >= 0.95:
        most_likely_commit = indices_commits[new_index - most_likely_index].decode()[:8]
        msg = (
            f"Bisection converged to {most_likely_commit} ({most_likely_prob:.1%}) "
            f"after {bisector.num_total_observations} observations\n"
            f"Subsequent failure rate is {most_likely_p_obs_new:.1%}, "
            f"prior failure rate is {most_likely_p_obs_old:.1%}"
        )
        msg = msg.rstrip()
        print(msg)
    else:
        msg = (
            f"Bisection narrowed to `{p90_right_commit}^...{p90_left_commit}` "
            f"({p90_range} commits) with 90% confidence "
            f"after {bisector.num_total_observations} observations\n"
        )
        msg += f"New failure rate estimate: {p_obs_new:.1%}, old failure rate estimate: {p_obs_old:.1%}\n\n"
        if most_likely_prob >= max(0.1, 2 / (new_index - old_index + 1)):
            msg += f"Most likely commit: {most_likely_commit} ({most_likely_prob:.1%})\n"
            msg += f"Subsequent failure rate is {most_likely_p_obs_new:.1%}, "
            msg += f"prior failure rate is {most_likely_p_obs_old:.1%}\n"

        msg = msg.rstrip()
        print(msg)

    print("=" * 80)


def select_and_checkout(repo_path: Path, state: State, bisector: Bisector) -> None:
    new_index = state.commit_indices[state.new_sha]

    relative_index = bisector.select()
    commit_index = new_index - relative_index
    commit_sha = {c: i for i, c in state.commit_indices.items()}[commit_index]

    print(f"Checking out next commit to test: {commit_sha.decode()[:8]}")
    subprocess.run(
        ["git", "checkout", commit_sha.decode()], cwd=repo_path, check=True, capture_output=True
    )


def cli_start(old: str, new: str | bytes | None) -> None:
    repo_path = Path.cwd()
    if new is None:
        new = get_current_commit(repo_path)

    commit_indices = get_commit_indices(repo_path, new)

    old_sha = resolve_commit(commit_indices, old)
    new_sha = resolve_commit(commit_indices, new)

    state = State(
        old_sha=old_sha,
        new_sha=new_sha,
        priors={},
        results=[],
        commit_indices=commit_indices,
    )
    state.dump(repo_path)

    bisector = get_bisector(state)
    print_status(state, bisector)
    select_and_checkout(repo_path, state, bisector)


def cli_reset() -> None:
    repo_path = Path.cwd()
    (repo_path / ".git" / STATE_FILENAME).unlink(missing_ok=True)


def cli_fail(commit: str | bytes | None) -> None:
    repo_path = Path.cwd()
    if commit is None:
        commit = get_current_commit(repo_path)

    state = State.from_git_state(repo_path)
    state.results.append((resolve_commit(state.commit_indices, commit), Result.FAIL))
    state.dump(repo_path)

    bisector = get_bisector(state)
    print_status(state, bisector)
    select_and_checkout(repo_path, state, bisector)


def cli_pass(commit: str | bytes | None) -> None:
    repo_path = Path.cwd()
    if commit is None:
        commit = get_current_commit(repo_path)

    state = State.from_git_state(repo_path)
    state.results.append((resolve_commit(state.commit_indices, commit), Result.PASS))
    state.dump(repo_path)

    bisector = get_bisector(state)
    print_status(state, bisector)
    select_and_checkout(repo_path, state, bisector)


def cli_log() -> None:
    repo_path = Path.cwd()
    state = State.from_git_state(repo_path)

    bisector = get_bisector(state)
    new_index = state.commit_indices[state.new_sha]

    dist = bisector.distribution
    dist_p_obs_new, dist_p_obs_old = bisector.empirical_p_obs
    (yes_new, total_new), (yes_old, total_old) = bisector.empirical_counts

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
                commit.decode()[:8],
                f"{dist[relative_index]:.1%}",
                observations,
                f"{dist_p_obs_new[relative_index]:.1%}",
                f"({yes_new[relative_index]}/{total_new[relative_index]})",
                f"{dist_p_obs_old[relative_index]:.1%}",
                f"({yes_old[relative_index]}/{total_old[relative_index]})",
            )
        )
        if commit == state.old_sha:
            break

    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]

    for commit_str, likelihood, observations, p_obs_new, c_obs_new, p_obs_old, c_obs_old in rows:
        print(
            f"{commit_str:<{widths[0]}} "
            f"likelihood {likelihood:<{widths[1]}}, "
            f"observed {observations:<{widths[2]}} failures, "
            f"subsequent failure rate {p_obs_new:<{widths[3]}} "
            f"{c_obs_new:<{widths[4]}}, "
            f"prior failure rate {p_obs_old:<{widths[5]}} "
            f"{c_obs_old:<{widths[6]}}"
        )

    print_status(state, bisector)


def parse_options(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(required=True)

    subparser = subparsers.add_parser("start")
    subparser.set_defaults(command=cli_start)
    subparser.add_argument("--old", help="Old commit hash", required=True)
    subparser.add_argument("--new", help="New commit hash", default=None)

    subparser = subparsers.add_parser("fail", aliases=["failure"])
    subparser.set_defaults(command=cli_fail)
    subparser.add_argument("--commit", default=None)

    subparser = subparsers.add_parser("pass", aliases=["success"])
    subparser.set_defaults(command=cli_pass)
    subparser.add_argument("--commit", default=None)

    subparser = subparsers.add_parser("reset")
    subparser.set_defaults(command=cli_reset)

    subparser = subparsers.add_parser("log")
    subparser.set_defaults(command=cli_log)

    return parser.parse_args(argv)


def main() -> None:
    args = parse_options(sys.argv[1:])
    command = args.__dict__.pop("command")
    command(**args.__dict__)


if __name__ == "__main__":
    main()
