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
import json
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
        alpha_new: float = 0.9,
        beta_new: float = 0.1,
        alpha_old: float = 0.05,
        beta_old: float = 0.95,
    ) -> None:
        if isinstance(prior_weights, list):
            prior_weights = np.array(prior_weights, dtype=np.float64)
        assert isinstance(prior_weights, np.ndarray)
        if np.any(prior_weights < 0):
            raise ValueError("prior_weights must be >= 0")
        self.prior_weights = prior_weights

        self.obs_yes = np.zeros_like(prior_weights, dtype=np.int64)
        self.obs_total = np.zeros_like(prior_weights, dtype=np.int64)

        # E.g. p_obs_new ~ Beta(0.9, 0.1), so E[p_obs_new] = 0.9
        self.alpha_new = alpha_new
        self.beta_new = beta_new

        # E.g. p_obs_old ~ Beta(0.05, 0.95), so E[p_obs_old] = 0.05
        self.alpha_old = alpha_old
        self.beta_old = beta_old

        self.post_weights: ndarray | None = None

    def _maybe_update_posteriors(self) -> None:
        if self.post_weights is None:
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
        right = min(right, len(cumsum) - 1)  # type: ignore[arg-type]

        return int(left), int(right)


# ==============================
# State logic
# ==============================


class BayesectError(Exception):
    pass


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


STATE_FILENAME = "BAYESECT_STATE"
STATE_VERSION = 2


class State:
    def __init__(
        self,
        old_sha: bytes,
        new_sha: bytes,
        beta_priors: BetaPriors,
        priors: dict[bytes, float],
        results: list[tuple[bytes, Result]],
        commit_indices: dict[bytes, int],
    ) -> None:
        self.old_sha = old_sha
        self.new_sha = new_sha
        self.beta_priors = beta_priors
        self.priors = priors
        self.results = results
        self.commit_indices = commit_indices

    def dump(self, repo_path: Path) -> None:
        state_dict = {
            "version": STATE_VERSION,
            "old_sha": self.old_sha.decode(),
            "new_sha": self.new_sha.decode(),
            "beta_priors": self.beta_priors.as_dict(),
            "priors": {k.decode(): v for k, v in self.priors.items()},
            "results": [(k.decode(), v.value) for k, v in self.results],
        }
        with open(git_dir(repo_path) / STATE_FILENAME, "w") as f:
            json.dump(state_dict, f)

    @classmethod
    def from_git_state(cls, repo_path: Path) -> State:
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
            "priors",
            "results",
        }

        old_sha: bytes = state_dict["old_sha"].encode()
        new_sha: bytes = state_dict["new_sha"].encode()
        beta_priors: BetaPriors = BetaPriors(**state_dict["beta_priors"])
        priors: dict[bytes, float] = {k.encode(): float(v) for k, v in state_dict["priors"].items()}
        results: list[tuple[bytes, Result]] = [
            (k.encode(), Result(v)) for k, v in state_dict["results"]
        ]

        commit_indices = get_commit_indices(repo_path, new_sha.decode())

        return cls(
            old_sha=old_sha,
            new_sha=new_sha,
            beta_priors=beta_priors,
            priors=priors,
            results=results,
            commit_indices=commit_indices,
        )


# ==============================
# Git logic
# ==============================


def smolsha(commit: bytes) -> str:
    return commit.decode()[:10]


def git_dir(path: Path) -> Path:
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


# ==============================
# CLI logic
# ==============================


def get_bisector(state: State) -> Bisector:
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


def print_status(repo_path: Path, state: State, bisector: Bisector) -> None:
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
    most_likely_commit = smolsha(indices_commits[new_index - most_likely_index])
    p90_left_commit = smolsha(indices_commits[new_index - p90_left])
    p90_right_commit = smolsha(indices_commits[new_index - p90_right])

    if most_likely_prob >= 0.95:
        most_likely_commit = smolsha(indices_commits[new_index - most_likely_index])
        msg = (
            f"Bisection converged to {most_likely_commit} ({most_likely_prob:.1%}) "
            f"after {bisector.num_total_observations} observations\n"
            f"Subsequent failure rate is {most_likely_p_obs_new:.1%}, "
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
            msg += f"Subsequent failure rate is {most_likely_p_obs_new:.1%}, "
            msg += f"prior failure rate is {most_likely_p_obs_old:.1%}\n"

        msg = msg.rstrip()
        print("=" * 80)
        print(msg)
        print("=" * 80)


def select_and_checkout(repo_path: Path, state: State, bisector: Bisector) -> bytes:
    new_index = state.commit_indices[state.new_sha]

    relative_index = bisector.select()
    commit_index = new_index - relative_index
    commit_sha = {c: i for i, c in state.commit_indices.items()}[commit_index]

    print(f"Checking out next commit to test: {smolsha(commit_sha)}")
    subprocess.run(
        ["git", "checkout", commit_sha.decode()], cwd=repo_path, check=True, capture_output=True
    )
    return commit_sha


def cli_start(old: str, new: str | None) -> None:
    repo_path = Path.cwd()
    new_sha = parse_commit(repo_path, new)
    old_sha = parse_commit(repo_path, old)
    commit_indices = get_commit_indices(repo_path, new_sha)

    state = State(
        old_sha=old_sha,
        new_sha=new_sha,
        beta_priors=BetaPriors(alpha_new=0.9, beta_new=0.1, alpha_old=0.05, beta_old=0.95),
        priors={},
        results=[],
        commit_indices=commit_indices,
    )
    state.dump(repo_path)

    bisector = get_bisector(state)
    print_status(repo_path, state, bisector)
    select_and_checkout(repo_path, state, bisector)


def cli_reset() -> None:
    repo_path = Path.cwd()
    (git_dir(repo_path) / STATE_FILENAME).unlink(missing_ok=True)


def cli_fail(commit: str | bytes | None) -> None:
    repo_path = Path.cwd()
    commit = parse_commit(repo_path, commit)

    state = State.from_git_state(repo_path)
    state.results.append((commit, Result.FAIL))
    state.dump(repo_path)

    bisector = get_bisector(state)
    print_status(repo_path, state, bisector)
    select_and_checkout(repo_path, state, bisector)


def cli_pass(commit: str | bytes | None) -> None:
    repo_path = Path.cwd()
    commit = parse_commit(repo_path, commit)

    state = State.from_git_state(repo_path)
    state.results.append((commit, Result.PASS))
    state.dump(repo_path)

    bisector = get_bisector(state)
    print_status(repo_path, state, bisector)
    select_and_checkout(repo_path, state, bisector)


def cli_undo() -> None:
    repo_path = Path.cwd()

    state = State.from_git_state(repo_path)
    if state.results:
        commit, result = state.results.pop()
        match result:
            case Result.FAIL:
                print(f"Undid last observation: git bayesect fail {smolsha(commit)}")
            case Result.PASS:
                print(f"Undid last observation: git bayesect pass {smolsha(commit)}")
            case Result.SKIP:
                print(f"Undid last observation: git bayesect skip {smolsha(commit)}")
    else:
        raise BayesectError("No observation to undo")
    state.dump(repo_path)

    bisector = get_bisector(state)
    print_status(repo_path, state, bisector)
    select_and_checkout(repo_path, state, bisector)


def cli_run(cmd: list[str]) -> None:
    repo_path = Path.cwd()

    if not cmd:
        raise BayesectError("No command to run")

    state = State.from_git_state(repo_path)
    bisector = get_bisector(state)

    old_index = state.commit_indices[state.old_sha]
    new_index = state.commit_indices[state.new_sha]
    assert new_index >= old_index

    try:
        while True:
            commit = select_and_checkout(repo_path, state, bisector)
            proc = subprocess.run(cmd, cwd=repo_path, check=False)
            result = Result.PASS if proc.returncode == 0 else Result.FAIL

            state.results.append((commit, result))
            relative_index = new_index - state.commit_indices[commit]
            assert 0 <= relative_index <= new_index - old_index
            bisector.record(relative_index, result == Result.FAIL)

            print_status(repo_path, state, bisector)
            if bisector.distribution.max() >= 0.95:
                break
    finally:
        state.dump(repo_path)


def cli_prior(commit: str | bytes, weight: float) -> None:
    repo_path = Path.cwd()
    commit = parse_commit(repo_path, commit)

    state = State.from_git_state(repo_path)
    state.priors[commit] = weight
    state.dump(repo_path)
    print(f"Updated prior for {smolsha(commit)} to {weight}")


def cli_priors_from_filenames(filenames_callback: str) -> None:
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
            assert isinstance(prior, (int, float))
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
    print(f"Updated beta priors to {state.beta_priors.as_dict()}")

    bisector = get_bisector(state)
    print_status(repo_path, state, bisector)
    select_and_checkout(repo_path, state, bisector)


def cli_checkout() -> None:
    repo_path = Path.cwd()
    state = State.from_git_state(repo_path)

    bisector = get_bisector(state)
    print_status(repo_path, state, bisector)
    select_and_checkout(repo_path, state, bisector)


def cli_status() -> None:
    repo_path = Path.cwd()
    state = State.from_git_state(repo_path)

    bisector = get_bisector(state)
    new_index = state.commit_indices[state.new_sha]
    old_index = state.commit_indices[state.old_sha]

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
                smolsha(commit),
                f"{dist[relative_index]:.1%}",
                observations,
                f"{dist_p_obs_new[relative_index]:.1%}",
                f"({yes_new[relative_index]}/{total_new[relative_index]})",
                f"{dist_p_obs_old[relative_index]:.1%}",
                f"({yes_old[relative_index]}/{total_old[relative_index]})",
                "yes" if dist[relative_index] > max(0.1, 2 / (new_index - old_index + 1)) else "",
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


def cli_log() -> None:
    repo_path = Path.cwd()
    state = State.from_git_state(repo_path)
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

    subparser = subparsers.add_parser("undo")
    subparser.set_defaults(command=cli_undo)

    subparser = subparsers.add_parser("reset")
    subparser.set_defaults(command=cli_reset)

    subparser = subparsers.add_parser("prior")
    subparser.add_argument("--commit", required=True)
    subparser.add_argument("--weight", type=float, required=True)
    subparser.set_defaults(command=cli_prior)

    subparser = subparsers.add_parser("priors_from_filenames", aliases=["priors-from-filenames"])
    subparser.add_argument(
        "--filenames-callback", help="Python code returning a float given filenames", required=True
    )
    subparser.set_defaults(command=cli_priors_from_filenames)

    subparser = subparsers.add_parser("beta_priors", aliases=["beta-priors"])
    subparser.add_argument("--alpha-new", type=float)
    subparser.add_argument("--beta-new", type=float)
    subparser.add_argument("--alpha-old", type=float)
    subparser.add_argument("--beta-old", type=float)
    subparser.set_defaults(command=cli_beta_priors)

    subparser = subparsers.add_parser("checkout")
    subparser.set_defaults(command=cli_checkout)

    subparser = subparsers.add_parser("status")
    subparser.set_defaults(command=cli_status)

    subparser = subparsers.add_parser("log")
    subparser.set_defaults(command=cli_log)

    subparser = subparsers.add_parser("run")
    subparser.set_defaults(command=cli_run)
    subparser.add_argument("cmd", nargs=argparse.REMAINDER)

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
