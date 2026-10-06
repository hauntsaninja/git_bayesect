import os
import random
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from git_bayesect import Bisector, SafeExperiment, SafeStopping


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep git away from the developer's repository and configuration. For example, git hooks
    export GIT_DIR and GIT_INDEX_FILE, and a global config may require signed commits."""
    for name in list(os.environ):
        if name.startswith("GIT_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


def test_bisector_posteriors() -> None:
    b = Bisector([1] * 5)
    # b.record(2, observation=True, p_obs_new=1, p_obs_old=0)
    for _ in range(100):
        b.record(0, True)
        b.record(4, False)
    b.record(2, True)
    assert np.allclose(b.distribution, [0, 0, 0.5, 0.5, 0], atol=0.01)

    b = Bisector([1] * 5)
    # b.record(2, observation=True, p_obs_new=1, p_obs_old=0.1)
    for _ in range(100):
        b.record(0, True)
    for _ in range(10):
        b.record(4, True)
    for _ in range(90):
        b.record(4, False)

    b.record(2, True)
    assert np.allclose(b.distribution, [0.0455, 0.0455, 0.4545, 0.4545, 0], atol=0.01)

    b = Bisector([1] * 5)
    # b.record(2, observation=True, p_obs_new=0.8, p_obs_old=0.2)
    for _ in range(80):
        b.record(0, True)
    for _ in range(20):
        b.record(0, False)
    for _ in range(20):
        b.record(4, True)
    for _ in range(80):
        b.record(4, False)

    b.record(2, True)
    assert np.allclose(b.distribution, [0.1, 0.1, 0.4, 0.4, 0], atol=0.01)

    b = Bisector([1] * 5)
    # b.record(2, observation=False, p_obs_new=1, p_obs_old=0)
    for _ in range(100):
        b.record(0, True)
    for _ in range(100):
        b.record(4, False)

    b.record(2, False)
    assert np.allclose(b.distribution, [0.5, 0.5, 0, 0, 0], atol=0.01)

    b = Bisector([1] * 5)
    # b.record(2, observation=False, p_obs_new=1, p_obs_old=0.1)
    for _ in range(100):
        b.record(0, True)
    for _ in range(10):
        b.record(4, True)
    for _ in range(90):
        b.record(4, False)

    b.record(2, False)
    assert np.allclose(b.distribution, [0.5, 0.5, 0, 0, 0], atol=0.01)

    b = Bisector([1] * 5)
    # b.record(2, observation=False, p_obs_new=0.8, p_obs_old=0.2)
    for _ in range(80):
        b.record(0, True)
    for _ in range(20):
        b.record(0, False)
    for _ in range(20):
        b.record(4, True)
    for _ in range(80):
        b.record(4, False)

    b.record(2, False)
    assert np.allclose(b.distribution, [0.4, 0.4, 0.1, 0.1, 0], atol=0.01)


def test_bisector_central_range() -> None:
    b = Bisector([1] * 3)
    b._post_weights = np.array([0.1, 0.8, 0.1])
    assert b.central_range(0) == (1, 1)
    assert b.central_range(0.5) == (1, 1)
    assert b.central_range(0.799) == (1, 1)
    assert b.central_range(0.801) == (0, 2)
    assert b.central_range(1.0) == (0, 2)

    b = Bisector([1] * 5)
    b._post_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    assert b.central_range(0) == (2, 2)
    assert b.central_range(0.3) == (1, 3)
    assert b.central_range(0.7) == (0, 4)
    assert b.central_range(1.0) == (0, 4)

    b = Bisector([1] * 5)
    b._post_weights = np.array([0.4, 0.4, 0.1, 0.1, 0.2])
    assert b.central_range(0) == (1, 1)
    assert b.central_range(0.2) == (0, 1)
    assert b.central_range(0.599) == (0, 1)
    assert b.central_range(0.601) == (0, 2)
    assert b.central_range(0.8) == (0, 3)
    assert b.central_range(1.0) == (0, 4)


def test_bisector_select_tie_breaks_to_middle_commit() -> None:
    # Given we currently have symmetric beta priors, we should tie break towards middle
    assert Bisector([1] * 5).select() == 2
    assert Bisector([1] * 6).select() == 3


def test_bisector_bisect() -> None:
    gen = random.Random()
    seed = gen.randbytes(16)
    print(f"seed: {seed!r}")
    gen = random.Random(seed)

    N = 32
    # [0, N - 2] to ensure that there is a commit before the change point
    B = gen.randint(0, N - 2)

    p_obs_new = gen.random() * 0.5 + 0.5
    p_obs_old = gen.random() * 0.2

    print("=" * 80)
    print(f"N: {N}")
    print(f"B: {B}")
    print(f"p_obs_new: {p_obs_new:.4f}")
    print(f"p_obs_old: {p_obs_old:.4f}")
    print("=" * 80)

    # Using a per-commit random generator lets us record the same sequence of observations even if
    # we experiment with different selections
    randgens = [random.Random(gen.randbytes(64)) for _ in range(N)]

    np.set_printoptions(formatter={"float": lambda f: f"{f:.4f}"})
    bisect = Bisector([1] * N)

    for iteration in range(1000):
        print("=" * 80)
        print(f"iteration: {iteration}")

        dist = bisect.distribution

        print(dist)
        print(f"answer prob:        {dist[B]:.4f}")
        print(f"likeliest prob:     {dist[np.argmax(dist)]:.4f}")
        print(f"index vs answer:    {np.argmax(dist)} vs {B}")
        print(f"entropy:            {bisect.entropy:.4f}")

        p_obs_new_dist, p_obs_old_dist = bisect.empirical_p_obs
        print(f"estimate p_obs_new: {(p_obs_new_dist * dist).sum():.4f}")
        print(f"estimate p_obs_old: {(p_obs_old_dist * dist).sum():.4f}")

        if dist[np.argmax(dist)] > 0.99:
            print("=" * 80)
            print("converged!")
            print("=" * 80)
            break

        select = bisect.select()
        print(f"selected index:    {select}")

        choice = ""  # input("index: ")
        index = eval(choice) if choice.strip() else select

        p_obs = p_obs_new if index <= B else p_obs_old
        obs = randgens[index].random() < p_obs
        print(f"tested index: {index}, observation: {obs}, p_obs: {p_obs:0.4f}")
        if (obs and p_obs < 0.5) or (not obs and p_obs > 0.5):
            print("(observation was less likely)")
        bisect.record(index, obs)
    else:
        raise RuntimeError("failed to converge")

    print("=" * 80)
    print(f"N: {N}")
    print(f"B: {B}")
    print(f"p_obs_new: {p_obs_new:.4f}")
    print(f"p_obs_old: {p_obs_old:.4f}")
    print("=" * 80)

    dist = bisect.distribution
    likeliest_index = np.argmax(dist)
    assert likeliest_index == B


def test_safe_stopping_rules_out_what_the_data_contradicts() -> None:
    b = Bisector([1] * 6)
    g = SafeStopping(6)
    rng = np.random.default_rng(42)
    # Only the ends are measured in the first announced randomized experiment.
    for _ in range(120):
        index = int(rng.choice([0, 5]))
        experiment = SafeExperiment(index, [0, 5], [0.5, 0.5], [0.9, 0.1])
        b.record(index, index == 0)
        g.record(b, index, index == 0, experiment)
    assert list(g.surviving(0.95)) == [0, 1, 2, 3, 4]

    # The next pair distinguishes every boundary except the one between 2 and 3.
    for _ in range(120):
        index = int(rng.choice([2, 3]))
        experiment = SafeExperiment(index, [2, 3], [0.5, 0.5], [0.9, 0.1])
        b.record(index, index == 2)
        g.record(b, index, index == 2, experiment)
    assert list(g.surviving(0.95)) == [2]


def test_safe_stopping_assumption_check() -> None:
    def run(check_assumptions: bool, bursty: bool) -> SafeStopping:
        gen = random.Random(0)
        b = Bisector([1] * 4)
        g = SafeStopping(4, check_assumptions=check_assumptions)
        last = [False] * 4
        for _ in range(400):
            index = gen.randrange(4)
            p = 0.6 if index <= 1 else 0.2
            fresh = gen.random() < p
            # with bursty reruns, most runs just repeat the previous outcome at that commit
            obs = last[index] if bursty and gen.random() < 0.8 else fresh
            last[index] = obs
            b.record(index, obs)
            g.record(b, index, obs)
        return g

    assert not run(check_assumptions=True, bursty=False).misfit(0.95)
    assert run(check_assumptions=True, bursty=True).misfit(0.95)
    assert not run(check_assumptions=False, bursty=True).misfit(0.95)


def test_safe_stopping_is_valid() -> None:
    # The true B should be ruled out in at most 1 - confidence of bisections, at any point. With
    # confidence 0.8 the bound is 20%; the actual rate is far lower, so this check is not flaky.
    gen = random.Random(1234)
    N, confidence, trials = 8, 0.8, 100
    ruled_out = 0
    rng = np.random.default_rng(1234)
    for _ in range(trials):
        B = gen.randint(0, N - 2)
        p_obs_new, p_obs_old = 0.6, 0.1
        randgens = [random.Random(gen.randbytes(16)) for _ in range(N)]
        b = Bisector([1] * N)
        g = SafeStopping(N)
        for _ in range(2000):
            experiment = SafeExperiment.choose(b, rng)
            index = experiment.index
            obs = randgens[index].random() < (p_obs_new if index <= B else p_obs_old)
            b.record(index, obs)
            g.record(b, index, obs, experiment)
            if len(g.surviving(confidence)) <= 1:
                break
        ruled_out += B not in g.surviving(confidence)
    assert ruled_out / trials <= 0.1


def make_repo(path: Path, failing: list[bool]) -> tuple[str, str]:
    """A repository with one commit per entry, oldest first, whose file `fails` says whether a
    test at that commit fails. Returns the oldest and newest commits."""
    import subprocess

    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=path, text=True).strip()

    git("init", "-q")
    git("config", "user.name", "test")
    git("config", "user.email", "test@example.com")
    for i, fails in enumerate(failing):
        (path / "fails").write_text("1" if fails else "0")
        git("add", "fails")
        git("commit", "-q", "--allow-empty", "-m", f"commit {i}")
    return git("rev-list", "--max-parents=0", "HEAD"), git("rev-parse", "HEAD")


def test_cli_start_options_imply_safe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from git_bayesect import State, parse_options

    old, new = make_repo(tmp_path, [False] * 5)
    monkeypatch.chdir(tmp_path)
    for options, safe, max_range, check in [
        ([], False, 1, False),
        (["--safe"], True, 1, False),
        (["--max-range", "1"], True, 1, False),
        (["--max-range", "3"], True, 3, False),
        (["--check-assumptions"], True, 1, True),
    ]:
        args = parse_options(["start", "--old", old, "--new", new, *options]).__dict__
        args.pop("command")(**args)
        stopping = State.from_git_state(tmp_path).stopping
        assert (stopping.safe, stopping.max_range, stopping.check_assumptions) == (
            safe,
            max_range,
            check,
        ), options


def test_cli_run_stops_safely(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from git_bayesect import State, cli_run, cli_start, get_safe_stopping, safe_outcome

    # commits 0..5 pass, 6..9 fail: the culprit is commit 6, 3 back from the newest
    old, new = make_repo(tmp_path, [i >= 6 for i in range(10)])
    monkeypatch.chdir(tmp_path)
    test = ["sh", "-c", "test $(cat fails) = 0"]

    cli_start(old=old, new=new, safe=True)
    cli_run(test)
    state = State.from_git_state(tmp_path)
    observed = len(state.results)
    assert observed > 0
    safe = get_safe_stopping(state)
    assert safe is not None
    assert safe_outcome(safe, 0.95) == ("converged", 3, 3)
    # already safe to stop: running again records nothing more
    cli_run(test)
    assert len(State.from_git_state(tmp_path).results) == observed

    # a range covering every commit is reported before any test runs
    cli_start(old=old, new=new, max_range=10)
    cli_run(test)
    assert State.from_git_state(tmp_path).results == []


def test_cli_safe_stopping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from git_bayesect import (
        State,
        cli_beta_priors,
        cli_fail,
        cli_log,
        cli_prior,
        cli_start,
        cli_undo,
    )

    old, new = make_repo(tmp_path, [False] * 5)
    monkeypatch.chdir(tmp_path)
    cli_start(old=old, new=new, max_range=3)
    state = State.from_git_state(tmp_path)
    # --max-range implies --safe
    assert state.stopping.safe
    assert state.stopping.max_range == 3
    assert not state.stopping.check_assumptions

    capsys.readouterr()
    cli_log()
    assert '"max_range": 3' in capsys.readouterr().out

    # setting priors before any observation keeps the bound: no warning
    cli_prior(commit="HEAD", weight=2)
    assert "WARNING" not in capsys.readouterr().err
    # New priors affect future forecasts; saved experiments keep the old evidence valid.
    cli_fail(commit=None)
    capsys.readouterr()
    cli_beta_priors(alpha_new=0.9, beta_new=None, alpha_old=None, beta_old=None)
    assert "WARNING" not in capsys.readouterr().err
    # Raw RIPr conservatively accounts for an undone observation.
    cli_undo()
    assert "WARNING" not in capsys.readouterr().err


def test_randomized_factors_and_canceled_draws() -> None:
    """Exhaust the next arm/outcome, including selective cancellation after seeing either."""
    indices, weights, forecasts = [0, 2, 5], [0.2, 0.5, 0.3], [0.8, 0.45, 0.1]
    for boundary in range(6):
        expectation = 0.0
        selective_expectation = 0.0
        for index, weight in zip(indices, weights):
            experiment = SafeExperiment(index, indices, weights, forecasts)
            rate = 0.3 if index <= boundary else 0.7
            for y, probability in [(False, 1 - rate), (True, rate)]:
                side = [k for k, i in enumerate(indices) if (i <= boundary) == (index <= boundary)]
                projected = sum(weights[k] * forecasts[k] for k in side) / sum(
                    weights[k] for k in side
                )
                predicted = forecasts[indices.index(index)]
                direct = predicted / projected if y else (1 - predicted) / (1 - projected)
                actual = np.exp(experiment.log_factor(6, y)[boundary])
                assert actual == pytest.approx(direct)
                expectation += weight * probability * actual
                # Allow cancellation to depend even on the otherwise unrecorded outcome.
                factor = experiment.log_factor(6, None if index == 2 or y else y)[boundary]
                selective_expectation += weight * probability * np.exp(factor)
        assert expectation == pytest.approx(1.0)
        assert selective_expectation <= 1 + 1e-14
    for index in range(6):
        experiment = SafeExperiment(index, [index], [1.0], [0.3])
        for outcome in [False, True, None]:
            assert np.allclose(experiment.log_factor(6, outcome), 0)


def test_safe_resume_prior_changes_and_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import subprocess
    from git_bayesect import (
        State,
        append_result,
        cli_beta_priors,
        cli_checkout,
        cli_log,
        cli_prior,
        cli_start,
        get_bisector,
        get_safe_stopping,
        git_dir,
        STATE_FILENAME,
    )

    old, new = make_repo(tmp_path, [False] * 4 + [True] * 4)
    monkeypatch.chdir(tmp_path)
    cli_start(old, new, safe=True, check_assumptions=True)
    state = State.from_git_state(tmp_path)
    rng = np.random.default_rng(9832)
    # Exercise persisted, informative queries, without subprocess overhead per observation.
    commits = {i: c for c, i in state.commit_indices.items()}
    for _ in range(35):
        b = get_bisector(state)
        if state.pending is None:
            state.pending = SafeExperiment.choose(b, rng)
        commit = commits[state.commit_indices[state.new_sha] - state.pending.index]
        from git_bayesect import Result

        result = Result.FAIL if state.pending.index < 4 else Result.PASS
        append_result(state, commit, result)
    state.dump(tmp_path)
    safe = get_safe_stopping(state)
    assert safe is not None
    before = safe.log_evidence.copy()
    before_check = safe.max_log_misfit
    cli_prior(new, 20)
    cli_beta_priors(0.9, 0.1, 0.05, 0.95)
    changed = State.from_git_state(tmp_path)
    after = get_safe_stopping(changed)
    assert after is not None
    assert np.array_equal(after.log_evidence, before)
    assert after.max_log_misfit == before_check
    pending = changed.pending
    cli_checkout()
    assert State.from_git_state(tmp_path).pending == pending
    capsys.readouterr()
    cli_log()
    log = capsys.readouterr().out
    # A safe log is a snapshot: only comments come before the script that restores it.
    script = log.index("python3 -I - <<")
    assert all(not line or line.startswith("#") for line in log[:script].splitlines())
    path = git_dir(tmp_path) / STATE_FILENAME
    head = lambda: subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path)
    expected = path.read_text()
    expected_head = head()
    path.unlink()
    subprocess.run(["git", "checkout", old], check=True, cwd=tmp_path)
    # Replay the log exactly as printed, from a different checkout. Modules in the bisected
    # working tree must not shadow the standard library.
    (tmp_path / "random.py").write_text("raise ImportError('shadows the standard library')\n")
    subprocess.run(
        ["sh", "-e"], input=log, text=True, check=True, cwd=tmp_path, capture_output=True
    )
    assert path.read_text() == expected
    assert head() == expected_head
    restored = get_safe_stopping(State.from_git_state(tmp_path))
    assert restored is not None
    assert np.array_equal(restored.log_evidence, before)

    # A typical longer history exceeds Linux's per-argument limit; stdin still restores it.
    state = State.from_git_state(tmp_path)
    b = get_bisector(state)
    for _ in range(1000):
        if state.pending is None:
            state.pending = SafeExperiment.choose(b, rng)
        index = state.pending.index
        append_result(state, commits[state.commit_indices[state.new_sha] - index], Result.FAIL)
        b.record(index, True)
    assert state.pending is None
    state.dump(tmp_path)
    cli_log()
    long_log = capsys.readouterr().out
    assert len(long_log.encode()) > 128 * 1024
    expected = path.read_text()
    path.unlink()
    # With no pending query, the restore checks out the commit that was checked out at the time.
    expected_head = head()
    subprocess.run(["git", "checkout", old], check=True, cwd=tmp_path, capture_output=True)
    subprocess.run(
        ["sh", "-e"], input=long_log, text=True, check=True, cwd=tmp_path, capture_output=True
    )
    assert path.read_text() == expected
    assert head() == expected_head


def test_safe_manual_override_accounts_for_missing_draw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from git_bayesect import (
        State,
        Result,
        append_result,
        get_bisector,
        get_safe_stopping,
        cli_start,
    )

    old, new = make_repo(tmp_path, [False, False, True, True])
    monkeypatch.chdir(tmp_path)
    cli_start(old, new, safe=True)
    state = State.from_git_state(tmp_path)
    append_result(state, new.encode(), Result.FAIL)
    # Announce a nontrivial query after learning that the newest commit failed.
    state.pending = SafeExperiment.choose(get_bisector(state), np.random.default_rng(9))
    experiment = state.pending
    before = get_safe_stopping(state)
    assert before is not None
    different = next(
        c
        for c, i in state.commit_indices.items()
        if state.commit_indices[state.new_sha] - i != experiment.index
    )
    append_result(state, different, Result.PASS)
    after = get_safe_stopping(state)
    assert after is not None
    assert np.allclose(after.log_evidence, before.log_evidence + experiment.log_factor(4, None))
    assert np.any(after.log_evidence < before.log_evidence)
    assert get_bisector(state).num_total_observations == 2


def test_resume_upstream_posterior_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from git_bayesect import State, cli_start, get_bisector, git_dir, STATE_FILENAME

    old, new = make_repo(tmp_path, [False, True, True])
    monkeypatch.chdir(tmp_path)
    cli_start(old, new)
    path = git_dir(tmp_path) / STATE_FILENAME
    data = json.loads(path.read_text())
    data["version"] = 2
    for key in ["pending", "pending_started", "safe_history", "check_prior", "check_cache"]:
        data.pop(key)
    data.pop("stopping")
    path.write_text(json.dumps(data))
    resumed = State.from_git_state(tmp_path)
    assert not resumed.stopping.safe
    assert get_bisector(resumed).select() == get_bisector(State.from_git_state(tmp_path)).select()
    resumed.dump(tmp_path)
    assert not State.from_git_state(tmp_path).stopping.safe


def test_interrupted_test_preserves_missing_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess
    from typing import Any
    from git_bayesect import State, Result, cli_run, cli_start, append_result, get_bisector

    old, new = make_repo(tmp_path, [False, False, True, True])
    monkeypatch.chdir(tmp_path)
    cli_start(old, new, safe=True)
    state = State.from_git_state(tmp_path)
    commits = {i: c for c, i in state.commit_indices.items()}
    rng = np.random.default_rng(42)
    for _ in range(2):
        if state.pending is None:
            state.pending = SafeExperiment.choose(get_bisector(state), rng)
        index = state.pending.index
        append_result(
            state,
            commits[state.commit_indices[state.new_sha] - index],
            Result.FAIL if index < 2 else Result.PASS,
        )
    state.dump(tmp_path)
    real_run = subprocess.run

    def interrupted(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        if args == ["simulated-interrupted-test"]:
            # The query must have reached disk before the test was started.
            assert State.from_git_state(tmp_path, read_only=True).pending_started
            raise KeyboardInterrupt
        return real_run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", interrupted)
    with pytest.raises(KeyboardInterrupt):
        cli_run(["simulated-interrupted-test"])
    resumed = State.from_git_state(tmp_path)
    assert resumed.pending is None
    assert resumed.results == state.results
    assert len(resumed.safe_history) == len(state.safe_history) + 1
    assert resumed.safe_history[-1]["observation"] is None


def test_cli_undo_checks_out_the_same_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from git_bayesect import (
        State,
        cli_fail,
        cli_pass,
        cli_start,
        cli_undo,
        get_current_commit,
        get_safe_stopping,
    )

    old, new = make_repo(tmp_path, [False] * 4 + [True] * 4)
    monkeypatch.chdir(tmp_path)
    cli_start(old, new, safe=True)
    for _ in range(8):
        tested = get_current_commit(tmp_path)
        cli_fail(commit=None)  # oops, meant to run git bayesect pass
        cli_undo()
        assert get_current_commit(tmp_path) == tested
        before = get_safe_stopping(State.from_git_state(tmp_path))
        cli_pass(commit=None)
        after = get_safe_stopping(State.from_git_state(tmp_path))
        assert before is not None and after is not None
        # the corrected result adds no evidence: it couldn't have been recorded anywhere else
        assert np.array_equal(after.log_evidence, before.log_evidence)


@pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX signals")
@pytest.mark.parametrize("signal_name", ["KILL", "TERM"])
def test_cli_run_charges_stopped_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, signal_name: str
) -> None:
    import signal
    import subprocess
    import git_bayesect
    from git_bayesect import State, cli_start

    old, new = make_repo(tmp_path, [False, False, True, True])
    monkeypatch.chdir(tmp_path)
    cli_start(old, new, safe=True)
    pending = State.from_git_state(tmp_path).pending
    assert pending is not None
    # The test stops `run` itself, before `run` can record its result.
    test = f"kill -{signal_name} $PPID"
    proc = subprocess.run(
        [sys.executable, git_bayesect.__file__, "run", "sh", "-c", test],
        cwd=tmp_path,
        capture_output=True,
        timeout=30,
    )
    assert proc.returncode == -getattr(signal, f"SIG{signal_name}")
    resumed = State.from_git_state(tmp_path)
    assert resumed.results == []
    assert resumed.pending is None
    assert resumed.safe_history[-1] == {
        "experiment": pending.as_dict(),
        "observation": None,
        "result_index": None,
    }


def test_safe_confidence_is_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from git_bayesect import State, cli_run, cli_start, cli_status

    old, new = make_repo(tmp_path, [i >= 6 for i in range(10)])
    monkeypatch.chdir(tmp_path)
    test = ["sh", "-c", "test $(cat fails) = 0"]
    cli_start(old, new, safe=True)
    assert State.from_git_state(tmp_path).stopping.confidence == 0.95
    cli_run(test, confidence=0.99)
    assert State.from_git_state(tmp_path).stopping.confidence == 0.99
    capsys.readouterr()
    cli_status()
    assert "safe at 99% confidence" in capsys.readouterr().out
    # a later plain run keeps the saved level
    cli_run(test)
    assert State.from_git_state(tmp_path).stopping.confidence == 0.99


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_state_file_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import stat
    from git_bayesect import cli_prior, cli_start, git_dir, STATE_FILENAME

    old, new = make_repo(tmp_path, [False, True, True])
    monkeypatch.chdir(tmp_path)
    path = git_dir(tmp_path) / STATE_FILENAME
    umask = os.umask(0o002)
    try:
        cli_start(old, new, safe=True)
        cli_prior(new, 2)
    finally:
        os.umask(umask)
    # the state file follows the umask, e.g. for a group-shared repository
    assert stat.S_IMODE(path.stat().st_mode) == 0o664
    assert [p.name for p in path.parent.glob(f"{STATE_FILENAME}*")] == [STATE_FILENAME]


def test_format_percent() -> None:
    from git_bayesect import format_percent

    assert format_percent(0.95) == "95%"
    assert format_percent(0.975) == "97.5%"
    assert format_percent(0.995) == "99.5%"
    assert format_percent(0.9999999) == "99.99999%"
    assert format_percent(1 - 1e-15) == "99.9999999999999%"


def test_safe_status_misfit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from git_bayesect import Result, State, append_result, cli_start, cli_status

    old, new = make_repo(tmp_path, [False, False, True, True])
    monkeypatch.chdir(tmp_path)
    cli_start(old, new, check_assumptions=True)
    state = State.from_git_state(tmp_path)
    commits = {i: c for c, i in state.commit_indices.items()}
    new_index = state.commit_indices[state.new_sha]
    # bursty reruns, as in test_safe_stopping_assumption_check
    gen = random.Random(0)
    last = [False] * 4
    for _ in range(400):
        index = gen.randrange(4)
        fresh = gen.random() < (0.6 if index <= 1 else 0.2)
        obs = last[index] if gen.random() < 0.8 else fresh
        last[index] = obs
        append_result(state, commits[new_index - index], Result.FAIL if obs else Result.PASS)
    state.dump(tmp_path)
    capsys.readouterr()
    cli_status()
    out = capsys.readouterr().out
    assert "do not fit a single change" in out
    assert "No commit is reported." in out
    # no commit is highlighted as likely when none is reported
    assert "\033[103m" not in out


def test_safe_draw_past_rounded_total() -> None:
    class LargestDraw:
        def random(self) -> float:
            # the largest value numpy's random() returns
            return float(np.nextafter(1.0, 0.0))

    # Rounding leaves these probabilities summing to just below one, and the newest commit has none
    b = Bisector([1] * 11)
    b.record(7, False)
    experiment = SafeExperiment.choose(b, cast(Any, LargestDraw()))
    assert np.cumsum(experiment.probabilities)[-1] < 1
    assert experiment.indices[-1] < 10
    assert experiment.index == experiment.indices[-1]
    for outcome in [False, True, None]:
        experiment.log_factor(11, outcome)


def test_assumption_check_reuses_saved_maximum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from git_bayesect import Result, State, append_result, cli_start, cli_undo, get_safe_stopping

    old, new = make_repo(tmp_path, [False] * 3 + [True] * 3)
    monkeypatch.chdir(tmp_path)
    cli_start(old, new, check_assumptions=True)
    state = State.from_git_state(tmp_path)
    commits = {i: c for c, i in state.commit_indices.items()}
    new_index = state.commit_indices[state.new_sha]
    gen = random.Random(5)

    def add(count: int) -> None:
        for _ in range(count):
            index = gen.randrange(6)
            fails = gen.random() < (0.7 if index <= 2 else 0.2)
            append_result(state, commits[new_index - index], Result.FAIL if fails else Result.PASS)

    def replayed(state: State) -> float:
        """The running maximum, recomputed after every result."""
        saved, state.check_cache = state.check_cache, None
        safe = get_safe_stopping(state)
        state.check_cache = saved
        assert safe is not None
        return safe.max_log_misfit

    checks = 0
    original = SafeStopping._log_check_likelihood

    def counted(self: SafeStopping, bisector: Bisector) -> float:
        nonlocal checks
        checks += 1
        return original(self, bisector)

    monkeypatch.setattr(SafeStopping, "_log_check_likelihood", counted)
    add(50)
    get_safe_stopping(state)
    assert state.check_cache is not None and state.check_cache["results"] == 50
    add(30)
    checks = 0
    safe = get_safe_stopping(state)
    assert safe is not None
    assert checks == 30
    assert safe.max_log_misfit == replayed(state)

    # undo replays every result, and saves the rebuilt check for later commands
    state.dump(tmp_path)
    cli_undo()
    undone = State.from_git_state(tmp_path)
    assert undone.check_cache is not None
    assert undone.check_cache["results"] == len(undone.results) == 79
    checks = 0
    safe = get_safe_stopping(undone)
    assert safe is not None
    assert checks == 0
    assert safe.max_log_misfit == replayed(undone)


@pytest.mark.parametrize("mode", ["default", "safe", "check"])
def test_run_models_match_a_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """After each test, the models that `run` updates match a replay of its state."""
    import contextlib
    import copy
    import io
    import subprocess
    import git_bayesect
    from git_bayesect import (
        State,
        cli_pass,
        cli_prior,
        cli_run,
        cli_start,
        cli_undo,
        get_bisector,
        get_safe_stopping,
    )

    class Killed(Exception):
        """Like SIGKILL, stops `run` without recording the test as missing."""

    _, new = make_repo(tmp_path, [False] * 6 + [True] * 4)
    monkeypatch.chdir(tmp_path)
    quiet = contextlib.ExitStack()
    quiet.enter_context(contextlib.redirect_stdout(io.StringIO()))
    quiet.enter_context(contextlib.redirect_stderr(io.StringIO()))
    with quiet:
        # the two oldest commits are outside the bisection range
        cli_start("HEAD~7", new, safe=mode != "default", check_assumptions=mode == "check")
    commits = list(State.from_git_state(tmp_path).commit_indices)
    rng = np.random.default_rng(3)
    gen = random.Random(3)
    tests = compared = 0
    real_run = subprocess.run
    real_print_status = git_bayesect.print_status

    def test_command(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        nonlocal tests
        if args != ["the-test"]:
            return real_run(args, **kwargs)
        tests += 1
        if tests % 13 == 0:
            raise KeyboardInterrupt
        if tests % 17 == 0:
            raise Killed
        fails = (tmp_path / "fails").read_text() == "1"
        return subprocess.CompletedProcess(args, int(fails != (rng.random() < 0.2)))

    def compare(
        repo_path: Path,
        state: State,
        bisector: Bisector,
        confidence: float = 0.95,
        safe: SafeStopping | None = None,
    ) -> None:
        nonlocal compared
        replay = copy.deepcopy(state)
        replay.check_cache = None  # recompute the assumption check after every result
        rebuilt = get_bisector(replay)
        assert np.array_equal(bisector.obs_yes, rebuilt.obs_yes)
        assert np.array_equal(bisector.obs_total, rebuilt.obs_total)
        assert np.array_equal(bisector.distribution, rebuilt.distribution)
        if safe is not None:
            rebuilt_safe = get_safe_stopping(replay)
            assert rebuilt_safe is not None
            assert np.array_equal(safe.log_evidence, rebuilt_safe.log_evidence)
            assert np.array_equal(safe.max_log_evidence, rebuilt_safe.max_log_evidence)
            assert safe.max_log_misfit == rebuilt_safe.max_log_misfit
        compared += 1
        real_print_status(repo_path, state, bisector, confidence, safe)

    monkeypatch.setattr(subprocess, "run", test_command)
    monkeypatch.setattr(git_bayesect, "print_status", compare)
    with quiet:
        while tests < 120:
            before = tests
            with contextlib.suppress(KeyboardInterrupt, Killed):
                cli_run(["the-test"], confidence=0.999999)
            if tests == before:
                break  # already safe to stop
            # between runs: a result for another commit, an undo or a prior change
            operation = gen.choice(["override", "undo", "prior", "none"])
            if operation == "override":
                cli_pass(commit=gen.choice(commits))
            elif operation == "undo":
                cli_undo()
            elif operation == "prior":
                cli_prior(gen.choice(commits), gen.choice([0.5, 2, 10]))
    assert compared >= 100


def test_status_during_a_live_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import subprocess
    import time
    import git_bayesect
    from git_bayesect import State, cli_start, cli_status

    old, new = make_repo(tmp_path, [False, False, True, True])
    monkeypatch.chdir(tmp_path)
    cli_start(old, new, safe=True)
    milestones = tmp_path / "milestones"
    milestones.mkdir()
    # The test signals that it started, then waits until it may finish.
    test = tmp_path / "wait.py"
    test.write_text(
        "import sys, time\n"
        "from pathlib import Path\n"
        f"milestones = Path({str(milestones)!r})\n"
        "(milestones / 'started').touch()\n"
        "while not (milestones / 'finish').exists():\n"
        "    time.sleep(0.01)\n"
        "sys.exit(1 if Path('fails').read_text() == '1' else 0)\n"
    )
    run = subprocess.Popen(
        [sys.executable, git_bayesect.__file__, "run", sys.executable, str(test)],
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 30
        while not (milestones / "started").exists():
            assert run.poll() is None and time.monotonic() < deadline
            time.sleep(0.01)
        capsys.readouterr()
        cli_status()
        captured = capsys.readouterr()
        assert "`run` may still be running" in captured.out
        assert "counting it as a missing result" not in captured.err
        assert State.from_git_state(tmp_path, read_only=True).pending_started
    finally:
        (milestones / "finish").touch()
        _, err = run.communicate(timeout=60)
    assert run.returncode == 0, err
    # status charged nothing: every draw of the run was tested and recorded
    final = State.from_git_state(tmp_path)
    assert final.results
    assert all(event["observation"] is not None for event in final.safe_history)
