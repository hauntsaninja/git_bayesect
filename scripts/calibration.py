import random
from statistics import median
from typing import Any

import numpy as np

from git_bayesect import Bisector, SafeExperiment, SafeStopping, safe_outcome


def run_trial(
    *,
    gen: random.Random,
    p_obs_new: float,
    p_obs_old: float,
    confidence: float,
    commits: int,
    max_iterations: int,
    safe: bool = False,
    max_range: int = 1,
    check_assumptions: bool = False,
) -> dict[str, Any]:
    b = gen.randint(0, commits - 2)

    # Using a per-commit random generator lets us record the same sequence of observations even if
    # we experiment with different selections.
    randgens = [random.Random(gen.randbytes(64)) for _ in range(commits)]
    bisect = Bisector([1] * commits)
    g = SafeStopping(commits, check_assumptions) if safe else None
    rng = np.random.default_rng(gen.getstate()[1]) if safe else None

    for iteration in range(max_iterations):
        if g is not None:
            outcome, left, right = safe_outcome(g, confidence, max_range)
            if outcome in ("misfit", "none"):
                return {
                    "iterations": iteration + 1,
                    "converged": True,
                    "correct": False,
                    "abstained": True,
                }
            if outcome != "narrowed":
                correct = left <= b <= right
                return {
                    "iterations": iteration + 1,
                    "converged": True,
                    "correct": correct,
                    "abstained": False,
                }
        else:
            dist = bisect.distribution
            likeliest_index = int(np.argmax(dist))
            likeliest_confidence = float(dist[likeliest_index])
            if likeliest_confidence >= confidence:
                return {
                    "iterations": iteration + 1,
                    "converged": True,
                    "correct": likeliest_index == b,
                    "abstained": False,
                }

        experiment = SafeExperiment.choose(bisect, rng) if safe else None
        index = experiment.index if experiment is not None else bisect.select()
        p_obs = p_obs_new if index <= b else p_obs_old
        obs = randgens[index].random() < p_obs
        bisect.record(index, obs)
        if g is not None:
            g.record(bisect, index, obs, experiment)

    dist = bisect.distribution
    likeliest_index = int(np.argmax(dist))
    return {
        "iterations": iteration + 1,
        "converged": False,
        "correct": likeliest_index == b,
        "abstained": False,
    }


def run_trials(
    *,
    p_obs_new: float,
    p_obs_old: float,
    confidence: float,
    commits: int,
    trials: int,
    seed: bytes | int,
    max_iterations: int,
    safe: bool = False,
    max_range: int = 1,
    check_assumptions: bool = False,
) -> list[dict[str, Any]]:
    gen = random.Random(seed)
    return [
        run_trial(
            gen=gen,
            p_obs_new=p_obs_new,
            p_obs_old=p_obs_old,
            confidence=confidence,
            commits=commits,
            max_iterations=max_iterations,
            safe=safe,
            max_range=max_range,
            check_assumptions=check_assumptions,
        )
        for _ in range(trials)
    ]


def main(
    seed: int | bytes | None = None,
    p_obs_new: float = 0.8,
    p_obs_old: float = 0.2,
    confidence: float = 0.95,
    commits: int = 128,
    trials: int = 1000,
    safe: bool = False,
    max_range: int | None = None,
    check_assumptions: bool = False,
) -> None:
    """Measure how often the reported commit is right at the confidence threshold.

    With safe=True, stop as `git bayesect start --safe` does (also max_range and
    check_assumptions). Runs with the same seed see the same observations at each commit.
    """
    if seed is None:
        gen = random.Random()
        seed = gen.randbytes(16)
    print(f"seed: {seed!r}")

    results = run_trials(
        trials=trials,
        seed=seed,
        commits=commits,
        p_obs_new=p_obs_new,
        p_obs_old=p_obs_old,
        confidence=confidence,
        max_iterations=100_000,
        safe=safe or max_range is not None or check_assumptions,
        max_range=1 if max_range is None else max_range,
        check_assumptions=check_assumptions,
    )
    iterations = [int(result["iterations"]) for result in results]
    num_converged = sum(int(result["converged"]) for result in results)
    num_correct = sum(int(result["correct"]) for result in results)
    num_abstained = sum(int(result["abstained"]) for result in results)

    print(f"trials: {len(results)}")
    print(f"converged: {num_converged}")
    print()
    print(f"confidence: {confidence:.2f}")
    print(f"accuracy: {num_correct / len(results):.4f}")
    if num_abstained:
        print(f"abstained: {num_abstained / len(results):.4f}")
    print()
    print(f"mean iterations: {sum(iterations) / len(iterations):.2f}")
    print(f"p50 iterations: {median(iterations)}")
    print(f"p95 iterations: {np.percentile(iterations, 95):.2f}")


if __name__ == "__main__":
    import chz

    chz.entrypoint(main)
