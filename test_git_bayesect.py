import random

import numpy as np

from git_bayesect import Bisector


def test_bisector_posteriors() -> None:
    b = Bisector([1] * 5)
    # b.record(1, observation=True, p_obs_new=1, p_obs_old=0)
    for _ in range(100):
        b.record(0, True)
        b.record(4, False)
    b.record(2, True)
    assert np.allclose(b.distribution, [0, 0, 0.5, 0.5, 0], atol=0.01)

    b = Bisector([1] * 5)
    # b.record(1, observation=True, p_obs_new=1, p_obs_old=0.1)
    for _ in range(100):
        b.record(0, True)
    for _ in range(10):
        b.record(4, True)
    for _ in range(90):
        b.record(4, False)

    b.record(2, True)
    assert np.allclose(b.distribution, [0.0455, 0.0455, 0.4545, 0.4545, 0], atol=0.01)

    b = Bisector([1] * 5)
    # b.record(1, observation=True, p_obs_new=0.8, p_obs_old=0.2)
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
    # b.record(1, observation=False, p_obs_new=1, p_obs_old=0)
    for _ in range(100):
        b.record(0, True)
    for _ in range(100):
        b.record(4, False)

    b.record(2, False)
    assert np.allclose(b.distribution, [0.5, 0.5, 0, 0, 0], atol=0.01)

    b = Bisector([1] * 5)
    # b.record(1, observation=False, p_obs_new=1, p_obs_old=0.1)
    for _ in range(100):
        b.record(0, True)
    for _ in range(10):
        b.record(4, True)
    for _ in range(90):
        b.record(4, False)

    b.record(2, False)
    assert np.allclose(b.distribution, [0.5, 0.5, 0, 0, 0], atol=0.01)

    b = Bisector([1] * 5)
    # b.record(1, observation=False, p_obs_new=0.8, p_obs_old=0.2)
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
    b.post_weights = np.array([0.1, 0.8, 0.1])
    assert b.central_range(0) == (1, 1)
    assert b.central_range(0.5) == (1, 1)
    assert b.central_range(0.799) == (1, 1)
    assert b.central_range(0.801) == (0, 2)
    assert b.central_range(1.0) == (0, 2)

    b = Bisector([1] * 5)
    b.post_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    assert b.central_range(0) == (2, 2)
    assert b.central_range(0.3) == (1, 3)
    assert b.central_range(0.7) == (0, 4)
    assert b.central_range(1.0) == (0, 4)

    b = Bisector([1] * 5)
    b.post_weights = np.array([0.4, 0.4, 0.1, 0.1, 0.2])
    assert b.central_range(0) == (1, 1)
    assert b.central_range(0.2) == (0, 1)
    assert b.central_range(0.599) == (0, 1)
    assert b.central_range(0.601) == (0, 2)
    assert b.central_range(0.8) == (0, 3)
    assert b.central_range(1.0) == (0, 4)


def test_bisector_bisect() -> None:
    gen = random.Random()
    seed = gen.randbytes(16)
    print(f"seed: {seed!r}")
    gen = random.Random(seed)

    N = 32
    B = gen.randrange(N)

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

        print(dist := bisect.distribution)

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
        if obs and p_obs < 0.5:
            print("!!! observation was unlikely")
        bisect.record(index, obs)
    else:
        print("=" * 80)
        print("failed to converge")
        print("=" * 80)

    print("=" * 80)
    print(f"N: {N}")
    print(f"B: {B}")
    print(f"p_obs_new: {p_obs_new:.4f}")
    print(f"p_obs_old: {p_obs_old:.4f}")
    print("=" * 80)

    dist = bisect.distribution
    likeliest_index = np.argmax(dist)
    assert likeliest_index == B
