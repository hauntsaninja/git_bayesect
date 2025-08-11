#!/usr/bin/env python3
import argparse
import os
import random
import shutil
import subprocess as sp
from pathlib import Path


def get_flaky_script(prob: float) -> str:
    return f"""#!/usr/bin/env python3
import random, sys

FAIL_PROBABILITY = {prob}

if random.random() < FAIL_PROBABILITY:
    print(f"failure (p={{FAIL_PROBABILITY:.4f}})", file=sys.stderr)
    sys.exit(1)
else:
    print(f"success (p={{1 - FAIL_PROBABILITY:.4f}})")
    sys.exit(0)
"""


def write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    fake_repo_dir = Path("fake_repo") if args.dir is None else Path(args.dir)
    if args.seed is not None:
        random.seed(args.seed)

    del args

    if fake_repo_dir.exists():
        shutil.rmtree(fake_repo_dir)
    fake_repo_dir.mkdir(parents=True)

    sp.run(["git", "init"], cwd=fake_repo_dir, check=True)
    sp.run(["git", "config", "user.name", "fake"], cwd=fake_repo_dir, check=True)
    sp.run(["git", "config", "user.email", "fake@example.com"], cwd=fake_repo_dir, check=True)

    sp.run(["git", "commit", "--allow-empty", "-m", "init"], cwd=fake_repo_dir, check=True)

    # first_prob = random.random() * 0.8
    # second_prob = random.random() * (1 - first_prob - 0.1) + first_prob + 0.1

    first_prob = random.random()
    second_prob = random.random()
    while max(first_prob / second_prob, second_prob / first_prob) < 1.5:
        first_prob = random.random()
        second_prob = random.random()

    write_file(fake_repo_dir / "flaky.py", get_flaky_script(first_prob))
    os.chmod(fake_repo_dir / "flaky.py", 0o755)

    sp.run(["git", "add", "flaky.py"], cwd=fake_repo_dir, check=True)
    sp.run(
        ["git", "commit", "-m", f"flaky.py fails with p={first_prob:.4f}"],
        cwd=fake_repo_dir,
        check=True,
    )

    old_commit = sp.check_output(["git", "rev-parse", "HEAD"], cwd=fake_repo_dir).decode().strip()

    for _ in range(random.randrange(10)):
        sp.run(
            ["git", "commit", "--allow-empty", "-m", "empty commit"], cwd=fake_repo_dir, check=True
        )

    write_file(fake_repo_dir / "flaky.py", get_flaky_script(second_prob))
    os.chmod(fake_repo_dir / "flaky.py", 0o755)

    sp.run(["git", "add", "flaky.py"], cwd=fake_repo_dir, check=True)
    sp.run(
        ["git", "commit", "-m", f"flaky.py fails with p={second_prob:.4f}"],
        cwd=fake_repo_dir,
        check=True,
    )

    change_commit = (
        sp.check_output(["git", "rev-parse", "HEAD"], cwd=fake_repo_dir).decode().strip()
    )

    for _ in range(random.randrange(10)):
        sp.run(
            ["git", "commit", "--allow-empty", "-m", "empty commit"], cwd=fake_repo_dir, check=True
        )

    latest_commit = (
        sp.check_output(["git", "rev-parse", "HEAD"], cwd=fake_repo_dir).decode().strip()
    )

    print("\n\n\n")
    print("=" * 80)
    print(f"path: {fake_repo_dir.absolute()}")
    print(f"initial failure probability p1: {first_prob:.4f}")
    print(f"second failure probability  p2: {second_prob:.4f}")
    print(f"initial commit: {old_commit}")
    print(f"change commit: {change_commit}")
    print(f"latest commit: {latest_commit}")
    print("=" * 80)


if __name__ == "__main__":
    main()
