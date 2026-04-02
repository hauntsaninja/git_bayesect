---
name: git-bayesect
description: Bayesian git bisect for locating the commit that changed test failure rates. Use when flaky tests became worse, when a probabilistic regression must be isolated, or when prior weights should be derived from filenames or diff text. Triggers on: "flaky test", "git bayesect", "bayesian bisect", "failure rate changed", "probabilistic regression".
---

# git-bayesect Skill

`git-bayesect` is a Bayesian git bisect tool for identifying the commit that introduced a change
in test failure rates. Unlike `git bisect`, it handles probabilistic (flaky) tests naturally.

## When to use this skill

- Flaky test failure rate increased at some point (e.g., 2% → 40%)
- Normal `git bisect` is unreliable because the test is non-deterministic
- You want to run automated bisection with a command via `uvx` (no pre-installation needed)

## Prerequisites

- `uv` must be installed (used to invoke via `uvx`)
- Must be run inside a git repository
- `git-bayesect` itself does **not** need to be pre-installed — `uvx` downloads and caches it on first use

## Invocation via uvx

```bash
# All commands follow this pattern
uvx git_bayesect <subcommand> [options]
```

- **First run**: downloads `git_bayesect` from PyPI and caches it in `~/.cache/uv/`
- **Subsequent runs**: uses the cache, no network required

## Basic Workflow

### 1. Start bisection

```bash
# old = a commit known to be before the change (required)
uvx git_bayesect start --old <commit-or-ref>

# Optionally specify the new endpoint (defaults to HEAD)
uvx git_bayesect start --old <old-commit> --new <new-commit>
```

### 2. Record observations (manual mode)

```bash
# Test failed at the current commit
uvx git_bayesect fail

# Test passed at the current commit
uvx git_bayesect pass

# Record against a specific commit
uvx git_bayesect fail --commit <sha>
uvx git_bayesect pass --commit <sha>
```

### 3. Automated bisection (recommended)

```bash
# Runs the command repeatedly until convergence
uvx git_bayesect run <test-command>

# Example with pytest
uvx git_bayesect run pytest tests/test_flaky.py

# Custom confidence threshold (default: 0.95)
uvx git_bayesect run --confidence 0.99 python flaky.py
```

### 4. Check status

```bash
uvx git_bayesect status
```

## Setting Priors

### Based on changed filenames

```bash
uvx git_bayesect priors_from_filenames \
  --filenames-callback "return 10 if any('network' in f for f in filenames) else 1"
```

### Based on commit message and diff text

```bash
uvx git_bayesect priors_from_text \
  --text-callback "return 10 if 'timeout' in text.lower() else 1"
```

### Set prior for a specific commit directly

```bash
uvx git_bayesect prior --commit <sha> --weight 10
```

## Utility Commands

```bash
uvx git_bayesect undo      # Undo the last recorded observation
uvx git_bayesect log       # Print commands to reproduce the current state
uvx git_bayesect reset     # Reset bisection state
uvx git_bayesect checkout  # Checkout the next suggested commit to test
```

## Key Concepts

- **fail** = the "new" behavior (higher failure rate, occurring on or after the bad commit)
- **pass** = the "old" behavior (lower failure rate, occurring before the bad commit)
- Uses Beta-Bernoulli conjugacy to handle unknown and probabilistic failure rates
- Next commit to test is chosen by greedy minimization of expected posterior entropy

## Example Convergence Output

```
================================================================================
Bisection converged to a1b2c3d4e5 (97.3%) after 18 observations
Observed subsequent failure rate is 42.0%, prior failure rate is 3.0%
================================================================================
```

## Typical Usage Pattern

```bash
cd <your-git-repo>

# Start bisection from 2 commits back and run automatically
OLD_COMMIT=$(git rev-list HEAD --reverse | head -n 2 | tail -n 1)
uvx git_bayesect start --old $OLD_COMMIT
uvx git_bayesect run python flaky_test.py
```

## AI Guidance Notes

- If the user hasn't identified the `old` commit yet, suggest running `git log --oneline` first
- Test commands are interpreted as: exit code 0 = pass, non-zero = fail
- Warn the user that `run` will leave the repository in a detached HEAD state
- After bisection completes, guide the user to run `git checkout <branch>` to return to their branch
