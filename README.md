# git bayesect

Bayesian git bisection!

Use this to detect changes in likelihoods of events, for instance, to isolate a commit where
a slightly flaky test became very flaky.

You don't need to know the likelihoods (although you can provide priors), just that something
has changed at some point in some direction

## Installation

```
pip install git_bayesect
```
Or:
```
uv tool install git_bayesect
```

## How it works

`git_bayesect` uses Bayesian inference to identify the commit introducing a change, with
commit selection performed via greedy minimisation of expected entropy, and using a Beta-Bernoulli
conjugacy trick while calculating posterior probabilities to make handling unknown failure rates
tractable.

See https://hauntsaninja.github.io/git_bayesect.html for a write up.

## Usage

Start a Bayesian bisection:
```
git bayesect start --old $COMMIT
```

Record an observation on the current commit:
```
git bayesect fail
```

Or on a specific commit:
```
git bayesect pass --commit $COMMIT
```

Check the overall status of the bisection:
```
git bayesect status
```

Reset:
```
git bayesect reset
```

## Safe stopping

By default the bisection stops once the most likely commit reaches the confidence threshold (95%,
or `git bayesect run --confidence`). That confidence is a Bayesian average over the priors: for
particular failure rates, or when the test doesn't behave the way the model assumes, the reported
commit can be wrong more often than the confidence says.

Start with `--safe` to only stop once every other commit has been ruled out. The reported
commit is then wrong with probability at most 1 - confidence, whatever the failure rates and
however many commits there are, as long as the failure rate changes at a single commit. The
rates may even change during the bisection, for example when CI gets busier, provided commits on
the same side of the culprit keep sharing a failure rate. Safe mode picks the commit to test at
random, so test the commits it checks out. It typically takes 18–35% more test runs.
```
git bayesect start --old $COMMIT --safe
```

The confidence level defaults to 95%. A safe bisection saves the level given to
`git bayesect run --confidence`, so every command reports at the same level.

Allow stopping with a range of up to N adjacent commits, which contains the culprit with the same
bound. A range can also help when the failure rate ramps up over a few commits, although the bound
does not cover that case:
```
git bayesect start --old $COMMIT --max-range 4
```

Stop without reporting a commit if the results don't fit a single change in failure rate, for
instance failures that come in bursts because reruns on the same commit aren't independent:
```
git bayesect start --old $COMMIT --check-assumptions
```

`--max-range` and `--check-assumptions` imply `--safe`. The bound doesn't cover reruns that
depend on each other, or failure rates that differ between commits on the same side of the
culprit. The assumption check catches some of these cases, but not reliably.

Priors can be changed at any time: they affect later choices, not the evidence already collected.
Restarting reuses the pending choice of commit. A result recorded for a commit other than the one
checked out still informs later choices, but cancels the pending draw, which can lower the current
evidence. `undo` keeps the bound; with `--check-assumptions`, only undo results to correct mistakes.
After `undo`, the same commit is checked out again. A test that is stopped during `git bayesect run`
counts as a missing result, and `git bayesect log` prints a snapshot that can't be edited.

## More usage

Set the prior for a given commit:
```
git bayesect prior --commit $COMMIT --weight 10
```

Set prior for all commits based on filenames:
```
git bayesect priors_from_filenames --filenames-callback "return 10 if any('suspicious' in f for f in filenames) else 1"
```

Set prior for all commits based on the text in the commit message + diff:
```
git bayesect priors_from_text --text-callback "return 10 if 'timeout' in text.lower() else 1"
```

Get a log of commands to let you reconstruct the state:
```
git bayesect log
```

Undo the last observation:
```
git bayesect undo
```

Run the bisection automatically using a command to make observations:
```
git bayesect run $CMD
```
While it runs, only use `git bayesect status` and `git bayesect log`: other commands change the
bisection or the checkout under the running test.

Checkout the best commit to test:
```
git bayesect checkout
```

## Demo

This repository contains a little demo, in case you'd like to play around:
```
# Create a fake repository with a history to bayesect over
python scripts/generate_fake_repo.py
cd fake_repo

# The fake repo contains a script called flaky.py
# This is a simple script that fails some fraction of the time
# At some point in the history of the repo, that fraction was changed
python flaky.py
git log --oneline

# Start the bayesection
OLD_COMMIT=$(git rev-list HEAD --reverse | head -n 2 | tail -n 1)
git bayesect start --new main --old $OLD_COMMIT

# Run a bayesection to find the commit that introduced the change
git bayesect run python flaky.py
```
