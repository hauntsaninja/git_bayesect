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

Checkout the best commmit to test:
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
