# git bayesect

Bayesian git bisection!

Use this to detect changes in likelihoods of events, for instance, to isolate a commit where
a slightly flaky test became very flaky.

You don't need to know the likelihoods (although you can provide priors), just that something
has changed at some point in some direction

## Installation

```
uv pip install git+https://github.com/hauntsaninja/git_bayesect
```

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
git bayesect success --commit $COMMIT
```

Check the status:
```
git bayesect status
```

Reset:
```
git bayesect reset
```

TODO: add commands that let you influence priors

## How it works

TODO: talk about math
