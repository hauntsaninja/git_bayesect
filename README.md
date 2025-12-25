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

Set the beta priors:
```
git bayesect beta_priors --alpha-new 0.9 --beta-new 0.1 --alpha-old 0.05 --beta-old 0.95
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

## How it works

TODO: talk about math
