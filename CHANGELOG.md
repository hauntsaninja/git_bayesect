# Changelog

## [v1.2]

- Switch to uninformative beta priors to avoid calibration issues
- Better validation for confidence in `git bayesect run`
- Introduce better tie breaking logic for commit selection and maximum a posteriori determination
- Improve error messages, performance and testing

## [v1.1]

- Add `priors_from_text` command for setting priors based on commit message + diff text
- Make confidence configurable in `git bayesect run`
- Print the beta priors on bayesect start
- Improve help output

## [v1.0]

- Initial release
