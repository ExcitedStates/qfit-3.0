# Contributing to qFit

:+1::tada: Thanks for taking the time to contribute, and to read this guide! :tada::+1:

The following is a set of guidelines for contributing to qFit through the [GitHub repository](https://github.com/ExcitedStates/qfit-3.0). They are guidelines, not rules---but they exist to make maintaining and contributing to the code as smooth as possible for all parties involved. Please use your best judgement.

----

#### Table Of Contents

[Code of Conduct](#code-of-conduct)

[How Can I Contribute?](#how-can-i-contribute)
  * [Reporting Bugs](#reporting-bugs)
  * [Pull Requests](#pull-requests)

[Styleguides](#styleguides)
  * [Git Commit Messages](#git-commit-messages)
  * [Code Styleguide](#code-styleguide)
  * [Documentation Styleguide](#documentation-styleguide)

## Code of Conduct

This project and everyone participating in it is governed by the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project managers listed in the Code of Conduct.

## How Can I Contribute?

### Reporting Bugs

This section guides you through submitting a bug report for qFit. Following these guidelines helps maintainers and the community understand your report :pencil:, reproduce the behavior :computer: :computer:, and find related reports :mag_right:.

1. Before creating bug reports, please **search the [issue tracker](https://github.com/search?q=+is%3Aissue+user%3Aatom)** to see if the problem has already been reported. If it has **and the issue is still open**, add a comment to the existing issue instead of opening a new one.
    * **Note:** If you find a **Closed** issue that seems like it is the same thing that you're experiencing, open a new issue and include a link to the original issue in the body of your new one.

2. **Submit a bug report** to our issue tracker.
    * Explain the problem and include additional details to help maintainers reproduce the problem:
        * **Use a clear and descriptive title** for the issue to identify the problem.
        * **Describe your configuration and environment.** Which version of qFit are you using? What's the name and version of the OS you're using?
        * **Describe the exact steps which reproduce the problem** in as many details as possible.
        * **Describe the behavior you observed after following the steps** and point out what exactly is the problem with that behavior.
        * **Explain which behavior you expected to see instead and why.**
        * **Provide the debug log file**. Run the same command again, with a `--debug` flag. This provides a lot of extra detail that will help project maintainers identify where in the program an issue occurred.
        * **Provide files to reproduce the issue**. Attach input files if possible---if you would prefer not to share research data publically, then please email the project maintainers with links instead. If you're providing snippets in the issue, use [Markdown code blocks](https://help.github.com/articles/markdown-basics/#multiple-lines).


### Pull Requests

The process described here has several goals:

- Improve qFit's quality
- Fix problems that are important to users
- Enable a sustainable system for qFit's maintainers to review contributions

Please follow these steps to help streamline the review process:

1. Follow all instructions in [the template](.github/pull_request_template.md)
2. Follow the [styleguides](#styleguides)
3. After you submit your pull request, verify that all [status checks](https://help.github.com/articles/about-status-checks/) are passing
    * **What if the status checks are failing?** If a status check is failing, and you believe that the failure is unrelated to your change, please leave a comment on the pull request explaining why you believe the failure is unrelated. A maintainer will re-run the status check for you. If we conclude that the failure was a false positive, then we will open an issue to track that problem with our status check suite.

While the prerequisites above must be satisfied prior to having your pull request reviewed, the reviewer(s) may ask you to complete additional design work, tests, or other changes before your pull request can be ultimately accepted.

## Styleguides

### Git Commit Messages

* Complete the sentence "This commit/PR will..."  
  (e.g. "Add feature" not "Added feature")
* Reference issues and pull requests liberally after the first line
* When only changing documentation, include `[ci skip]` in the commit title
* Consider starting the commit message with an applicable emoji:
    * :art: `:art:` when improving the format/structure of the code
    * :racehorse: `:racehorse:` when improving performance
    * :non-potable_water: `:non-potable_water:` when plugging memory leaks
    * :memo: `:memo:` when writing docs
    * :bug: `:bug:` when fixing a bug
    * :fire: `:fire:` when removing code or files
    * :green_heart: `:green_heart:` when fixing the CI build
    * :white_check_mark: `:white_check_mark:` when adding tests

### Code Styleguide

One of the key language features of Python is its readability. This was an important factor in deciding to move qFit from a C codebase to Python.

Maintaining code readability, however, requires conscious effort.

* **Follow [PEP8](https://www.python.org/dev/peps/pep-0008/) guidelines** for code that you are improving/adding.
  Tools like [pycodestyle](https://pypi.org/project/pycodestyle/) and [flake8](https://pypi.org/project/flake8/) will tell you where you could improve readability; tools like [yapf](https://github.com/google/yapf) (`yapf --diff file.py`) will _advise_ you on changes to make.
    * On its own, PEP8ifying someone else's code should not be a reason to submit a PR. [Follow it yourself sooner than impose it on others](https://medium.com/@drb/pep-8-beautiful-code-and-the-tyranny-of-guidelines-f96499f5ac17).
* **Add descriptive docstrings**, especially to new functions.  
  More time is spent reading code than writing it. Docstrings make the intention of a block of code obvious.
    * We try to create [Google-style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) ← (lots of fantastic examples to follow!)
    * Docstrings _will_ become a part of our documentation through Sphinx.
* **Write unit tests** (Not required, but this will make you a hero developer).  
  This helps to make sure we don't break your code with future changes. Also, unit tests are good examples of how the code is intended to be run.
    * We use [pytest](https://docs.pytest.org/en/stable/contents.html#toc) to run our tests.


### Documentation Styleguide

* Use [Markdown](https://daringfireball.net/projects/markdown).

----

<small>This contributing guide borrows heavily from the [Atom project](https://github.com/atom/atom/blob/master/CONTRIBUTING.md).</small>
