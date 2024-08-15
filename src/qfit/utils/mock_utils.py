import os

def is_github_pull_request():
    """
    Attempt to detect the branch being run in GitHub CI actions; this allows
    us to mark tests for running post-merge only.
    https://docs.github.com/en/actions/learn-github-actions/variables
    """
    branch_name = os.environ.get("GITHUB_REF", "main").split("/")[-1]
    return branch_name in {"merge"}
