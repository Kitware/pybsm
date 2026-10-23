CI modernization
----------------

* Made GitLab CI standalone and migrated jobs to Kubernetes runners, with small
  CPU as the default and medium CPU for tests, examples, and notebooks.
* Removed the compliance child pipeline.
* Migrated development tooling to Poetry 2.2 and standard dependency groups.
* Added tox environments for Python 3.10 through 3.14 tests, pre-commit checks,
  coverage, the original example, and notebook execution.
* Added internal and external Pyright checks to pre-commit, matching NRTK, and
  annotated exported constants and sensor attributes for complete public types.
* Enforced the combined 90% coverage threshold and preserved reports on test failures.
* Reused the documentation build for link checks and updated Markdown and PDF checks.
* Preserved interpolation input validation with newer NumPy versions and made
  scalar conversions and plotting arguments explicit for type checking.
* Matched NRTK's development dependency constraints and Ruff configuration, keeping
  pybsm-specific exclusions and existing numerical API signatures.
* Removed the old shared-template region markers throughout the repository.
* Removed unused development dependencies and preserved NRTK's dependency comments.
* Matched NRTK's tox job names and enabled nightly tests, documentation, and security scans.
* Matched NRTK's LaTeX font and indexing setup and Pyright deprecation and override checks.
