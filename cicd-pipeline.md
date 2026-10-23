# CI/CD Pipeline

pyBSM defines its GitLab CI jobs locally in `.gitlab-ci.yml` and `.gitlab-ci/`.
The application jobs run independently of devel-jatic. GitLab's standard
security scanning templates are included from `.gitlab-ci/.gitlab-security.yml`.

## Jobs and runners

The global runner is `k8s-small-cpu`. Unit tests, the original example, and the
notebook override it with `k8s-medium-cpu`.

| Stage      | Jobs                                                                                 | Behavior                                                                                                                                                   |
| ---------- | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test`     | `tox:pytest:*`, `tox:example`, `tox:coverage`, `tox:notebooks:examples`              | Tests and examples run on Python 3.10–3.14. Coverage combines the test reports and requires 90%. The notebook is manual on MRs and automatic on schedules. |
| `quality`  | `precommit`                                                                          | Pre-commit runs formatting, linting, and internal and external type checks on Python 3.13, matching nrtk.                                                  |
| `docs`     | `release-notes`, `pages-branch`, `pages`, `pdf-build`, `markdownlint`, `check-links` | Build documentation previews and Markdown checks. Lychee checks the existing Pages build. Release notes, PDF, and link checks remain advisory.             |
| `security` | GitLab SAST, dependency scanning, and secret detection                               | Run the configured GitLab security scanners.                                                                                                               |
| `mirror`   | `mirror`                                                                             | Mirror the main and release branches, including Git LFS objects, to GitHub.                                                                                |
| `publish`  | `publish`                                                                            | Build and publish to PyPI when an upstream version tag matches the package version.                                                                        |

Tests, quality checks, documentation, and security scans also run on nightly
schedules, matching nrtk.

Jobs using shared setup retry once. Publishing overrides that policy with zero
retries. The shared setup uses Poetry 2.2 or newer and disables job caches.

## Local checks

Install the supported Python interpreters and `tox>=4.22` in an isolated
environment. The atmosphere data and example images require Git LFS:

```bash
git lfs install
git lfs pull
pipx install 'tox>=4.22'
tox -e precommit
tox -e py313-pytest
tox -e coverage
tox -e py313-example
tox -e papermill -- pybsm_visualization.ipynb /tmp/pybsm-visualization.ipynb
```

Run `tox` without `-e` to test all five supported Python versions. Each test
environment writes a separate coverage database and JUnit report; CI preserves
these artifacts even when a test fails. Coverage aggregation runs on small CPU
and installs only the coverage utility.

The notebook environment builds and installs a wheel from the checkout. It runs
from `docs/examples` so notebook data paths resolve the same way locally and in
CI. Matplotlib uses a noninteractive backend in the example and notebook
environments.

Tox installs dependencies from the constraints in `pyproject.toml`. For the
locked Poetry development environment and documentation build, use:

```bash
poetry sync --only main,linting,tests,docs
poetry run make -C docs html
```
