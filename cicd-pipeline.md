# CI/CD Pipeline (as of April 19, 2025)

Kitware’s CI/CD system for the JATIC program is modular and highly reusable
across repositories. It is designed to standardize workflows while allowing for
project-specific overrides.

The root pipeline entrypoint is `.gitlab-ci.yml`, which includes most job
definitions from the `.gitlab-ci/` directory. These are further broken down by
purpose.

## Structure

- **`.gitlab-ci.yml`** The top-level orchestrator file that defines pipeline
  stages, includes jobs, and enforces scheduling and skip rules. Other JATIC
  repositories typically include it like this:

  ```yaml
  include:
    - project: "jatic/kitware/devel-jatic"
      ref: "main"
      file: ".gitlab-ci.yml"

    # Repo-specific job overrides or additions
    - local: .gitlab-ci/.gitlab-docs.yml
    - local: .gitlab-ci/.gitlab-test.yml
    - local: .gitlab-ci/.gitlab-container.yml
  ```

- **`.gitlab-ci/`** Contains shared CI job definitions:

  - `.gitlab-test.yml`: Unit tests, extras, notebooks, coverage
    - notebooks are run manually and tested nightly
  - `.gitlab-quality.yml`: Linters, type checks, Sphinx linting
  - `.gitlab-docs.yml`: Sphinx build and GitLab Pages deployment
  - `.gitlab-security.yml`: SAST, dependency scanning, secret detection
  - `.gitlab-mirror.yml`: GitHub mirror including Git LFS
  - `.gitlab-publish.yml`: PyPI publishing on tag
  - `.gitlab-devel.yml`: Automation and multi-repo content propagation
  - `.gitlab-shared.yml`: Shared setup used across jobs

- **`.gitlab-ci/pipelines/`** Contains child pipeline entrypoints used for:

  - **Compliance Scanning**: `.gitlab-compliance.yml` includes components for
    SR, TR, DR compliance, unit tests, and pipeline validation
  - **Container Build/Scan**: `.gitlab-container.yml` builds and scans Docker
    images using Harbor and Trivy

## Pipeline Stages Overview

| Stage | Purpose | Merge Request | Required |
|-------------|-------------------------------------------------------------------------|-------------|-------------|
| `test` | Run unit tests, notebook validation (manual), and coverage reporting | ✅ | ✅ |
| `quality` | Code linting (`ruff`), type checks (`pyright`), and doc linting (`sphinx`) | ✅ | ✅ |
| `docs` | Build and preview documentation | ✅ | ✅ |
| `mirror` | Push code and LFS objects to GitHub | ❌ | ❌ |
| `publish` | Publish to PyPI if tag matches version | ❌ | ❌ |
| `devel` | Propagate files across Kitware-managed repos | ❌ | ❌ |
| `security` | Run GitLab SAST, Dependency, and Secret detection scanners | ✅ | ❌ |
| `container` | Build Docker images and run vulnerability scans | Manual |
| `compliance`| Trigger DevSecOps child pipeline (`.gitlab-compliance.yml`) | ✅ | ❌ |

> Each repo can selectively override, disable, or add jobs by defining its own
> `.gitlab-ci/` entries alongside the shared pipeline.

## Where to Start

Most developers interact with:

- `test` and `quality` stages to validate code
- `docs` to verify documentation previews
- `container` when working on tools with a Docker interface

For infrastructure contributors, review `.gitlab-ci.yml` and child pipelines in
`.gitlab-ci/pipelines/` for advanced setup and compliance jobs.

<!--
.. Code Style Consistency (``test-py-lint``)
.. -----------------------------------------
.. Runs ``flake8`` to quality check the code style.
.. You can run this check manually in your local repository with
.. ``poetry run flake8``.

.. Passage of this check is strictly required.

.. Static Type Analysis (``test-py-typecheck``)
.. --------------------------------------------
.. Performs static type analysis.
.. You can run this check manually in your local repository with ``poetry run
.. mypy``.

.. Passage of this check is strictly required.

.. Documentation Build (``test-docs-build``)
.. -----------------------------------------
.. Performs a build of our Sphinx documentation.

.. Passage of this check is strictly required.

.. Unit Tests (``test-pytest``)
.. ----------------------------
.. Runs the unittests created under ``tests/`` as well as any doctests found in
.. docstrings in the package code proper.
.. You can run this check manually  in your local repository with ``poetry run
.. pytest``.

.. Passage of these checks is strictly required.

.. Regression Tests
.. ^^^^^^^^^^^^^^^^
.. Regression test snapshots are generated using
.. `syrupy <https://github.com/syrupy-project/syrupy>`_. To generate new snapshots,
.. run ``poetry run pytest --snapshot-update path/to/test_file.py``. Ensure the full
.. filepath is included so that irrelevant snapshots are not erroneously updated.
.. Once a snapshot is generated, regression test results will be included in the
.. ``test-pytest`` job.

.. Code Coverage (``test-coverage-percent``)
.. -----------------------------------------
.. This job checks that the lines of code covered by our Unit Tests checks meet or
.. exceed certain thresholds.

.. Passage of this check is not strictly required but highly encouraged.

.. Release Notes Check (``test-release-notes-check``)
.. --------------------------------------------------
.. Checks that the current branch's release notes has modifications relative to
.. the marge target's.

.. Passage of this check is not strictly required but highly encouraged.

.. Example Notebooks Execution (``test-notebooks``)
.. ------------------------------------------------
.. This check executes included example notebooks to ensure their proper
.. functionality with the package with respect to a merge request.
.. Not all notebooks may be run, as some may be set up to use too many resources
.. or run for an extended period of time.

.. Passage of these checks is strictly required.
-->
