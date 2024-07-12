Pending Release Notes
=====================

Updates / New Features
----------------------

CI/CD

* Optimized to not run anything but `publish` when `tag`.

* Created a shared `python-version` job for `python` version matrices.

* Updated scanning to properly report the vulnerabilities.

* Updated scanning to properly scan used packages

* Added caching of packages to pipeline.

* Changed check release notes to only fetch last commit from main.

* Added examples to `black` scan.

* Added `jupyter` notebook extra to `black`.

Documentation

* Added a section to the README about using the pre-commit hooks

Fixes
-----

* Updated `poetry.lock` file to remove a development environment vulnerability.

* Modified security scanning to not use latest but instead the stable version.