Pending Release Notes
=====================

Updates / New Features
----------------------

CI/CD

* Added a mirroring job to replace builtin gitlab mirroring due to LFS issue.

* Numerous changes to help automated the CI/CD process.

* `poetry.lock` file updated for the dev environment.

* Updates to dependencies to support the new CI/CD.

* Changed `opencv-python` to an optional dependency.

* Added `opencv-python-headless` as an optional dependency.

* Added two extras (graphics and headless) for `opencv-python` and `opencv-python-headless` compatibility.

* Changed CI to use headless extra.

Documentation

* Added sphinx's autosummary template for recursively populating
  docstrings from the module level down to the class method level.

Fixes
-----

* Added tests to for geospatial.py to be in line with
  coverage requirements

* Added test for noise_gain function
