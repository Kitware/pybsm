Pending Release Notes
=====================

Updates / New Features
----------------------

CI/CD

* Major overhaul of pipeline to improve efficiency and `yml` readability.

* Fixed a typo in the `coverage` location job.

Documentation

* Added additional documentation on the relationships between GSD/altitude/pixel pitch/focal length and
  dax/day/speed/altitude
  
* Added `ruff` and `black` check to CI/CD (currently optional).

* Updated documents to reflect new refactor.

Other

* Added `git pre-hook` to assist in linting.

* Refactored package into 'src/pybsm' instead of 'pybsm'

* Add `prefer-active-python=true` to `poetry.toml` to use system `Python`.

Dependencies

* Added new linting `black` and `ruff`.

Fixes
-----

* Updated git lfs to properly track large files in any directory.
