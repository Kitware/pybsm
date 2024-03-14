Pending Release Notes
=====================

Updates / New Features
----------------------

Documentation

* Added baseline documentation for all pyBSM functionality and instruction pages
  for package installation, the review process, and the release process.

* Added Read the Docs configuration files.

* Added Apache license file.

* Added a style sheet to guide future documentation and text updates.

* Completed the code documentation sections and updated citations.

* Added an introduction section.

CI/CD

* Added pytest and coverage reports to CI.

* Added a conditional case of ``opencv-python-headless`` installation.

* Added release notes modification check.

* Added python3.12 to test matrix.

Tests

* Documented exceptions for `reflectance2photoelectrons` and its downstream functions.

* Added tests for `reflectance2photoelectrons` and its downstream functions.

Fixes
-----

* Minor ``pybsm_example.py`` fixes: properly display ground range in kilometers,
  initialize GSD correctly, remove save operation at non-existent path.
