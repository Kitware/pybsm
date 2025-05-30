Pending Release Notes
=====================

Updates / New Features
----------------------

CI/CD

* Removed ``mypy`` check and dependency.

Documentation

* Moved ``examples`` directory to ``docs/examples``

* Updated ``index.rst``, ``installation.rst``, and ``README.md``  based on ``devel-jatic``.

* Added warning to use Poetry only in a virtual environment per Poetry documentation.

* Clarified that ``poetry<2.0`` is currently required.

* Added CI/CD pipeline documentation.

Other

* Refactored codebase to make use of the ``*`` keyword-only separator.

Fixes
-----

* Fixed pyright errors

* Removed ``Optional`` and ``Union`` type hints.

* Update pytest and ruff configurations
