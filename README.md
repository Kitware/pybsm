# pyBSM

pyBSM is a Python-based tool for sensor modeling. It provides common components useful for simulating the image formation process through different imaging systems.

This repo builds off of the work done by LeMaster and Eismann in creating the original pyBSM package for modeling image systems [[1]](#1) as well as LeMaster, et al. in their work in validating the pyBSM package [[2]](#2).

# Documentation

Documentation snapshots for releases as well as the latest master are hosted on
[ReadTheDocs](https://pybsm.readthedocs.io/en/latest/).

The Sphinx-based documentation may also be built locally for the most
up-to-date reference:
```bash
# Install dependencies
poetry install --sync --with dev-linting,dev-testing,dev-docs
# Navigate to the documentation root.
cd docs
# Build the docs.
poetry run make html
# Open in your favorite browser!
firefox _build/html/index.html
```

# Developer tools

**pre-commit hooks**  
pre-commit hooks are used to ensure that any code meets all linting and formatting guidelines required.  
After installing, this will always run before committing to ensure that any commits are following the standards, 
but you can also manually run the check without committing. If you want to commit despite there being errors, you 
can add `--no-verify` to your commit command.  
Installing pre-commit hooks:  
```bash
# Ensure that all dependencies are installed  
poetry install --sync --with dev-linting,dev-testing,dev-docs  
# Initialize pre-commit for the repository  
poetry run pre-commit install  
# Run pre-commit check on all files  
poetry run pre-commit run --all-files  
```

# References
<a id="1">[1]</a>
LeMaster, Daniel A., and Michael T. Eismann. ‘pyBSM: A Python Package for Modeling Imaging Systems’. Society of Photo-Optical Instrumentation Engineers (SPIE) Conference Series, edited by Eric J. Kelmelis, vol. 10204, 2017, p. 1020405, https://doi.org10.1117/12.2262561. Society of Photo-Optical Instrumentation Engineers (SPIE) Conference Series.

<a id="2">[2]</a>
LeMaster, Daniel, et al. Validating pyBSM: A Python Package for Modeling Imaging Systems. 05 2018, p. 19, https://doi.org10.1117/12.2305228.
