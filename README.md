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
poetry install
# Navigate to the documentation root.
cd docs
# Build the docs.
poetry run make html
# Open in your favorite browser!
firefox _build/html/index.html
```

# References
<a id="1">[1]</a>
LeMaster, Daniel A., and Michael T. Eismann. ‘pyBSM: A Python Package for Modeling Imaging Systems’. Society of Photo-Optical Instrumentation Engineers (SPIE) Conference Series, edited by Eric J. Kelmelis, vol. 10204, 2017, p. 1020405, https://doi.org10.1117/12.2262561. Society of Photo-Optical Instrumentation Engineers (SPIE) Conference Series.

<a id="2">[2]</a>
LeMaster, Daniel, et al. Validating pyBSM: A Python Package for Modeling Imaging Systems. 05 2018, p. 19, https://doi.org10.1117/12.2305228.
