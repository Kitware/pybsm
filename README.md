# pyBSM

pyBSM is a Python-based tool for sensor modeling. It provides common components
useful for simulating the image formation process through different imaging
systems.

This repo builds off of the work done by LeMaster and Eismann in creating the
original pyBSM package for modeling image systems [1] as well as LeMaster, et
al. in their work in validating the pyBSM package [2].

NOTE: A set of functions which infer OTFs from user data has been removed from
the current distribution of pyBSM. They are archived under the v0.7.0 tag if
they are needed.

## References

<a id="1">[1]</a> LeMaster, Daniel A., and Michael T. Eismann. ‘pyBSM: A Python
Package for Modeling Imaging Systems’. Society of Photo-Optical Instrumentation
Engineers (SPIE) Conference Series, edited by Eric J. Kelmelis, vol. 10204,
2017, p. 1020405, <https://doi.org/10.1117/12.2262561>. Society of Photo-Optical
Instrumentation Engineers (SPIE) Conference Series.

<a id="2">[2]</a> LeMaster, Daniel, et al. Validating pyBSM: A Python Package
for Modeling Imaging Systems. 05 2018, p. 19,
<https://doi.org/10.1117/12.2305228>.

## Installation

`pybsm` installation has been tested on Unix and Linux systems.

To install the current version via `pip`:

```bash
pip install pybsm
```

To install the current version via `conda-forge`:

```bash
conda install -c conda-forge pybsm
```

For more detailed installation instructions, visit the
[installation documentation](https://pybsm.readthedocs.io/en/latest/installation.html).

## Getting Started

Explore usage examples of the `pybsm` package in various contexts using the
Jupyter notebooks provided in the `./docs/examples/` directory.

## Documentation

Documentation for both release snapshots and the latest main branch is available
on [ReadTheDocs](https://pybsm.readthedocs.io/en/latest/).

To build the Sphinx-based documentation locally for the latest reference:

```bash
# Install dependencies
poetry sync --only main,linting,tests,docs
# Navigate to the documentation root
cd docs
# Build the documentation
poetry run make html
# Open the generated documentation in your browser
firefox _build/html/index.html
```

## Contributing

Contributions are encouraged!

The following points help ensure contributions follow development practices.

- Follow the
  [JATIC Design Principles](https://cdao.pages.jatic.net/public/program/design-principles/).
- Adopt the Git Flow branching strategy.
- Detailed release information is available in
  [docs/release_process.rst](./docs/release_process.rst).
- Additional contribution guidelines and issue reporting steps can be found in
  [CONTRIBUTING.md](./CONTRIBUTING.md).

### Developer Tools

Ensure the source tree is acquired locally before proceeding.

#### Poetry Install

You can install using [Poetry](https://python-poetry.org/):

Development requires `poetry>=2.2`. See the
[installation documentation](https://pybsm.readthedocs.io/en/latest/installation.html#from-source)
for setup details.

```bash
poetry sync --only main,linting,tests,docs
```

#### Pre-commit Hooks

Pre-commit hooks ensure that code complies with required linting and formatting
guidelines. These hooks run automatically before commits but can also be
executed manually. To bypass checks during a commit, use the `--no-verify` flag.

To install and use pre-commit hooks:

```bash
# Install required dependencies
poetry sync --only main,linting,tests,docs
# Initialize pre-commit hooks for the repository
poetry run pre-commit install
# Run pre-commit checks on all files
poetry run pre-commit run --all-files
```

#### CI Environments

CI uses [tox](https://tox.wiki/) to run the same checks locally and on GitLab.
Install `tox>=4.22` in an isolated environment, for example with `pipx`:

```bash
pipx install 'tox>=4.22'
tox -e precommit
tox -e py313-pytest
tox -e coverage
tox -e py313-example
tox -e papermill -- pybsm_visualization.ipynb /tmp/pybsm-visualization.ipynb
```

The test environments cover Python 3.10 through 3.14; each interpreter must be
installed to run its corresponding environment. Running `tox` without `-e` runs
all five test environments. The coverage environment combines their reports and
requires at least 90% coverage. The example and notebook environments use a
noninteractive Matplotlib backend, and the notebook environment installs a wheel
built from the checkout. Pre-commit runs both internal type checking and public
type completeness checks, matching nrtk.

## License

[Apache 2.0](./LICENSE)

## Contacts

**Principal Investigator**: Brian Hu (Kitware) @brian.hu

**Project Manager / Product Owner**: Keith Fieldhouse (Kitware)
@keith.fieldhouse

**Scrum Master / Maintainer**: Brandon RichardWebster (Kitware)
@b.richardwebster

**Deputy Scrum Master / Deputy Maintainer**: Emily Veenhuis (Kitware)
@emily.veenhuis

**Program Representative**: Austin Whitesell (MITRE) @awhitesell

## Acknowledgment

This material is based upon work supported by the Chief Digital and Artificial
Intelligence Office under Contract No. 519TC-23-9-2032. The views and
conclusions contained herein are those of the author(s) and should not be
interpreted as necessarily representing the official policies or endorsements,
either expressed or implied, of the U.S. Government.
