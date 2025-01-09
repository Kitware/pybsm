# pyBSM

pyBSM is a Python-based tool for sensor modeling. It provides common components useful for simulating the image formation process through different imaging systems.

This repo builds off of the work done by LeMaster and Eismann in creating the original pyBSM package for modeling image systems \[1] as well as LeMaster, et al. in their work in validating the pyBSM package \[2].

NOTE: A set of functions which infer OTFs from user data has been removed from
the current distribution of pyBSM. They are archived under the v0.7.0 tag if
they are needed.

## References
<a id="1">[1]</a>
LeMaster, Daniel A., and Michael T. Eismann. ‘pyBSM: A Python Package for Modeling Imaging Systems’. Society of Photo-Optical Instrumentation Engineers (SPIE) Conference Series, edited by Eric J. Kelmelis, vol. 10204, 2017, p. 1020405, https://doi.org10.1117/12.2262561. Society of Photo-Optical Instrumentation Engineers (SPIE) Conference Series.

<a id="2">[2]</a>
LeMaster, Daniel, et al. Validating pyBSM: A Python Package for Modeling Imaging Systems. 05 2018, p. 19, https://doi.org10.1117/12.2305228.

<!-- :auto installation: -->
## Installation
Ensure the source tree is acquired locally before proceeding.

To install the current version via `pip`:
```bash
pip install pybsm
```

Alternatively, you can use [Poetry](https://python-poetry.org/):
```bash
poetry install --with main,linting,tests,docs
```

For more detailed installation instructions, visit the [installation documentation](https://pybsm.readthedocs.io/en/latest/installation.html).
<!-- :auto installation: -->

<!-- :auto getting-started: -->
## Getting Started
Explore usage examples of the `pybsm` package in various contexts using the Jupyter notebooks provided in the `./examples/` directory.

Contributions are encouraged! For more details, refer to the [CONTRIBUTING.md](./CONTRIBUTING.md) file.
<!-- :auto getting-started: -->

<!-- :auto documentation: -->
## Documentation
Documentation for both release snapshots and the latest master branch is available on [ReadTheDocs](https://pybsm.readthedocs.io/en/latest/).

To build the Sphinx-based documentation locally for the latest reference:
```bash
# Install dependencies
poetry install --sync --with main,linting,tests,docs
# Navigate to the documentation root
cd docs
# Build the documentation
poetry run make html
# Open the generated documentation in your browser
firefox _build/html/index.html
```
<!-- :auto documentation: -->

<!-- :auto developer-tools: -->
## Developer Tools

### Pre-commit Hooks
Pre-commit hooks ensure that code complies with required linting and formatting guidelines. These hooks run automatically before commits but can also be executed manually. To bypass checks during a commit, use the `--no-verify` flag.

To install and use pre-commit hooks:
```bash
# Install required dependencies
poetry install --sync --with main,linting,tests,docs
# Initialize pre-commit hooks for the repository
poetry run pre-commit install
# Run pre-commit checks on all files
poetry run pre-commit run --all-files
```
<!-- :auto developer-tools: -->

<!-- :auto contributing: -->
## Contributing
- Follow the [JATIC Design Principles](https://cdao.pages.jatic.net/public/program/design-principles/).
- Adopt the Git Flow branching strategy.
- Detailed release information is available in [docs/release_process.rst](./docs/release_process.rst).
- Additional contribution guidelines can be found in [CONTRIBUTING.md](./CONTRIBUTING.md).
<!-- :auto contributing: -->

<!-- :auto license: -->
## License
[Apache 2.0](./LICENSE)
<!-- :auto license: -->

<!-- :auto contacts: -->
## Contacts

**Principal Investigator**: Brian Hu (Kitware) @brian.hu

**Product Owner**: Austin Whitesell (MITRE) @awhitesell

**Scrum Master / Tech Lead**: Brandon RichardWebster (Kitware) @b.richardwebster

**Deputy Tech Lead**: Emily Veenhuis (Kitware) @emily.veenhuis
<!-- :auto contacts: -->
