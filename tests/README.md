# Tests for OptSWMM

This directory contains unit tests for the OptSWMM package, which provides tools for optimizing SWMM (Storm Water Management Model) parameters.

## Test Structure

- **test_calparams.py**: Unit tests for the `CalParam` class, ensuring proper initialization and processing of calibration parameters.
- **test_calibutils.py**: Tests for calibration utility functions, verifying their functionality and correctness.
- **test_swmmutils.py**: Tests for SWMM utility functions, ensuring correct interaction with SWMM models.
- **test_runutils.py**: Tests for run utility functions, checking behavior in managing and summarizing simulation runs.
- **test_optconfig.py**: Tests for the `OptConfig` class, ensuring correct setup and validation of optimization configurations.
- **test_datasets.py**: Tests for dataset handling functions, verifying correct loading and processing of data.

## Fixtures

The `fixtures` directory contains sample models, data, and configuration files used in the tests. These fixtures help simulate various scenarios for testing purposes.

## Running Tests

To run the tests, navigate to the root directory of the OptSWMM project and use the following command:

```
pytest tests/
```

Ensure that all dependencies are installed and that the SWMM executable is accessible in your environment.