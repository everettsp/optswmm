# optswmm

A Python package for optimizing SWMM (Storm Water Management Model) parameters through automated calibration techniques.

## Description

`optswmm` provides tools for calibrating SWMM models by automatically adjusting model parameters to match observed data. The package supports various optimization algorithms and provides utilities for handling SWMM input/output operations.

## Features

- **Parameter Calibration**: Automated calibration of SWMM model parameters
- **Multiple Optimization Algorithms**: Support for differential evolution and other optimization techniques
- **Flexible Parameter Types**: Handle both distributed and non-distributed parameters
- **Data Processing**: Built-in utilities for processing SWMM timeseries data
- **Configuration Management**: Easy-to-use configuration system for optimization settings

## Installation

### From Source

1. Clone the repository:
```bash
git clone https://github.com/everettsp/optswmm.git
cd optswmm
```

2. Install in development mode:
```bash
pip install -e .
```

### Dependencies

- numpy
- pandas
- matplotlib
- swmm-toolkit
- pyswmm
- swmmio
- scikit-learn

## Quick Start

```python
import optswmm

# Define calibration parameters
cal_params = optswmm.CalParams()
cal_params.add_param(
    section="SUBCATCHMENTS",
    attribute="Width",
    element="S1",
    bounds=(10, 100)
)

# Configure optimization
opt_config = optswmm.OptConfig(
    model_file="model.inp",
    optimization_method="differential_evolution"
)

# Run calibration
results = optswmm.calibrate(
    model_file="model.inp",
    cal_params=cal_params,
    cal_targets=observed_data,
    opt_config=opt_config
)
```

## Main Components

### CalParam and CalParams

Classes for defining calibration parameters:

```python
from optswmm import CalParam, CalParams

# Create individual parameter
param = CalParam(
    section="SUBCATCHMENTS",
    attribute="Width", 
    element="S1",
    bounds=(10, 100),
    relative=True
)

# Create parameter collection
params = CalParams()
params.add_param(param)
```

### OptConfig

Configuration class for optimization settings:

```python
from optswmm import OptConfig

config = OptConfig(
    model_file="model.inp",
    optimization_method="differential_evolution",
    max_iterations=100,
    population_size=50
)
```

### SWMM Utilities

Functions for working with SWMM models:

```python
from optswmm import get_node_timeseries, run_swmm

# Get timeseries data
data = get_node_timeseries(
    model_file="model.inp",
    nodes=["J1", "J2"],
    variable="Total_inflow"
)

# Run SWMM simulation
run_swmm(model_file="model.inp")
```

## API Reference

### Core Functions

- `calibrate()`: Main calibration function
- `get_cal_params()`: Helper for creating calibration parameters
- `run_swmm()`: Execute SWMM simulation
- `summarize_runs()`: Analyze optimization results

### Data Processing

- `get_node_timeseries()`: Extract node timeseries data
- `get_link_timeseries()`: Extract link timeseries data  
- `get_subcatchment_timeseries()`: Extract subcatchment timeseries data
- `normalise()` / `denormalise()`: Data normalization utilities

### Configuration

- `OptConfig`: Optimization configuration class
- `CalParam`: Individual calibration parameter
- `CalParams`: Collection of calibration parameters

## Examples

### Basic Calibration

```python
import optswmm

# Define what to calibrate
cal_params = optswmm.CalParams()
cal_params.add_param(
    section="SUBCATCHMENTS",
    attribute="Width",
    bounds=(10, 100)
)

# Load observed data
observed_data = pd.read_csv("observed.csv")

# Run calibration
results = optswmm.calibrate(
    model_file="model.inp",
    cal_params=cal_params,
    cal_targets=observed_data
)
```

### Advanced Configuration

```python
import optswmm

# Create optimization configuration
config = optswmm.OptConfig(
    model_file="model.inp",
    optimization_method="differential_evolution",
    max_iterations=200,
    population_size=100,
    tolerance=1e-6
)

# Run with custom configuration
results = optswmm.calibrate(
    model_file="model.inp",
    cal_params=cal_params,
    cal_targets=observed_data,
    opt_config=config
)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Author

Everett Snieder (everett.snieder@gmail.com)

## Changelog

### Version 0.1.0
- Initial release
- Basic calibration functionality
- Support for differential evolution optimization
- SWMM utilities for data extraction