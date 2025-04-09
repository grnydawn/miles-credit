# Preparing your dataset for CREDIT models

CREDIT supports `netCDF` and `zarr` data formats as yearly files. For example, you can have:
```
train_data_2000.zarr, train_data_2001.zarr, …, train_data_2024.zarr
```
Files will be referenced in YAML config file using regular expression:
```
data:
  # upper-air variables
  variables: ['your_varnames',]
  save_loc: '/your_path/train_data_*.zarr'
```
## Supported variable types

The following variable types are supported:

* Upper-air variables: variables with coordinates of `(time, level, latitude, longitude)` and as both inputs and outputs. e.g., air temperature.
* Surface variables: variables with coordinates of `(time, latitude, longitude)` and as both inputs and outputs. e.g., surface pressure.
* Dynamic forcing variables: variables with coordinates of `(time, latitude, longitude)` and as **input only**. e.g., solar radiation.
* Diagnostic variables: variables with coordinates of `(time, latitude, longitude)` and as **output only**. e.g., precipitation.
* Periodic forcing variables: variables with coordinates of `(time, latitude, longitude)` with the `time` coordinate covering 366 days of a year. This variable type will be used repeatedly as annual cycles and for **inputs only**. e.g., periodic sea surface temperature.
* Static variables: variables with coordinates of `(latitude, longitude)` and as input only. e.g., 'terrain elevation'.

These variable information needs to be added in the `data` section of YAML config file:

```
data
    # upper-air variables
    variables: ['your_varnames',]
    save_loc: '/your_path/train_data_*.zarr'
    
    # surface variables
    surface_variables: ['your_varnames',]
    save_loc_surface: '/your_path/train_data_*.zarr'
  
    # dynamic forcing variables
    dynamic_forcing_variables: ['your_varnames',]
    save_loc_dynamic_forcing: '/your_path/train_data_*.zarr'
  
    # diagnostic variables
    diagnostic_variables: ['your_varnames',]
    save_loc_diagnostic: '/your_path/train_data_*.zarr'
    
    # periodic forcing variables
    forcing_variables: ['your_varnames',]
    save_loc_forcing: '/your_path/forcing_*.zarr'
    
    # static variables
    static_variables: ['your_varnames',]
    save_loc_static: '/your_path/static_*.zarr'
```

## Mandatory steps
* Coordinates
* Upper-air variables
* No NaN values

## Recommended steps
* Chunking
* All variables in one file

## Test the validity of your data preparation

