# waves

This repo is for developing and testing wave and maybe sediment transport functions.

### Main functions
`wave_funcs.py` contain most of the callable functions. This will eventually become a package, but right now, just put
```#run -1 wave_funcs.py``` at the top of scripts that use the functions.

### Tests
```wave_calcs.ipynb``` is a handy interfaces for doing basic wave calculations

```airy_waves.ipynb``` is a work in progress that generates Stokes waves

```test_qkhfs.ipynb``` calculates some wavenumbers to check the Python version of the dispersion equation solution

```test_qhkfs.m``` is the Matlab version to generate the same numbers and calls `qhkfs.m`

```qkhfs.m``` is the Matlab version of the quick, iterative solution of the dispersion equation

```matlab_results.txt``` is the output of the Matlab version.

```wave_assymtery.ipynb``` is a mispelled notebook used to develop the wave asymmetry calcs.

###TODO

* Rename and clean up wave_assymtery!

