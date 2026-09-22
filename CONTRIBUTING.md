# Contributing

Contributions that improve the reproducibility, documentation, analysis, or
physical modelling of the ECRIPAC simulations are welcome.

## Before opening an issue

Please check the README and existing issues first. For a reproducibility
problem, include:

- the operating system and machine or HPC system;
- the WarpX version or commit;
- the Python version and relevant dependency versions;
- the input file used;
- the complete error message;
- enough information to reproduce the problem without sharing private data.

Do not upload checkpoints, diagnostic output, field maps, or other large
simulation data to an issue unless their redistribution is permitted.

## Pull requests

1. Create a branch from `main`.
2. Keep changes focused and explain their scientific or technical purpose.
3. Update the documentation when the way to run a simulation changes.
4. Preserve attribution and license notices for third-party code.
5. Check that Python files compile and that imports do not depend on private
   absolute paths.

This repository does not currently require a full simulation for every pull
request. When a full run is impractical, describe the validation that was
performed and the reason.

## Scientific changes

Changes to field definitions, particle distributions, numerical resolution,
boundary conditions, or diagnostics should document the affected input files
and, where possible, provide a comparison with the previous behaviour.
