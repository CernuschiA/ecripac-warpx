# ecripac-pic-warpx

## Description

This project aims to simulate the ECRIPAC accelerator concept through
Particle-In-Cell simulations using the open-source electromagnetic PIC code
[WarpX](https://warpx.readthedocs.io/en/latest/index.html).

The accelerator design follows the concept described in
[this paper](https://journals.aps.org/pre/abstract/10.1103/xfhl-nxgx).
The simulations can currently run either in 3D or in cylindrical mode. The
external magnetic fields, including the induced electric field from magnetic
field variation, are implemented using field maps. The injected microwave is
implemented using the TE111 mode.

The project structure is:

- `irene_job.sh`: shell script for launching simulations on the TGCC Irene
  Rome partition.
- `analysis.py`: Python entry point for analysing simulation results.
- `res/`: field-map conversion utilities and related resources.
- `ecripac_3D/`: input files for 3D ECRIPAC simulations.
- `ecripac_rz/`: input files for cylindrical (`RZ`) ECRIPAC simulations.
- `functions/`: Python functions used for data analysis.
- `data/`: documentation for field maps and other external simulation data.

## Installation

The simulations require an installation of the open-source electromagnetic PIC
code [WarpX](https://warpx.readthedocs.io/en/latest/install/users.html).
Python dependencies used by the analysis and field-map utilities are listed in
[`environment.yml`](environment.yml). WarpX and `pywarpx` must be installed
according to the WarpX documentation and the capabilities required by the
selected input deck.

The supplied shell script is configured for the Irene Rome partition of TGCC
and may require local changes for another machine or allocation.

The field maps required by some input files are not currently distributed in
this repository. See [`data/README.md`](data/README.md) before attempting to
run a configuration that loads external HDF5 maps.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff).

The associated publications are:

1. A. Cernuschi, T. Thuillier and L. Garrigues (2026), “Theoretical study of
   the electron cyclotron resonance ion plasma accelerator concept,”
   *Physical Review E* **113**, 045211.
   [doi:10.1103/5hk4-3md2](https://doi.org/10.1103/5hk4-3md2)
2. A. Cernuschi, T. Thuillier and L. Garrigues (2026), “Milestone toward an
   ECRIPAC accelerator demonstrator,” *Physical Review E* **113**, L043202.
   [doi:10.1103/xfhl-nxgx](https://doi.org/10.1103/xfhl-nxgx)

## Support

For questions about this project, contact
[Andrea Cernuschi](mailto:andrea.cernuschi1998@gmail.com). Questions about
WarpX itself should be directed to the
[WarpX documentation and support channels](https://warpx.readthedocs.io/).

## Authors and acknowledgments

Thanks to Thomas Thuillier and Laurent Garrigues, who helped develop this
project as supervisors during my PhD thesis. Thanks also to the WarpX community
for support through the support chat and GitHub discussions.

This project was provided with computer and storage resources by GENCI at TGCC
thanks to grant `20XXAD010517004` on the Joliot-Curie supercomputer, Rome
partition.

This research used the open-source particle-in-cell code WarpX. Primary WarpX
contributors are with LBNL, LLNL, CEA-LIDYL, SLAC, DESY, CERN, Helion Energy,
TAE Technologies, and Realta Fusion. We acknowledge all WarpX contributors.

## Contributing

Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for issue and pull-request
guidance.

## Project status

The project can currently fully simulate the GA and PC phases in 3D. The best
way to simulate the PLEIADE phase is still being investigated because of its
large computational cost. Particle restarts have been tested, but no
definitive solution has been found.

## License

The original code in this repository is released under the
[MIT License](LICENSE). WarpX, Python dependencies, external field maps, and
other third-party materials remain subject to their respective licenses and
terms of use.
