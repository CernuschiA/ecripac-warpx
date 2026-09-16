# ecripac-pic-warpx


## Description
This project aim at simulation the ECRIPAC accelerator concept through a Particle-In-Cell simulation using the open source electromagnetic PIC code Warpx (https://warpx.readthedocs.io/en/latest/index.html).
The accelerator design follows the one described in this paper: https://journals.aps.org/pre/abstract/10.1103/xfhl-nxgx .
The simulations can currently run either in 3D or in cylindrical mode.
The external magnetic fields (including an induced electric field by the magnetic field variation) are implemented using field maps. The injected microwave is implemented either with TE111 mode.
The structure of the project is the following:
- irene_job.sh: shell script to launch simulations on TGCC (Irene Rome partition).
- analysis.py: python file to analyze the simulations results.
- res: include the txt files for the fieldmaps and the functions to generate field maps compatible with WarpX starting from output of COMSOL multiphysics.
- ecripac_3D: includes the namelist to simulate the ECRIPAC in 3D.
- ecripac_rz: includes the namelist to simulate the ECRIPAC in rz geometry (cylindrical).
- functions: includes the Python functions to run the data analysis.

## Installation
The only required installation is the open source electromagnetic PIC code WarpX (https://warpx.readthedocs.io/en/latest/install/users.html).
The shell script available are developed to run on the Irene Rome partition of TGCC (irene_job.sh).

## Support
Help can be provided by the Warpx support or writing me an email (andrea.cernuschi1998@gmail.com).

## Authors and acknowledgment
Thanks to Thomas Thuillier and Laurent Garrigues, who helped me develop this projects as supervisors during my PhD thesis.
Thanks to all the Warpx community for the help provided in the support chat and GitHub thread.

## Project status
Currently able to fully simulate the GA and PC phases in 3D. Currently searching for the best way to simulate the PLEIADE phase due to the large computational cost. Particle restarts have been tested, but no definitive solution has been found.
[//]: # - [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
[//]: # - [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
[//]: # - [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
[//]: # - [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)
