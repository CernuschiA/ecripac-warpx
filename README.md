# ecripac-pic-warpx


## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/topics/git/add_files/#add-files-to-a-git-repository) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gricad-gitlab.univ-grenoble-alpes.fr/cernusca/ecripac-warpx.git
git branch -M main
git push -uf origin main
```

***

## Description
This project aim at simulation the ECRIPAC accelerator concept through a Particle-In-Cell simulation using the open source electromagnetic PIC code Warpx (https://warpx.readthedocs.io/en/latest/index.html).
The accelerator design follows the one described in this paper: https://arxiv.org/abs/2507.07827 .
The simulations can currently run either in 3D or in azimuthal mode decomposition (AMD, consisting in quasi cylindrically symmetric simulations with 3D propagation for particles).
The external fields are implemented using Applied Fields. The external magnetic fields have been implemented through either analytical formulation or magnetic field maps interpolation (including an induced electric field by the magnetic field variation). The injected microwave is implemented either with plane wave with circular polarization.
The structure of the project is the following:
- (TODO) irene_job.sh: shell script to launch simulations on TGCC (Irene Rome partition).
- postprocess.py: python file to analyze the simulations results.
- res: include the txt files for the fieldmaps.
- ecripac_3D: includes the namelist to simulate the first phase (gyromagnetic autoresonance) of ECRIPAC in 3D.
- ecripac_rz: includes the namelist to simulate the first phase (gyromagnetic autoresonance) of ECRIPAC in rz geometry.

## Installation
The only required installation is the open source electromagnetic PIC code Smilei (https://warpx.readthedocs.io/en/latest/install/users.html).
The shell script available are developed to run on the Irene Rome partition of TGCC (irene_job.sh).

## Support
Help can be provided by the Warpx support or writing me an email (andrea.cernuschi@lpsc.in2p3.fr).

## Authors and acknowledgment
Thanks to Thomas Thuillier and Laurent Garrigues, who helped me develop this projects as supervisors during my PhD thesis.
Thanks to all the Warpx community for the help provided in the support chat and GitHub thread.

## Project status
First version of the project need to be tested.

[//]: # ## Badges
[//]: # On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.
[//]: # ## Visuals
[//]: # Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.
[//]: # ## Usage
[//]: # Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.
[//]: # ## Roadmap
[//]: # If you have ideas for releases in the future, it is a good idea to list them in the README.
[//]: # ## Contributing
[//]: # State if you are open to contributions and what your requirements are for accepting them.
[//]: # For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.
[//]: # You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.
[//]: # ## License
[//]: # For open source projects, say how it is licensed.
[//]: # ## Integrate with your tools
[//]: # - [ ] [Set up project integrations](https://gricad-gitlab.univ-grenoble-alpes.fr/cernusca/test/-/settings/integrations)
[//]: # ## Collaborate with your team
[//]: # - [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
[//]: # - [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
[//]: # - [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
[//]: # - [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
[//]: # - [ ] [Set auto-merge](https://docs.gitlab.com/user/project/merge_requests/auto_merge/)
[//]: # ## Test and Deploy
[//]: # Use the built-in continuous integration in GitLab.
[//]: # - [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/)
[//]: # - [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
[//]: # - [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
[//]: # - [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
[//]: # - [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)
