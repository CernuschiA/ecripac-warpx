#!/bin/bash
#MSUB -r ecripac_warpx        # Job name
#MSUB -N 4                  # Total number of nodes
#MSUB -n 32                  # Number of tasks to use (MPI tasks) - N.B. due to IRENE architecture, use 8 MPI tasks per node
#MSUB -c 16                  # Number of cores (or threads) per task to use (OpenMP threads) - In IRENE rome each node has 8 socket with 16 core each
#MSUB -T 70000               # Elapsed time limit in seconds of the job (default: 7200)
#MSUB -o ecripac_warpx_%I.o    # Standard output. %I is the job id
#MSUB -e ecripac_warpx_%I.e    # Error output. %I is the job id
#MSUB -A gen17004            # Project ID
#MSUB -q rome               # Partition name (see ccc_mpinfo)
#MSUB -m work,scratch

cd && module switch dfldatadir dfldatadir/gen17004 && cd ${CCCSCRATCHDIR}/ecripac/ecripac_rz

# Setup the Warpx environment
source ${CCCSCRATCHDIR}/.warpx_env
#WARPX_ADD=${CCCWORKDIR}/install_warpx/warpx/bin
WARPX_ADD=${CCCWORKDIR}/install_warpx/warpx_nocuda/bin

export OMP_NUM_THREADS=16
export OMP_SCHEDULE=dynamic
export OMP_PROC_BIND=true

ccc_mprun $WARPX_ADD/warpx.3d.MPI.OMP.DP.PDP.OPMD.EB.QED inputs_pl_rz_picmi
# warpx.break_signals=HUP