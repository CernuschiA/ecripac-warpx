from pathlib import Path
import os

sys_path = "/home/cernuschi/pole_acc_storage/"
address_shared = sys_path + "PhD/Simulation results/WarpX/TE111_200kHz/test7/diags/"
address_run = sys_path + "PhD/Simulation results/WarpX/TE111_200kHz/test7_tr3_pc2/diags/"
diag_types = [
    "smallparticle_diag/",
    "particle_diag/",
    "field_diag/",
]

for diag_type in diag_types:
    shared = Path(address_shared + diag_type)
    run = Path(address_run + diag_type)

    # Ensure destination directory exists
    run.mkdir(parents=True, exist_ok=True)
    
    for file_path in shared.glob("*.h5"):
        filename = file_path.name
        dst = run / filename
        # Remove existing file/symlink only if present
        if dst.exists() or dst.is_symlink():
            dst.unlink()

        # Create symlink to source file
        os.symlink(file_path, dst)