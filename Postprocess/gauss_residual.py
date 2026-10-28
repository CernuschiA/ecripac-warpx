#!/usr/bin/env python3
"""
Gauss's-law check for WarpX RZ runs.

Reads the `gauss_diag` openPMD output (fields divE, rho, optionally Ez) and,
for every iteration, computes the residual

    R = eps0 * div(E) - rho        [C/m^3]

A correct simulation has R ~ 0 at all times. A run started with E = 0 on the
grid while rho != 0 has R(step 0) = -rho(step 0), and with a charge-conserving
deposition (Esirkepov) R then stays *frozen*: R(step n) == R(step 0).

Usage:
    python gauss_residual.py diags/gauss_diag            # prints the table
    python gauss_residual.py diags/gauss_diag --plot     # + gauss_residual.png

Requires: numpy, openpmd_api, (matplotlib for --plot).
"""
import argparse
import glob
import os
import sys

import numpy as np

EPS0 = 8.8541878188e-12


# --------------------------------------------------------------------------- #
# I/O (openPMD). Everything returns mode-0 arrays indexed [r, z].
# --------------------------------------------------------------------------- #
def open_series(diag_dir):
    import openpmd_api as io
    files = sorted(glob.glob(os.path.join(diag_dir, "*.h5")) +
                   glob.glob(os.path.join(diag_dir, "*.bp*")))
    if not files:
        sys.exit(f"No openPMD files found in {diag_dir}")
    import re
    stem = os.path.basename(files[0])
    m = re.match(r"^(.*?)(\d+)(\.\w+)$", stem)          # e.g. openpmd_002500.h5
    if m is None:
        return io, io.Series(files[0], io.Access.read_only)  # single file holding all steps
    pattern = os.path.join(diag_dir, f"{m.group(1)}%0{len(m.group(2))}T{m.group(3)}")
    return io, io.Series(pattern, io.Access.read_only)


def load_mode0(io, series, it, name, comp=None):
    mesh = it.meshes[name]
    rc = mesh[comp] if comp is not None else mesh[io.Mesh_Record_Component.SCALAR]
    data = rc.load_chunk()
    series.flush()
    data = np.asarray(data, dtype=float) * rc.unit_SI
    labels = list(mesh.axis_labels)
    if data.ndim == 3:            # thetaMode: leading index = mode component
        data = data[0]            # m = 0
    spacing = np.asarray(mesh.grid_spacing, dtype=float) * mesh.grid_unit_SI
    offset = np.asarray(mesh.grid_global_offset, dtype=float) * mesh.grid_unit_SI
    pos = np.asarray(rc.position, dtype=float)
    ax = [offset[k] + (np.arange(data.shape[k]) + pos[k]) * spacing[k] for k in range(2)]
    if labels[0] == "z":          # C-order (z, r) -> (r, z)
        data = data.T
        ax = ax[::-1]
    return data, ax[0], ax[1]


# --------------------------------------------------------------------------- #
# Analysis (independent of the file format)
# --------------------------------------------------------------------------- #
def residual(divE, rho, r, skip_axis_cells=2, skip_edge_cells=2):
    """R = eps0 divE - rho.
    Masked: the first cells off-axis (different on-axis stencil) and a frame of
    cells along the other edges of the output window. Next to a pec wall
    div(E) != rho/eps0 legitimately (field lines end on the induced surface
    charge of the wall, which is not part of rho), and the step-0 output also
    shows a spurious value in the wall cells."""
    R = EPS0 * divE - rho
    R[:skip_axis_cells, :] = 0.0
    if skip_edge_cells:
        n = skip_edge_cells
        R[-n:, :] = 0.0
        R[:, :n] = 0.0
        R[:, -n:] = 0.0
    return R


def summarise(records):
    """records: list of dicts with keys step, t, rho, R, (Ez)."""
    R0 = records[0]["R"]
    n0 = np.linalg.norm(R0) or 1.0
    print(f"{'step':>8} {'t [ns]':>9} {'max|rho|':>11} {'max|R|':>11} "
          f"{'max|R|/max|rho0|':>17} {'|R-R0|/|R0|':>12}")
    rho0max = np.abs(records[0]["rho"]).max() or 1.0
    for rec in records:
        R = rec["R"]
        print(f"{rec['step']:>8d} {rec['t']*1e9:>9.3f} {np.abs(rec['rho']).max():>11.3e} "
              f"{np.abs(R).max():>11.3e} {np.abs(R).max()/rho0max:>17.3e} "
              f"{np.linalg.norm(R-R0)/n0:>12.3e}")
    last = records[-1]
    frozen = (np.abs(last["R"]).max() > 0.1 * rho0max and
              np.linalg.norm(last["R"] - R0) / n0 < 0.05 and len(records) > 1)
    if frozen:
        print("\n=> FROZEN Gauss-law error: R is large and does not change in time.\n"
              "   The grid fields were not initialised consistently with rho.")
    elif np.abs(last["R"]).max() < 0.01 * rho0max:
        print("\n=> Gauss's law satisfied to < 1 % of max|rho0| at the last output.")
    else:
        print("\n=> Residual neither frozen nor negligible: inspect the maps (--plot).")


def plot(records, r, z, fname="gauss_residual.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    first, last = records[0], records[-1]
    panels = [(first["rho"], f"rho, step {first['step']}", "C/m$^3$"),
              (first["R"], f"R = eps0 divE - rho, step {first['step']}", "C/m$^3$"),
              (last["R"], f"R, step {last['step']}", "C/m$^3$")]
    if last.get("Ez") is not None:
        panels.append((last["Ez"], f"Ez, step {last['step']}", "V/m"))
    fig, axs = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6), constrained_layout=True)
    ext = [z[0] * 100, z[-1] * 100, r[0] * 100, r[-1] * 100]
    for a, (d, title, unit) in zip(axs, panels):
        v = np.abs(d).max() or 1.0
        im = a.imshow(d, origin="lower", extent=ext, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
        a.set_title(title, fontsize=9)
        a.set_xlabel("z [cm]")
        a.set_ylabel("r [cm]")
        fig.colorbar(im, ax=a, label=unit)
    fig.savefig(fname, dpi=130)
    print(f"saved {fname}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("diag_dir")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    io, series = open_series(args.diag_dir)
    records = []
    for step, it in sorted(series.iterations.items(), key=lambda kv: kv[0]):
        rho, r, z = load_mode0(io, series, it, "rho")
        divE, _, _ = load_mode0(io, series, it, "divE")
        try:
            Ez, _, _ = load_mode0(io, series, it, "E", "z")
        except Exception:
            Ez = None
        records.append(dict(step=step, t=it.time * it.time_unit_SI, rho=rho,
                            R=residual(divE, rho, r), Ez=Ez))
    summarise(records)
    if args.plot:
        plot(records, r, z)


if __name__ == "__main__":
    main()
