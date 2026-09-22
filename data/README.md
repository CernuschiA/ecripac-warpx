# Simulation data

The field maps required by some input files are not distributed in this
repository at present. In particular, the input decks may refer to HDF5 files
under `res/` containing static, pulsed, induced, or TE111 fields.

These files must be obtained separately from the project author or generated
from the relevant COMSOL Multiphysics exports using the conversion utilities
in [`res/`](../res/). Do not assume that COMSOL source files or generated field
maps may be redistributed without checking their applicable terms.

When the field maps are published, this document should be updated with:

- a stable download or archive URL;
- the dataset version and release date;
- checksums for each file;
- any required file placement or naming conventions;
- the license and attribution for the source data.

Simulation checkpoints, diagnostics, and generated plots should remain outside
the Git repository unless a small, redistributable example is intentionally
provided.
