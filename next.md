# Next Tasks / Backlog

## Orbital Viewer Display Issue
- **Observation**: When switching tautomer filter in `frontier_orbitals.html`, only one orbital (HOMO) renders while LUMO remains blank or unrendered on certain viewports/state transitions, even though trace data is present.
- **Root Cause Hypotheses**:
  - Plotly WebGL (`gl3d`) context viewport handling when traces are toggled or scene domains are modified via `relayout` vs `restyle`.
  - Trace indexing offset between `surfaces` traces and molecular bond/atom traces.
  - Camera synchronization (`tickCameraSync`) potentially suppressing the redraw of `sB` (scene2).
- **Target Resolution**:
  - Refactor the orbital viewer rendering to instantiate dedicated, static subplots per calculation state, or migrate the volumetric CUBE isosurface rendering directly into the high-performance WebGL viewer (e.g. 3Dmol.js) alongside the QM cluster adduct viewer.

