"""Comprehensive standalone HTML dossier generator for SharK.

Compiles docking scores, quantum chemical (ORCA DFT) properties, frontier
orbital 3D isosurfaces, and molecular dynamics (GROMACS) trajectory metrics
into a single, zero-server, offline-ready HTML report.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
import math
import os
from pathlib import Path
from typing import Optional, Sequence
from .. import __version__

try:
    from plotly.offline import get_plotlyjs
    HAVE_PLOTLY_JS = True
except ImportError:
    HAVE_PLOTLY_JS = False


def _load_plotly_js() -> str:
    """Retrieves plotly.js from package or local bundled asset."""
    if HAVE_PLOTLY_JS:
        try:
            return get_plotlyjs()
        except Exception:
            pass

    search_paths = [Path(__file__).parent / "plotly.min.js"]
    for p in search_paths:
        if p.is_file():
            try:
                return p.read_text(encoding="utf-8")
            except Exception:
                pass

    raise RuntimeError('Plotly.js unavailable; install the declared Plotly dependency for offline reports')


def generate_html_dossier(
    project_name: str,
    poses_data: list[dict],
    out_html: str | Path,
    qm_summary: Optional[dict] = None,
    md_summary: Optional[dict] = None,
    orbital_mesh: Optional[dict] = None,
    notes: str = ""
) -> Path:
    """Builds a complete, interactive HTML report."""
    out_path = Path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plotly_code = _load_plotly_js()
    project_name = escape(str(project_name))
    notes = escape(str(notes))

    # Build Pose Table Rows
    table_rows = []
    for p in poses_data:
        lig = escape(str(p.get("ligand_id", "UNK")))
        idx = escape(str(p.get("pose_idx", 1)))
        score = p.get('score')
        vina = f"{score:.2f}" if isinstance(score, (int, float)) and math.isfinite(score) else 'N/A'
        qm_e = escape(str(p.get('delta_e_bind_kcal', 'N/A')))
        if isinstance(p.get('delta_e_bind_kcal'), (int, float)):
            qm_e = f"{p.get('delta_e_bind_kcal'):.2f} kcal/mol"
        homo = escape(str(p.get('homo_ev', 'N/A')))
        if isinstance(p.get('homo_ev'), (int, float)):
            homo = f"{p.get('homo_ev'):.2f} eV"
        lumo = escape(str(p.get('lumo_ev', 'N/A')))
        if isinstance(p.get('lumo_ev'), (int, float)):
            lumo = f"{p.get('lumo_ev'):.2f} eV"
        gap = escape(str(p.get('gap_ev', 'N/A')))
        if isinstance(p.get('gap_ev'), (int, float)):
            gap = f"{p.get('gap_ev'):.2f} eV"

        table_rows.append(f"""
        <tr>
            <td><strong>{lig}</strong> (Pose {idx})</td>
            <td><span class="badge badge-vina">{vina} kcal/mol</span></td>
            <td><span class="badge badge-qm">{qm_e}</span></td>
            <td>{homo}</td>
            <td>{lumo}</td>
            <td>{gap}</td>
        </tr>
        """)

    table_html = "\n".join(table_rows) if table_rows else "<tr><td colspan='6'>No poses evaluated.</td></tr>"

    # MD Summary Block
    md_html = ""
    if md_summary:
        bb_rmsd = escape(str(md_summary.get("backbone_rmsd_final", "N/A")))
        mean_rmsf = escape(str(md_summary.get("mean_rmsf", "N/A")))
        coord_dist = escape(str(md_summary.get("mean_coord_dist", "N/A")))
        sim_time = escape(str(md_summary.get("sim_time_ns", "N/A")))

        md_html = f"""
        <div class="card">
            <h3>Molecular Dynamics Stability Profile (GROMACS)</h3>
            <p class="muted">Simulation Length: {sim_time} ns</p>
            <div class="grid grid-3">
                <div class="stat-box">
                    <span class="stat-label">Backbone RMSD</span>
                    <span class="stat-val">{bb_rmsd} Å</span>
                </div>
                <div class="stat-box">
                    <span class="stat-label">Mean C-alpha RMSF</span>
                    <span class="stat-val">{mean_rmsf} Å</span>
                </div>
                <div class="stat-box">
                    <span class="stat-label">Coordination Distance</span>
                    <span class="stat-val">{coord_dist} Å</span>
                </div>
            </div>
        </div>
        """

    qm_rows = []
    for job in (qm_summary or {}).get('jobs', []):
        result = job.get('results') or {}
        energy = result.get('electronic_energy_hartree', 'Not calculated')
        frontier = []
        for spin, values in result.get('orbitals', {}).items():
            frontier.append(f"Spin {escape(str(spin))}: HOMO {values['homo']['energy_eV']:.4f} eV; "
                            f"LUMO {values['lumo']['energy_eV']:.4f} eV; gap {values['gap_ev']:.4f} eV")
        if job.get('viewer_link'):
            frontier.append(f'<a href="{escape(job["viewer_link"], quote=True)}">Open 3D orbital viewer</a>')
        if job.get('error'):
            frontier.append(escape(job['error']))
        if job.get('orbital_export', {}).get('status') == 'failed':
            frontier.append('Orbital export failed: ' + escape(job['orbital_export']['error']))
        qm_rows.append(f"<tr><td>{escape(str(job['selection']['ligand_id']))}</td>"
                       f"<td>{escape(str(job['status']))}</td><td>{escape(str(energy))}</td>"
                       f"<td>{'<br>'.join(frontier) or 'Not calculated'}</td></tr>")
    qm_html = ('<div class="card"><h2>Isolated-ligand DFT</h2>'
               '<p>Electronic energies are in Hartree. These are not protein binding energies. '
               'Optimization alone does not verify a stationary minimum; frequencies are required.</p>'
               '<table><tr><th>Ligand</th><th>Status</th><th>Electronic energy (Eh)</th><th>Orbitals</th></tr>'
               + ''.join(qm_rows) + '</table><details><summary>Calculation provenance</summary><pre>'
               + escape(json.dumps(qm_summary, indent=2)) + '</pre></details></div>') if qm_rows else ''
    mesh_json = json.dumps(orbital_mesh or {}).replace('<', '\\u003c')
    poses_json = json.dumps(poses_data).replace('<', '\\u003c')

    html_content = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SharK Quantum & MD Dossier — {project_name}</title>
<style>
:root {{
    --bg: #0f172a;
    --card: #1e293b;
    --border: #334155;
    --text: #f8fafc;
    --muted: #94a3b8;
    --primary: #38bdf8;
    --accent: #818cf8;
    --success: #34d399;
    --warning: #fbbf24;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 24px;
    line-height: 1.5;
}}
.container {{ max-width: 1300px; margin: 0 auto; }}
header {{
    border-bottom: 1px solid var(--border);
    padding-bottom: 16px;
    margin-bottom: 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
h1 {{ font-size: 26px; font-weight: 700; color: var(--primary); }}
.subtitle {{ color: var(--muted); font-size: 14px; margin-top: 4px; }}
.card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 24px;
}}
a {{ color: var(--primary); }}
pre {{ overflow-x: auto; font-size: 12px; margin-top: 12px; }}
h2, h3 {{ font-size: 18px; margin-bottom: 12px; color: #f1f5f9; }}
.grid {{ display: grid; gap: 16px; }}
.grid-2 {{ grid-template-columns: 1fr 1fr; }}
.grid-3 {{ grid-template-columns: 1fr 1fr 1fr; }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; text-align: left; }}
th, td {{ padding: 12px; border-bottom: 1px solid var(--border); }}
th {{ color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: 12px; letter-spacing: 0.05em; }}
tr:hover td {{ background: rgba(255,255,255,0.02); }}
.badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 12px; }}
.badge-vina {{ background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }}
.badge-qm {{ background: rgba(52, 211, 153, 0.15); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.3); }}
.stat-box {{ background: rgba(15, 23, 42, 0.6); padding: 16px; border-radius: 6px; border: 1px solid var(--border); }}
.stat-label {{ display: block; font-size: 12px; color: var(--muted); text-transform: uppercase; margin-bottom: 4px; }}
.stat-val {{ font-size: 22px; font-weight: 700; color: var(--primary); }}
.controls {{ display: flex; gap: 12px; align-items: center; margin-bottom: 14px; }}
select, button {{
    background: #0f172a;
    border: 1px solid var(--border);
    color: var(--text);
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 13px;
    cursor: pointer;
}}
button:hover {{ background: #1e293b; border-color: var(--primary); }}
#viewer-container {{ width: 100%; height: 550px; background: #0b0f19; border-radius: 6px; position: relative; }}
.footer {{ font-size: 12px; color: var(--muted); text-align: center; margin-top: 30px; border-top: 1px solid var(--border); padding-top: 16px; }}
</style>
<script>{plotly_code}</script>
</head>
<body>
<div class="container">
    <header>
        <div>
            <h1>SharK Quantum & MD Dossier</h1>
            <div class="subtitle">Project: <strong>{project_name}</strong> | Characterization Suite</div>
        </div>
        <div style="text-align: right;">
            <span class="badge" style="background: rgba(129, 140, 248, 0.15); color: #818cf8; border: 1px solid rgba(129, 140, 248, 0.3);">SharK {__version__}</span>
        </div>
    </header>

    <div class="card" style="display: {'block' if poses_data else 'none'};">
        <h2>Docking Poses & Supplied QM Results</h2>
        <p class="muted" style="margin-bottom: 14px;">Docking scores and any supplied quantum results. Missing calculations are marked N/A.</p>
        <div style="overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th>Ligand & Pose</th>
                        <th>Vina Score</th>
                        <th>QM Binding (Delta E)</th>
                        <th>HOMO</th>
                        <th>LUMO</th>
                        <th>HOMO-LUMO Gap</th>
                    </tr>
                </thead>
                <tbody>
                    {table_html}
                </tbody>
            </table>
        </div>
    </div>

    {qm_html}
    {md_html}

    <div class="card" id="orbital-card" style="display: {'block' if orbital_mesh else 'none'};">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <h2>3D Frontier Molecular Orbitals</h2>
            <div class="controls">
                <label style="font-size: 13px; color: var(--muted);">View:
                    <select id="camera-select">
                        <option value="default">Default 3D</option>
                        <option value="top">Top (XY)</option>
                        <option value="side">Side (XZ)</option>
                    </select>
                </label>
                <label style="font-size: 13px; color: var(--muted);">Isovalue:
                    <select id="iso-select">
                        <option value="0.03" selected>+/- 0.03</option>
                        <option value="0.02">+/- 0.02</option>
                        <option value="0.04">+/- 0.04</option>
                    </select>
                </label>
            </div>
        </div>
        <div id="viewer-container"></div>
        <p class="muted" style="font-size: 12px; margin-top: 8px;">
            Blue: Positive orbital phase (+psi) | Red: Negative orbital phase (-psi). Click and drag to rotate, scroll to zoom.
        </p>
    </div>

    <div class="footer">
        Generated by SharK (Automated Quantum Chemical Workflow Engine). Zero-server offline dossier.
    </div>
</div>

<script>
const meshData = {mesh_json};
const poses = {poses_json};

function render3D() {{
    const container = document.getElementById('viewer-container');
    if (!meshData.pos && !meshData.atoms) {{
        container.textContent = 'No orbital meshes embedded. Use the calculated ligand viewer links above when available.';
        document.getElementById('camera-select').disabled = true;
        document.getElementById('iso-select').disabled = true;
        return;
    }}
    if (!window.Plotly) {{
        container.innerHTML = '<div style="padding: 20px; color: #ef4444;">Plotly library not loaded.</div>';
        return;
    }}

    const traces = [];

    // Check if meshData has triangles
    if (meshData && meshData.pos && meshData.pos.x) {{
        traces.push({{
            type: 'mesh3d',
            x: meshData.pos.x, y: meshData.pos.y, z: meshData.pos.z,
            i: meshData.pos.i, j: meshData.pos.j, k: meshData.pos.k,
            color: '#2563eb', opacity: 0.85, name: 'Phase (+)',
            flatshading: false, lighting: {{ ambient: 0.5, diffuse: 0.7 }}
        }});
        traces.push({{
            type: 'mesh3d',
            x: meshData.neg.x, y: meshData.neg.y, z: meshData.neg.z,
            i: meshData.neg.i, j: meshData.neg.j, k: meshData.neg.k,
            color: '#dc2626', opacity: 0.85, name: 'Phase (-)',
            flatshading: false, lighting: {{ ambient: 0.5, diffuse: 0.7 }}
        }});
    }}

    // Add atoms if present
    if (meshData && meshData.atoms) {{
        traces.push({{
            type: 'scatter3d',
            mode: 'markers+text',
            x: meshData.atoms.x, y: meshData.atoms.y, z: meshData.atoms.z,
            text: meshData.atoms.text,
            textposition: 'top center',
            marker: {{ size: meshData.atoms.size || 5, color: meshData.atoms.color || '#94a3b8' }},
            name: 'Atoms'
        }});
    }}

    const layout = {{
        margin: {{ l: 0, r: 0, b: 0, t: 0 }},
        paper_bgcolor: '#0b0f19',
        scene: {{
            xaxis: {{ visible: false }},
            yaxis: {{ visible: false }},
            zaxis: {{ visible: false }},
            aspectratio: {{ x: 1, y: 1, z: 1 }},
            camera: {{ eye: {{ x: 1.4, y: 1.4, z: 1.2 }} }}
        }},
        legend: {{ font: {{ color: '#f8fafc' }} }}
    }};

    Plotly.react(container, traces, layout, {{ responsive: true, displayModeBar: true }});
}}

document.getElementById('camera-select').addEventListener('change', function(e) {{
    const val = e.target.value;
    let eye = {{ x: 1.4, y: 1.4, z: 1.2 }};
    let up = {{ x: 0, y: 0, z: 1 }};
    if (val === 'top') {{ eye = {{ x: 0, y: 0, z: 2.2 }}; up = {{ x: 0, y: 1, z: 0 }}; }}
    else if (val === 'side') {{ eye = {{ x: 0, y: -2.2, z: 0 }}; up = {{ x: 0, y: 0, z: 1 }}; }}
    Plotly.relayout('viewer-container', {{ 'scene.camera.eye': eye, 'scene.camera.up': up }});
}});

window.addEventListener('DOMContentLoaded', render3D);
if (document.readyState !== 'loading') render3D();
</script>
</body>
</html>
"""
    out_path.write_text(html_content, encoding="utf-8")
    return out_path
