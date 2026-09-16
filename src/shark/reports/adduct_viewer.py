"""Interactive 3D Active-Site Adduct & Reaction Geometry Visualizer using 3Dmol.js.

Visualizes:
1. Catalytic nucleophile residue (e.g., Thr309) with element-colored sticks.
2. Catalytic dyad / activating partner residue (e.g., Asp199).
3. Electrophilic ligand / warhead in high-contrast representation.
4. Reactive trajectory: dashed vector, attack distance (d <= 3.5 A), and Bürgi-Dunitz angle (θ_BD).
5. Frontier orbital isosurface overlay (from ORCA CUBE volumetric data) if available.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union


def _get_3dmol_js() -> str:
    """Loads bundled 3Dmol-min.js from local package directory."""
    local_path = Path(__file__).parent / "3Dmol-min.js"
    if local_path.is_file():
        return local_path.read_text(encoding="utf-8", errors="replace")
    # Fallback to CDN script tag if not bundled
    return ""


def build_adduct_pdb(
    ligand_pdb_or_xyz: str | Path,
    receptor_pdb: Optional[str | Path] = None,
    target_residue: str = "THR309",
    dyad_residue: Optional[str] = "ASP199",
    radius: float = 8.0,
) -> str:
    """Builds a compact PDB snippet containing the ligand and active-site residues."""
    lines = []
    # 1. If receptor PDB is provided, extract target residue and dyad within radius
    if receptor_pdb and Path(receptor_pdb).is_file():
        rec_lines = Path(receptor_pdb).read_text(encoding="utf-8", errors="replace").splitlines()
        target_num = None
        import re
        m = re.search(r"(\d+)", target_residue)
        if m:
            target_num = int(m.group(1))

        dyad_num = None
        if dyad_residue:
            dm = re.search(r"(\d+)", dyad_residue)
            if dm:
                dyad_num = int(dm.group(1))

        for line in rec_lines:
            if line.startswith(("ATOM  ", "HETATM")):
                res_seq_str = line[22:26].strip()
                try:
                    res_seq = int(res_seq_str)
                except ValueError:
                    continue
                if res_seq in (target_num, dyad_num):
                    lines.append(line)

    # 2. Append ligand lines
    if ligand_pdb_or_xyz:
        lig_p = Path(ligand_pdb_or_xyz)
        if lig_p.is_file():
            lig_text = lig_p.read_text(encoding="utf-8", errors="replace")
            for line in lig_text.splitlines():
                if line.startswith(("ATOM  ", "HETATM")):
                    # Ensure residue name is LIG
                    line_mod = line[:17] + "LIG " + line[21:]
                    lines.append(line_mod)
        elif isinstance(ligand_pdb_or_xyz, str):
            for line in ligand_pdb_or_xyz.splitlines():
                if line.startswith(("ATOM  ", "HETATM")):
                    lines.append(line)

    lines.append("END")
    return "\n".join(lines) + "\n"


def generate_adduct_viewer_html(
    pdb_data: str,
    target_residue: str = "THR309",
    nucl_atom_coords: Optional[Tuple[float, float, float]] = None,
    el_atom_coords: Optional[Tuple[float, float, float]] = None,
    attack_distance: Optional[float] = None,
    burgi_dunitz_angle: Optional[float] = None,
    dyad_residue: Optional[str] = "ASP199",
    cube_data: Optional[str] = None,
    isovalue: float = 0.03,
    title: str = "Covalent Adduct & Reaction Geometry",
    standalone: bool = False,
) -> str:
    """Generates an interactive 3D WebGL adduct viewer component.

    Parameters
    ----------
    pdb_data : str
        PDB string containing the active site residues (Thr309, Asp199) and ligand.
    target_residue : str
        Residue label of the nucleophile, e.g. 'THR309'.
    nucl_atom_coords : tuple of 3 floats, optional
        Coordinates of the nucleophile reactive atom (e.g. Thr309 OG1).
    el_atom_coords : tuple of 3 floats, optional
        Coordinates of the electrophilic warhead center.
    attack_distance : float, optional
        Attack distance in Angstroms.
    burgi_dunitz_angle : float, optional
        Bürgi-Dunitz attack angle in degrees.
    dyad_residue : str, optional
        Residue label of activating dyad partner, e.g. 'ASP199'.
    cube_data : str, optional
        Raw text of an ORCA CUBE file for frontier orbital isosurface mapping.
    isovalue : float
        Isovalue for CUBE isosurface (default: 0.03).
    title : str
        Widget title.
    standalone : bool
        If True, generates a full standalone HTML page with 3Dmol script included.
    """
    import re
    res_num_m = re.search(r"(\d+)", target_residue)
    target_resi = int(res_num_m.group(1)) if res_num_m else 309

    dyad_resi = None
    if dyad_residue:
        dyad_m = re.search(r"(\d+)", dyad_residue)
        if dyad_m:
            dyad_resi = int(dyad_m.group(1))

    dist_str = f"{attack_distance:.2f} Å" if attack_distance is not None else "N/A"
    angle_str = f"{burgi_dunitz_angle:.1f}°" if burgi_dunitz_angle is not None else "N/A"
    angle_suffix = f" (θ={angle_str})" if burgi_dunitz_angle is not None else ""
    full_dist_label = f"d = {dist_str}{angle_suffix}"
    full_dist_json = json.dumps(full_dist_label)
    dyad_display = html.escape(dyad_residue) if dyad_residue else "None (Isolated Nucleophile)"

    safe_pdb = json.dumps(pdb_data)
    safe_cube = json.dumps(cube_data) if cube_data else "null"

    nucl_json = json.dumps(nucl_atom_coords) if nucl_atom_coords else "null"
    el_json = json.dumps(el_atom_coords) if el_atom_coords else "null"

    viewer_js = _get_3dmol_js()
    script_source = f"<script>{viewer_js}</script>" if viewer_js else '<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>'

    widget_id = "adduct_viewer_3dmol"

    html_content = f"""
<div class="adduct-viewer-container" style="background:#fff;border:1px solid #d9e2ec;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.04);margin-top:20px;">
  <div class="viewer-header" style="background:#f8fafc;border-bottom:1px solid #e2e8f0;padding:14px 20px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
    <div>
      <h3 style="margin:0;font-size:16px;color:#16283f;display:flex;align-items:center;gap:8px;">
        <span style="display:inline-block;width:10px;height:10px;background:#087b70;border-radius:50%;"></span>
        {html.escape(title)}
      </h3>
      <div style="font-size:12px;color:#64748b;margin-top:3px;">
        Nucleophile: <strong style="color:#059669;">{html.escape(target_residue)}</strong> |
        Dyad Partner: <strong style="color:#2563eb;">{dyad_display}</strong> |
        Reaction Distance: <strong style="color:#dc2626;">{dist_str}</strong> |
        Bürgi-Dunitz Angle (θ_BD): <strong style="color:#d97706;">{angle_str}</strong>
      </div>
    </div>
    <div class="viewer-controls" style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
      <button type="button" id="{widget_id}_spin_btn" style="min-height:34px;padding:4px 12px;font-size:12px;background:#fff;border:1px solid #cbd5e1;border-radius:6px;cursor:pointer;">Spin</button>
      <button type="button" id="{widget_id}_reset_btn" style="min-height:34px;padding:4px 12px;font-size:12px;background:#fff;border:1px solid #cbd5e1;border-radius:6px;cursor:pointer;">Reset View</button>
      <button type="button" id="{widget_id}_labels_btn" style="min-height:34px;padding:4px 12px;font-size:12px;background:#fff;border:1px solid #cbd5e1;border-radius:6px;cursor:pointer;">Toggle Labels</button>
      <button type="button" id="{widget_id}_png_btn" style="min-height:34px;padding:4px 12px;font-size:12px;background:#fff;border:1px solid #cbd5e1;border-radius:6px;cursor:pointer;">Save PNG</button>
    </div>
  </div>

  <div id="{widget_id}" style="width:100%;height:520px;position:relative;background:#0f172a;" role="img" aria-label="3D Active Site and Attack Geometry"></div>

  <div class="viewer-legend" style="background:#f8fafc;border-top:1px solid #e2e8f0;padding:10px 20px;font-size:12px;color:#475569;display:flex;justify-content:space-between;flex-wrap:wrap;gap:14px;">
    <div style="display:flex;gap:16px;align-items:center;">
      <span style="display:flex;align-items:center;gap:5px;"><span style="display:inline-block;width:12px;height:12px;background:#10b981;border-radius:2px;"></span> Thr309 (Green Sticks)</span>
      <span style="display:flex;align-items:center;gap:5px;"><span style="display:inline-block;width:12px;height:12px;background:#3b82f6;border-radius:2px;"></span> Asp199 Dyad (Blue Sticks)</span>
      <span style="display:flex;align-items:center;gap:5px;"><span style="display:inline-block;width:12px;height:12px;background:#f59e0b;border-radius:2px;"></span> Ligand Warhead (Orange Sticks)</span>
      <span style="display:flex;align-items:center;gap:5px;"><span style="display:inline-block;width:12px;height:3px;background:#ef4444;border-top:2px dashed #ef4444;"></span> Near-Attack Vector</span>
    </div>
    <div style="color:#64748b;">Rotate: Left click + Drag | Zoom: Scroll wheel | Pan: Right click + Drag</div>
  </div>
</div>

<script>
(function() {{
  function init3DmolViewer() {{
    if (typeof $3Dmol === 'undefined') {{
      setTimeout(init3DmolViewer, 100);
      return;
    }}
    const container = document.getElementById('{widget_id}');
    if (!container) return;

    const config = {{ backgroundColor: '#0f172a' }};
    const viewer = $3Dmol.createViewer(container, config);

    const pdbStr = {safe_pdb};
    if (pdbStr && pdbStr.trim().length > 0) {{
      viewer.addModel(pdbStr, 'pdb');
    }}

    // 1. Base style for all atoms in active site
    viewer.setStyle({{}}, {{
      stick: {{ radius: 0.18, colorscheme: 'Jmol' }}
    }});

    // 2. Nucleophile (e.g. Thr309): green carbons
    viewer.addStyle({{ resi: ['{target_resi}', '{target_residue}', {target_resi}] }}, {{
      stick: {{ radius: 0.26, colorscheme: 'greenCarbon' }},
      sphere: {{ radius: 0.42, color: '#10b981' }}
    }});

    // 3. Catalytic Dyad partner (e.g. Asp199): blue carbons if present
    {f"viewer.addStyle({{ resi: ['{dyad_resi}', {dyad_resi}] }}, {{ stick: {{ radius: 0.22, colorscheme: 'blueCarbon' }} }});" if dyad_resi else ""}

    // 4. Ligand: orange carbons
    viewer.addStyle({{ resn: ['LIG', 'UNL', 'MOL'] }}, {{
      stick: {{ radius: 0.26, colorscheme: 'yellowCarbon' }},
      sphere: {{ radius: 0.40, color: '#f59e0b' }}
    }});

    const nuclCrd = {nucl_json};
    const elCrd = {el_json};

    let labels = [];

    // Attack trajectory cylinder & labels
    if (nuclCrd && elCrd) {{
      viewer.addCylinder({{
        start: {{ x: nuclCrd[0], y: nuclCrd[1], z: nuclCrd[2] }},
        end: {{ x: elCrd[0], y: elCrd[1], z: elCrd[2] }},
        radius: 0.08,
        color: '#ef4444',
        dashed: true
      }});

      const midX = (nuclCrd[0] + elCrd[0]) / 2;
      const midY = (nuclCrd[1] + elCrd[1]) / 2;
      const midZ = (nuclCrd[2] + elCrd[2]) / 2;

      const dLabel = viewer.addLabel({full_dist_json}, {{
        position: {{ x: midX, y: midY, z: midZ + 0.35 }},
        backgroundColor: 'rgba(15, 23, 42, 0.88)',
        fontColor: '#fca5a5',
        fontSize: 12,
        borderColor: '#ef4444',
        borderThickness: 1
      }});
      labels.push(dLabel);

      const nuclLabel = viewer.addLabel('{html.escape(target_residue)} (OG1)', {{
        position: {{ x: nuclCrd[0], y: nuclCrd[1], z: nuclCrd[2] + 0.45 }},
        backgroundColor: 'rgba(16, 185, 129, 0.88)',
        fontColor: '#ffffff',
        fontSize: 11
      }});
      labels.push(nuclLabel);

      const elLabel = viewer.addLabel('Electrophile (Ligand)', {{
        position: {{ x: elCrd[0], y: elCrd[1], z: elCrd[2] - 0.45 }},
        backgroundColor: 'rgba(245, 158, 11, 0.88)',
        fontColor: '#ffffff',
        fontSize: 11
      }});
      labels.push(elLabel);
    }}

    setTimeout(function() {{
      viewer.resize();
      viewer.zoomTo();
      viewer.render();
    }}, 150);

    window.addEventListener('resize', function() {{
      viewer.resize();
      viewer.render();
    }});

    // Overlay ORCA CUBE isosurface if provided
    const cubeData = {safe_cube};
    if (cubeData) {{
      try {{
        const voldata = new $3Dmol.VolumeData(cubeData, 'cube');
        viewer.addIsosurface(voldata, {{
          isoval: {isovalue},
          color: '#38bdf8',
          alpha: 0.65,
          smoothness: 2
        }});
        viewer.addIsosurface(voldata, {{
          isoval: -{isovalue},
          color: '#f87171',
          alpha: 0.65,
          smoothness: 2
        }});
      }} catch (e) {{
        console.warn('Could not load volumetric CUBE in 3Dmol viewer:', e);
      }}
    }}

    viewer.zoomTo();
    viewer.render();

    // UI Controls
    let spinning = false;
    const spinBtn = document.getElementById('{widget_id}_spin_btn');
    if (spinBtn) {{
      spinBtn.addEventListener('click', function() {{
        spinning = !spinning;
        viewer.spin(spinning ? 'y' : false, 1.0);
        spinBtn.textContent = spinning ? 'Pause Spin' : 'Spin';
        spinBtn.style.background = spinning ? '#e2e8f0' : '#fff';
      }});
    }}

    const resetBtn = document.getElementById('{widget_id}_reset_btn');
    if (resetBtn) {{
      resetBtn.addEventListener('click', function() {{
        viewer.zoomTo();
        viewer.render();
      }});
    }}

    let showLabels = true;
    const labelsBtn = document.getElementById('{widget_id}_labels_btn');
    if (labelsBtn) {{
      labelsBtn.addEventListener('click', function() {{
        showLabels = !showLabels;
        labels.forEach(l => l.show = showLabels);
        viewer.render();
        labelsBtn.style.background = showLabels ? '#fff' : '#e2e8f0';
      }});
    }}

    const pngBtn = document.getElementById('{widget_id}_png_btn');
    if (pngBtn) {{
      pngBtn.addEventListener('click', function() {{
        const pngUrl = viewer.pngURI();
        const a = document.createElement('a');
        a.href = pngUrl;
        a.download = 'shark_adduct_geometry.png';
        a.click();
      }});
    }}
  }}

  if (document.readyState === 'loading') {{
    document.addEventListener('DOMContentLoaded', init3DmolViewer);
  }} else {{
    init3DmolViewer();
  }}
}})();
</script>
"""

    if standalone:
        return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body {{ margin: 0; padding: 20px; font-family: system-ui, -apple-system, sans-serif; background: #f1f5f9; }}
.wrapper {{ max-width: 1200px; margin: 0 auto; }}
</style>
{script_source}
</head>
<body>
<div class="wrapper">
{html_content}
</div>
</body>
</html>
"""
    return html_content

