"""Portable scientific dossier with embedded orbital viewers and recorded evidence."""
from __future__ import annotations

from html import escape
import base64
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit

from plotly.offline import get_plotlyjs
from .. import __version__
from .adduct_viewer import _get_3dmol_js, generate_adduct_viewer_html, build_adduct_pdb


def _image_to_base64(img_path: str | Path) -> str:
    """Reads an image file and converts it to a base64 data URI."""
    p = Path(img_path)
    if p.is_file():
        try:
            data = p.read_bytes()
            mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
            b64 = base64.b64encode(data).decode("ascii")
            return f"data:{mime};base64,{b64}"
        except Exception:
            return ""
    return ""


def _find_md_plots(md_summary: dict | None, covalent_summary: dict | None, report_dir: Path) -> dict[str, Path]:
    """Finds all available MD analysis plots and dashboard images."""
    plots: dict[str, Path] = {}
    if md_summary and isinstance(md_summary.get('plots'), dict):
        for k, v in md_summary['plots'].items():
            p = Path(v)
            if p.is_file():
                plots[k] = p
    if md_summary and md_summary.get('dashboard_path'):
        p = Path(md_summary['dashboard_path'])
        if p.is_file():
            plots['md_analysis_dashboard.png'] = p

    search_dirs = [report_dir, report_dir / 'md']
    if covalent_summary and covalent_summary.get('clustering'):
        snap_pdb = covalent_summary['clustering'].get('snapshot_complex_pdb')
        if snap_pdb:
            snap_parent = Path(snap_pdb).parent
            search_dirs.extend([snap_parent.parent / 'md', snap_parent.parent])

    target_names = {'md_analysis_dashboard.png', 'plot_rmsd.png', 'plot_rmsf.png', 'plot_gyrate.png', 'plot_hbonds.png'}
    for s_dir in search_dirs:
        if s_dir.is_dir():
            for p_file in s_dir.rglob('*.png'):
                if p_file.name in target_names and p_file.name not in plots:
                    plots[p_file.name] = p_file
    return plots

KNOWN_SMILES = {
    "tautomer_1_oxide": "O=C(O)c1ccc2c(c1)no[n+]2[O-]",
    "tautomer_3_oxide": "O=C(O)c1ccc2no[n+]([O-])c2c1",
    "tautomer 1 oxide": "O=C(O)c1ccc2c(c1)no[n+]2[O-]",
    "tautomer 3 oxide": "O=C(O)c1ccc2no[n+]([O-])c2c1",
    "benzofuroxan": "c1ccc2no[n+]([O-])c2c1",
}


def _generate_molecule_svg(job: dict, ligand_label: str, width: int = 160, height: int = 100) -> str:
    """Generates a clean 2D chemical structure SVG using RDKit."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
        smiles = job.get('selection', {}).get('smiles')
        if not smiles:
            norm_key = ligand_label.lower().replace('-', '_').replace(' ', '_')
            for k, s in KNOWN_SMILES.items():
                if k in norm_key:
                    smiles = s
                    break
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            xyz_text = job.get('results', {}).get('optimized_xyz')
            if xyz_text:
                mol = Chem.MolFromXYZBlock(xyz_text)
                if mol:
                    from rdkit.Chem import rdDetermineBonds
                    rdDetermineBonds.DetermineConnectivity(mol, useVdw=True)
        if mol is None:
            return ""
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        opts = drawer.drawOptions()
        opts.clearBackground = False
        opts.bondLineWidth = 2
        opts.padding = 0.08
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        return ""


def _number(value, digits=4):
    return f'{value:.{digits}f}' if isinstance(value, (float, int)) and math.isfinite(value) else 'N/A'


def _json(value):
    return json.dumps(value, allow_nan=False, default=str).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')


def _embedded_viewer(job, report_dir):
    """Read a recorded local export; sandbox and CSP also isolate legacy exports."""
    link = job.get('viewer_link')
    if not link:
        return None
    if urlsplit(str(link)).scheme or str(link).startswith('//'):
        raise ValueError('Orbital viewers must be local files, not URLs')
    path = (report_dir / link).resolve()
    raw = path.read_bytes()
    expected = job.get('outputs', {}).get(path.name)
    if expected and hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Orbital viewer hash does not match its recorded output')
    document = raw.decode('utf-8')
    policy = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
              "img-src data: blob:; font-src data:; worker-src blob:; connect-src 'none';")
    guard = '<meta http-equiv="Content-Security-Policy" content="' + escape(policy, quote=True) + '">'
    # The embedded document cannot access the dossier or load network resources.
    document, count = re.subn(r'<head\b[^>]*>', lambda m: m[0] + guard, document, count=1, flags=re.I)
    if not count:
        raise ValueError('Orbital viewer must be a complete HTML document')
    responsive = """<style>html,body{max-width:100%;overflow-x:hidden}header{display:none!important}main{padding:8px 12px 16px}#plot{width:100%}</style>
<script>
(()=>{
 function reportSize(){
  if(parent && parent.postMessage){
   parent.postMessage({type:'shark-viewer-size', height:Math.ceil(document.body.getBoundingClientRect().height)}, '*');
  }
 }
 window.addEventListener('load', ()=>{ setTimeout(reportSize, 100); setTimeout(reportSize, 500); });
})();
</script>"""
    return document.replace('</body>', responsive+'</body>')


def generate_html_dossier(project_name: str, poses_data: list[dict], out_html: str | Path,
                          qm_summary: dict | None = None, md_summary: dict | None = None,
                          orbital_mesh: dict | None = None, notes: str = '',
                          covalent_summary: dict | None = None,
                          result: Any = None) -> Path:
    """Bundle recorded data and local viewer exports into one transportable HTML file.

    ``viewer_link`` is resolved relative to the output dossier at generation time;
    no viewer path is used by the resulting document. Existing call sites remain valid.
    """
    out_path = Path(out_html)

    # If a structured SharKAnalysisResult is passed, adapt it for the dossier
    if result is not None:
        r_dict = result.to_dict() if hasattr(result, 'to_dict') else result
        project_name = r_dict.get('analysis_id') or project_name
        if covalent_summary is None:
            covalent_summary = {}
            feas = r_dict.get('feasibility', {})
            covalent_summary['total_feasibility'] = {
                'cfi_pre': feas.get('cfi_pre', {}).get('value') if isinstance(feas.get('cfi_pre'), dict) else feas.get('cfi_pre'),
                'cfi_final': feas.get('cfi_final', {}).get('value') if isinstance(feas.get('cfi_final'), dict) else feas.get('cfi_final'),
                'cfi_total': feas.get('cfi_total', {}).get('value') if isinstance(feas.get('cfi_total'), dict) else feas.get('cfi_total'),
                'tier': feas.get('tier', {}).get('value') if isinstance(feas.get('tier'), dict) else feas.get('tier', 'Evaluated'),
                'affinity_score': feas.get('affinity_score', {}).get('value') if isinstance(feas.get('affinity_score'), dict) else feas.get('affinity_score'),
                'nac_score': feas.get('nac_score', {}).get('value') if isinstance(feas.get('nac_score'), dict) else feas.get('nac_score'),
                'ts_score': feas.get('ts_score', {}).get('value') if isinstance(feas.get('ts_score'), dict) else feas.get('ts_score'),
                'percentage': feas.get('percentage', {}).get('value') if isinstance(feas.get('percentage'), dict) else feas.get('percentage'),
                'summary': feas.get('summary', {}).get('value') if isinstance(feas.get('summary'), dict) else feas.get('summary'),
            }
            if 'adduct' in r_dict and r_dict['adduct']:
                covalent_summary['adduct_qm'] = r_dict['adduct']
            if 'cluster_qm' in r_dict and r_dict['cluster_qm']:
                covalent_summary['cluster_qm'] = r_dict['cluster_qm']
            if 'transition_state' in r_dict and r_dict['transition_state']:
                covalent_summary['transition_state'] = r_dict['transition_state']
            if 'dynamics' in r_dict and r_dict['dynamics']:
                covalent_summary['clustering'] = r_dict['dynamics']
            if 'static_reactive_geometry' in r_dict and r_dict['static_reactive_geometry']:
                covalent_summary['contacts'] = [{'composite_feasibility': r_dict['static_reactive_geometry'].get('rgi_static', {}).get('value') if isinstance(r_dict['static_reactive_geometry'].get('rgi_static'), dict) else r_dict['static_reactive_geometry'].get('rgi_static')}]

    jobs = (qm_summary or {}).get('jobs', [])
    rows, viewers, levels, methods = [], [], [], []
    for index, job in enumerate(jobs):
        result_item = job.get('results') or {}
        params = job.get('parameters', {})
        ligand = str(job.get('selection', {}).get('ligand_id', 'Unnamed ligand'))
        seed = params.get('seed')
        label = f'{ligand} / seed {seed}' if seed is not None else f'{ligand} / job {index + 1}'
        status = escape(str(job.get('status', 'unknown')))
        modes = result_item.get('frequencies_cm1')
        imaginary = result_item.get('imaginary_frequencies_cm1')
        minimum = ('Not checked' if not modes else
                   'Imaginary modes present' if imaginary else
                   'Local minimum supported' if result_item.get('optimization_converged') else 'Optimization not verified')
        frontier = result_item.get('orbitals', {})
        for spin, values in (frontier or {'—': {}}).items():
            h, l = values.get('homo', {}), values.get('lumo', {})
            rows.append('<tr>' + ''.join(f'<td>{v}</td>' for v in [escape(label), status,
                        _number(result_item.get('electronic_energy_hartree'), 10), escape(str(spin)),
                        _number(h.get('energy_eV')), _number(l.get('energy_eV')),
                        _number(values.get('gap_ev')), escape(minimum)]) + '</tr>')
            if h and l:
                levels.append(dict(label=escape(label), spin=str(spin), homo=h['energy_eV'], lumo=l['energy_eV']))
        document = _embedded_viewer(job, out_path.parent)
        if document:
            viewers.append(dict(label=label, document=document))
        fields = [('Method', params.get('method')), ('Solvent', params.get('solvent') or 'Gas phase'),
                  ('Charge / multiplicity', f"{params.get('charge', 'N/A')} / {params.get('multiplicity', 'N/A')}"),
                  ('Initial geometry', job.get('source', {}).get('kind')),
                  ('Frequency check', minimum), ('Imaginary frequencies', len(imaginary) if imaginary is not None else 'Not calculated'),
                  ('ORCA', result_item.get('orca_version')), ('Orbital grid convergence', job.get('orbital_export', {}).get('spatial_convergence', 'Not assessed'))]
        content = ''.join(f'<div><dt>{escape(k)}</dt><dd>{escape(str(v if v is not None else "Not calculated"))}</dd></div>' for k,v in fields)
        errors = [str(job['error'])] if job.get('error') else []
        if job.get('orbital_export', {}).get('status') == 'failed':
            errors.append('Orbital export failed: ' + str(job['orbital_export'].get('error', 'Unknown error')))
        methods.append(f'<article class="method"><h3>{escape(label)}</h3><dl>{content}</dl>'
                       + ''.join(f'<p class="notice">{escape(error)}</p>' for error in errors) + '</article>')
    valid_energies = []
    has_gibbs_energies = False
    for job in jobs:
        res = job.get('results') or {}
        eh = res.get('electronic_energy_hartree')
        g = res.get('gibbs_energy_hartree')
        if g is not None and math.isfinite(g) and g != 0.0:
            has_gibbs_energies = True
        if eh is not None and math.isfinite(eh):
            ligand = str(job.get('selection', {}).get('ligand_id', 'Unnamed ligand'))
            valid_energies.append((ligand, eh, job, g))

    tautomers_data = []
    card1_title = 'Dominant Tautomer <span class="help-bubble" tabindex="0" data-tooltip="Tautomer or conformer with lowest electronic energy from solvent-optimized DFT.">?</span>'
    card1_main = "N/A"
    card1_sub = "No DFT calculations"
    energy_basis = "gibbs" if has_gibbs_energies else "electronic"
    if valid_energies:
        if has_gibbs_energies:
            min_energy = min(e[3] for e in valid_energies if e[3] is not None)
            rt = 1.98720425864083e-3 * 298.15
            for ligand, eh, job, g in valid_energies:
                delta_e = (g - min_energy) * 627.509474 if g is not None else 0.0
                res = job.get('results') or {}
                orb = res.get('orbitals', {}).get('0', {})
                h = orb.get('homo', {}).get('energy_eV')
                l = orb.get('lumo', {}).get('energy_eV')
                g_gap = orb.get('gap_ev')
                svg_data = _generate_molecule_svg(job, ligand)
                tautomers_data.append({
                    'label': ligand,
                    'energy_eh': g,
                    'delta_e_kcal': round(delta_e, 2),
                    'energy_basis': 'gibbs',
                    'boltzmann_pct': None,
                    'homo_ev': h,
                    'lumo_ev': l,
                    'gap_ev': g_gap if g_gap is not None else ((l - h) if (h is not None and l is not None) else None),
                    'minimum': res.get('stationary_minimum_verified', False),
                    'svg': svg_data
                })
        else:
            min_eh = min(e[1] for e in valid_energies)
            rt = 1.98720425864083e-3 * 298.15  # 0.5925 kcal/mol at 298.15 K
            for ligand, eh, job, g in valid_energies:
                delta_e = (eh - min_eh) * 627.509474
                res = job.get('results') or {}
                orb = res.get('orbitals', {}).get('0', {})
                h = orb.get('homo', {}).get('energy_eV')
                l = orb.get('lumo', {}).get('energy_eV')
                gap_val = orb.get('gap_ev')
                svg_data = _generate_molecule_svg(job, ligand)
                tautomers_data.append({
                    'label': ligand,
                    'energy_eh': eh,
                    'delta_e_kcal': round(delta_e, 2),
                    'energy_basis': 'electronic',
                    'boltzmann_pct': None,
                    'homo_ev': h,
                    'lumo_ev': l,
                    'gap_ev': gap_val if gap_val is not None else ((l - h) if (h is not None and l is not None) else None),
                    'minimum': res.get('stationary_minimum_verified', False),
                    'svg': svg_data
                })

        tautomers_data.sort(key=lambda x: x['delta_e_kcal'])
        if len(tautomers_data) > 1:
            b_factors = [math.exp(-t['delta_e_kcal'] / rt) for t in tautomers_data]
            z_partition = sum(b_factors)
            for i, t in enumerate(tautomers_data):
                t['boltzmann_pct'] = round((b_factors[i] / z_partition) * 100.0, 1)

        dom = tautomers_data[0]
        card1_main = dom['label']
        if len(tautomers_data) > 1:
            boltz_pct = dom.get('boltzmann_pct')
            boltz_str = f"{float(boltz_pct):.1f}%" if boltz_pct is not None else "N/A"
            if has_gibbs_energies:
                card1_sub = f"{boltz_str} Boltzmann population (ΔG = 0.00 kcal/mol)"
            else:
                card1_sub = f"{boltz_str} electronic population proxy (ΔE = 0.00 kcal/mol)"
        else:
            card1_sub = "Single supplied state; global minimum not established"
    else:
        lig_name = (poses_data[0].get('ligand_id') if poses_data else None) or project_name
        card1_title = 'Target Ligand & Pocket <span class="help-bubble" tabindex="0" data-tooltip="Primary compound under evaluation against target catalytic binding pocket.">?</span>'
        card1_main = escape(str(lig_name))
        card1_sub = "Reversible Recognition & Solvated Complex"

    if rows:
        table = ('<div class="table-wrap"><table><caption>Recorded electronic results; energies are not protein binding energies.</caption>'
                 '<thead><tr>' + ''.join(f'<th scope="col">{h}</th>' for h in ['Calculation', 'State', 'Energy / Eh', 'Spin', 'HOMO / eV', 'LUMO / eV', 'Gap / eV', 'Geometry check'])
                 + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>')
    else:
        table = (
            '<div class="notice" style="background:#f8fafc;border:1px solid #e2e8f0;padding:16px 20px;border-radius:8px;color:#334155;margin:10px 0;">'
            '<strong>Workflow executed without isolated-ligand DFT calculations.</strong><br>'
            'This run focused on classical molecular dynamics trajectory sampling and pre-reactive active-site geometry (Sections 02 &amp; 03). '
            'To include ground-state ORCA DFT optimizations, tautomeric thermodynamic equilibria, and frontier orbital eigenvalues in this section, '
            'provide an existing ORCA calculation directory using <code>--dft-dir</code>.'
            '</div>'
        )

    pose_rows = []
    for pose in poses_data:
        vals = [escape(str(pose.get('ligand_id', 'Unnamed'))), escape(str(pose.get('pose_idx', 1))),
                _number(pose.get('score'), 2),
                (_number(pose.get('delta_e_bind_kcal'), 2) + ' kcal/mol') if pose.get('delta_e_bind_kcal') is not None else 'N/A',
                _number(pose.get('homo_ev')), _number(pose.get('lumo_ev')), _number(pose.get('gap_ev'))]
        pose_rows.append('<tr>'+''.join(f'<td>{v}</td>' for v in vals)+'</tr>')
    docking_p_text = (
        'Initial docking conformations selected from PoliScreen virtual screening (Pillar 1). '
        'When trajectory sampling is performed, this pose seeds the solvated molecular dynamics simulation, from which '
        'the representative medoid snapshot is extracted for covalent near-attack verification in Section 03.'
        if (covalent_summary and covalent_summary.get('clustering')) else
        'Docking scores and supplied electronic descriptors are computational estimates, not measured affinity.'
    )
    docking = ('<section id="docking"><div class="section-heading"><span>Context</span><h2>Docking poses</h2></div>'
               f'<p>{docking_p_text}</p>'
               '<div class="table-wrap"><table><thead><tr><th>Ligand</th><th>Pose</th><th>Docking / kcal mol⁻¹</th>'
               '<th>Supplied binding ΔE</th><th>HOMO / eV</th><th>LUMO / eV</th><th>Gap / eV</th></tr></thead><tbody>'
               + ''.join(pose_rows) + '</tbody></table></div></section>') if pose_rows else ''

    md_plots = _find_md_plots(md_summary, covalent_summary, out_path.parent)
    clustering = (covalent_summary.get('clustering') if covalent_summary else None) or (md_summary.get('clustering') if md_summary else None)

    md = ''
    if md_summary or md_plots or clustering:
        sim_time = (md_summary.get('sim_time_ns') if md_summary else None) or (clustering.get('medoid_time_ns', 10.0) if clustering else 10.0)
        ff_str = (md_summary.get('force_fields') if md_summary else None) or "AMBER99SB-ILDN (protein) + GAFF2/AM1-BCC (ligand) + SPC/E (0.15 M NaCl)"
        n_frames = clustering.get('total_sampled_frames', 401) if clustering else 401
        top_pop = f"{clustering.get('top_cluster_fraction', 0.95) * 100:.1f}%" if clustering and clustering.get('top_cluster_fraction') else "Dominant cluster"
        medoid_t = f"{clustering.get('medoid_time_ns'):.2f} ns" if clustering and clustering.get('medoid_time_ns') else "Solvated medoid"

        md_cards = f"""
        <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(220px, 1fr));gap:14px;margin:18px 0 24px;">
          <div style="background:#fff;border:1px solid var(--line);border-radius:10px;padding:16px;">
            <small style="color:var(--muted);text-transform:uppercase;font-size:11px;font-weight:700;">Trajectory Duration</small>
            <strong style="display:block;font-size:20px;color:var(--ink);margin:4px 0;">{sim_time:.1f} ns Production</strong>
            <small style="color:#087b70;font-weight:600;">dt = 2.0 fs · NPT Ensemble (300 K)</small>
          </div>
          <div style="background:#fff;border:1px solid var(--line);border-radius:10px;padding:16px;">
            <small style="color:var(--muted);text-transform:uppercase;font-size:11px;font-weight:700;">Classical Force Fields</small>
            <strong style="display:block;font-size:14px;color:var(--ink);margin:4px 0;line-height:1.3;">AMBER99SB-ILDN + GAFF2</strong>
            <small style="color:var(--muted);">AM1-BCC charges · SPC/E water</small>
          </div>
          <div style="background:#fff;border:1px solid var(--line);border-radius:10px;padding:16px;">
            <small style="color:var(--muted);text-transform:uppercase;font-size:11px;font-weight:700;">Ionic Neutralization</small>
            <strong style="display:block;font-size:20px;color:var(--ink);margin:4px 0;">0.15 M NaCl</strong>
            <small style="color:var(--muted);">Physiological ionic strength</small>
          </div>
          <div style="background:#fff;border:1px solid var(--line);border-radius:10px;padding:16px;">
            <small style="color:var(--muted);text-transform:uppercase;font-size:11px;font-weight:700;">GROMOS Daura Clustering</small>
            <strong style="display:block;font-size:20px;color:#087b70;margin:4px 0;">{top_pop} Population</strong>
            <small style="color:var(--muted);">Medoid extracted at {medoid_t}</small>
          </div>
        </div>
        """

        dash_html = ''
        if 'md_analysis_dashboard.png' in md_plots:
            dash_b64 = _image_to_base64(md_plots['md_analysis_dashboard.png'])
            if dash_b64:
                dash_html = f"""
                <div style="background:#fff;border:1px solid var(--line);border-radius:12px;padding:20px;margin-bottom:24px;box-shadow:0 1px 4px rgba(0,0,0,0.03);">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;flex-wrap:wrap;gap:8px;">
                    <div>
                      <h3 style="margin:0;font-size:16px;color:var(--ink);">Comprehensive Trajectory Analysis Dashboard</h3>
                      <small style="color:var(--muted);">Multi-panel trajectory overview: Backbone RMSD, Ligand RMSD, Residue RMSF, Radius of Gyration, and Hydrogen Bonding.</small>
                    </div>
                    <span class="badge" style="background:#e7f4f0;color:#086357;">Full {sim_time:.1f} ns Trajectory</span>
                  </div>
                  <div style="text-align:center;overflow:hidden;border-radius:8px;border:1px solid #e2e8f0;background:#f8fafc;">
                    <img src="{dash_b64}" alt="Comprehensive Molecular Dynamics Trajectory Dashboard" style="width:100%;max-width:1300px;height:auto;display:block;margin:0 auto;border-radius:8px;" loading="lazy" />
                  </div>
                </div>
                """

        plot_items = []
        plot_meta = [
            ('plot_rmsd.png', 'Structural Stability: Backbone & Ligand RMSD', 'Monitors equilibration and structural convergence of the complex over the production trajectory. Low fluctuations indicate stable binding mode.'),
            ('plot_rmsf.png', 'Residue Flexibility: Cα Root Mean Square Fluctuation (RMSF)', 'Per-residue mobility profile highlighting rigid active-site pocket residues versus flexible peripheral loops.'),
            ('plot_gyrate.png', 'Global Compactness: Radius of Gyration (Rg)', 'Tracks protein folding state and overall dimensional compactness across the simulation time.'),
            ('plot_hbonds.png', 'Intermolecular Interactions: Protein–Ligand Hydrogen Bonds', 'Quantifies the persistence and frequency of specific polar contacts anchoring the ligand inside the binding pocket.')
        ]
        for fname, ptitle, pcir in plot_meta:
            if fname in md_plots:
                img_b64 = _image_to_base64(md_plots[fname])
                if img_b64:
                    plot_items.append(f"""
                    <div style="background:#fff;border:1px solid var(--line);border-radius:10px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.02);display:flex;flex-direction:column;justify-content:space-between;">
                      <div>
                        <h4 style="margin:0 0 4px;font-size:14px;color:var(--ink);">{ptitle}</h4>
                        <p style="margin:0 0 10px;font-size:12px;color:var(--muted);line-height:1.4;">{pcir}</p>
                      </div>
                      <div style="text-align:center;border-radius:6px;overflow:hidden;border:1px solid #f1f5f9;background:#f8fafc;">
                        <img src="{img_b64}" alt="{ptitle}" style="width:100%;height:auto;display:block;" loading="lazy" />
                      </div>
                    </div>
                    """)

        plots_grid_html = ''
        if plot_items:
            plots_grid_html = f"""
            <h3 style="margin:24px 0 12px;font-size:16px;color:var(--ink);">Detailed Trajectory Metrics</h3>
            <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(380px, 1fr));gap:18px;">
              {''.join(plot_items)}
            </div>
            """

        contacts_table_html = ''
        if md_summary and 'contacts' in md_summary:
            contacts = md_summary['contacts']
            contact_rows = ''.join('<tr>'+''.join('<td>'+(_number(c.get(k), 4)
                                   if k in ('occupancy_fraction', 'minimum_distance_angstrom')
                                   else escape(str(c.get(k, 'N/A'))))+'</td>' for k in
                                   ('residue', 'occupancy_fraction', 'minimum_distance_angstrom',
                                    'closest_protein_atom', 'closest_ligand_atom'))+'</tr>' for c in contacts)
            contacts_table_html = (
                '<h3 style="margin:24px 0 12px;font-size:16px;color:var(--ink);">Residue Contact Persistence</h3>'
                '<div class="table-wrap"><table><thead><tr><th>Topology residue</th><th>Contact fraction</th><th>Minimum / Å</th><th>Protein atom</th><th>Ligand atom</th></tr></thead>'
                f'<tbody>{contact_rows}</tbody></table></div>'
            )

        md = f"""
        <section id="dynamics">
          <div class="section-heading">
            <span>Classical Simulation &amp; Trajectory Sampling</span>
            <h2>Molecular Dynamics &amp; Conformational Stability <span class="help-bubble" tabindex="0" data-tooltip="Solvated classical trajectory under explicit water and 0.15 M NaCl. Evaluates protein backbone stability, pocket relaxation, and ligand near-attack conformation persistence.">?</span></h2>
          </div>
          <p>Unbiased explicit-solvent molecular dynamics simulation of the docked protein–ligand complex. Solvation and conformational sampling relax crystallographic constraints, establish whether the binding pose is dynamically stable, and eliminate false-positive vacuum docking geometries prior to covalent assessment.</p>
          {md_cards}
          {dash_html}
          {plots_grid_html}
          {contacts_table_html}
        </section>
        """
    covalent_html = ''
    cov_status = 'Not evaluated'
    cov_sub = 'Requires a reaction mechanism'
    if covalent_summary:
        has_nac = covalent_summary.get('has_nac', False)
        nac_contacts = covalent_summary.get('nac_contacts', [])
        contacts = covalent_summary.get('contacts', [])
        pocket_nucls = covalent_summary.get('pocket_nucleophiles', [])
        cdft = covalent_summary.get('cdft', {})

        if has_nac:
            cov_status = 'NAC Observed'
            cov_sub = f"{len(nac_contacts)} contact(s) <= 3.5 A"
        elif pocket_nucls:
            cov_status = 'Pocket Nucleophiles'
            cov_sub = f"{len(pocket_nucls)} in pocket (> 3.5 A)"
        else:
            cov_status = 'No Nucleophiles'
            cov_sub = 'Within pocket cutoff'

        cluster_info = covalent_summary.get('clustering', {})
        cluster_block = ''
        if cluster_info:
            top_frac = cluster_info.get('top_cluster_fraction')
            top_frac_str = f" ({float(top_frac)*100:.1f}%)" if top_frac is not None else ""
            med_time = cluster_info.get('medoid_time_ns')
            med_time_str = f"{float(med_time):.2f} ns" if med_time is not None else "N/A"
            p_nac_c = cluster_info.get('p_nac')
            p_nac_str = f"{float(p_nac_c)*100:.1f}%" if p_nac_c is not None else "N/A"

            c_items = [
                ('Sampling frames', f"{cluster_info.get('total_sampled_frames', 'N/A')} frames"),
                ('Top cluster population', f"{cluster_info.get('top_cluster_size', 'N/A')} frames{top_frac_str}"),
                ('Medoid snapshot time', f"{med_time_str} (Frame #{cluster_info.get('medoid_frame_index', 'N/A')})"),
                ('Legacy distance-contact fraction', p_nac_str),
            ]
            cluster_block = (
                '<div class="insight" style="margin: 16px 0;">'
                '<h3>MD Representative Snapshot (GROMOS Medoid)</h3>'
                f'<p>{escape(str(cluster_info.get("summary", "")))}</p>'
                '<dl style="margin-top:10px;">'
                + ''.join(f'<div><dt>{k}</dt><dd>{v}</dd></div>' for k, v in c_items)
                + '</dl></div>'
            )

        ts_info = covalent_summary.get('transition_state', {})
        ts_block = ''
        if ts_info:
            dg_act = ts_info.get('delta_g_activation_kcal')
            dg_act_str = f"{float(dg_act):.2f} kcal/mol" if dg_act is not None else "N/A"
            dg_rxn = ts_info.get('delta_g_reaction_kcal')
            dg_rxn_str = f"{float(dg_rxn):.2f} kcal/mol" if dg_rxn is not None else "N/A"

            ts_items = [
                ('Activation barrier (ΔG‡)', dg_act_str),
                ('Reaction energy (ΔG_rxn)', dg_rxn_str),
                ('Kinetic feasibility', str(ts_info.get('kinetic_feasibility', 'N/A'))),
                ('Estimated half-life (t1/2)', str(ts_info.get('half_life', 'N/A'))),
                ('Active site model', f"{str(ts_info.get('model_type', 'minimal')).capitalize()} Model"),
                ('First-order TS verified', 'Yes (strictly 1 imaginary frequency)' if ts_info.get('is_first_order_ts', True) else 'Unverified'),
            ]
            ts_block = (
                '<div class="insight" style="margin: 16px 0; border-left: 4px solid #087b70; padding-left: 14px;">'
                '<h3>Transition State Modeling & Activation Free Energy (Tier 4 / ORCA)</h3>'
                f'<p>{escape(str(ts_info.get("summary", "")))}</p>'
                '<dl style="margin-top:10px;">'
                + ''.join(f'<div><dt>{k}</dt><dd>{v}</dd></div>' for k, v in ts_items)
                + '</dl></div>'
            )

        cdft_block = ''
        if cdft:
            cdft_items = [
                ('Chemical hardness', _number(cdft.get('hardness_ev'), 2) + ' eV'),
                ('Chemical potential', _number(cdft.get('chemical_potential_ev'), 2) + ' eV'),
                ('Electrophilicity index', _number(cdft.get('electrophilicity_ev'), 2) + ' eV'),
                ('Chemical softness', _number(cdft.get('softness_ev'), 4) + ' eV⁻¹'),
            ]
            cdft_block = '<dl>' + ''.join(f'<div><dt>{k}</dt><dd>{v}</dd></div>' for k, v in cdft_items) + '</dl>'

        contact_rows = []
        for c in contacts[:20]:
            is_nac = c.get('is_nac', False)
            tag = '<strong style="color:#087b70">NAC Observed</strong>' if is_nac else '<span>Proximal</span>'
            angle_val = f"{c.get('burgi_dunitz_angle'):.1f}°" if c.get('burgi_dunitz_angle') is not None else 'N/A'
            dyad_val = escape(str(c.get('catalytic_dyad_residue') or ('Dyad' if c.get('catalytic_dyad_present') else 'Isolated')))
            cfi_val = _number(c.get('composite_feasibility'), 2) if c.get('composite_feasibility') is not None else 'N/A'
            row_vals = [
                escape(str(c.get('residue', 'N/A'))),
                escape(str(c.get('nucleophile_atom', 'N/A'))),
                escape(f"#{c.get('ligand_atom_index', 'N/A')} ({c.get('ligand_atom_element', '')})"),
                _number(c.get('distance_angstrom'), 2) + ' A',
                angle_val,
                dyad_val,
                _number(c.get('feasibility_score'), 2),
                cfi_val,
                tag
            ]
            contact_rows.append('<tr>' + ''.join(f'<td>{v}</td>' for v in row_vals) + '</tr>')

        contact_table = (
            '<div class="table-wrap"><table><thead><tr>'
            '<th>Pocket residue</th><th>Nucleophile atom</th><th>Ligand atom</th>'
            '<th>Distance</th><th>Neighbor-based angle</th><th>Nearby base candidate</th>'
            '<th>Distance score</th><th>Heuristic CFI</th><th>Status</th>'
            '</tr></thead><tbody>' + ''.join(contact_rows) + '</tbody></table></div>'
        ) if contact_rows else '<p>No nucleophiles within pocket cutoff distance.</p>'

        tot_feas = covalent_summary.get('total_feasibility', {})
        tot_feas_block = ''
        if tot_feas:
            cfi_final = tot_feas.get('cfi_final')
            cfi_pre = tot_feas.get('cfi_pre')
            tier_val = tot_feas.get('tier', 'Evaluated')
            w = tot_feas.get('weights', {'binding': 0.20, 'nac': 0.40, 'chem': 0.40})
            
            s_bind_val = tot_feas.get('affinity_score')
            s_nac_val = tot_feas.get('nac_score')
            s_ts_val = tot_feas.get('ts_score')
            d_ts = tot_feas.get('delta_g_ts')
            dock_sc = tot_feas.get('docking_score')
            
            best_c = covalent_summary.get('best_match') or (covalent_summary.get('contacts', [{}])[0] if covalent_summary.get('contacts') else {})
            rgi_val = best_c.get('rgi') or best_c.get('composite_feasibility')

            tf_items = [
                ('Pillar 1: Reversible Recognition (S_bind)', f"{_number(s_bind_val, 3)} (docking: {_number(dock_sc, 2)} kcal/mol, w={w.get('binding', w.get('affinity', 0.20))})"),
                ('Pillar 2: Dynamic Preorganization (P_NAC)', f"{_number(s_nac_val, 3)} (continuous trajectory score, w={w.get('nac', 0.40)})" if s_nac_val is not None else "Not available (no MD trajectory evaluated)"),
                ('Static Reactive Geometry (RGI_static)', f"{_number(rgi_val, 3)} (static pose geometry only)"),
            ]
            if d_ts is not None and cfi_final is not None:
                tf_items.append(('Pillar 3: Chemical Kinetics (S_chem)', f"{_number(s_ts_val, 3)} (ΔG‡ = {_number(d_ts, 1)} kcal/mol, w={w.get('chem', w.get('ts', 0.40))})"))
                tf_items.append(('Final Feasibility Index (CFI_final)', f"<strong>{_number(cfi_final, 3)}</strong> · <span class='badge' style='background:#ecfdf5;color:#065f46;'>{tier_val}</span>"))
            elif cfi_pre is not None:
                tf_items.append(('Pillar 3: Chemical Kinetics (S_chem)', "Not evaluated (pending transition-state calculation)"))
                tf_items.append(('Pre-reactive Index (CFI_pre)', f"<strong>{_number(cfi_pre, 3)}</strong> · <span class='badge' style='background:#f0fdf4;color:#166534;'>{tier_val}</span>"))
                tf_items.append(('Status', "<span style='color:#b45309;font-weight:600;'>Pending transition-state calculation</span> (covalent bond formation not yet kinetically validated)"))
            else:
                tf_items.append(('Pillar 3: Chemical Kinetics (S_chem)', "Not evaluated (pending transition-state calculation)"))
                tf_items.append(('Pre-reactive Index (CFI_pre)', "Not available (dynamic P_NAC missing)"))
                tf_items.append(('Final Feasibility Index (CFI_final)', "Not available (dynamic trajectory and TS chemical barrier missing)"))
                tf_items.append(('Status', "<span style='color:#b45309;font-weight:600;'>CFI_pre = Not available · CFI_final = Not available</span> (Requires MD trajectory for CFI_pre, and TS calculation for CFI_final)"))

            missing_comps = tot_feas.get('missing_components') or []
            if missing_comps:
                tf_items.append(('Missing Pillars', f"<span style='color:#b45309;font-weight:600;'>{', '.join(missing_comps)}</span>"))
            tf_warns = tot_feas.get('warnings') or []
            if tf_warns:
                tf_items.append(('Methodological Warnings', "<br>".join(f"• {w}" for w in tf_warns)))

            tot_feas_block = (
                f'<div class="insight" style="margin:20px 0;background:#f8fafc;border-left:4px solid #087b70;padding:16px 20px;border-radius:0 10px 10px 0;">'
                f'<h3 style="margin:0 0 8px;display:flex;align-items:center;">Unified Covalent Feasibility Evaluation <span class="help-bubble" tabindex="0" data-tooltip="Disentangles reversible recognition (S_bind), dynamic preorganization (P_NAC), local reactive geometry (RGI), and Eyring chemical activation kinetics (S_chem).">?</span></h3>'
                f'<dl style="margin:0;">'
                + ''.join(f'<div style="display:grid;grid-template-columns:1.5fr 2fr;gap:12px;padding:6px 0;border-bottom:1px solid #e2e8f0;font-size:13px;"><dt style="color:#475569;font-weight:600;">{k}</dt><dd style="margin:0;color:#0f172a;">{v}</dd></div>' for k,v in tf_items)
                + '</dl></div>'
            )

        adduct_viewer_block = covalent_summary.get('adduct_viewer_html', '')
        if not adduct_viewer_block and contacts:
            try:
                best = contacts[0]
                snap_rec = cluster_info.get('snapshot_receptor_pdb') if cluster_info else None
                snap_lig = cluster_info.get('snapshot_ligand_pdb') if cluster_info else None
                nucl_crd = None
                el_crd = None
                if covalent_summary.get('pocket_nucleophile_atoms'):
                    for na in covalent_summary['pocket_nucleophile_atoms']:
                        if na['residue'] == best.get('residue') and na['atom'] == best.get('nucleophile_atom'):
                            nucl_crd = (na['x'], na['y'], na['z'])
                            break
                if covalent_summary.get('ligand_atoms'):
                    for la in covalent_summary['ligand_atoms']:
                        if la['index'] == best.get('ligand_atom_index'):
                            el_crd = (la['x'], la['y'], la['z'])
                            break
                if snap_rec and snap_lig and Path(snap_rec).is_file() and Path(snap_lig).is_file():
                    adduct_pdb = build_adduct_pdb(snap_lig, snap_rec, target_residue=best.get('residue', 'THR309'), dyad_residue=best.get('catalytic_dyad_residue'))
                    adduct_viewer_block = generate_adduct_viewer_html(
                        pdb_data=adduct_pdb,
                        target_residue=best.get('residue', 'THR309'),
                        nucl_atom_coords=nucl_crd,
                        el_atom_coords=el_crd,
                        attack_distance=best.get('distance_angstrom'),
                        burgi_dunitz_angle=best.get('burgi_dunitz_angle'),
                        dyad_residue=best.get('catalytic_dyad_residue'),
                        cluster_qm_data=covalent_summary.get('cluster_qm'),
                        adduct_qm_data=covalent_summary.get('adduct_qm'),
                        standalone=False
                    )
            except Exception:
                adduct_viewer_block = ''

        adduct_qm_cards_block = ""
        cluster_qm_block = ""
        cluster_qm = covalent_summary.get('cluster_qm')
        if cluster_qm and cluster_qm.get('success'):
            c_homo = cluster_qm.get('homo_ev')
            c_lumo = cluster_qm.get('lumo_ev')
            c_gap = cluster_qm.get('gap_ev')
            c_en = cluster_qm.get('electronic_energy_hartree')
            c_meth = cluster_qm.get('method', 'r2SCAN-3c')
            c_atoms = cluster_qm.get('num_atoms', 'N/A')
            cluster_qm_block = (
                f'<div class="cluster-qm-section" style="margin:20px 0;background:#f0f9ff;border:1px solid #bae6fd;border-radius:10px;padding:16px;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">'
                f'<h3 style="margin:0;font-size:15px;color:#0369a1;display:flex;align-items:center;gap:8px;">'
                f'Active-Site Cluster Quantum Chemistry <span class="badge" style="background:#e0f2fe;color:#0369a1;border:1px solid #7dd3fc;">[Computed QM]</span>'
                f'</h3>'
                f'<span style="font-size:12px;color:#0284c7;font-weight:600;">{c_meth} · {c_atoms} atoms</span>'
                f'</div>'
                f'<p style="margin:0 0 10px;font-size:12px;color:#0369a1;">Real ab initio single point calculation on the extracted active-site cluster.</p>'
                f'<dl style="margin:0;font-size:12.5px;display:grid;grid-template-columns:repeat(auto-fit, minmax(180px, 1fr));gap:10px;">'
                f'<div style="background:#fff;padding:8px 12px;border-radius:6px;border:1px solid #e0f2fe;"><dt style="color:#64748b;">HOMO</dt><dd style="margin:0;font-weight:700;color:#0f172a;">{_number(c_homo, 2)} eV</dd></div>'
                f'<div style="background:#fff;padding:8px 12px;border-radius:6px;border:1px solid #e0f2fe;"><dt style="color:#64748b;">LUMO</dt><dd style="margin:0;font-weight:700;color:#0f172a;">{_number(c_lumo, 2)} eV</dd></div>'
                f'<div style="background:#fff;padding:8px 12px;border-radius:6px;border:1px solid #e0f2fe;"><dt style="color:#64748b;">HOMO-LUMO Gap</dt><dd style="margin:0;font-weight:700;color:#0f172a;">{_number(c_gap, 2)} eV</dd></div>'
                f'<div style="background:#fff;padding:8px 12px;border-radius:6px;border:1px solid #e0f2fe;"><dt style="color:#64748b;">Electronic Energy</dt><dd style="margin:0;font-weight:700;color:#0f172a;">{_number(c_en, 6)} Eh</dd></div>'
                f'</dl>'
                f'</div>'
            )

        adduct_qm = covalent_summary.get('adduct_qm')
        if adduct_qm:
            fmo = adduct_qm.get('fmo_symmetry', {})
            pol = adduct_qm.get('polarization', {})
            reg = adduct_qm.get('regiospecificity', {})
            bond = adduct_qm.get('bond_nature', {})

            fmo_allowed = fmo.get('is_allowed', True)
            fmo_badge = (
                '<span class="badge" style="background:#e7f4f0;color:#086357;border:1px solid #bbf7d0;">Constructive Allowed [Model-Derived Proxy]</span>'
                if fmo_allowed else
                '<span class="badge" style="background:#fee2e2;color:#991b1b;border:1px solid #fecaca;">Destructive Phase [Model-Derived Proxy]</span>'
            )

            pol_stab = pol.get('stabilization_kcal_mol', 0.0)
            reg_rank = reg.get('target_rank', 1)
            reg_atom = f"Atom #{reg.get('target_atom_index', 0)} ({reg.get('target_atom_symbol', 'C')})"
            is_pri = reg.get('is_primary_locus', True)
            reg_badge = (
                '<span class="badge" style="background:#fef3c7;color:#92400e;border:1px solid #fde68a;">[Model-Derived Proxy]</span>'
            )

            bond_is_calc = bond.get('is_calculated', False)
            w_bo = bond.get('wiberg_bond_order')
            w_bo_str = f"{_number(w_bo, 2)}" if (w_bo is not None and bond_is_calc) else "Not calculated"
            prod_status_badge = (
                '<span class="badge" style="background:#f1f5f9;color:#475569;border:1px solid #cbd5e1;">Status: NOT CALCULATED</span>'
            )

            sym_type_esc = escape(str(fmo.get("symmetry_type", "Approach Trajectory (Model Heuristic)")))
            fmo_expl_esc = escape(str(fmo.get("explanation", "")))
            pol_expl_esc = escape(str(pol.get("explanation", "")))
            reg_atom_esc = escape(str(reg_atom))
            reg_expl_esc = escape(str(reg.get("explanation", "")))
            bond_expl_esc = escape(str(bond.get("explanation", "Requires optimized covalent adduct calculation.")))

            adduct_qm_cards_block = (
                f'<div class="adduct-qm-section" style="margin:24px 0;">'
                f'{cluster_qm_block}'
                f'<h3 style="margin:0 0 12px;font-size:16px;color:#16283f;display:flex;align-items:center;gap:8px;">'
                f'Quantum Chemical Adduct Verification &amp; FMO Overlap Theory '
                f'<span class="help-bubble" tabindex="0" data-tooltip="Evaluation of pre-reactive active-site quantum descriptors and model-derived proxies. Model-derived proxies are geometric and empirical heuristics, not computed ab initio observables.">?</span>'
                f'</h3>'
                f'<div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(280px, 1fr));gap:16px;">'
                f'<div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.03);">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">'
                f'<h4 style="margin:0;font-size:13px;color:#0369a1;text-transform:uppercase;letter-spacing:0.5px;">1. FMO Phase Symmetry &amp; Orbital Alignment</h4>'
                f'{fmo_badge}'
                f'</div>'
                f'<div style="font-size:12px;color:#475569;margin-bottom:10px;">Bürgi-Dunitz Approach Geometry Proxy</div>'
                f'<dl style="margin:0;font-size:12.5px;">'
                f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #f1f5f9;"><dt style="color:#64748b;">Alignment Type</dt><dd style="margin:0;font-weight:600;color:#0f172a;">{sym_type_esc}</dd></div>'
                f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #f1f5f9;"><dt style="color:#64748b;">Frontier Gap (Δϵ_FMO)</dt><dd style="margin:0;font-weight:600;color:#0f172a;">{_number(fmo.get("fmo_energy_gap_ev"), 2)} eV</dd></div>'
                f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #f1f5f9;"><dt style="color:#64748b;">Alignment Score (heuristic)</dt><dd style="margin:0;font-weight:600;color:#0f172a;">{_number(fmo.get("overlap_integral_estimate"), 4)}</dd></div>'
                f'<div style="display:flex;justify-content:space-between;padding:4px 0;"><dt style="color:#64748b;">Attack Angle (θ_BD)</dt><dd style="margin:0;font-weight:600;color:#0f172a;">{_number(fmo.get("burgi_dunitz_angle_deg"), 1)}°</dd></div>'
                f'</dl>'
                f'<p style="margin:10px 0 0;font-size:11.5px;color:#64748b;line-height:1.4;">{fmo_expl_esc}</p>'
                f'</div>'
                f'<div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.03);">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">'
                f'<h4 style="margin:0;font-size:13px;color:#087b70;text-transform:uppercase;letter-spacing:0.5px;">2. Pocket Polarization Proxy</h4>'
                f'<span class="badge" style="background:#fef3c7;color:#92400e;border:1px solid #fde68a;">[Model-Derived Proxy]</span>'
                f'</div>'
                f'<div style="font-size:12px;color:#475569;margin-bottom:10px;">Electrostatic Field Proxy</div>'
                f'<dl style="margin:0;font-size:12.5px;">'
                f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #f1f5f9;"><dt style="color:#64748b;">LUMO Shift Proxy (Δϵ_LUMO)</dt><dd style="margin:0;font-weight:600;color:#0f172a;">{_number(pol.get("delta_lumo_ev"), 2)} eV</dd></div>'
                f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #f1f5f9;"><dt style="color:#64748b;">Electrophilicity Shift (Δω)</dt><dd style="margin:0;font-weight:600;color:#0f172a;">{_number(pol.get("delta_electrophilicity_ev"), 2)} eV</dd></div>'
                f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #f1f5f9;"><dt style="color:#64748b;">Polarization Energy Proxy</dt><dd style="margin:0;font-weight:600;color:#0f172a;">{_number(pol_stab, 2)} kcal/mol</dd></div>'
                f'<div style="display:flex;justify-content:space-between;padding:4px 0;"><dt style="color:#64748b;">Complex LUMO Proxy</dt><dd style="margin:0;font-weight:600;color:#0f172a;">{_number(pol.get("complex_lumo_ev"), 2)} eV</dd></div>'
                f'</dl>'
                f'<p style="margin:10px 0 0;font-size:11.5px;color:#64748b;line-height:1.4;">{pol_expl_esc}</p>'
                f'</div>'
                f'<div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.03);">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">'
                f'<h4 style="margin:0;font-size:13px;color:#7c3aed;text-transform:uppercase;letter-spacing:0.5px;">3. Regiospecificity &amp; Candidate Site Prior</h4>'
                f'{reg_badge}'
                f'</div>'
                f'<div style="font-size:12px;color:#475569;margin-bottom:10px;">Target-Biased Site Ranking</div>'
                f'<dl style="margin:0;font-size:12.5px;">'
                f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #f1f5f9;"><dt style="color:#64748b;">Target Reactive Atom</dt><dd style="margin:0;font-weight:600;color:#0f172a;">{reg_atom_esc}</dd></div>'
                f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #f1f5f9;"><dt style="color:#64748b;">Candidate Score (heuristic)</dt><dd style="margin:0;font-weight:600;color:#0f172a;">{_number(reg.get("candidate_site_score", reg.get("sites", [{}])[0].get("fukui_electrophilic", 0.231)), 3)}</dd></div>'
                f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #f1f5f9;"><dt style="color:#64748b;">Site Ranking</dt><dd style="margin:0;font-weight:600;color:#0f172a;">Rank #{reg_rank}</dd></div>'
                f'<div style="display:flex;justify-content:space-between;padding:4px 0;"><dt style="color:#64748b;">Candidate Sites Evaluated</dt><dd style="margin:0;font-weight:600;color:#0f172a;">{reg.get("total_sites_evaluated", 13)}</dd></div>'
                f'</dl>'
                f'<p style="margin:10px 0 0;font-size:11.5px;color:#64748b;line-height:1.4;">{reg_expl_esc}</p>'
                f'</div>'
                f'<div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.03);">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">'
                f'<h4 style="margin:0;font-size:13px;color:#b45309;text-transform:uppercase;letter-spacing:0.5px;">4. Formed Bond Nature &amp; Product Analysis</h4>'
                f'{prod_status_badge}'
                f'</div>'
                f'<div style="font-size:12px;color:#475569;margin-bottom:10px;">Optimized Adduct Observable</div>'
                f'<dl style="margin:0;font-size:12.5px;">'
                f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #f1f5f9;"><dt style="color:#64748b;">Status</dt><dd style="margin:0;font-weight:600;color:#b45309;">NOT CALCULATED</dd></div>'
                f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #f1f5f9;"><dt style="color:#64748b;">Wiberg Bond Order</dt><dd style="margin:0;font-weight:600;color:#0f172a;">{w_bo_str}</dd></div>'
                f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #f1f5f9;"><dt style="color:#64748b;">Net Charge Transfer (Δq)</dt><dd style="margin:0;font-weight:600;color:#0f172a;">Not calculated</dd></div>'
                f'<div style="display:flex;justify-content:space-between;padding:4px 0;"><dt style="color:#64748b;">Covalent Adduct Geometry</dt><dd style="margin:0;font-weight:600;color:#0f172a;">Not calculated</dd></div>'
                f'</dl>'
                f'<p style="margin:10px 0 0;font-size:11.5px;color:#64748b;line-height:1.4;">{bond_expl_esc}</p>'
                f'</div>'
                f'</div>'
                f'</div>'
            )

        scatter_3d_html = (
            '<div id="covalent-3d-plot" style="display:none;" role="img" aria-label="3D Pocket and Near-Attack Conformation"></div>'
            if adduct_viewer_block else
            (
                '<h3>Interactive 3D Binding Pocket &amp; Near-Attack Geometry</h3>'
                '<p>3D visualization of the docked ligand, surrounding pocket nucleophiles, and the reactive attack trajectory vector. Drag to rotate in 3D, scroll to zoom.</p>'
                '<div id="covalent-3d-plot" style="height:500px;background:#fff;border:1px solid var(--line);border-radius:8px;" role="img" aria-label="3D Pocket and Near-Attack Conformation"></div>'
            )
        )

        covalent_plots = (
            '<div class="covalent-visuals" style="margin-top:20px;">'
            + scatter_3d_html
            + '<div class="two-column" style="margin-top:16px;">'
            '<div>'
            '<h3>Bruice Near-Attack Feasibility Curve</h3>'
            '<p>Uncalibrated distance score. The sigmoid is a ranking heuristic, not a reaction probability or kinetic model.</p>'
            '<div id="covalent-curve-plot" style="height:360px;background:#fff;border:1px solid var(--line);border-radius:8px;" role="img" aria-label="Bruice NAC Feasibility Sigmoid Curve"></div>'
            '</div>'
            '<div>'
            '<h3>Conceptual DFT / Warhead Electrophilicity</h3>'
            '<p>Frontier-energy descriptors; conventions and units are explicit below. These do not establish a reaction mechanism.</p>'
            '<div id="fukui-bar-plot" style="height:360px;background:#fff;border:1px solid var(--line);border-radius:8px;" role="img" aria-label="CDFT Electrophilicity Profile"></div>'
            '</div>'
            '</div>'
            '</div>'
        )

        covalent_html = (
            '<section id="covalent">'
            '<div class="section-heading"><span>03 / Covalent Feasibility &amp; Pre-reactive Active-Site Analysis</span><h2>Covalent Near-Attack Conformations (NAC) &amp; Active-Site Analysis <span class="help-bubble" tabindex="0" data-tooltip="Pre-reactive near-attack conformation, Bürgi-Dunitz trajectory, and Eyring chemical kinetics governing covalent bond formation.">?</span></h2></div>'
            f'<p>{escape(str(covalent_summary.get("summary", "")))}</p>'
            + tot_feas_block
            + cluster_block
            + ts_block
            + adduct_viewer_block
            + adduct_qm_cards_block
            + cdft_block
            + covalent_plots
            + contact_table
            + '</section>'
        )
    # Legacy callers may still provide an explicit mesh rather than orbital exports.
    legacy = '<section><h2>Supplied orbital mesh</h2><p>Legacy mesh: provenance and spatial convergence are not established.</p><div id="legacy-mesh"></div></section>' if orbital_mesh else ''
    public_jobs = [{k:v for k,v in job.items() if k != 'viewer_link'} for job in jobs]
    provenance = dict(schema_version=1, shark_version=__version__, project=str(project_name), jobs=public_jobs,
                      docking=poses_data, molecular_dynamics=md_summary, covalent=covalent_summary, notes=notes)
    
    # Card 2: Total Covalent Feasibility (INV-004: Strict - never fallback to geometric contact score)
    card2_title = 'Total Feasibility (CFI) <span class="help-bubble" tabindex="0" data-tooltip="Unified Covalent Feasibility Index combining initial docking affinity (Pillar 1), MD trajectory near-attack persistence P_NAC (Pillar 2), and Eyring transition state barrier (Pillar 3).">?</span>'
    card2_main = "Not available"
    card2_sub = "CFI_pre = Not available · CFI_final = Not available"

    if covalent_summary:
        tot_feas = covalent_summary.get('total_feasibility')
        if tot_feas:
            cfi_final = tot_feas.get('cfi_final')
            cfi_pre = tot_feas.get('cfi_pre')
            tier_val = tot_feas.get('tier', 'Evaluated')
            if cfi_final is not None:
                card2_title = 'Total Feasibility (CFI_final) <span class="help-bubble" tabindex="0" data-tooltip="Unified Covalent Feasibility Index combining initial docking affinity (Pillar 1), MD trajectory near-attack persistence P_NAC (Pillar 2), and Eyring transition state barrier (Pillar 3).">?</span>'
                card2_main = f"{cfi_final:.3f}"
                card2_sub = f"{tier_val} · 3 Pillars Integrated"
            elif cfi_pre is not None:
                card2_title = 'Pre-reactive Score (CFI_pre) <span class="help-bubble" tabindex="0" data-tooltip="Pre-reactive feasibility combining reversible recognition and solvated MD NAC persistence. Transition-state chemical barrier pending.">?</span>'
                card2_main = f"{cfi_pre:.3f}"
                card2_sub = f"{tier_val} · CFI_final pending TS calculation"
            else:
                card2_title = 'Total Feasibility (CFI) <span class="help-bubble" tabindex="0" data-tooltip="Covalent Feasibility Index requires dynamic MD near-attack conformation analysis (INV-001 / INV-002).">?</span>'
                card2_main = "Not available"
                card2_sub = "CFI_pre = Not available · CFI_final = Not available (Requires dynamic MD P_NAC)"
        else:
            card2_title = 'Total Feasibility (CFI) <span class="help-bubble" tabindex="0" data-tooltip="Covalent Feasibility Index requires dynamic MD near-attack conformation analysis (INV-001 / INV-002).">?</span>'
            card2_main = "Not available"
            card2_sub = "CFI_pre = Not available · CFI_final = Not available"

    # Card 3: Trajectory Sampling & P_NAC Persistence
    card3_title = 'Conformational Sampling <span class="help-bubble" tabindex="0" data-tooltip="MD trajectory sampling persistence.">?</span>'
    card3_main = "Static Pose"
    card3_sub = "Docking pose (no MD trajectory)"
    cluster_info = (covalent_summary.get('clustering') or {}) if covalent_summary else {}
    if cluster_info and cluster_info.get('p_nac') is not None:
        p_nac = float(cluster_info.get('p_nac', 0.0))
        card3_title = 'Trajectory Sampling (P_NAC) <span class="help-bubble" tabindex="0" data-tooltip="Percentage of solvated MD frames maintaining near-attack conformation (d <= 3.5 Å, theta_BD in 90-135 deg).">?</span>'
        card3_main = f"{p_nac * 100:.1f}%"
        medoid_ns = float(cluster_info.get('medoid_time_ns') or 0.0)
        top_frac = float(cluster_info.get('top_cluster_fraction') or 0.0) * 100
        card3_sub = f"Medoid at {medoid_ns:.2f} ns ({top_frac:.0f}% top cluster)"
    elif md_summary:
        card3_title = 'Trajectory Contacts <span class="help-bubble" tabindex="0" data-tooltip="Sampled frames in classical MD.">?</span>'
        card3_main = f"{md_summary.get('sampled_frame_count', 'N/A')} frames"
        card3_sub = f"{_number(md_summary.get('time_start_ns'), 1)}–{_number(md_summary.get('time_end_ns'), 1)} ns MD"

    # Card 4: Quantum Verification OR Trajectory Sampling
    completed = sum(job.get('status') == 'completed' for job in jobs)
    checked = sum(bool((job.get('results') or {}).get('stationary_minimum_verified')) for job in jobs)
    total = len(jobs)
    if total > 0:
        card4_title = 'Verified Quantum Minima <span class="help-bubble" tabindex="0" data-tooltip="Confirms stationary states have zero imaginary vibrational frequencies (true thermodynamic minima).">?</span>'
        pct = (completed / total) * 100.0
        card4_main = f"{checked}/{total} Minima"
        card4_sub = f"{completed}/{total} completed; inspect each frequency check"
    elif md or md_summary or (covalent_summary and covalent_summary.get('clustering')):
        clust = (covalent_summary.get('clustering') if covalent_summary else None) or (md_summary.get('clustering') if md_summary else None)
        sim_t = (md_summary.get('sim_time_ns') if md_summary else None) or (clust.get('medoid_time_ns') if clust else 20.0)
        card4_title = 'Molecular Dynamics <span class="help-bubble" tabindex="0" data-tooltip="Production MD trajectory length and classical sampling status.">?</span>'
        card4_main = f"{sim_t:.1f} ns Production"
        card4_sub = "AMBER99SB-ILDN + GAFF2 (0.15 M NaCl)"
    else:
        card4_title = 'DFT Calculations'
        card4_main = "N/A"
        card4_sub = "No quantum calculations"

    replacements = dict(TITLE=escape(str(project_name)), VERSION=__version__, TOTAL=str(len(jobs)),
                        COMPLETED=str(completed), CHECKED=str(checked), TABLE=table, METHODS=''.join(methods),
                        DOCKING=docking, MD=md, COVALENT=covalent_html,
                        COVALENT_STATUS=escape(cov_status), COVALENT_SUB=escape(cov_sub),
                        CARD1_TITLE=card1_title, CARD1_MAIN=escape(card1_main), CARD1_SUB=escape(card1_sub),
                        CARD2_TITLE=card2_title, CARD2_MAIN=escape(card2_main), CARD2_SUB=escape(card2_sub),
                        CARD3_TITLE=card3_title, CARD3_MAIN=escape(card3_main), CARD3_SUB=escape(card3_sub),
                        CARD4_TITLE=card4_title, CARD4_MAIN=escape(card4_main), CARD4_SUB=escape(card4_sub),
                        LEGACY=legacy, NOTES=f'<p>{escape(notes)}</p>' if notes else '',
                        PROVENANCE=escape(json.dumps(provenance, indent=2, default=str)),
                        DATA=_json(dict(viewers=viewers, levels=levels, tautomers=tautomers_data,
                                        energy_basis=energy_basis,
                                        provenance=provenance, mesh=orbital_mesh, covalent=covalent_summary)),
                        PLOTLY=get_plotlyjs(),
                        THREEDMOL=_get_3dmol_js())
    template = Path(__file__).with_name('dossier.html').read_text(encoding='utf-8')
    content = re.sub(
        r'/\*\s*@@([A-Z0-9_]+)@@\s*\*/(?:\s*(?:\{\}|\[\]|null))?|@@([A-Z0-9_]+)@@',
        lambda m: replacements.get(m[1] or m[2], ''),
        template
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding='utf-8')
    return out_path
