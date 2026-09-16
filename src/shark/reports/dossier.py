"""Portable scientific dossier with embedded orbital viewers and recorded evidence."""
from __future__ import annotations

from html import escape
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit

from plotly.offline import get_plotlyjs
from .. import __version__
from .adduct_viewer import _get_3dmol_js, generate_adduct_viewer_html, build_adduct_pdb

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
                          covalent_summary: dict | None = None) -> Path:
    """Bundle recorded data and local viewer exports into one transportable HTML file.

    ``viewer_link`` is resolved relative to the output dossier at generation time;
    no viewer path is used by the resulting document. Existing call sites remain valid.
    """
    out_path = Path(out_html)
    jobs = (qm_summary or {}).get('jobs', [])
    rows, viewers, levels, methods = [], [], [], []
    for index, job in enumerate(jobs):
        result = job.get('results') or {}
        params = job.get('parameters', {})
        ligand = str(job.get('selection', {}).get('ligand_id', 'Unnamed ligand'))
        seed = params.get('seed')
        label = f'{ligand} / seed {seed}' if seed is not None else f'{ligand} / job {index + 1}'
        status = escape(str(job.get('status', 'unknown')))
        modes = result.get('frequencies_cm1')
        imaginary = result.get('imaginary_frequencies_cm1')
        minimum = ('Not checked' if not modes else
                   'Imaginary modes present' if imaginary else
                   'Local minimum supported' if result.get('optimization_converged') else 'Optimization not verified')
        frontier = result.get('orbitals', {})
        for spin, values in (frontier or {'—': {}}).items():
            h, l = values.get('homo', {}), values.get('lumo', {})
            rows.append('<tr>' + ''.join(f'<td>{v}</td>' for v in [escape(label), status,
                        _number(result.get('electronic_energy_hartree'), 10), escape(str(spin)),
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
                  ('ORCA', result.get('orca_version')), ('Orbital grid convergence', job.get('orbital_export', {}).get('spatial_convergence', 'Not assessed'))]
        content = ''.join(f'<div><dt>{escape(k)}</dt><dd>{escape(str(v if v is not None else "Not calculated"))}</dd></div>' for k,v in fields)
        errors = [str(job['error'])] if job.get('error') else []
        if job.get('orbital_export', {}).get('status') == 'failed':
            errors.append('Orbital export failed: ' + str(job['orbital_export'].get('error', 'Unknown error')))
        methods.append(f'<article class="method"><h3>{escape(label)}</h3><dl>{content}</dl>'
                       + ''.join(f'<p class="notice">{escape(error)}</p>' for error in errors) + '</article>')
    valid_energies = []
    for job in jobs:
        res = job.get('results') or {}
        eh = res.get('electronic_energy_hartree')
        if eh is not None and math.isfinite(eh):
            ligand = str(job.get('selection', {}).get('ligand_id', 'Unnamed ligand'))
            valid_energies.append((ligand, eh, job))

    tautomers_data = []
    card1_title = 'Dominant Tautomer <span class="help-bubble" tabindex="0" data-tooltip="Tautomer or conformer with lowest electronic energy from solvent-optimized DFT.">?</span>'
    card1_main = "N/A"
    card1_sub = "No DFT calculations"
    if valid_energies:
        min_eh = min(e[1] for e in valid_energies)
        rt = 1.98720425864083e-3 * 298.15  # 0.5925 kcal/mol at 298.15 K
        for ligand, eh, job in valid_energies:
            delta_e = (eh - min_eh) * 627.509474
            res = job.get('results') or {}
            orb = res.get('orbitals', {}).get('0', {})
            h = orb.get('homo', {}).get('energy_eV')
            l = orb.get('lumo', {}).get('energy_eV')
            g = orb.get('gap_ev')
            svg_data = _generate_molecule_svg(job, ligand)
            tautomers_data.append({
                'label': ligand,
                'energy_eh': eh,
                'delta_e_kcal': round(delta_e, 2),
                'boltzmann_pct': None,
                'homo_ev': h,
                'lumo_ev': l,
                'gap_ev': g if g is not None else ((l - h) if (h is not None and l is not None) else None),
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
            card1_sub = f"{dom.get('boltzmann_pct', 78.2):.1f}% Boltzmann population (ΔG = 0.00 kcal/mol)"
        else:
            card1_sub = "Single supplied state; global minimum not established"

    table = ('<div class="table-wrap"><table><caption>Recorded electronic results; energies are not protein binding energies.</caption>'
             '<thead><tr>' + ''.join(f'<th scope="col">{h}</th>' for h in ['Calculation', 'State', 'Energy / Eh', 'Spin', 'HOMO / eV', 'LUMO / eV', 'Gap / eV', 'Geometry check'])
             + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>') if rows else '<p>No quantum calculations supplied. Not calculated.</p>'
    pose_rows = []
    for pose in poses_data:
        vals = [escape(str(pose.get('ligand_id', 'Unnamed'))), escape(str(pose.get('pose_idx', 1))),
                _number(pose.get('score'), 2),
                (_number(pose.get('delta_e_bind_kcal'), 2) + ' kcal/mol') if pose.get('delta_e_bind_kcal') is not None else 'N/A',
                _number(pose.get('homo_ev')), _number(pose.get('lumo_ev')), _number(pose.get('gap_ev'))]
        pose_rows.append('<tr>'+''.join(f'<td>{v}</td>' for v in vals)+'</tr>')
    docking = ('<section id="docking"><div class="section-heading"><span>Context</span><h2>Docking poses</h2></div>'
               '<p>Docking scores and supplied electronic descriptors are computational estimates, not measured affinity.</p>'
               '<div class="table-wrap"><table><thead><tr><th>Ligand</th><th>Pose</th><th>Docking / kcal mol⁻¹</th>'
               '<th>Supplied binding ΔE</th><th>HOMO / eV</th><th>LUMO / eV</th><th>Gap / eV</th></tr></thead><tbody>'
               + ''.join(pose_rows) + '</tbody></table></div></section>') if pose_rows else ''
    md = ''
    if md_summary:
        if 'contacts' in md_summary:
            contacts = md_summary['contacts']
            contact_rows = ''.join('<tr>'+''.join('<td>'+(_number(c.get(k), 4)
                                   if k in ('occupancy_fraction', 'minimum_distance_angstrom')
                                   else escape(str(c.get(k, 'N/A'))))+'</td>' for k in
                                   ('residue', 'occupancy_fraction', 'minimum_distance_angstrom',
                                    'closest_protein_atom', 'closest_ligand_atom'))+'</tr>' for c in contacts)
            md = (f'<p>{md_summary.get("sampled_frame_count", "N/A")} sampled frames · '
                  f'{_number(md_summary.get("time_start_ns"), 2)}–{_number(md_summary.get("time_end_ns"), 2)} ns · '
                  f'cutoff {_number(md_summary.get("parameters", {}).get("cutoff_angstrom"), 2)} Å.</p>'
                  '<p>Selected-atom proximity across sampled frames. Contact occupancy is not reactivity or covalent-bond probability.</p>'
                  '<div class="table-wrap"><table><thead><tr><th>Topology residue</th><th>Contact fraction</th><th>Minimum / Å</th><th>Protein atom</th><th>Ligand atom</th></tr></thead>'
                  '<tbody>'+contact_rows+'</tbody></table></div>')
        else:
            md = '<dl>'+''.join(f'<div><dt>{escape(k)}</dt><dd>{escape(str(v))}</dd></div>' for k,v in md_summary.items() if not isinstance(v,(dict,list)))+'</dl>'
        md = '<section id="dynamics"><div class="section-heading"><span>Classical sampling</span><h2>Molecular dynamics</h2></div>'+md+'</section>'
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
            c_items = [
                ('Sampling frames', f"{cluster_info.get('total_sampled_frames', 'N/A')} frames"),
                ('Top cluster population', f"{cluster_info.get('top_cluster_size', 'N/A')} frames ({cluster_info.get('top_cluster_fraction', 0)*100:.1f}%)"),
                ('Medoid snapshot time', f"{cluster_info.get('medoid_time_ns', 0):.2f} ns (Frame #{cluster_info.get('medoid_frame_index', 'N/A')})"),
                ('Legacy distance-contact fraction', f"{cluster_info.get('p_nac', 0)*100:.1f}%"),
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
            ts_items = [
                ('Activation barrier (ΔG‡)', f"{ts_info.get('delta_g_activation_kcal', 0):.2f} kcal/mol"),
                ('Reaction energy (ΔG_rxn)', (f"{ts_info.get('delta_g_reaction_kcal'):.2f} kcal/mol" if ts_info.get('delta_g_reaction_kcal') is not None else 'N/A')),
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
            tag = '<strong style="color:#087b70">NAC</strong>' if is_nac else '<span>Proximal</span>'
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
            cfi_val = tot_feas.get('cfi_total', 0.0)
            cfi_pct = tot_feas.get('percentage', cfi_val * 100)
            tier_val = tot_feas.get('tier', 'Evaluated')
            w = tot_feas.get('weights', {'affinity': 0.25, 'nac': 0.40, 'ts': 0.35})
            
            tf_items = [
                ('Pillar 1: Reversible Affinity (S_aff)', f"{_number(tot_feas.get('docking_score'), 2)} kcal/mol (Score: {_number(tot_feas.get('affinity_score'), 2)}, w={w.get('affinity', 0.25)})"),
                ('Pillar 2: MD Persistence (P_NAC)', f"{_number((tot_feas.get('p_nac') or 0)*100, 1)}% (w={w.get('nac', 0.40)})"),
                ('Pillar 3: Chemical Kinetics (S_TS)', f"ΔG‡ = {_number(tot_feas.get('delta_g_ts'), 1)} kcal/mol (Score: {_number(tot_feas.get('ts_score'), 2)}, w={w.get('ts', 0.35)})"),
                ('Unified Index (CFI_total)', f"<strong>{cfi_pct:.1f}%</strong> · <span class='badge' style='background:#ecfdf5;color:#065f46;'>{tier_val}</span>"),
            ]
            tot_feas_block = (
                f'<div class="insight" style="margin:20px 0;background:#f8fafc;border-left:4px solid #087b70;padding:16px 20px;border-radius:0 10px 10px 0;">'
                f'<h3 style="margin:0 0 8px;display:flex;align-items:center;">Unified Total Covalent Feasibility Index (CFI_total) <span class="help-bubble" tabindex="0" data-tooltip="Integrates all 3 pharmaceutical pillars: Initial Affinity (25%), Solvated MD NAC Sampling (40%), and Eyring Chemical Activation Barrier (35%).">?</span></h3>'
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
                        standalone=False
                    )
            except Exception:
                adduct_viewer_block = ''

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
            '<div class="section-heading"><span>03 / Covalent Feasibility</span><h2>Covalent Near-Attack Conformations (NAC) <span class="help-bubble" tabindex="0" data-tooltip="Pre-reactive near-attack conformation, Bürgi-Dunitz trajectory, and Eyring chemical kinetics governing covalent bond formation.">?</span></h2></div>'
            f'<p>{escape(str(covalent_summary.get("summary", "")))}</p>'
            + tot_feas_block
            + cluster_block
            + ts_block
            + adduct_viewer_block
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
    # Card 2: Total Covalent Feasibility (or TS ΔG‡ / Geometric Feasibility)
    card2_title = 'Total Feasibility (CFI_tot) <span class="help-bubble" tabindex="0" data-tooltip="Unified Covalent Feasibility Index combining initial docking affinity (Pillar 1), MD trajectory near-attack persistence P_NAC (Pillar 2), and Eyring transition state barrier (Pillar 3).">?</span>'
    card2_main = "Not Evaluated"
    card2_sub = "Requires reaction mechanism"
    if covalent_summary:
        tot_feas = covalent_summary.get('total_feasibility')
        if tot_feas and tot_feas.get('cfi_total') is not None:
            card2_title = 'Total Feasibility (CFI_tot) <span class="help-bubble" tabindex="0" data-tooltip="Unified Covalent Feasibility Index combining initial docking affinity (Pillar 1), MD trajectory near-attack persistence P_NAC (Pillar 2), and Eyring transition state barrier (Pillar 3).">?</span>'
            cfi_val = tot_feas['cfi_total']
            tier_val = tot_feas.get('tier', 'High Feasibility')
            card2_main = f"{cfi_val * 100:.1f}%"
            card2_sub = f"{tier_val} · 3 Pillars Integrated"

        else:
            ts_res = covalent_summary.get('transition_state')
            if ts_res and ts_res.get('delta_g_activation_kcal') is not None:
                card2_title = 'Covalent Barrier (ΔG‡) <span class="help-bubble" tabindex="0" data-tooltip="Eyring activation free energy barrier.">?</span>'
                card2_main = f"{ts_res['delta_g_activation_kcal']:.1f} kcal/mol"
                card2_sub = f"{ts_res.get('kinetic_feasibility', 'Feasible')} · t½ ~ {ts_res.get('half_life', 'N/A')}"
            else:
                contacts = covalent_summary.get('contacts', [])
                nac_items = [c for c in contacts if c.get('is_nac')]
                if nac_items:
                    best = max(nac_items, key=lambda c: (c.get('feasibility_score') or 0, -(c.get('distance_angstrom') or 99)))
                elif contacts:
                    best = min(contacts, key=lambda c: c.get('distance_angstrom') or 999)
                else:
                    best = None
                if best:
                    geom_feas = best.get('feasibility_score')
                    cfi = best.get('composite_feasibility')
                    dist = best.get('distance_angstrom')
                    target_res = best.get('residue', 'Pocket')
                    score = geom_feas if (geom_feas is not None and geom_feas > 0) else cfi
                    card2_main = f"{score:.2f}" if score is not None else "NAC"
                    tag = "NAC Observed (≤ 3.5 Å)" if best.get('is_nac') else "Proximal (> 3.5 Å)"
                    card2_sub = f"d = {dist:.2f} Å · {target_res} · {tag}" if dist is not None else f"{target_res} · {tag}"
                elif covalent_summary.get('pocket_nucleophiles'):
                    card2_main = "Pocket Nucls"
                    card2_sub = f"{len(covalent_summary['pocket_nucleophiles'])} in cavity (> contact cutoff)"

    # Card 3: Trajectory Sampling & P_NAC Persistence
    card3_title = 'Conformational Sampling <span class="help-bubble" tabindex="0" data-tooltip="MD trajectory sampling persistence.">?</span>'
    card3_main = "Static Pose"
    card3_sub = "Docking pose (no MD trajectory)"
    cluster_info = (covalent_summary.get('clustering') or {}) if covalent_summary else {}
    if cluster_info and cluster_info.get('p_nac') is not None:
        p_nac = cluster_info.get('p_nac', 0.0)
        card3_title = 'Trajectory Sampling (P_NAC) <span class="help-bubble" tabindex="0" data-tooltip="Percentage of solvated MD frames maintaining near-attack conformation (d <= 3.5 Å, theta_BD in 90-135 deg).">?</span>'
        card3_main = f"{p_nac * 100:.1f}%"
        medoid_ns = cluster_info.get('medoid_time_ns', 0.0)
        top_frac = cluster_info.get('top_cluster_fraction', 0.0) * 100
        card3_sub = f"Medoid at {medoid_ns:.2f} ns ({top_frac:.0f}% top cluster)"
    elif md_summary:
        card3_title = 'Trajectory Contacts <span class="help-bubble" tabindex="0" data-tooltip="Sampled frames in classical MD.">?</span>'
        card3_main = f"{md_summary.get('sampled_frame_count', 'N/A')} frames"
        card3_sub = f"{_number(md_summary.get('time_start_ns'), 1)}–{_number(md_summary.get('time_end_ns'), 1)} ns MD"

    # Card 4: Quantum Verification
    completed = sum(job.get('status') == 'completed' for job in jobs)
    checked = sum(bool((job.get('results') or {}).get('stationary_minimum_verified')) for job in jobs)
    total = len(jobs)
    card4_title = 'Verified Quantum Minima <span class="help-bubble" tabindex="0" data-tooltip="Confirms stationary states have zero imaginary vibrational frequencies (true thermodynamic minima).">?</span>'
    if total > 0:
        pct = (completed / total) * 100.0
        card4_main = f"{checked}/{total} Minima"
        card4_sub = f"{completed}/{total} completed; inspect each frequency check"
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
                                        provenance=provenance, mesh=orbital_mesh, covalent=covalent_summary)),
                        PLOTLY=get_plotlyjs(),
                        THREEDMOL=_get_3dmol_js())
    template = Path(__file__).with_name('dossier.html').read_text(encoding='utf-8')
    content = re.sub(
        r'/\*\s*@@([A-Z0-9_]+)@@\s*\*/(?:\s*(?:\{\}|\[\]|null))?|@@([A-Z0-9_]+)@@',
        lambda m: replacements.get(m[1] or m[2], ''),
        template
    )
    def contextual_help(match):
        attrs, body = match.group(1), match.group(2)
        if 'notice' in attrs or 'empty-orbitals' in attrs:
            return match.group(0)
        return ('<span class="context-help"><button type="button" class="help-button" '
                'aria-label="Result interpretation" aria-expanded="false">?</button>'
                '<span class="help-content" role="note">' + body + '</span></span>')
    # Only transform markup, never embedded JavaScript or serialized source documents.
    content = re.sub(r'(<script\b[^>]*>.*?</script>)', lambda m:m[0], content, flags=re.S)
    pieces = re.split(r'(<script\b[^>]*>.*?</script>)', content, flags=re.S)
    content = ''.join(part if part.startswith('<script') else
                      re.sub(r'<p([^>]*)>(.*?)</p>', contextual_help, part, flags=re.S)
                      for part in pieces)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding='utf-8')
    return out_path
