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
    responsive = """<style>html,body{max-width:100%;overflow-x:hidden}header,main{padding:16px}h1{font-size:22px}#plot{width:100%}</style>
<script>
(()=>{let original=null,timer;async function fit(){
 const plot=document.getElementById('plot');if(!window.sharkReady||!plot){timer=setTimeout(fit,100);return;}
 const names=Object.keys(plot.layout).filter(k=>/^scene[0-9]*$/.test(k));
 if(!original)original={annotations:structuredClone(plot.layout.annotations),domains:names.map(n=>structuredClone(plot.layout[n].domain)),height:plot.layout.height};
 const narrow=innerWidth<650,annotations=structuredClone(original.annotations),update={width:Math.max(240,plot.clientWidth),height:narrow?names.length*370+140:original.height,'title.text':'',margin:{l:12,r:12,t:75,b:90}};
 names.forEach((name,i)=>{const domain=narrow?{x:[0,1],y:[(names.length-i-1)/names.length+.03,(names.length-i)/names.length-.06]}:original.domains[i];update[name+'.domain']=domain;if(narrow&&annotations[i]){annotations[i].x=.5;annotations[i].y=domain.y[1]+.015;}});
 if(narrow)annotations.forEach((a,i)=>{a.font={...a.font,size:10};if(i>=names.length)a.text=a.text.replaceAll(' | ','<br>');});
 update.annotations=annotations;await Plotly.relayout(plot,update);window.sharkEmbeddedReady=true;
 parent.postMessage({type:'shark-viewer-size',height:Math.ceil(document.body.getBoundingClientRect().height)},'*');
 }addEventListener('resize',()=>{clearTimeout(timer);timer=setTimeout(fit,150);});fit();})();
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
    table = ('<div class="table-wrap"><table><caption>Recorded electronic results; energies are not protein binding energies.</caption>'
             '<thead><tr>' + ''.join(f'<th scope="col">{h}</th>' for h in ['Calculation', 'State', 'Energy / Eh', 'Spin', 'HOMO / eV', 'LUMO / eV', 'Gap / eV', 'Geometry check'])
             + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>') if rows else '<p>No quantum calculations supplied. Not calculated.</p>'
    pose_rows = []
    for pose in poses_data:
        vals = [escape(str(pose.get('ligand_id', 'Unnamed'))), escape(str(pose.get('pose_idx', 1))),
                _number(pose.get('score'), 2), _number(pose.get('delta_e_bind_kcal'), 2) + ' kcal/mol',
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
            row_vals = [
                escape(str(c.get('residue', 'N/A'))),
                escape(str(c.get('nucleophile_atom', 'N/A'))),
                escape(f"#{c.get('ligand_atom_index', 'N/A')} ({c.get('ligand_atom_element', '')})"),
                _number(c.get('distance_angstrom'), 2) + ' A',
                _number(c.get('feasibility_score'), 2),
                tag
            ]
            contact_rows.append('<tr>' + ''.join(f'<td>{v}</td>' for v in row_vals) + '</tr>')

        contact_table = (
            '<div class="table-wrap"><table><thead><tr>'
            '<th>Pocket residue</th><th>Nucleophile atom</th><th>Ligand atom</th>'
            '<th>Distance</th><th>Feasibility</th><th>Status</th>'
            '</tr></thead><tbody>' + ''.join(contact_rows) + '</tbody></table></div>'
        ) if contact_rows else '<p>No nucleophiles within pocket cutoff distance.</p>'

        covalent_html = (
            '<section id="covalent">'
            '<div class="section-heading"><span>03 / Warhead & Reactivity</span><h2>Covalent Near-Attack Conformations (NAC)</h2></div>'
            f'<p>{escape(str(covalent_summary.get("summary", "")))}</p>'
            + cdft_block
            + contact_table
            + '</section>'
        )
    # Legacy callers may still provide an explicit mesh rather than orbital exports.
    legacy = '<section><h2>Supplied orbital mesh</h2><p>Legacy mesh: provenance and spatial convergence are not established.</p><div id="legacy-mesh"></div></section>' if orbital_mesh else ''
    public_jobs = [{k:v for k,v in job.items() if k != 'viewer_link'} for job in jobs]
    provenance = dict(schema_version=1, shark_version=__version__, project=str(project_name), jobs=public_jobs,
                      docking=poses_data, molecular_dynamics=md_summary, covalent=covalent_summary, notes=notes)
    completed = sum(job.get('status') == 'completed' for job in jobs)
    checked = sum(bool((job.get('results') or {}).get('stationary_minimum_verified')) for job in jobs)
    replacements = dict(TITLE=escape(str(project_name)), VERSION=__version__, TOTAL=str(len(jobs)),
                        COMPLETED=str(completed), CHECKED=str(checked), TABLE=table, METHODS=''.join(methods),
                        DOCKING=docking, MD=md, COVALENT=covalent_html,
                        COVALENT_STATUS=escape(cov_status), COVALENT_SUB=escape(cov_sub),
                        LEGACY=legacy, NOTES=f'<p>{escape(notes)}</p>' if notes else '',
                        PROVENANCE=escape(json.dumps(provenance, indent=2, default=str)),
                        DATA=_json(dict(viewers=viewers, levels=levels, provenance=provenance, mesh=orbital_mesh)),
                        PLOTLY=get_plotlyjs())
    template = Path(__file__).with_name('dossier.html').read_text(encoding='utf-8')
    content = re.sub(r'@@([A-Z_]+)@@', lambda m: replacements.get(m[1], ''), template)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding='utf-8')
    return out_path
