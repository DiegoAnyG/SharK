"""Shared, reproducible isosurface geometry for offline HTML and static figures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import argparse
import copy
import html
import importlib.metadata
import json
import math

import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdFMCS
from rdkit.Chem.Draw.MolDrawing import DrawingOptions
from skimage.measure import marching_cubes, mesh_surface_area
from scipy.spatial import cKDTree, ConvexHull

from ..core.cube import BOHR_TO_ANGSTROM, parse_cube_file
from ..core.frontier import OrbitalField, discover_frontier_fields, file_hash


@dataclass
class ViewSettings:
    """Visualization defaults, all overridable through JSON or the Python API."""
    isovalues: tuple[float, ...] = (0.02, 0.03, 0.04)
    isovalue: float = 0.03
    positive_color: str = '#2878B5'
    negative_color: str = '#D64B40'
    opacity: float = 1.0
    alignment: str = 'reference'
    camera: dict = field(default_factory=lambda: dict(
        eye=dict(x=0.4, y=-0.7, z=0.95), up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0), projection=dict(type='orthographic')))
    width: int = 1400
    panel_height: int = 480
    columns: int = 2
    export_scale: float = 2.5
    bond_factor: float = 1.25
    atom_radius_scale: float = 0.25
    atom_resolution: int = 12
    atom_colors: dict[str, str] = field(default_factory=dict)
    padding_angstrom: float = 0.8
    scene_scale: float = 2.0
    title: str = 'Frontier molecular orbitals'

    def validate(self):
        self.isovalues = tuple(sorted(set(float(v) for v in (*self.isovalues, self.isovalue))))
        if not self.isovalues or not all(np.isfinite(v) and v > 0 for v in self.isovalues):
            raise ValueError('Isovalues must be finite positive amplitudes')
        if not np.isfinite(self.opacity) or not 0 < self.opacity <= 1:
            raise ValueError('Opacity must be between zero and one')
        if self.alignment not in ('reference', 'principal', 'none'):
            raise ValueError('Alignment must be reference, principal, or none')
        for name in ('width', 'panel_height', 'columns', 'atom_resolution'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.atom_resolution < 4:
            raise ValueError('atom_resolution must be at least four')
        for name in ('export_scale', 'bond_factor', 'atom_radius_scale', 'padding_angstrom', 'scene_scale'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        # Plotly validates the camera schema and CSS colors before generation.
        go.layout.scene.Camera(**self.camera)
        go.Mesh3d(color=self.positive_color)
        go.Mesh3d(color=self.negative_color)


def principal_transform(positions: np.ndarray, atomic_numbers: list[int]) -> dict:
    selected = positions[np.array(atomic_numbers) > 1]
    if len(selected) < 2:
        selected = positions
    center = selected.mean(axis=0)
    if len(selected) < 2:
        rotation = np.eye(3)
    else:
        _, _, vt = np.linalg.svd(selected - center, full_matrices=True)
        rotation = vt.T
        projected = (selected - center) @ rotation
        for axis in range(2):
            if projected[np.argmax(abs(projected[:, axis])), axis] < 0:
                rotation[:, axis] *= -1
        rotation[:, 2] = np.cross(rotation[:, 0], rotation[:, 1])
    return dict(center=center, rotation=rotation, translation=np.zeros(3), kind='principal')


def transform_points(points: np.ndarray, transform: dict) -> np.ndarray:
    return (points - transform['center']) @ transform['rotation'] + transform['translation']


def molecule_from_cube(cube: dict, bond_factor: float):
    text = f'{len(cube["atoms"])}\nCUBE geometry\n' + '\n'.join(
        a['element'] + ' ' + ' '.join(str(v) for v in a['pos_ang']) for a in cube['atoms'])
    mol = Chem.MolFromXYZBlock(text)
    if mol is None:
        raise ValueError('Cannot construct a molecular geometry from this cube')
    rdDetermineBonds.DetermineConnectivity(mol, useVdw=True, covFactor=bond_factor)
    Chem.GetSymmSSSR(mol)
    return mol


def atom_spheres(atoms: list[dict], positions: np.ndarray, settings: ViewSettings):
    """Use physical 3D sphere geometry so atom size and occlusion survive export."""
    phi = np.linspace(0, np.pi, settings.atom_resolution+1)[1:-1]
    theta = np.linspace(0, 2*np.pi, settings.atom_resolution*2, endpoint=False)
    phi, theta = np.meshgrid(phi, theta, indexing='ij')
    unit = np.column_stack([(np.sin(phi)*np.cos(theta)).ravel(),
                            (np.sin(phi)*np.sin(theta)).ravel(), np.cos(phi).ravel()])
    unit = np.vstack([unit, [0, 0, 1], [0, 0, -1]])
    triangles = ConvexHull(unit).simplices.copy()
    # Orient every face outward for consistent lighting.
    tri = unit[triangles]
    inward = np.einsum('ij,ij->i', np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]), tri.mean(1)) < 0
    triangles[inward] = triangles[inward][:, [0, 2, 1]]
    vertices, faces, colors, labels = [], [], [], []
    palette = DrawingOptions.elemDict
    for index, (atom, position) in enumerate(zip(atoms, positions)):
        vertices.append(position + unit*atom['radius']*settings.atom_radius_scale)
        faces.append(triangles + index*len(unit))
        rgb = palette.get(atom['atno'], palette[0])
        default = 'rgb(' + ','.join(str(round(c*255)) for c in rgb) + ')'
        colors.extend([settings.atom_colors.get(atom['element'], default)]*len(unit))
        labels.extend([f'{atom["element"]} {index}']*len(unit))
    vertices, faces = np.concatenate(vertices), np.concatenate(faces)
    return go.Mesh3d(x=vertices[:,0].tolist(), y=vertices[:,1].tolist(), z=vertices[:,2].tolist(),
                     i=faces[:,0].tolist(), j=faces[:,1].tolist(), k=faces[:,2].tolist(),
                     vertexcolor=colors, text=labels, hovertemplate='%{text}<extra></extra>',
                     showlegend=False, lighting=dict(ambient=.6, diffuse=.8, specular=.2))


def reference_transform(mobile, reference, reference_positions: np.ndarray, fallback: dict) -> dict:
    """Proper rigid fit using a discovered common heavy-atom graph; never reflect."""
    # Bond orders cannot be established from a cube alone, so match connectivity.
    heavy_mobile = Chem.RemoveAllHs(mobile, sanitize=False)
    heavy_reference = Chem.RemoveAllHs(reference, sanitize=False)
    match = rdFMCS.FindMCS([heavy_mobile, heavy_reference], timeout=3,
                          bondCompare=rdFMCS.BondCompare.CompareAny,
                          atomCompare=rdFMCS.AtomCompare.CompareElements)
    if match.canceled or match.numAtoms < 3:
        return {**fallback, 'kind': 'principal (no adequate common scaffold)'}
    query = Chem.MolFromSmarts(match.smartsString)
    mobile_indices = heavy_mobile.GetSubstructMatch(query)
    reference_indices = heavy_reference.GetSubstructMatch(query)
    # Removing hydrogens preserves the relative order of heavy atoms.
    ref_heavy = [a.GetIdx() for a in reference.GetAtoms() if a.GetAtomicNum() != 1]
    target = reference_positions[np.array(ref_heavy)[list(reference_indices)]]
    source = heavy_mobile.GetConformer().GetPositions()[list(mobile_indices)]
    center, translation = source.mean(0), target.mean(0)
    if np.linalg.matrix_rank(source - center, tol=1e-6) < 2:
        return {**fallback, 'kind': 'principal (collinear common scaffold)'}
    u, _, vt = np.linalg.svd((source - center).T @ (target - translation))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    fitted = (source - center) @ rotation + translation
    return dict(center=center, rotation=rotation, translation=translation, kind='common scaffold',
                matched_atoms=len(source), fit_rms_angstrom=float(np.sqrt(np.mean(np.sum((fitted-target)**2, axis=1)))))


def extract_surface(cube: dict, level: float, transform: dict) -> dict:
    """Extract in index space, then apply the full CUBE affine transformation."""
    grid = cube['grid']
    if min(grid.shape) < 2:
        raise ValueError('Isosurfaces require at least two samples per axis')
    if not grid.min() < level < grid.max():
        return dict(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=int), clipped=False, area_angstrom2=0.0)
    vertices, faces, _, _ = marching_cubes(grid, level=level, allow_degenerate=False)
    clipped = bool(np.any(vertices <= 1e-5) or np.any(vertices >= np.array(grid.shape) - 1 - 1e-5))
    world = (cube['origin_bohr'] + vertices @ np.array(cube['voxel_vectors_bohr'])) * BOHR_TO_ANGSTROM
    transformed = transform_points(world, transform)
    return dict(vertices=transformed, faces=faces, clipped=clipped,
                area_angstrom2=float(mesh_surface_area(transformed, faces)))


def orbital_diagnostics(cube: dict) -> dict:
    grid = cube['grid']
    voxel_volume = abs(np.linalg.det(cube['voxel_vectors_bohr']))
    boundary = np.concatenate([grid[0].ravel(), grid[-1].ravel(), grid[:, 0].ravel(),
                               grid[:, -1].ravel(), grid[:, :, 0].ravel(), grid[:, :, -1].ravel()])
    return dict(norm_integral=float(np.sum(grid**2) * voxel_volume),
                boundary_max_abs_amplitude=float(np.max(abs(boundary))),
                shape=list(grid.shape), voxel_vectors_bohr=np.array(cube['voxel_vectors_bohr']).tolist(),
                origin_bohr=cube['origin_bohr'].tolist(),
                grid_convergence='not established by a single grid')


def compare_orbital_grids(first_path: str | Path, second_path: str | Path, isovalues=(0.03,)) -> dict:
    """Measure grid sensitivity without declaring a universal convergence tolerance.

    Surface distances are symmetric nearest-vertex distances, not an exact
    continuous Hausdorff distance. Compare exports of the same wavefunction.
    """
    first, second = parse_cube_file(first_path), parse_cube_file(second_path)
    if first['orbital_index'] is None or second['orbital_index'] is None:
        raise ValueError('Grid comparison requires identified orbital amplitudes')
    if (first['orbital_index'], first['operator']) != (second['orbital_index'], second['operator']):
        raise ValueError('Grid comparison requires the same orbital index and operator')
    if [a['atno'] for a in first['atoms']] != [a['atno'] for a in second['atoms']]:
        raise ValueError('Grid comparison requires the same atoms')
    a = np.array([atom['pos_ang'] for atom in first['atoms']])
    b = np.array([atom['pos_ang'] for atom in second['atoms']])
    if not np.allclose(a, b, atol=2e-5, rtol=0):
        raise ValueError('Grid comparison requires matching source coordinates')
    transform = dict(center=np.zeros(3), rotation=np.eye(3), translation=np.zeros(3))
    result = dict(first_file=Path(first_path).name, second_file=Path(second_path).name,
                  first_sha256=file_hash(Path(first_path)), second_sha256=file_hash(Path(second_path)),
                  first=orbital_diagnostics(first), second=orbital_diagnostics(second), surfaces={})
    for value in isovalues:
        if not np.isfinite(value) or value <= 0:
            raise ValueError('Comparison isovalues must be finite and positive')
        for level in (value, -value):
            meshes = [extract_surface(cube, level, transform) for cube in (first, second)]
            if any(m['clipped'] or not len(m['vertices']) for m in meshes):
                result['surfaces'][str(level)] = dict(status='absent or clipped surface')
                continue
            vertices_a, vertices_b = [m['vertices'] for m in meshes]
            distances = np.concatenate([cKDTree(vertices_a).query(vertices_b)[0], cKDTree(vertices_b).query(vertices_a)[0]])
            result['surfaces'][str(level)] = dict(
                symmetric_vertex_rms_angstrom=float(np.sqrt(np.mean(distances**2))),
                symmetric_vertex_max_angstrom=float(distances.max()),
                first_area_angstrom2=meshes[0]['area_angstrom2'], second_area_angstrom2=meshes[1]['area_angstrom2'])
    return result


def build_orbital_figure(fields: list[OrbitalField], settings: ViewSettings | None = None):
    """Build one common Plotly figure and its provenance, without rendering files."""
    settings = copy.deepcopy(settings or ViewSettings())
    settings.validate()
    if not fields:
        raise ValueError('No orbital fields supplied')
    for f in fields:
        if not f.cube['atoms']:
            raise ValueError('A molecular overlay requires atomic coordinates')
    columns = min(settings.columns, len(fields))
    rows = math.ceil(len(fields) / columns)
    titles = []
    for f in fields:
        energy = f'{f.energy_eV:.4f} eV' if f.energy_eV is not None else 'Energy unavailable'
        spin = f' / operator {f.operator}'
        title = f'{html.escape(f.label)} | {f.role}<br>MO {f.index}{spin} | {energy} | occupation {f.occupation:g}'
        if not f.metadata.get('normal_termination'):
            title += '<br>Normal termination not verified'
        titles.append(title)
    fig = make_subplots(rows=rows, cols=columns, specs=[[{'type': 'scene'}]*columns for _ in range(rows)],
                        subplot_titles=titles, horizontal_spacing=0.025, vertical_spacing=min(0.12, 0.3/rows))
    figure_metadata = dict(schema_version=1, settings=asdict(settings), orbitals=[],
                           scientific_notes=[
                               'Colors encode orbital phase, not charge; global sign is arbitrary.',
                               'Surfaces are amplitude contours in bohr^(-3/2), not electron-density surfaces.',
                               'One grid does not establish spatial convergence.',
                               'Cube headers and geometry were checked; binary hashes do not prove field provenance.',
                               'Bond lines are inferred connectivity, not assigned bond orders.'],
                           software={name: importlib.metadata.version(name) for name in ('numpy', 'plotly', 'scikit-image', 'rdkit')})
    surfaces_by_trace = []
    transformations, ref_mol, ref_positions = {}, None, None
    all_points = []
    for panel, f in enumerate(fields):
        row, col = divmod(panel, columns)
        atoms = f.cube['atoms']
        positions = np.array([a['pos_ang'] for a in atoms])
        mol = molecule_from_cube(f.cube, settings.bond_factor)
        key = (tuple(a['atno'] for a in atoms), positions.tobytes())
        if key not in transformations:
            transform = principal_transform(positions, [a['atno'] for a in atoms])
            if settings.alignment == 'none':
                transform = dict(center=np.zeros(3), rotation=np.eye(3), translation=np.zeros(3), kind='source coordinates')
            elif settings.alignment == 'reference' and ref_mol is not None:
                transform = reference_transform(mol, ref_mol, ref_positions, transform)
            transformations[key] = transform
        transform = transformations[key]
        positioned = transform_points(positions, transform)
        if ref_mol is None:
            ref_mol, ref_positions = mol, positioned
        all_points.append(positioned)
        diagnostic = orbital_diagnostics(f.cube)
        record = dict(label=f.label, role=f.role, orbital_index=f.index, operator=f.operator,
                      occupation=f.occupation, energy_eV=f.energy_eV, cube_file=f.cube_path.name,
                      cube_sha256=file_hash(f.cube_path), calculation=f.metadata, diagnostics=diagnostic,
                      transform={k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in transform.items()},
                      surfaces={})
        for sign, color, phase in ((1, settings.positive_color, 'Positive phase'), (-1, settings.negative_color, 'Negative phase')):
            variants = {}
            for value in settings.isovalues:
                surface = extract_surface(f.cube, sign * value, transform)
                if surface['clipped']:
                    raise ValueError(f'{f.cube_path.name}: surface at {sign*value:g} touches the grid boundary; export a larger cube')
                vertices, faces = surface['vertices'], surface['faces']
                if len(vertices):
                    all_points.append(vertices)
                variants[str(value)] = dict(x=vertices[:, 0].tolist(), y=vertices[:, 1].tolist(), z=vertices[:, 2].tolist(),
                                            i=faces[:, 0].tolist(), j=faces[:, 1].tolist(), k=faces[:, 2].tolist())
                record['surfaces'][str(sign*value)] = dict(vertex_count=len(vertices), area_angstrom2=surface['area_angstrom2'])
            trace_index = len(fig.data)
            fig.add_trace(go.Mesh3d(**variants[str(settings.isovalue)], color=color, opacity=settings.opacity,
                                   name=phase, legendgroup=phase, showlegend=panel == 0, flatshading=False,
                                   hoverinfo='skip', lighting=dict(ambient=0.65, diffuse=0.8, specular=0.2, roughness=0.8)), row=row+1, col=col+1)
            surfaces_by_trace.append(dict(trace=trace_index, variants=variants))
        bond_coords = []
        for bond in mol.GetBonds():
            bond_coords.extend([positioned[bond.GetBeginAtomIdx()].tolist(), positioned[bond.GetEndAtomIdx()].tolist(), [None]*3])
        if bond_coords:
            bond_array = np.array(bond_coords, dtype=object)
            fig.add_trace(go.Scatter3d(x=bond_array[:, 0].tolist(), y=bond_array[:, 1].tolist(), z=bond_array[:, 2].tolist(),
                                      mode='lines', line=dict(color='#444444', width=6), hoverinfo='skip', showlegend=False), row=row+1, col=col+1)
        fig.add_trace(atom_spheres(atoms, positioned, settings), row=row+1, col=col+1)
        figure_metadata['orbitals'].append(record)
    bounds = np.concatenate(all_points)
    midpoint = (bounds.max(0) + bounds.min(0)) / 2
    half_span = (bounds.max(0) - bounds.min(0)).max()/2 + settings.padding_angstrom
    axis = lambda i: dict(range=[float(midpoint[i]-half_span), float(midpoint[i]+half_span)], visible=False)
    scene = dict(xaxis=axis(0), yaxis=axis(1), zaxis=axis(2), aspectmode='manual',
                 aspectratio=dict(x=settings.scene_scale, y=settings.scene_scale, z=settings.scene_scale), camera=settings.camera,
                 bgcolor='white', dragmode='orbit')
    scenes = []
    for panel in range(len(fields)):
        name = 'scene' if panel == 0 else f'scene{panel+1}'
        scenes.append(name)
        fig.update_layout(**{name: scene})
    fig.update_layout(width=settings.width, height=rows*settings.panel_height+140,
                      title=dict(text=html.escape(settings.title), x=0.5, font=dict(size=25)),
                      paper_bgcolor='white', font=dict(family='Arial, sans-serif', size=14, color='#203044'),
                      margin=dict(l=20, r=20, t=85, b=95),
                      legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.04),
                      meta=dict(isovalue=settings.isovalue))
    methods = {(f.metadata.get('orca_version'), f.metadata.get('input_keywords')) for f in fields}
    if len(methods) == 1:
        version, keywords = next(iter(methods))
        method_note = f'ORCA {version or "version unavailable"} | {keywords or "Method unavailable"}'
    else:
        method_note = 'Calculation methods are recorded individually in the companion metadata.'
    fig.add_annotation(text=html.escape(method_note), x=0.5, y=-0.14, xref='paper', yref='paper',
                       showarrow=False, font=dict(size=11))
    figure_metadata['scene_names'] = scenes
    figure_metadata['axis_ranges_angstrom'] = [axis(i)['range'] for i in range(3)]
    return fig, figure_metadata, surfaces_by_trace


def _safe_json(value) -> str:
    return json.dumps(value, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')


def write_viewer(fig, metadata: dict, surfaces: list[dict], path: Path) -> Path:
    """Bundle Plotly and all meshes locally; the viewer makes no network requests."""
    template = Path(__file__).with_name('orbital_viewer.html').read_text(encoding='utf-8')
    # Round-trip Plotly's own encoder (handles its trace schema), then escape HTML.
    replacements = dict(TITLE=html.escape(metadata['settings']['title']),
                        PLOTLY=get_plotlyjs(), FIGURE=_safe_json(json.loads(fig.to_json())),
                        METADATA=_safe_json(metadata), SURFACES=_safe_json(surfaces))
    for key, value in replacements.items():
        template = template.replace(f'@@{key}@@', value)
    path.write_text(template, encoding='utf-8')
    return path


def export_orbital_report(fields: list[OrbitalField], output_directory: str | Path,
                          settings: ViewSettings | None = None, formats=('html', 'png', 'pdf')) -> dict[str, Path]:
    settings = copy.deepcopy(settings or ViewSettings())
    settings.validate()
    unknown = set(formats) - {'html', 'png', 'pdf', 'svg'}
    if unknown:
        raise ValueError(f'Unsupported export formats: {sorted(unknown)}')
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    fig, metadata, surfaces = build_orbital_figure(fields, settings)
    files = {}
    metadata_path = output / 'frontier_orbitals.json'
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    files['metadata'] = metadata_path
    for extension in formats:
        path = output / f'frontier_orbitals.{extension}'
        if extension == 'html':
            write_viewer(fig, metadata, surfaces, path)
        else:
            try:
                fig.write_image(path, format=extension, scale=settings.export_scale)
            except (RuntimeError, ValueError) as exc:
                raise RuntimeError('Static export needs kaleido and a compatible Chrome/Chromium installation. '
                                   'Set BROWSER_PATH to its executable, or request --formats html.') from exc
        files[extension] = path
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Directory containing ORCA output and orbital CUBE files')
    parser.add_argument('--output', type=Path, required=True, help='Report directory')
    parser.add_argument('--config', type=Path, help='Settings JSON, saved viewer state, or calculation manifest')
    parser.add_argument('--isovalue', type=float, help='Displayed absolute orbital amplitude in bohr^(-3/2)')
    parser.add_argument('--formats', nargs='+', choices=('html', 'png', 'pdf', 'svg'), default=['html', 'png', 'pdf'])
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text()) if args.config else {}
    options = dict(config.get('settings', {}))
    if args.isovalue is not None:
        options['isovalue'] = args.isovalue
    settings = ViewSettings(**options)
    fields = discover_frontier_fields(args.input, config.get('calculations'))
    if not args.output.exists():
        print(f'Creating report directory: {args.output}. Generated reports can be removed and regenerated.')
    paths = export_orbital_report(fields, args.output, settings, formats=args.formats)
    for kind, path in paths.items():
        print(f'{kind}: {path}')


if __name__ == '__main__':
    main()
