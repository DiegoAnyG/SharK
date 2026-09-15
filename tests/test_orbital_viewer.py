"""Numerical and provenance checks independent of the benchmark's energies."""

from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from shark.core.cube import BOHR_TO_ANGSTROM, parse_cube_file
from shark.core.frontier import discover_frontier_fields, read_orbital_metadata
from shark.reports.orbital_viewer import (
    ViewSettings, build_orbital_figure, extract_surface, principal_transform,
    transform_points, write_viewer, compare_orbital_grids, atom_spheres,
)


def cube_text(grid, origin=(0, 0, 0), vectors=None, orbital=None, values_per_line=6):
    vectors = np.eye(3) if vectors is None else vectors
    comment = f'Molecular orbital {orbital} of operator 0' if orbital is not None else 'Scalar field'
    lines = ['Synthetic field', comment, f'{-1 if orbital is not None else 1} ' + ' '.join(map(str, origin))]
    lines += [str(n) + ' ' + ' '.join(map(str, vector)) for n, vector in zip(grid.shape, vectors)]
    lines += ['1 1 0 0 0']
    if orbital is not None:
        lines += [f'1 {orbital}']
    flat = grid.ravel()
    lines += [' '.join(f'{v:.12E}' for v in flat[i:i+values_per_line]) for i in range(0, len(flat), values_per_line)]
    return '\n'.join(lines) + '\n'


def orbital_table(rows, spin=None):
    heading = f'SPIN {spin} ORBITALS\n' if spin else ''
    return heading + 'NO OCC E(Eh) E(eV)\n' + '\n'.join(f'{i} {occ:.4f} {ev / 27.2114:.6f} {ev:.6f}' for i, occ, ev in rows) + '\n\n'


class CubeReaderTests(unittest.TestCase):
    def test_positive_atom_count_does_not_discard_short_data_lines(self):
        grid = np.arange(8).reshape(2, 2, 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'scalar.cube'
            path.write_text(cube_text(grid, values_per_line=1))
            np.testing.assert_array_equal(parse_cube_file(path)['grid'], grid)

    def test_negative_atom_header_and_fortran_exponents(self):
        grid = np.arange(8).reshape(2, 2, 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'orbital.cube'
            path.write_text(cube_text(grid, orbital=7).replace('E+', 'D+'))
            cube = parse_cube_file(path)
            self.assertEqual(cube['orbital_index'], 7)
            np.testing.assert_array_equal(cube['grid'], grid)

    def test_multi_dataset_selection_and_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'multi.cube'
            header = '\n'.join(cube_text(np.zeros((2, 2, 2)), orbital=4).splitlines()[:7])
            values = np.column_stack([np.arange(8), -np.arange(8)]).ravel()
            path.write_text(header + '\n2 4\n9\n' + ' '.join(map(str, values)))
            # Multi-dataset headers do not have a unique orbital comment.
            path.write_text(path.read_text().replace('Molecular orbital 4 of operator 0', 'Multiple molecular orbitals'))
            with self.assertRaisesRegex(ValueError, 'Multiple datasets'):
                parse_cube_file(path)
            np.testing.assert_array_equal(parse_cube_file(path, dataset_id=9)['grid'].ravel(), -np.arange(8))
            path.write_text(cube_text(np.zeros((2, 2, 2))).rsplit(' ', 1)[0])
            with self.assertRaisesRegex(ValueError, 'Scalar count'):
                parse_cube_file(path)

    def test_angstrom_convention_and_malformed_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'units.cube'
            lines = cube_text(np.zeros((2, 2, 2)), origin=(1, 2, 3)).splitlines()
            for i in range(3, 6):
                lines[i] = '-' + lines[i]
            path.write_text('\n'.join(lines))
            np.testing.assert_allclose(parse_cube_file(path)['origin_ang'], [1, 2, 3])
            lines[3] = lines[3][1:]
            path.write_text('\n'.join(lines))
            with self.assertRaisesRegex(ValueError, 'consistent'):
                parse_cube_file(path)


class FrontierDiscoveryTests(unittest.TestCase):
    def test_last_table_and_nonbenchmark_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / 'unrelated_molecule.out'
            output.write_text('ORBITAL ENERGIES\n' + orbital_table([(0, 2, -5), (1, 0, 1)]) +
                              'ORBITAL ENERGIES\n' + orbital_table([(2, 2, -3.5), (3, 0, 2.2)]) + 'MULLIKEN\n')
            for index in (2, 3):
                (root / f'unrelated_molecule.custom{index}.cube').write_text(cube_text(np.zeros((2, 2, 2)), orbital=index))
            fields = discover_frontier_fields(root)
            self.assertEqual([f.index for f in fields], [2, 3])
            self.assertEqual([f.energy_eV for f in fields], [-3.5, 2.2])
            self.assertEqual([f.role for f in fields], ['HOMO', 'LUMO'])

    def test_spin_resolved_and_fractional_occupations(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'radical.out'
            path.write_text('ORBITAL ENERGIES\n' + orbital_table([(0, 1, -2), (1, 0, 1)], 'UP') +
                            orbital_table([(0, 1, -3), (1, 0, 2)], 'DOWN'))
            self.assertEqual(set(read_orbital_metadata(path)['channels']), {0, 1})
            path.write_text('ORBITAL ENERGIES\n' + orbital_table([(0, 0.4, -2), (1, 0.6, 1)]))
            with self.assertRaisesRegex(ValueError, 'fractional'):
                read_orbital_metadata(path)

    def test_missing_field_and_geometry_mismatch_are_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'sample.out').write_text('ORBITAL ENERGIES\n' + orbital_table([(0, 2, -2), (1, 0, 1)]))
            (root / 'sample.h.cube').write_text(cube_text(np.zeros((2, 2, 2)), orbital=0))
            with self.assertRaisesRegex(FileNotFoundError, 'LUMO'):
                discover_frontier_fields(root)
            text = cube_text(np.zeros((2, 2, 2)), orbital=1).replace('1 1 0 0 0', '1 1 1 0 0')
            (root / 'sample.l.cube').write_text(text)
            with self.assertRaisesRegex(ValueError, 'geometry'):
                discover_frontier_fields(root)


class SurfaceGeometryTests(unittest.TestCase):
    def test_atom_spheres_use_scaled_radii_and_preserve_centers(self):
        atoms = [dict(element='C', atno=6, radius=.76), dict(element='O', atno=8, radius=.66)]
        positions = np.array([[1.,2,3], [-1.,0,2]])
        settings = ViewSettings(atom_radius_scale=.3)
        mesh = atom_spheres(atoms, positions, settings)
        vertices = np.column_stack([mesh.x, mesh.y, mesh.z]).reshape(2, -1, 3)
        for atom, center, points in zip(atoms, positions, vertices):
            np.testing.assert_allclose(np.linalg.norm(points-center, axis=1), atom['radius']*.3, atol=1e-12)
            np.testing.assert_allclose(points.mean(0), center, atol=1e-12)

    def test_grid_comparison_has_zero_difference_for_identical_fields(self):
        axis = np.linspace(-3, 3, 15)
        x, y, z = np.meshgrid(axis, axis, axis, indexing='ij')
        grid = z*np.exp(-(x*x+y*y+z*z))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'test.cube'
            path.write_text(cube_text(grid, origin=(-3,-3,-3), vectors=np.eye(3)*(axis[1]-axis[0]), orbital=8))
            result = compare_orbital_grids(path, path)
            for phase in result['surfaces'].values():
                self.assertEqual(phase['symmetric_vertex_rms_angstrom'], 0.)
                self.assertEqual(phase['first_area_angstrom2'], phase['second_area_angstrom2'])

    def test_affine_transform_on_oblique_grid_has_no_extra_voxel(self):
        grid = np.indices((5, 6, 7))[0].astype(float)
        vectors = np.array([[.2, .1, 0], [0, .3, .05], [.02, 0, .4]])
        origin = np.array([3., -2., 1.])
        rotation = Rotation.from_euler('xyz', [20, 40, 15], degrees=True).as_matrix()
        transform = dict(center=np.array([1., 2., 3.]), rotation=rotation, translation=np.array([4., 0., 0.]))
        cube = dict(grid=grid, voxel_vectors_bohr=vectors, origin_bohr=origin)
        surface = extract_surface(cube, 2.25, transform)
        world = (surface['vertices']-transform['translation']) @ rotation.T + transform['center']
        indices = (world / BOHR_TO_ANGSTROM-origin) @ np.linalg.inv(vectors)
        np.testing.assert_allclose(indices[:, 0], 2.25, atol=1e-6)
        np.testing.assert_allclose(indices[:, 1:].max(0), [5, 6], atol=1e-6)

    def test_sphere_surface_converges_with_refinement(self):
        errors = []
        for n in (15, 41):
            x = np.linspace(-2, 2, n)
            xyz = np.stack(np.meshgrid(x, x, x, indexing='ij'), axis=-1)
            cube = dict(grid=np.exp(-np.sum(xyz**2, axis=-1)), origin_bohr=np.repeat(-2., 3),
                        voxel_vectors_bohr=np.eye(3)*(x[1]-x[0]))
            transform = dict(center=np.zeros(3), rotation=np.eye(3), translation=np.zeros(3))
            surface = extract_surface(cube, np.exp(-1), transform)
            radius = np.linalg.norm(surface['vertices'], axis=1)/BOHR_TO_ANGSTROM
            errors.append(np.mean(abs(radius-1)))
            self.assertFalse(surface['clipped'])
        self.assertLess(errors[1], errors[0]/3)

    def test_alignment_is_rigid_and_never_reflects(self):
        positions = np.array([[0.,0,0], [2,0,0], [1,1,0], [0,0,1]])
        transform = principal_transform(positions, [6]*len(positions))
        shifted = transform_points(positions, transform)
        np.testing.assert_allclose(np.linalg.norm(positions[:,None]-positions,axis=-1),
                                   np.linalg.norm(shifted[:,None]-shifted,axis=-1), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(transform['rotation']), 1.)


class BenchmarkViewerTests(unittest.TestCase):
    def test_shared_geometry_offline_bundle_and_provenance(self):
        import os
        root = Path(__file__).resolve().parents[1]
        data = Path(os.environ.get('SHARK_TEST_DATA', os.environ.get('TOPICS_TEST_DATA', root / 'dft_benzofuroxan')))
        if not list(data.glob('*.cube')):
            self.skipTest('Private benchmark unavailable; set SHARK_TEST_DATA')
        fields = discover_frontier_fields(data)
        fields[0].label = '</script><script>alert(1)</script>'
        settings = ViewSettings(isovalues=(.03,), isovalue=.03, export_scale=1)
        fig, metadata, meshes = build_orbital_figure(fields, settings)
        for record in metadata['orbitals']:
            self.assertAlmostEqual(record['diagnostics']['norm_integral'], 1., delta=.01)
            self.assertEqual(len(record['cube_sha256']), 64)
        for surface in meshes:
            trace = fig.data[surface['trace']]
            np.testing.assert_array_equal(trace.x, surface['variants']['0.03']['x'])
        with tempfile.TemporaryDirectory() as tmp:
            path = write_viewer(fig, metadata, meshes, Path(tmp)/'viewer.html')
            text = path.read_text()
            self.assertNotIn('<script src=', text)
            self.assertNotIn('</script><script>alert(1)</script>', text)
            self.assertIn('Save settings', text)
        self.assertEqual(metadata['scene_names'], ['scene', 'scene2', 'scene3', 'scene4'])


if __name__ == '__main__':
    unittest.main()
