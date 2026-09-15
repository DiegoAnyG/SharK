"""Unit tests for SharK visualization modules (IR spectra, Boltzmann equilibrium, Orbitals)."""

import tempfile
import unittest
import os
from pathlib import Path

from shark.core.parser import parse_orca_results
from shark.analysis.thermo import calculate_relative_thermo
from shark.reports.visualizer import plot_boltzmann_equilibrium, plot_ir_comparison
from shark.reports.orbitals import plot_frontier_orbitals_panel, plot_electron_density_comparison


class TestSharKVisualizers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parent.parent
        cls.dft_dir = Path(os.environ.get('SHARK_TEST_DATA', os.environ.get('TOPICS_TEST_DATA', root / 'dft_benzofuroxan')))
        if not (cls.dft_dir / 'tautomer_1_oxide.out').is_file():
            raise unittest.SkipTest('Private benchmark unavailable; set SHARK_TEST_DATA')
        cls.r1 = parse_orca_results(cls.dft_dir / "tautomer_1_oxide", name="tautomer_1")
        cls.r3 = parse_orca_results(cls.dft_dir / "tautomer_3_oxide", name="tautomer_3")

    def test_ir_transmittance_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_png = Path(tmpdir) / "test_ir.png"
            res = plot_ir_comparison([self.r1, self.r3], out_png, mode="transmittance", dpi=100)
            self.assertTrue(res.exists())
            self.assertGreater(res.stat().st_size, 1000)

    def test_boltzmann_equilibrium_plot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_png = Path(tmpdir) / "test_boltzmann.png"
            df = calculate_relative_thermo([self.r1, self.r3])
            res = plot_boltzmann_equilibrium(df, out_png, dpi=100)
            self.assertTrue(res.exists())
            self.assertGreater(res.stat().st_size, 1000)

    def test_orbital_panel_generation(self):
        t1_h = self.dft_dir / "tautomer_1_oxide.mo45a.cube"
        t1_l = self.dft_dir / "tautomer_1_oxide.mo46a.cube"
        t3_h = self.dft_dir / "tautomer_3_oxide.mo45a.cube"
        t3_l = self.dft_dir / "tautomer_3_oxide.mo46a.cube"

        if all(p.exists() for p in [t1_h, t1_l, t3_h, t3_l]):
            with tempfile.TemporaryDirectory() as tmpdir:
                out_png = Path(tmpdir) / "test_orbitals.png"
                res = plot_frontier_orbitals_panel(t1_h, t1_l, t3_h, t3_l, out_png, dpi=100)
                self.assertTrue(res.exists())
                self.assertGreater(res.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
