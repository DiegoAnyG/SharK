"""Unit tests for SharK core parser, thermo calculations, and spectra convolution."""

import unittest
import os
from pathlib import Path
import numpy as np

from shark.core.parser import parse_orca_results
from shark.analysis.thermo import calculate_relative_thermo
from shark.analysis.spectra import simulate_ir_spectrum


class TestSharKCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parent.parent
        cls.dft_dir = Path(os.environ.get('SHARK_TEST_DATA', os.environ.get('TOPICS_TEST_DATA', root / 'dft_benzofuroxan')))
        if not (cls.dft_dir / 'tautomer_1_oxide.out').is_file():
            raise unittest.SkipTest('Private benchmark unavailable; set SHARK_TEST_DATA')

    def test_parse_orca_tautomer_1(self):
        res = parse_orca_results(self.dft_dir / "tautomer_1_oxide", name="tautomer_1")
        self.assertTrue(res.converged)
        self.assertEqual(len(res.imaginary_frequencies), 0)
        self.assertTrue(res.is_stationary_minimum)
        self.assertAlmostEqual(res.gibbs_energy, -678.540212, places=4)
        self.assertGreater(res.dipole_magnitude, 4.0)
        self.assertGreater(res.homo_lumo_gap, 3.0)
        self.assertEqual(len(res.ir_intensities), 45)

    def test_parse_orca_tautomer_3(self):
        res = parse_orca_results(self.dft_dir / "tautomer_3_oxide", name="tautomer_3")
        self.assertTrue(res.converged)
        self.assertEqual(len(res.imaginary_frequencies), 0)
        self.assertTrue(res.is_stationary_minimum)
        self.assertAlmostEqual(res.gibbs_energy, -678.541415, places=4)
        self.assertGreater(res.dipole_magnitude, 3.0)
        self.assertGreater(res.homo_lumo_gap, 3.0)
        self.assertEqual(len(res.ir_intensities), 45)

    def test_relative_thermo_calculation(self):
        r1 = parse_orca_results(self.dft_dir / "tautomer_1_oxide", name="tautomer_1")
        r3 = parse_orca_results(self.dft_dir / "tautomer_3_oxide", name="tautomer_3")

        df = calculate_relative_thermo([r1, r3], temperature=298.15)
        self.assertEqual(len(df), 2)

        # tautomer 3 must be the ground state (dG == 0.0)
        top = df.iloc[0]
        self.assertEqual(top["Name"], "tautomer_3")
        self.assertEqual(top["dG (kcal/mol)"], 0.0)
        self.assertGreater(top["Population (%)"], 70.0)

        second = df.iloc[1]
        self.assertEqual(second["Name"], "tautomer_1")
        self.assertAlmostEqual(second["dG (kcal/mol)"], 0.755, delta=0.05)
        self.assertAlmostEqual(top["Population (%)"] + second["Population (%)"], 100.0, delta=0.1)

    def test_simulate_ir_spectrum(self):
        freqs = [1000.0, 1700.0]
        intensities = [50.0, 200.0]
        wn, absorp = simulate_ir_spectrum(freqs, intensities, fwhm=10.0, wavenumber_range=(500, 2000), step=1.0)
        self.assertEqual(len(wn), 1501)
        self.assertEqual(len(absorp), 1501)
        idx_1700 = int(np.where(wn == 1700.0)[0][0])
        self.assertGreater(absorp[idx_1700], 10.0)

    def test_parse_population_analysis(self):
        res = parse_orca_results(self.dft_dir / "tautomer_1_oxide", name="tautomer_1")
        self.assertEqual(len(res.loewdin_charges), 17)
        self.assertEqual(len(res.mulliken_charges), 17)
        self.assertEqual(len(res.atomic_numbers), 17)
        self.assertEqual(len(res.atomic_symbols), 17)
        self.assertEqual(len(res.coordinates_angstrom), 17)
        self.assertEqual(res.atomic_symbols[0], "O")
        self.assertEqual(res.atomic_symbols[1], "C")
        self.assertAlmostEqual(res.loewdin_charges[0], -0.20976, places=3)
        self.assertAlmostEqual(res.mulliken_charges[0], -0.24615, places=3)


if __name__ == "__main__":
    unittest.main()
