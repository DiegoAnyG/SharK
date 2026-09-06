"""Vibrational infrared (IR) spectroscopy simulation and convolution."""

from __future__ import annotations
import numpy as np
from typing import List, Tuple


def simulate_ir_spectrum(
    frequencies: List[float],
    intensities: List[float],
    fwhm: float = 15.0,
    wavenumber_range: Tuple[float, float] = (400.0, 4000.0),
    step: float = 1.0,
    scale_factor: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate an infrared (IR) spectrum via Lorentzian lineshape convolution.

    Parameters:
    -----------
    frequencies : List[float]
        Harmonic vibrational frequencies (cm^-1) from Hessian calculation.
    intensities : List[float]
        Infrared absorption intensities (km/mol).
    fwhm : float
        Full Width at Half Maximum for Lorentzian peak broadening (default 15.0 cm^-1).
    wavenumber_range : Tuple[float, float]
        (min_cm, max_cm) frequency interval to simulate (default 400.0 to 4000.0 cm^-1).
    step : float
        Grid resolution in cm^-1 (default 1.0 cm^-1).
    scale_factor : float
        Empirical frequency scaling factor for the DFT functional/basis (default 1.0).

    Returns:
    --------
    wavenumbers : np.ndarray
        Array of wavenumbers in cm^-1.
    absorbance : np.ndarray
        Convoluted molar absorptivity / simulated absorption profile.
    """
    # Handle cases where full 3N modes (including 5 or 6 translational/rotational zeros)
    # were provided alongside 3N-6 vibrational intensities
    if len(frequencies) > len(intensities) and (len(frequencies) - len(intensities)) in (5, 6):
        frequencies = frequencies[-len(intensities):]

    if len(frequencies) != len(intensities):
        raise ValueError(
            f"Frequencies length ({len(frequencies)}) does not match intensities length ({len(intensities)})."
        )

    wavenumbers = np.arange(wavenumber_range[0], wavenumber_range[1] + step, step)
    absorbance = np.zeros_like(wavenumbers, dtype=float)

    # Filter out imaginary (negative) or zero (translational/rotational) frequencies
    valid_pairs = [
        (f * scale_factor, inten)
        for f, inten in zip(frequencies, intensities)
        if f > 10.0 and inten >= 0.0
    ]

    half_gamma = 0.5 * fwhm
    gamma_factor = half_gamma / np.pi

    for f_center, inten in valid_pairs:
        # Lorentzian: I * (gamma / (2 * pi)) / ((nu - nu0)^2 + (gamma/2)^2)
        lorentzian = inten * (gamma_factor / ((wavenumbers - f_center) ** 2 + half_gamma ** 2))
        absorbance += lorentzian

    return wavenumbers, absorbance
