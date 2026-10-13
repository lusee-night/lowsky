"""Measurement conversion and supported-domain checks for bright-source figures."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "examples/compare_bright_sources.py"
spec = importlib.util.spec_from_file_location("compare_bright_sources", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_kelvin_solid_angle_conversion_recovers_known_jy() -> None:
    frequency = np.array([10.0, 50.0])
    known_jy = np.array([1234.0, 5678.0])
    npix = 12
    omega = 4 * np.pi / npix
    integrated_k_sr = known_jy * 1e-26 * module.C_LIGHT**2 / (
        2 * module.K_BOLTZMANN * (frequency * 1e6) ** 2
    )
    # Uniform full-sky maps with the specified integrated source amplitudes.
    maps = np.repeat((integrated_k_sr / (npix * omega))[:, None], npix, axis=1)
    np.testing.assert_allclose(module.integrated_flux_jy(maps, frequency), known_jy, rtol=1e-12)


def test_out_of_band_measurements_have_no_model_prediction() -> None:
    frequency = np.array([1.0, 50.0])
    flux = {name: np.array([10.0, 20.0]) for name in module.SOURCES}
    records = [
        {"source_id": "lwa1_llfss_2017", "object": "CAS_A", "frequency_mhz_nominal": nominal,
         "frequency_mhz_center": center, "ratio_to_cyg_a": 1.0, "reported_pm": .1}
        for nominal, center in [(50, 50.005), (50, 50.02), (60, 59.985)]
    ]
    result = module.prepare_comparisons({"observational_records": records}, frequency, flux)
    near, beyond, sixty = result["lwa1_ratios"]
    assert near["comparison_mhz"] == 50.0
    assert near["note"] == "50.005-MHz band evaluated at nominal 50 MHz"
    assert near["model_ratio_source_only"] == 1.0
    assert not near["in_saved_model_range"]
    assert near["comparison_available"]
    assert near["nominal_frequency_approximation"]
    for row in (beyond, sixty):
        assert row["model_ratio_source_only"] is None
        assert row["comparison_mhz"] is None
        assert not row["in_saved_model_range"]
        assert not row["comparison_available"]
        assert not row["nominal_frequency_approximation"]


def test_conditional_cas_calibration_hits_reported_peak() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "analysis/constraints-20261013/calibration/cas_a/fit.json"
    fit = json.loads(path.read_text())
    m, target = fit["model"], fit["target"]
    f = target["peak_frequency_mhz"]
    tau_mw = m["foreground_tau_at_50mhz"] * (f / 50) ** -2.1
    value = m["new_s50_jy"] * (f / 50) ** m["fixed_alpha"] * np.exp(
        -m["new_tau10"] * ((f / 10) ** -2.1 - 5 ** -2.1) - tau_mw
    )
    logarithmic_derivative = m["fixed_alpha"] + 2.1 * (
        m["new_tau10"] * (f / 10) ** -2.1 + tau_mw
    )
    np.testing.assert_allclose(value, target["peak_flux_jy"], rtol=1e-12)
    np.testing.assert_allclose(logarithmic_derivative, 0, atol=1e-12)
