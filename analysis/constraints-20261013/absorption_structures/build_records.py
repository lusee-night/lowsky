"""Transcribe CDS fixed-width Su et al. (2018) and printed loop Table 2.

Only source-table values are copied; no map sampling or plot digitization.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "sources" / "su_2018_tablea1.dat"

# One-indexed columns from the accompanying CDS ReadMe.
COLS = {
    "name": (1, 15), "distance_kpc": (17, 20), "distance_error_kpc": (22, 24),
    "electron_temperature_1000k": (26, 30), "electron_temperature_error_1000k": (32, 35),
    "absorbed_brightness_1000k": (37, 40), "absorbed_brightness_error_1000k": (42, 44),
    "surrounding_brightness_1000k": (46, 49), "surrounding_brightness_error_1000k": (51, 53),
    "haslam_total_brightness_1000k": (55, 59), "haslam_total_brightness_error_1000k": (61, 64),
    "behind_brightness_1000k": (66, 69), "behind_error_plus_1000k": (71, 75),
    "behind_error_minus_1000k": (77, 81), "front_brightness_1000k": (83, 86),
    "front_error_plus_1000k": (88, 92), "front_error_minus_1000k": (94, 98),
    "missing_behind_1000k": (100, 103), "missing_behind_error_plus_1000k": (105, 109),
    "missing_behind_error_minus_1000k": (111, 115), "missing_front_1000k": (117, 120),
    "missing_front_error_plus_1000k": (122, 126), "missing_front_error_minus_1000k": (128, 132),
    "optical_depth_limit_flag": (134, 134), "optical_depth": (135, 138),
    "emissivity_behind_k_per_pc": (140, 142), "emissivity_behind_error_plus_k_per_pc": (144, 147),
    "emissivity_behind_error_minus_k_per_pc": (149, 152),
    "emissivity_front_k_per_pc": (154, 157), "emissivity_front_error_plus_k_per_pc": (159, 163),
    "emissivity_front_error_minus_k_per_pc": (165, 169),
    "emissivity_behind_simplified_k_per_pc": (171, 174),
    "emissivity_behind_simplified_error_k_per_pc": (176, 179),
}


def parse_line(line):
    values = {}
    for key, (start, stop) in COLS.items():
        raw = line[start - 1:stop].strip()
        values[key] = raw if key == "name" else (raw if key == "optical_depth_limit_flag" else (float(raw) if raw else None))
    match = re.fullmatch(r"G(\d{3}\.\d{3})([+-])(\d{2}\.\d{3})", values["name"])
    if not match:
        raise ValueError(values["name"])
    values["galactic_longitude_deg"] = float(match.group(1))
    values["galactic_latitude_deg"] = float(match.group(3)) * (1 if match.group(2) == "+" else -1)
    return values


def main():
    rows = [parse_line(line) for line in SOURCE.read_text().splitlines()]
    assert len(rows) == 152 and len({r["name"] for r in rows}) == 152
    out = {
        "dataset": "Su et al. 2018 GLEAM H II absorption, CDS J/MNRAS/479/4041/tablea1",
        "frequency_mhz": 76.2,
        "frequency_bandwidth_mhz": 7.68,
        "beam_fwhm_major_arcmin": 5.41,
        "beam_fwhm_minor_arcmin": 4.43,
        "source_url": "https://cdsarc.cds.unistra.fr/ftp/J/MNRAS/479/4041/tablea1.dat",
        "source_readme_url": "https://cdsarc.cds.unistra.fr/ftp/J/MNRAS/479/4041/ReadMe",
        "paper_url": "https://doi.org/10.1093/mnras/sty1732",
        "sampling": "Selected H II absorption features in -40 < l < 40 deg, -2 < b < 4 deg; one table row per feature.",
        "measured_fields": ["absorbed_brightness_1000k", "surrounding_brightness_1000k"],
        "derived_fields": ["haslam_total_brightness_1000k", "behind_brightness_1000k", "front_brightness_1000k", "missing_behind_1000k", "missing_front_1000k", "optical_depth", "emissivity_behind_k_per_pc", "emissivity_front_k_per_pc", "emissivity_behind_simplified_k_per_pc"],
        "calibration_and_model": "Brightness temperatures are GLEAM interferometer on-source and nearby-region measurements. Total power is reconstructed from 408-MHz Haslam dsds with brightness spectral index -2.7; emissivity path to Galactic edge assumes R_G=20 kpc. Optical depths and emissivities are fitted jointly to six frequencies, but this CDS table publishes 76.2 MHz only. Reported one-sigma errors exclude spectral-index systematics; -2.6 instead of -2.7 lowers emissivity by about 15% (CDS ReadMe).",
        "conditioning_status": "candidate_holdout_shared_haslam_lineage: measured GLEAM on/off columns are new to stated direct lowsky conditioning, but derived columns use Haslam; full ULSA/input lineage has not been audited",
        "records": rows,
    }
    (ROOT / "hii_absorption_76p2mhz.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
