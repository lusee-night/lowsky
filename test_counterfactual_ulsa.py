import numpy as np
import healpy as hp
import jax.numpy as jnp

from counterfactual_ulsa import (
    ATEAM,
    SkyConfig,
    _tau_coefficient,
    _smooth_broken_power_law_scale,
    exact_ateam_alms,
    local_bubble_boundary_kpc,
    make_local_partial_screens,
    make_local_shell_catalog,
    make_orion_partial_screens,
    make_random_fields,
    prepare_geometry,
    shell_ray_segments,
    source_templates,
)


def test_random_fields_are_reproducible_and_normalized():
    config = SkyConfig(nside=8, n_shells=3, seed=7)
    first = make_random_fields(config)
    second = make_random_fields(config)
    np.testing.assert_allclose(first["synch_shells"], second["synch_shells"])
    np.testing.assert_allclose(np.mean(first["synch_shells"], axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.std(first["synch_shells"], axis=1), 1.0, atol=1e-12)


def test_tau_increases_toward_lower_frequency():
    frequencies = jnp.asarray([1.0, 3.0, 10.0, 50.0])
    coefficients = np.asarray(_tau_coefficient(frequencies, 8_000.0))
    assert np.all(np.diff(coefficients) < 0.0)


def test_local_bubble_boundary_is_irregular_and_has_chimneys():
    x, y, z = hp.pix2vec(32, np.arange(hp.nside2npix(32)))
    radius = np.asarray(local_bubble_boundary_kpc(jnp.asarray(x), jnp.asarray(y), jnp.asarray(z)))
    assert 0.15 < np.mean(radius) < 0.19
    assert np.min(radius) >= 0.070
    assert np.percentile(radius, 95) > 0.45
    assert np.max(radius) > 0.60


def test_shell_spectrum_flattens_smoothly_below_break():
    frequencies = jnp.asarray([1.0, 3.0, 100.0, 300.0])
    scale = np.asarray(_smooth_broken_power_law_scale(frequencies, -2.9, -2.5, 30.0, 2.0))
    low_slope = np.log(scale[1] / scale[0]) / np.log(3.0 / 1.0)
    high_slope = np.log(scale[3] / scale[2]) / np.log(300.0 / 100.0)
    assert abs(low_slope + 2.5) < 0.03
    assert abs(high_slope + 2.9) < 0.03


def test_geometry_shape_and_galactic_center_direction():
    config = SkyConfig(nside=8, n_distance=16)
    geometry = prepare_geometry(config)
    assert geometry["radius"].shape == (hp.nside2npix(16), 16)
    gc_pix = hp.ang2pix(16, 0.0, 0.0, lonlat=True)
    assert np.min(geometry["radius"][gc_pix]) < 1.0


def test_orion_filaments_are_thin_partial_covering_screens():
    config = SkyConfig(nside=32, ray_oversample=1, sky_mode="ours")
    geometry = prepare_geometry(config)
    emission_measure, covering, distance = make_orion_partial_screens(
        config, geometry
    )
    l_deg, b_deg = geometry["l_deg"], geometry["b_deg"]
    vectors = np.asarray(hp.ang2vec(l_deg, b_deg, lonlat=True))
    center = np.asarray(hp.ang2vec(202.0, -38.0, lonlat=True))
    aperture = np.degrees(
        np.arccos(np.clip(vectors @ center, -1.0, 1.0))
    ) < 35.0
    effective_covering = 1.0 - np.prod(1.0 - covering, axis=0)

    assert emission_measure.shape == (2, hp.nside2npix(32))
    assert np.max(emission_measure) > 50.0
    assert np.all((covering >= 0.0) & (covering <= 0.46))
    assert 0.02 <= np.mean(effective_covering[aperture]) <= 0.05
    assert np.mean(effective_covering[aperture] > 0.25) <= 0.07
    assert np.all((distance > 0.0) & (distance < 0.651))


def test_gum_screen_is_diffuse_without_an_invented_high_covering_sector():
    config = SkyConfig(nside=32, ray_oversample=1, sky_mode="ours")
    geometry = prepare_geometry(config)
    emission_measure, covering, _distance = make_local_partial_screens(
        config, geometry
    )
    vectors = np.asarray(
        hp.ang2vec(geometry["l_deg"], geometry["b_deg"], lonlat=True)
    )
    center = np.asarray(hp.ang2vec(258.0, -6.6, lonlat=True))
    separation = np.degrees(np.arccos(np.clip(vectors @ center, -1.0, 1.0)))
    interior = separation < 15.0
    inner_edge = (separation >= 20.0) & (separation < 22.0)
    outer_edge = (separation >= 22.0) & (separation < 24.0)
    exterior = (separation >= 24.0) & (separation < 26.0)

    assert np.max(covering[0]) < 0.37
    assert np.percentile(covering[0, interior], 90) - np.percentile(
        covering[0, interior], 10
    ) < 0.10
    assert np.median(covering[0, inner_edge]) > np.median(
        covering[0, outer_edge]
    ) > np.median(covering[0, exterior])
    assert np.percentile(emission_measure[0, inner_edge], 50) > 500.0


def test_source_templates_have_unit_solid_angle_integral():
    config = SkyConfig(nside=16)
    omega = hp.nside2pixarea(config.nside)
    for template in source_templates(config).values():
        np.testing.assert_allclose(np.sum(template) * omega, 1.0, rtol=1e-12)


def test_exact_point_source_alms_have_flat_cl_and_exact_additivity():
    frequencies = np.asarray([10.0, 50.0])
    amplitudes = {
        name: np.asarray([1.25 + i, 2.5 + i]) for i, name in enumerate(ATEAM)
    }
    lmax = 24
    individual = exact_ateam_alms(frequencies, amplitudes, lmax)
    for name, alm_cube in individual.items():
        for fi, amplitude in enumerate(amplitudes[name]):
            cl = hp.alm2cl(alm_cube[fi])
            np.testing.assert_allclose(cl, amplitude**2 / (4.0 * np.pi), rtol=2e-12)
    summed = np.sum(list(individual.values()), axis=0)
    rebuilt = np.zeros_like(summed)
    for alm_cube in individual.values():
        rebuilt += alm_cube
    np.testing.assert_allclose(summed, rebuilt, rtol=0.0, atol=0.0)


def test_local_shell_catalog_and_ray_geometry_are_reproducible_and_physical():
    config = SkyConfig(nside=8, n_distance=16, seed=11)
    first = make_local_shell_catalog(config)
    second = make_local_shell_catalog(config)
    np.testing.assert_allclose(first.distance_kpc, second.distance_kpc)
    assert config.min_local_shells <= len(first.distance_kpc) <= config.max_local_shells
    assert np.all(first.radius_kpc > first.thickness_kpc)
    segments, indices = shell_ray_segments(first, prepare_geometry(config), config)
    assert segments.shape == (
        len(first.distance_kpc),
        config.shell_distance_bins,
        hp.nside2npix(16),
    )
    assert np.all(segments >= 0.0)
    assert np.any(segments > 0.0)
    assert np.all((indices >= 0) & (indices < config.n_distance))


def test_our_sky_mode_conditions_named_loops_without_fixing_exact_parameters():
    config = SkyConfig(nside=8, sky_mode="ours", seed=13)
    catalog = make_local_shell_catalog(config)
    assert catalog.name.tolist() == ["LOOP_I_S1", "LOOP_I_S2", "LOOP_II", "LOOP_III", "LOOP_IV"]
    np.testing.assert_allclose(catalog.center_l_deg, [346, 347, 100, 124, 315], atol=6.0)
    assert np.all(catalog.distance_kpc > 0.0)
    assert np.all(catalog.low_frequency_beta_temperature > catalog.beta_temperature)
    assert np.all((-catalog.beta_temperature >= 2.55) & (-catalog.beta_temperature <= 3.03))
    assert np.all((catalog.excess_408_k >= 3.0) & (catalog.excess_408_k <= 9.0))
