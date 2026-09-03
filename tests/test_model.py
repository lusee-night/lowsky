import jax
import jax.numpy as jnp
import numpy as np

from lowsky.model import SkyInputs, SkyParameters, generate_sky, generate_sky_components


def small_inputs() -> SkyInputs:
    pixels, distances = 3, 4
    return SkyInputs(
        emissivity_408=jnp.full((pixels, distances), 2.0),
        emission_measure_rate=jnp.full((pixels, distances), 0.4),
        fixed_emission_measure_rate=jnp.zeros((pixels, distances)),
        synchrotron_random_field=jnp.asarray(
            [[-1.0, 0.0, 1.0, 0.5], [0.5, -0.5, 1.0, -1.0], [1.0, 0.5, 0.0, -0.5]]
        ),
        emission_measure_random_field=jnp.asarray(
            [[0.5, -0.5, 0.0, 1.0], [0.0, 1.0, -1.0, 0.5], [-0.5, 0.0, 1.0, -1.0]]
        ),
        spectral_index=jnp.asarray([-2.45, -2.55, -2.65]),
        spectral_index_random_field=jnp.asarray([-1.0, 0.0, 1.0]),
        shell_emission_408=jnp.full((2, 2, pixels), 0.1),
        shell_foreground_emission_measure=jnp.full((2, 2, pixels), 0.3),
        shell_spectral_index=jnp.asarray([-2.7, -2.9]),
        shell_low_frequency_spectral_index=jnp.asarray([-2.5, -2.6]),
        shell_distance_kpc=jnp.full((2, 2, pixels), 0.15),
        shell_foreground_distance_index=-jnp.ones((2, 2, pixels), dtype=jnp.int32),
        partial_screen_emission_measure=jnp.zeros((1, pixels)),
        partial_screen_covering_fraction=jnp.zeros((1, pixels)),
        partial_screen_distance_kpc=jnp.full((1, pixels), 0.5),
        distance_midpoint_kpc=jnp.asarray([0.125, 0.375, 0.625, 0.875]),
        distance_step_kpc=jnp.asarray(0.25),
    )


def test_generate_sky_returns_frequency_by_pixel_jax_array():
    frequencies = jnp.asarray([5.0, 15.0, 40.0])
    sky = generate_sky(frequencies, small_inputs())

    assert isinstance(sky, jax.Array)
    assert sky.shape == (3, 3)
    assert np.all(np.asarray(sky) > 0.0)

    components = generate_sky_components(frequencies, small_inputs())
    np.testing.assert_allclose(
        components.total,
        components.stochastic_synchrotron
        + components.local_shells
        + components.free_free
        + components.extragalactic,
        rtol=1e-6,
    )


def test_generate_sky_is_jittable_and_differentiable_in_model_parameters():
    frequencies = jnp.asarray([10.0, 30.0])
    inputs = small_inputs()
    base = SkyParameters()

    def mean_temperature(emissivity_scale):
        parameters = base._replace(emissivity_scale=emissivity_scale)
        return jnp.mean(generate_sky(frequencies, inputs, parameters))

    eager = mean_temperature(jnp.asarray(1.1))
    compiled = jax.jit(mean_temperature)(jnp.asarray(1.1))
    gradient = jax.grad(mean_temperature)(jnp.asarray(1.1))

    np.testing.assert_allclose(eager, compiled, rtol=1e-6)
    assert np.isfinite(float(gradient))
    assert float(gradient) > 0.0


def test_random_realization_amplitudes_and_synchrotron_shape_are_differentiable():
    frequencies = jnp.asarray([10.0, 30.0])
    inputs = small_inputs()
    base = SkyParameters()

    def mean_temperature(values):
        parameters = base._replace(
            synchrotron_fluctuation_sigma=values[0],
            emission_measure_fluctuation_sigma=values[1],
            spectral_index_fluctuation_sigma=values[2],
            synchrotron_spectral_curvature=values[3],
            local_shell_scale=values[4],
            local_shell_spectral_index_offset=values[5],
        )
        return jnp.mean(generate_sky(frequencies, inputs, parameters))

    values = jnp.asarray([0.28, 0.32, 0.055, 0.0, 1.0, 0.0])
    gradient = jax.jit(jax.grad(mean_temperature))(values)

    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.all(np.asarray(gradient) != 0.0)


def test_precomputed_additive_components_stay_inside_jax_graph():
    frequencies = jnp.asarray([12.0, 24.0])
    inputs = small_inputs()
    baseline = generate_sky(frequencies, inputs)
    additive = jnp.full_like(baseline, 7.0)

    np.testing.assert_allclose(
        generate_sky(frequencies, inputs, additive_temperature=additive),
        baseline + 7.0,
    )


def test_opaque_partial_screen_beam_averages_in_transmission_space():
    inputs = small_inputs()._replace(
        emissivity_408=jnp.zeros((3, 4)),
        emission_measure_rate=jnp.zeros((3, 4)),
        shell_emission_408=jnp.zeros((2, 2, 3)),
        partial_screen_emission_measure=jnp.full((1, 3), 100.0),
        partial_screen_covering_fraction=jnp.full((1, 3), 0.1),
    )
    components = generate_sky_components(jnp.asarray([1.0]), inputs)
    # The filament itself is opaque, but covers only 10% of each beam. Its
    # thermal contribution is therefore 0.1 Te, not a full 8000-K screen.
    np.testing.assert_allclose(components.free_free[0], 800.0, rtol=1e-5)
