import lowsky


def test_public_api_and_version():
    assert lowsky.__version__ == "0.1.0"
    assert lowsky.SkyConfig().sky_mode == "random"
    assert callable(lowsky.generate)
    assert callable(lowsky.generate_sky)
    assert callable(lowsky.prepare_sky_inputs)
    assert callable(lowsky.power_from_alms)
