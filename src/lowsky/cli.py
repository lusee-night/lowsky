"""Console entry points for lowsky."""


def generate() -> None:
    """Generate a counterfactual low-frequency sky."""
    from counterfactual_ulsa import main

    main()


def power() -> None:
    """Compare diffuse and analytic point-source harmonic power."""
    from plot_harmonic_power import main

    main()


def validate() -> None:
    """Run the local-feature fidelity audit on a generated product."""
    from validate_feature_fidelity import main

    main()
