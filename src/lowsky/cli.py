"""Console entry points for lowsky."""


def generate() -> None:
    """Generate a counterfactual low-frequency sky."""
    from .pipeline import main

    main()


def power() -> None:
    """Compare diffuse and analytic point-source harmonic power."""
    from .power import main

    main()


def validate() -> None:
    """Run the local-feature fidelity audit on a generated product."""
    from .validation import main

    main()
