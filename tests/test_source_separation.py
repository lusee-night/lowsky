import numpy as np
import healpy as hp

from examples.one_night_source_separation import (
    fourier_nuisance_basis,
    recover_spectra,
    truncate_alms,
)


def test_truncate_alms_remaps_healpy_m_major_packing():
    alms = np.arange(30).reshape(2, 15)
    ell, emm = hp.Alm.getlm(2)
    source_indices = hp.Alm.getidx(4, ell, emm)
    assert np.array_equal(truncate_alms(alms, 2), alms[:, source_indices])
    assert not np.array_equal(source_indices, np.arange(6))


def test_fourier_basis_is_orthonormal():
    basis = fourier_nuisance_basis(41, 3)
    np.testing.assert_allclose(basis.T @ basis, np.eye(7), atol=1e-12)


def test_recover_spectra_ignores_unknown_fourier_foreground():
    rng = np.random.default_rng(2)
    nt, nf, ns = 61, 5, 3
    nuisance = fourier_nuisance_basis(nt, 2)
    templates = rng.normal(size=(ns, nt, nf))
    truth = rng.uniform(0.5, 2, size=(ns, nf))
    foreground = nuisance @ rng.normal(size=(nuisance.shape[1], nf))
    total = foreground + np.einsum("stf,sf->tf", templates, truth)
    recovered, reconstructed, condition = recover_spectra(total, templates, 2)
    np.testing.assert_allclose(recovered, truth, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(reconstructed, np.einsum("stf,sf->tf", templates, truth))
    assert np.all(np.isfinite(condition))


def test_recover_spectra_combines_multiple_channels():
    rng = np.random.default_rng(3)
    nt, nc, nf, ns = 45, 4, 3, 2
    templates = rng.normal(size=(ns, nt, nc, nf))
    truth = rng.uniform(1, 3, size=(ns, nf))
    nuisance = fourier_nuisance_basis(nt, 2)
    foreground = np.einsum("tk,kcf->tcf", nuisance,
                            rng.normal(size=(nuisance.shape[1], nc, nf)))
    total = foreground + np.einsum("stcf,sf->tcf", templates, truth)
    recovered, reconstructed, condition = recover_spectra(total, templates, 2)
    np.testing.assert_allclose(recovered, truth, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(reconstructed, np.einsum("stcf,sf->tcf", templates, truth))
    assert np.all(np.isfinite(condition))
