import numpy as np
import zfit
from fonctions.sec33_fit_masse import fit_donnees


def fit_duree_vie_simple(df, branche="D0_TAU", min_tau=0.0004, max_tau=0.004, nom_param="", colonne_poids=None):
    masque_tau = (df[branche] >= min_tau) & (df[branche] <= max_tau)
    donnees_tau = df[masque_tau][branche]

    if colonne_poids is not None:
        poids_tau = df[masque_tau][colonne_poids].to_numpy()
        n_total = np.sum(poids_tau)
    else:
        poids_tau = None
        n_total = len(donnees_tau)

    obs = zfit.Space(branche, min_tau, max_tau)
    lam = zfit.Parameter(f"lam{nom_param}", -2400.0, -15000.0, -100.0)
    n_evts = zfit.Parameter(f"n_evts{nom_param}", n_total, 0, abs(n_total) * 2.0)

    model = zfit.pdf.Exponential(obs=obs, lam=lam, extended=n_evts)
    result = fit_donnees(donnees_tau, obs, model, poids=poids_tau)

    return result, model, donnees_tau, lam, n_evts


def fit_duree_vie_complet(df, lam_bkg_fixe, branche="D0_TAU", min_tau=0.0004, max_tau=0.004):
    masque_tau = (df[branche] >= min_tau) & (df[branche] <= max_tau)
    donnees_tau = df[masque_tau][branche]
    n_total = len(donnees_tau)

    obs = zfit.Space(branche, min_tau, max_tau)
    lam_sig = zfit.Parameter("lam_sig_final", -2400.0, -15000.0, -100.0)
    lam_bkg = zfit.Parameter("lam_bkg_final", lam_bkg_fixe, floating=False)

    n_sig = zfit.Parameter("n_sig_tau", 0.8 * n_total, 0, n_total * 1.5)
    n_bkg = zfit.Parameter("n_bkg_tau", 0.2 * n_total, 0, n_total * 1.5)

    sig_model = zfit.pdf.Exponential(obs=obs, lam=lam_sig, extended=n_sig)
    bkg_model = zfit.pdf.Exponential(obs=obs, lam=lam_bkg, extended=n_bkg)
    model = zfit.pdf.SumPDF([sig_model, bkg_model])

    result = fit_donnees(donnees_tau, obs, model)
    return result, model, bkg_model, donnees_tau, lam_sig, n_sig, n_bkg