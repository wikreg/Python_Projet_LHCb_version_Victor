import numpy as np
import zfit
from fonctions.sec33_fit_masse import fit_donnees


def fit_masse_parabole_bkg(df, branche="D0_MM", min_fit=1816, max_fit=1914):
    masque_plage = (df[branche] >= min_fit) & (df[branche] <= max_fit)
    donnees_masse = df[masque_plage][branche]
    n_total = len(donnees_masse)

    obs = zfit.Space(branche, min_fit, max_fit)

    mu = zfit.Parameter("mu_parab", 1865.0, 1850.0, 1880.0)
    sigma = zfit.Parameter("sigma_parab", 8.0, 1.0, 25.0)
    c1 = zfit.Parameter("c1_parab", -0.1, -0.5, 0.5)
    c2 = zfit.Parameter("c2_parab", -0.05, -0.5, 0.0)

    n_sig = zfit.Parameter("n_sig_parab", 0.7 * n_total, 0, n_total * 1.5)
    n_bkg = zfit.Parameter("n_bkg_parab", 0.3 * n_total, 0, n_total * 1.5)

    sig_model = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma, extended=n_sig)
    bkg_model = zfit.pdf.Chebyshev(obs=obs, coeffs=[c1, c2], extended=n_bkg)

    model = zfit.pdf.SumPDF([sig_model, bkg_model])
    result = fit_donnees(donnees_masse, obs, model)

    return result, model, bkg_model, donnees_masse, mu, sigma, c1, c2, n_sig, n_bkg


def calcul_sweights_manuel(sig_pdf, bkg_pdf, donnees_masse, N_S, N_B):
    m_vals = donnees_masse.to_numpy()
    f_S = sig_pdf.pdf(m_vals).numpy()
    f_B = bkg_pdf.pdf(m_vals).numpy()
    denom = N_S * f_S + N_B * f_B

    M_SS = np.sum((f_S * f_S) / (denom ** 2))
    M_SB = np.sum((f_S * f_B) / (denom ** 2))
    M_BB = np.sum((f_B * f_B) / (denom ** 2))

    determinant = M_SS * M_BB - M_SB ** 2
    V_SS = M_BB / determinant
    V_SB = -M_SB / determinant

    w_sig_manuel = (V_SS * f_S + V_SB * f_B) / denom
    return w_sig_manuel