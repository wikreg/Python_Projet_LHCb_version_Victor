import numpy as np
import zfit
from fonctions.sec33_fit_masse import fit_donnees

# Fonction de la section 3.7 : ajuste la masse du D0 avec une gaussienne simple (signal)
# et une parabole négative (polynôme de Chebyshev d'ordre 2 avec c2 <= 0) pour le bruit de fond
# df -> dataframe pandas
# branche -> nom de la colonne de masse ("D0_MM")
# min_fit, max_fit -> bornes en MeV/c^2 de l'ajustement
def fit_masse_parabole_bkg(df, branche="D0_MM", min_fit=1816, max_fit=1914):
    # Garde les évènements compris dans la fenêtre de masse
    masque_plage = (df[branche] >= min_fit) & (df[branche] <= max_fit)
    donnees_masse = df[masque_plage][branche]
    n_total = len(donnees_masse)

    # Définit l'espace d'observation
    obs = zfit.Space(branche, min_fit, max_fit)

    # Paramètres du pic gaussien (signal)
    mu = zfit.Parameter("mu_parab", 1865.0, 1850.0, 1880.0)
    sigma = zfit.Parameter("sigma_parab", 8.0, 1.0, 25.0)
    # Paramètres de la parabole (Chebyshev ordre 2) : c2 borné jusqu'à 0.0 pour une parabole négative
    c1 = zfit.Parameter("c1_parab", -0.1, -0.5, 0.5)
    c2 = zfit.Parameter("c2_parab", -0.05, -0.5, 0.0)

    # Estimations initiales du nombre d'évènements de signal et de bruit
    n_sig = zfit.Parameter("n_sig_parab", 0.7 * n_total, 0, n_total * 1.5)
    n_bkg = zfit.Parameter("n_bkg_parab", 0.3 * n_total, 0, n_total * 1.5)

    # Construit le modèle étendu : Gaussienne + Chebyshev ordre 2
    sig_model = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma, extended=n_sig)
    bkg_model = zfit.pdf.Chebyshev(obs=obs, coeffs=[c1, c2], extended=n_bkg)

    model = zfit.pdf.SumPDF([sig_model, bkg_model])
    # Lance l'ajustement via la fonction générique
    result = fit_donnees(donnees_masse, obs, model)

    return result, model, bkg_model, donnees_masse, mu, sigma, c1, c2, n_sig, n_bkg

# Fonction de la section 3.7.2 : calcule les sWeights à la main (sans hepstats)
# par inversion directe de la matrice 2x2 des moments
# sig_pdf, bkg_pdf -> densités de probabilité du signal et du bruit (zfit)
# donnees_masse -> valeurs de masse des évènements
# N_S, N_B -> nombres ajustés d'évènements de signal et de bruit
def calcul_sweights_manuel(sig_pdf, bkg_pdf, donnees_masse, N_S, N_B):
    # Évalue les PDF du signal et du bruit pour chaque évènement
    m_vals = donnees_masse.to_numpy()
    f_S = sig_pdf.pdf(m_vals).numpy()
    f_B = bkg_pdf.pdf(m_vals).numpy()
    # Dénominateur commun : densité totale attendue pour chaque évènement
    denom = N_S * f_S + N_B * f_B

    # Éléments de la matrice M (2x2)
    M_SS = np.sum((f_S * f_S) / (denom ** 2))
    M_SB = np.sum((f_S * f_B) / (denom ** 2))
    M_BB = np.sum((f_B * f_B) / (denom ** 2))

    # Inverse analytique de la matrice 2x2 pour obtenir la matrice de covariance V = M^(-1)
    determinant = M_SS * M_BB - M_SB ** 2
    V_SS = M_BB / determinant
    V_SB = -M_SB / determinant

    # Formule analytique du poids sWeight de signal pour chaque évènement
    w_sig_manuel = (V_SS * f_S + V_SB * f_B) / denom
    return w_sig_manuel