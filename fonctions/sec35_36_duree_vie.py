import numpy as np
import zfit
from fonctions.sec33_fit_masse import fit_donnees

# Fonction qui ajuste la durée de vie par une seule exponentielle (avec poids optionnels pour sPlot)
# df -> dataframe pandas contenant les évènements
# branche -> nom de la colonne de temps de vol ("D0_TAU")
# min_tau, max_tau -> bornes en ns (on coupe à 0.0004 ns pour éviter l'effet du trigger)
# nom_param -> suffixe ajouté aux noms des paramètres pour éviter les doublons
# colonne_poids -> nom de la colonne de poids (ex. "w_sig" en section 3.6), ou None pour un fit non pondéré
def fit_duree_vie_simple(df, branche="D0_TAU", min_tau=0.0004, max_tau=0.004, nom_param="", colonne_poids=None):
    # Garde uniquement les évènements dans la fenêtre de temps de vol choisie
    masque_tau = (df[branche] >= min_tau) & (df[branche] <= max_tau)
    donnees_tau = df[masque_tau][branche]

    # Si une colonne de poids est fournie (sPlot), on récupère les poids et la somme des poids
    if colonne_poids is not None:
        poids_tau = df[masque_tau][colonne_poids].to_numpy()
        n_total = np.sum(poids_tau)
    # Sinon, fit classique où chaque évènement compte pour 1
    else:
        poids_tau = None
        n_total = len(donnees_tau)

    # Définit l'espace d'observation en temps de vol
    obs = zfit.Space(branche, min_tau, max_tau)
    # Paramètre de décroissance lambda (en ns^-1) : tau = -1 / lam
    lam = zfit.Parameter(f"lam{nom_param}", -2400.0, -15000.0, -100.0)
    # Nombre total d'évènements (paramètre étendu)
    n_evts = zfit.Parameter(f"n_evts{nom_param}", n_total, 0, abs(n_total) * 2.0)

    # Modèle exponentiel zfit et lancement de l'ajustement
    model = zfit.pdf.Exponential(obs=obs, lam=lam, extended=n_evts)
    result = fit_donnees(donnees_tau, obs, model, poids=poids_tau)

    return result, model, donnees_tau, lam, n_evts

# Fonction qui ajuste la durée de vie avec deux exponentielles : signal (pente libre) + bruit (pente fixée)
# df -> dataframe pandas (sur la région du pic)
# lam_bkg_fixe -> pente lambda du bruit de fond déterminée sur les bandes latérales (section 3.5.2)
# branche, min_tau, max_tau -> variable et bornes en ns
def fit_duree_vie_complet(df, lam_bkg_fixe, branche="D0_TAU", min_tau=0.0004, max_tau=0.004):
    # Filtre les données dans l'intervalle de temps de vol
    masque_tau = (df[branche] >= min_tau) & (df[branche] <= max_tau)
    donnees_tau = df[masque_tau][branche]
    n_total = len(donnees_tau)

    obs = zfit.Space(branche, min_tau, max_tau)
    # Pente du signal laissée libre ; pente du bruit bloquée avec floating=False
    lam_sig = zfit.Parameter("lam_sig_final", -2400.0, -15000.0, -100.0)
    lam_bkg = zfit.Parameter("lam_bkg_final", lam_bkg_fixe, floating=False)

    # Estimations initiales du nombre d'évènements de signal et de bruit dans la fenêtre de temps
    n_sig = zfit.Parameter("n_sig_tau", 0.8 * n_total, 0, n_total * 1.5)
    n_bkg = zfit.Parameter("n_bkg_tau", 0.2 * n_total, 0, n_total * 1.5)

    # Somme des deux exponentielles (signal + bruit de fond)
    sig_model = zfit.pdf.Exponential(obs=obs, lam=lam_sig, extended=n_sig)
    bkg_model = zfit.pdf.Exponential(obs=obs, lam=lam_bkg, extended=n_bkg)
    model = zfit.pdf.SumPDF([sig_model, bkg_model])

    # Lance l'ajustement
    result = fit_donnees(donnees_tau, obs, model)
    return result, model, bkg_model, donnees_tau, lam_sig, n_sig, n_bkg