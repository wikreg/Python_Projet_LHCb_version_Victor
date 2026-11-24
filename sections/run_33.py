from fonctions.sec31_32_donnees import configurer_style, charge_donnees
from fonctions.sec33_fit_masse import fit_masse, trace_fit

# Script exécutable pour la section 3.3 : sélection de base, ajustement de masse du D0 et tracé des pulls
# df_total -> DataFrame complet optionnel (si absent, on le recharge depuis le fichier .root)
# dossier -> dossier où sauvegarder les graphiques
def executer(df_total=None, dossier="histograms_plots"):
    # Initialise le style sombre et crée le dossier de sortie si nécessaire
    configurer_style(dossier)
    # Charge les données si le script est lancé seul sans passer df_total
    if df_total is None:
        df_total = charge_donnees("data/MasterclassData_2012_all.root")

    # Applique les coupures initiales : identification (kaon, pion), alignement (DIRA) et paramètre d'impact (IPCHI2)
    masque = (
        (df_total["Kplus_ProbNNk"] > 0.95) &
        (df_total["piminus_ProbNNpi"] > 0.8) &
        (df_total["D0_DIRA_OWNPV"] > 0.9998) &
        (df_total["D0_MINIPCHI2"] < 9)
    )
    df_propre = df_total[masque]

    # Ajuste la masse du D0 (gaussienne + droite) et récupère les paramètres ajustés
    result, model, sig_model, bkg_model, donnees_masse, mu, sigma, n_sig, n_bkg, S, B = fit_masse(df_propre)
    params = result.params
    mu_val = params[mu]["value"]
    mu_err = params[mu]["hesse"]["error"]
    n_sig_val = params[n_sig]["value"]
    n_bkg_val = params[n_bkg]["value"]

    # Affiche la masse mesurée, son incertitude et les effectifs de signal et de bruit
    print(f"\n[3.3] Masse du D0 : {mu_val:.2f} +/- {mu_err:.2f} MeV/c²")
    print(f"[3.3] Signal : {int(n_sig_val)} | Bruit : {int(n_bkg_val)}")

    # Trace le fit, les pulls et récupère le chi2 réduit
    comptes, centres, erreurs_data, largeur_bin, chi2_reduit = trace_fit(
        donnees_masse, model, bkg_model, mu_val, mu_err, n_sig_val, n_bkg_val, dossier=dossier
    )
    print(f"[3.3] Chi2 réduit : {chi2_reduit:.3f}")
    return df_propre, comptes, centres, erreurs_data, largeur_bin, chi2_reduit

# Permet d'exécuter cette section directement (ex. python -m sections.run_33)
if __name__ == "__main__":
    executer()