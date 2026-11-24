import numpy as np
import matplotlib.pyplot as plt
from hepstats.splot import compute_sweights
from fonctions.sec31_32_donnees import COULEUR_SIGNAL, COULEUR_POINTS, COULEUR_FIT, COULEUR_BRUIT, configurer_style, charge_donnees
from fonctions.sec33_fit_masse import fit_masse
from fonctions.sec35_36_duree_vie import fit_duree_vie_simple

# Script exécutable pour la section 3.6 : séparation signal/bruit par la méthode sPlot et fit de durée de vie
# df_optimal -> DataFrame filtré optionnel (si absent, on recharge et applique des coupures par défaut)
# dossier -> dossier où sauvegarder les graphiques
def executer(df_optimal=None, dossier="histograms_plots"):
    # Initialise le style sombre et crée le dossier de sortie si besoin
    configurer_style(dossier)
    if df_optimal is None:
        df_total = charge_donnees("data/MasterclassData_2012_all.root")
        df_optimal = df_total[
            (df_total["Kplus_ProbNNk"] > 0.95) & (df_total["piminus_ProbNNpi"] > 0.8) &
            (df_total["D0_DIRA_OWNPV"] > 0.9995) & (df_total["D0_MINIPCHI2"] < 25)
        ]

    # Ajuste d'abord la masse invariante pour obtenir le modèle total et les effectifs ajustés (variable discriminante)
    res_m_splot, mod_m_splot, sig_m_splot, bkg_m_splot, donnees_m_splot, mu_sp, sig_sp, n_sig_sp, n_bkg_sp, S_sp, B_sp = fit_masse(
        df_optimal, min_fit=1816, max_fit=1914, nom_param="_splot"
    )

    # Calcule les sWeights avec hepstats et extrait les poids associés au signal et au bruit
    poids_splot = compute_sweights(mod_m_splot, donnees_m_splot.to_numpy())
    w_sig = poids_splot[n_sig_sp]
    w_bkg = poids_splot[n_bkg_sp]

    # Vérifie que la somme des sWeights retrouve bien les nombres d'évènements ajustés du fit de masse
    print(f"\n[3.6.1] Somme poids signal : {np.sum(w_sig):.1f} | n_sig fit : {res_m_splot.params[n_sig_sp]['value']:.1f}")
    print(f"[3.6.1] Somme poids bruit  : {np.sum(w_bkg):.1f} | n_bkg fit : {res_m_splot.params[n_bkg_sp]['value']:.1f}")

    # Ajoute les colonnes de poids au sous-DataFrame dans la fenêtre de masse
    df_splot = df_optimal[(df_optimal["D0_MM"] >= 1816) & (df_optimal["D0_MM"] <= 1914)].copy()
    df_splot["w_sig"] = w_sig
    df_splot["w_bkg"] = w_bkg

    # Bornes en ns et découpage en barres pour le temps de vol D0_TAU
    min_tau, max_tau, nbins_tau = 0.0004, 0.004, 60
    largeur_bin_tau = (max_tau - min_tau) / nbins_tau
    masque_tau_sp = (df_splot["D0_TAU"] >= min_tau) & (df_splot["D0_TAU"] <= max_tau)

    tau_vals_sp = df_splot[masque_tau_sp]["D0_TAU"].to_numpy()
    w_sig_tau = df_splot[masque_tau_sp]["w_sig"].to_numpy()
    w_bkg_tau = df_splot[masque_tau_sp]["w_bkg"].to_numpy()

    # Construit les histogrammes pondérés pour le signal et le bruit ; l'incertitude est sqrt(somme des w^2)
    comptes_sig_sp, bords_sp = np.histogram(tau_vals_sp, bins=nbins_tau, range=(min_tau, max_tau), weights=w_sig_tau)
    somme_w2_sig, _ = np.histogram(tau_vals_sp, bins=nbins_tau, range=(min_tau, max_tau), weights=w_sig_tau ** 2)
    erreurs_sig_sp = np.sqrt(somme_w2_sig)

    comptes_bkg_sp, _ = np.histogram(tau_vals_sp, bins=nbins_tau, range=(min_tau, max_tau), weights=w_bkg_tau)
    somme_w2_bkg, _ = np.histogram(tau_vals_sp, bins=nbins_tau, range=(min_tau, max_tau), weights=w_bkg_tau ** 2)
    erreurs_bkg_sp = np.sqrt(somme_w2_bkg)

    centres_sp = np.array([(bords_sp[i] + bords_sp[i + 1]) / 2 for i in range(nbins_tau)])
    masque_sig_pos = comptes_sig_sp > 0
    masque_bkg_pos = comptes_bkg_sp > 0

    # Trace les distributions séparées de D0_TAU (signal extrait vs bruit extrait)
    plt.figure(figsize=(8, 5))
    plt.errorbar(centres_sp[masque_sig_pos], comptes_sig_sp[masque_sig_pos], yerr=erreurs_sig_sp[masque_sig_pos], fmt="o", color=COULEUR_SIGNAL, markersize=3.8, label="Signal seul (sPlot)")
    plt.errorbar(centres_sp[masque_bkg_pos], comptes_bkg_sp[masque_bkg_pos], yerr=erreurs_bkg_sp[masque_bkg_pos], fmt="s", color=COULEUR_BRUIT, markersize=3.8, label="Bruit de fond seul (sPlot)")
    plt.yscale("log")
    plt.ylim(bottom=0.5)
    plt.title("Distribution de D0_TAU séparée par la méthode sPlot")
    plt.xlabel("Temps de vol D0_TAU [ns]")
    plt.ylabel("Somme des poids (échelle log)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{dossier}/splot_tau_signal_vs_bruit.png")
    plt.close()

    # 3.6.4 : ajuste une seule exponentielle sur le signal extrait par sPlot (pondéré par w_sig)
    res_tau_sp, mod_tau_sp, donnees_tau_sp, lam_sp, n_evts_sp = fit_duree_vie_simple(
        df_splot, min_tau=min_tau, max_tau=max_tau, nom_param="_splot_tau", colonne_poids="w_sig"
    )
    lam_sp_val = res_tau_sp.params[lam_sp]["value"]
    lam_sp_err = res_tau_sp.params[lam_sp]["hesse"]["error"]
    n_sp_val = res_tau_sp.params[n_evts_sp]["value"]

    # Conversion de lambda (ns^-1) vers tau (ps)
    tau_splot_ps = (-1.0 / lam_sp_val) * 1000.0
    err_tau_splot_ps = (lam_sp_err / (lam_sp_val ** 2)) * 1000.0
    print(f"[3.6.4] Durée de vie sPlot (Signal seul) : {tau_splot_ps:.4f} +/- {err_tau_splot_ps:.4f} ps")

    # Trace l'ajustement exponentiel sur les évènements de signal pondérés par sPlot
    x_tau_lisse = np.linspace(min_tau, max_tau, 500)
    y_tau_sp_lisse = mod_tau_sp.pdf(x_tau_lisse).numpy() * n_sp_val * largeur_bin_tau

    plt.figure(figsize=(8, 5))
    plt.fill_between(x_tau_lisse, 0.5, y_tau_sp_lisse, color=COULEUR_FIT, alpha=0.14)
    plt.errorbar(centres_sp[masque_sig_pos], comptes_sig_sp[masque_sig_pos], yerr=erreurs_sig_sp[masque_sig_pos], fmt="o", color=COULEUR_POINTS, markersize=3.8, label="Signal extrait (sWeights)")
    plt.plot(x_tau_lisse, y_tau_sp_lisse, color=COULEUR_FIT, linewidth=2.4, label=f"Fit sPlot 1 exp (τ = {tau_splot_ps:.4f} ± {err_tau_splot_ps:.4f} ps)")
    plt.yscale("log")
    plt.ylim(bottom=0.5)
    plt.title("Fit de la durée de vie du D⁰ via sPlot (Signal seul)")
    plt.xlabel("Temps de vol D0_TAU [ns]")
    plt.ylabel("Évènements pondérés (échelle log)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{dossier}/fit_tau_splot.png")
    plt.close()

    # Renvoie les objets nécessaires à la section 3.7 (comparaison sPlot manuel et bandes latérales)
    return res_m_splot, sig_m_splot, bkg_m_splot, donnees_m_splot, mu_sp, sig_sp, n_sig_sp, n_bkg_sp, w_sig, centres_sp, comptes_sig_sp, erreurs_sig_sp

# Permet d'exécuter cette section directement (ex. python -m sections.run_36)
if __name__ == "__main__":
    executer()