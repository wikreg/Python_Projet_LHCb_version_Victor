import numpy as np
import zfit
import matplotlib.pyplot as plt
from fonctions.sec31_32_donnees import COULEUR_SIGNAL, COULEUR_POINTS, COULEUR_FIT, COULEUR_BRUIT, COULEUR_PULLS, configurer_style
from fonctions.sec37_avance import fit_masse_parabole_bkg, calcul_sweights_manuel
from sections import run_33, run_34, run_36

def executer(df_propre=None, df_optimal=None, comptes=None, centres=None, erreurs_data=None, largeur_bin=None, chi2_reduit=None, dossier="histograms_plots"):
    configurer_style(dossier)
    if df_propre is None:
        df_propre, comptes, centres, erreurs_data, largeur_bin, chi2_reduit = run_33.executer(dossier=dossier)
    if df_optimal is None:
        df_optimal, _, _ = run_34.executer(dossier=dossier)

    # 3.7.1 : Signal Gaussienne + Bruit Parabole négative
    res_pb, mod_pb, bkg_pb, donnees_m_pb, mu_pb, sig_pb, c1_pb, c2_pb, ns_pb, nb_pb = fit_masse_parabole_bkg(df_propre)

    mu_pb_val = res_pb.params[mu_pb]["value"]
    mu_pb_err = res_pb.params[mu_pb]["hesse"]["error"]
    c2_pb_val = res_pb.params[c2_pb]["value"]
    n_tot_pb = res_pb.params[ns_pb]["value"] + res_pb.params[nb_pb]["value"]
    n_bkg_pb_val = res_pb.params[nb_pb]["value"]

    nbins = len(centres)
    pred_pb = mod_pb.pdf(centres).numpy() * n_tot_pb * largeur_bin
    somme_pulls_pb = 0.0
    for i in range(nbins):
        pull_i = (pred_pb[i] - comptes[i]) / erreurs_data[i]
        somme_pulls_pb = somme_pulls_pb + pull_i ** 2
    chi2_reduit_pb = somme_pulls_pb / (nbins - 6)

    print(f"\n[3.7.1] Masse D0 (Gauss + Parabole) : {mu_pb_val:.2f} +/- {mu_pb_err:.2f} MeV/c²")
    print(f"[3.7.1] Courbure parabole c2 : {c2_pb_val:.4f}")
    print(f"[3.7.1] Chi2 réduit (Gauss + Parabole) : {chi2_reduit_pb:.3f} (contre {chi2_reduit:.3f})")

    x_lisse = np.linspace(1816, 1914, 500)
    y_pb_lisse = mod_pb.pdf(x_lisse).numpy() * n_tot_pb * largeur_bin
    y_bkg_pb_lisse = bkg_pb.pdf(x_lisse).numpy() * n_bkg_pb_val * largeur_bin

    plt.figure(figsize=(8.5, 5))
    plt.fill_between(x_lisse, 0, y_bkg_pb_lisse, color=COULEUR_BRUIT, alpha=0.15)
    plt.fill_between(x_lisse, y_bkg_pb_lisse, y_pb_lisse, color=COULEUR_FIT, alpha=0.15)
    plt.errorbar(centres, comptes, yerr=erreurs_data, fmt="o", color=COULEUR_POINTS, markersize=3.8, label="Données sélectionnées")
    plt.plot(x_lisse, y_pb_lisse, color=COULEUR_FIT, linewidth=2.4, label=f"Fit Gauss + Parabole (χ²/ndf = {chi2_reduit_pb:.2f})")
    plt.plot(x_lisse, y_bkg_pb_lisse, color=COULEUR_BRUIT, linestyle="--", linewidth=1.8, label="Bruit de fond (Parabole)")
    plt.ylim(bottom=0)
    plt.title("Modèle amélioré : Gaussienne + Parabole négative")
    plt.xlabel("Masse D⁰ [MeV/c²]")
    plt.ylabel(f"Évènements / {largeur_bin:.2f} MeV/c²")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{dossier}/fit_masse_parabole.png")
    plt.close()

    # 3.7.2 : sWeights manuels vs hepstats
    res_m_splot, sig_m_splot, bkg_m_splot, donnees_m_splot, mu_sp, sig_sp, n_sig_sp, n_bkg_sp, w_sig, centres_sp, comptes_sig_sp, erreurs_sig_sp = run_36.executer(df_optimal=df_optimal, dossier=dossier)

    w_sig_manuel = calcul_sweights_manuel(
        sig_m_splot, bkg_m_splot, donnees_m_splot,
        res_m_splot.params[n_sig_sp]["value"], res_m_splot.params[n_bkg_sp]["value"]
    )
    print(f"[3.7.2] Écart max entre sWeights manuels et hepstats : {np.max(np.abs(w_sig_manuel - w_sig)):.2e}")

    # 3.7.3 : Soustraction des bandes latérales
    mu_sp_val = res_m_splot.params[mu_sp]["value"]
    sig_sp_val = res_m_splot.params[sig_sp]["value"]
    min_pic, max_pic = mu_sp_val - 3 * sig_sp_val, mu_sp_val + 3 * sig_sp_val
    max_bg_gauche, min_bg_droite = mu_sp_val - 4 * sig_sp_val, mu_sp_val + 4 * sig_sp_val

    int_bruit_pic = bkg_m_splot.integrate(limits=zfit.Space("D0_MM", min_pic, max_pic)).numpy()[0]
    int_bruit_side = (
        bkg_m_splot.integrate(limits=zfit.Space("D0_MM", 1816, max_bg_gauche)).numpy()[0] +
        bkg_m_splot.integrate(limits=zfit.Space("D0_MM", min_bg_droite, 1914)).numpy()[0]
    )
    alpha_sb = int_bruit_pic / int_bruit_side

    df_sb_pic = df_optimal[(df_optimal["D0_MM"] >= min_pic) & (df_optimal["D0_MM"] <= max_pic)]
    df_sb_side = df_optimal[((df_optimal["D0_MM"] >= 1816) & (df_optimal["D0_MM"] <= max_bg_gauche)) |
                            ((df_optimal["D0_MM"] >= min_bg_droite) & (df_optimal["D0_MM"] <= 1914))]

    h_pic, _ = np.histogram(df_sb_pic["D0_TAU"], bins=60, range=(0.0004, 0.004))
    h_side, _ = np.histogram(df_sb_side["D0_TAU"], bins=60, range=(0.0004, 0.004))

    h_sig_sb = h_pic - alpha_sb * h_side
    err_sig_sb = np.sqrt(h_pic + (alpha_sb ** 2) * h_side)

    plt.figure(figsize=(8, 5))
    masque_sig_pos = comptes_sig_sp > 0
    masque_sb_pos = h_sig_sb > 0
    plt.errorbar(centres_sp[masque_sig_pos], comptes_sig_sp[masque_sig_pos], yerr=erreurs_sig_sp[masque_sig_pos], fmt="o", color=COULEUR_SIGNAL, markersize=3.8, label="Méthode sPlot")
    plt.errorbar(centres_sp[masque_sb_pos], h_sig_sb[masque_sb_pos], yerr=err_sig_sb[masque_sb_pos], fmt="^", color=COULEUR_PULLS, markersize=4.2, label="Soustraction bandes latérales")
    plt.yscale("log")
    plt.ylim(bottom=0.5)
    plt.title("Extraction du signal D0_TAU : sPlot vs Bandes latérales")
    plt.xlabel("Temps de vol D0_TAU [ns]")
    plt.ylabel("Évènements de signal (échelle log)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{dossier}/comparaison_splot_vs_sideband.png")
    plt.close()

if __name__ == "__main__":
    executer()