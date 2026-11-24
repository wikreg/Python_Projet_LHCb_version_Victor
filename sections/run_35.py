import numpy as np
import matplotlib.pyplot as plt
from fonctions.sec31_32_donnees import COULEUR_POINTS, COULEUR_FIT, COULEUR_BRUIT, configurer_style, charge_donnees
from fonctions.sec35_36_duree_vie import fit_duree_vie_simple, fit_duree_vie_complet

def executer(df_optimal=None, dossier="histograms_plots"):
    configurer_style(dossier)
    if df_optimal is None:
        df_total = charge_donnees("data/MasterclassData_2012_all.root")
        df_optimal = df_total[
            (df_total["Kplus_ProbNNk"] > 0.95) & (df_total["piminus_ProbNNpi"] > 0.8) &
            (df_total["D0_DIRA_OWNPV"] > 0.9995) & (df_total["D0_MINIPCHI2"] < 25)
        ]

    df_tau_pic = df_optimal[(df_optimal["D0_MM"] >= 1850) & (df_optimal["D0_MM"] <= 1882)]
    df_tau_bruit = df_optimal[((df_optimal["D0_MM"] >= 1815) & (df_optimal["D0_MM"] <= 1835)) |
                              ((df_optimal["D0_MM"] >= 1895) & (df_optimal["D0_MM"] <= 1915))]

    min_tau, max_tau, nbins_tau = 0.0004, 0.004, 60
    largeur_bin_tau = (max_tau - min_tau) / nbins_tau

    # 3.5.1 : 1 exponentielle
    res_simple, mod_simple, donnees_tau_pic, lam_simple, n_evts_simple = fit_duree_vie_simple(
        df_tau_pic, min_tau=min_tau, max_tau=max_tau, nom_param="_simple"
    )
    lam_s_val = res_simple.params[lam_simple]["value"]
    lam_s_err = res_simple.params[lam_simple]["hesse"]["error"]
    n_s_val = res_simple.params[n_evts_simple]["value"]

    tau_simple_ps = (-1.0 / lam_s_val) * 1000.0
    err_tau_simple_ps = (lam_s_err / (lam_s_val ** 2)) * 1000.0
    print(f"\n[3.5.1] Durée de vie (1 exp) : {tau_simple_ps:.4f} +/- {err_tau_simple_ps:.4f} ps")

    comptes_tau, bords_tau = np.histogram(donnees_tau_pic, bins=nbins_tau, range=(min_tau, max_tau))
    centres_tau, erreurs_tau = [], []
    for i in range(nbins_tau):
        centres_tau.append((bords_tau[i] + bords_tau[i + 1]) / 2)
        erreurs_tau.append(np.sqrt(comptes_tau[i]) if comptes_tau[i] > 0 else 1.0)
    centres_tau = np.array(centres_tau)
    erreurs_tau = np.array(erreurs_tau)

    x_tau_lisse = np.linspace(min_tau, max_tau, 500)
    y_tau_simple_lisse = mod_simple.pdf(x_tau_lisse).numpy() * n_s_val * largeur_bin_tau
    masque_non_nul = comptes_tau > 0

    plt.figure(figsize=(8, 5))
    plt.fill_between(x_tau_lisse, 0.5, y_tau_simple_lisse, color=COULEUR_FIT, alpha=0.14)
    plt.errorbar(centres_tau[masque_non_nul], comptes_tau[masque_non_nul], yerr=erreurs_tau[masque_non_nul], fmt="o", color=COULEUR_POINTS, markersize=3.8, label="Données région pic")
    plt.plot(x_tau_lisse, y_tau_simple_lisse, color=COULEUR_FIT, linewidth=2.4, label=f"Fit 1 exp (τ = {tau_simple_ps:.4f} ± {err_tau_simple_ps:.4f} ps)")
    plt.yscale("log")
    plt.ylim(bottom=0.5)
    plt.title("Fit de la durée de vie du D⁰ (une seule exponentielle)")
    plt.xlabel("Temps de vol D0_TAU [ns]")
    plt.ylabel("Évènements (échelle log)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{dossier}/fit_tau_simple.png")
    plt.close()

    # 3.5.2 : Pente du bruit
    res_bkg, mod_bkg_tau, donnees_tau_bkg, lam_bkg, n_evts_bkg = fit_duree_vie_simple(
        df_tau_bruit, min_tau=min_tau, max_tau=max_tau, nom_param="_bkg"
    )
    lam_bkg_val = res_bkg.params[lam_bkg]["value"]
    print(f"[3.5.2] Pente lam du bruit de fond : {lam_bkg_val:.1f} ns^-1")

    # 3.5.3 : Fit complet Signal + Bruit
    res_comp, mod_comp, bkg_comp, donnees_tau_pic, lam_sig, n_sig_tau, n_bkg_tau = fit_duree_vie_complet(
        df_tau_pic, lam_bkg_val, min_tau=min_tau, max_tau=max_tau
    )
    lam_sig_val = res_comp.params[lam_sig]["value"]
    lam_sig_err = res_comp.params[lam_sig]["hesse"]["error"]
    n_sig_t_val = res_comp.params[n_sig_tau]["value"]
    n_bkg_t_val = res_comp.params[n_bkg_tau]["value"]

    tau_final_ps = (-1.0 / lam_sig_val) * 1000.0
    err_tau_final_ps = (lam_sig_err / (lam_sig_val ** 2)) * 1000.0
    print(f"[3.5.3] Durée de vie finale (Signal + Bruit) : {tau_final_ps:.4f} +/- {err_tau_final_ps:.4f} ps")

    y_tau_comp_lisse = mod_comp.pdf(x_tau_lisse).numpy() * (n_sig_t_val + n_bkg_t_val) * largeur_bin_tau
    y_tau_bkg_lisse = bkg_comp.pdf(x_tau_lisse).numpy() * n_bkg_t_val * largeur_bin_tau

    plt.figure(figsize=(8, 5))
    plt.fill_between(x_tau_lisse, 0.5, y_tau_bkg_lisse, color=COULEUR_BRUIT, alpha=0.14)
    plt.fill_between(x_tau_lisse, y_tau_bkg_lisse, y_tau_comp_lisse, color=COULEUR_FIT, alpha=0.14)
    plt.errorbar(centres_tau[masque_non_nul], comptes_tau[masque_non_nul], yerr=erreurs_tau[masque_non_nul], fmt="o", color=COULEUR_POINTS, markersize=3.8, label="Données région pic")
    plt.plot(x_tau_lisse, y_tau_comp_lisse, color=COULEUR_FIT, linewidth=2.4, label=f"Fit total (τ = {tau_final_ps:.4f} ± {err_tau_final_ps:.4f} ps)")
    plt.plot(x_tau_lisse, y_tau_bkg_lisse, color=COULEUR_BRUIT, linestyle="--", linewidth=1.8, label="Contribution bruit de fond")
    plt.yscale("log")
    plt.ylim(bottom=0.5)
    plt.title("Fit de la durée de vie du D⁰ (Signal + Bruit de fond)")
    plt.xlabel("Temps de vol D0_TAU [ns]")
    plt.ylabel("Évènements (échelle log)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{dossier}/fit_tau_complet.png")
    plt.close()

if __name__ == "__main__":
    executer()