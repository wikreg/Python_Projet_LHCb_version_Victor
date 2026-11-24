import numpy as np
import matplotlib.pyplot as plt
from fonctions.sec31_32_donnees import COULEUR_FIT, COULEUR_PULLS
from fonctions.sec33_fit_masse import fit_masse


def FOM_variable(df, nom_variable, liste_seuils, type_coupure="superieur", suffixe_fichier="", dossier="histograms_plots"):
    liste_fom = []
    meilleure_fom = 0.0
    meilleur_seuil = liste_seuils[0]

    for i in range(len(liste_seuils)):
        seuil = liste_seuils[i]
        if type_coupure == "superieur":
            df_test = df[df[nom_variable] > seuil]
        else:
            df_test = df[df[nom_variable] < seuil]

        result, model, sig_model, bkg_model, donnees_masse, mu, sigma, n_sig, n_bkg, S, B = fit_masse(
            df_test, nom_param=f"_{nom_variable}{suffixe_fichier}_{i}"
        )

        fom = S / np.sqrt(S + B)
        liste_fom.append(fom)

        if fom > meilleure_fom:
            meilleure_fom = fom
            meilleur_seuil = seuil

    plt.figure(figsize=(8, 5))
    plt.plot(liste_seuils, liste_fom, marker="o", color=COULEUR_FIT, linewidth=2.4, label=f"FOM ({nom_variable})")
    plt.fill_between(liste_seuils, liste_fom, min(liste_fom) * 0.99, color=COULEUR_FIT, alpha=0.14)
    plt.axvline(meilleur_seuil, color=COULEUR_PULLS, linestyle="--", linewidth=1.3)
    plt.scatter([meilleur_seuil], [meilleure_fom], color=COULEUR_PULLS, s=80, zorder=5, label=f"Optimum : {meilleur_seuil} (FOM = {meilleure_fom:.1f})")
    plt.ylim(min(liste_fom) * 0.99, max(liste_fom) * 1.01)
    plt.title(f"Optimisation de la sélection : FOM pour {nom_variable}{suffixe_fichier}")
    plt.xlabel(f"Seuil sur {nom_variable}")
    plt.ylabel("FOM = S / √(S + B)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{dossier}/FOM_{nom_variable}{suffixe_fichier}.png")
    plt.close()

    return meilleur_seuil, meilleure_fom