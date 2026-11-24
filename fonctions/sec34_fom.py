import numpy as np
import matplotlib.pyplot as plt
from fonctions.sec31_32_donnees import COULEUR_FIT, COULEUR_PULLS
from fonctions.sec33_fit_masse import fit_masse

# Fonction qui teste plusieurs seuils de coupure sur une variable et calcule la Figure de Mérite (FOM)
# df -> dataframe pandas sur lequel on applique la coupure
# nom_variable -> nom de la colonne à tester (ex. "D0_DIRA_OWNPV" ou "D0_MINIPCHI2")
# liste_seuils -> liste des valeurs de seuil à essayer
# type_coupure -> "superieur" (on garde > seuil) ou "inferieur" (on garde < seuil)
# suffixe_fichier -> texte ajouté au nom du fichier image (ex. "_sans_PID")
# dossier -> dossier d'enregistrement du graphique
def FOM_variable(df, nom_variable, liste_seuils, type_coupure="superieur", suffixe_fichier="", dossier="histograms_plots"):
    # Listes et variables pour stocker l'évolution de la FOM et trouver le meilleur seuil
    liste_fom = []
    meilleure_fom = 0.0
    meilleur_seuil = liste_seuils[0]

    # Boucle sur chaque seuil candidat
    for i in range(len(liste_seuils)):
        seuil = liste_seuils[i]
        # Filtre les données selon le sens de la coupure choisi
        if type_coupure == "superieur":
            df_test = df[df[nom_variable] > seuil]
        else:
            df_test = df[df[nom_variable] < seuil]

        # Lance le fit de masse sur les données filtrées pour récupérer S (signal) et B (bruit sous le pic)
        result, model, sig_model, bkg_model, donnees_masse, mu, sigma, n_sig, n_bkg, S, B = fit_masse(
            df_test, nom_param=f"_{nom_variable}{suffixe_fichier}_{i}"
        )

        # Formule de la Figure de Mérite : maximise le rapport signal sur incertitude statistique totale
        fom = S / np.sqrt(S + B)
        liste_fom.append(fom)

        # Met à jour le meilleur seuil si on trouve une FOM plus élevée
        if fom > meilleure_fom:
            meilleure_fom = fom
            meilleur_seuil = seuil

    # Création de la figure pour visualiser la courbe FOM en fonction du seuil
    plt.figure(figsize=(8, 5))
    # Trace la courbe avec des marqueurs ronds et remplit légèrement en dessous
    plt.plot(liste_seuils, liste_fom, marker="o", color=COULEUR_FIT, linewidth=2.4, label=f"FOM ({nom_variable})")
    plt.fill_between(liste_seuils, liste_fom, min(liste_fom) * 0.99, color=COULEUR_FIT, alpha=0.14)
    # Repère vertical et point mis en évidence sur le seuil optimal
    plt.axvline(meilleur_seuil, color=COULEUR_PULLS, linestyle="--", linewidth=1.3)
    plt.scatter([meilleur_seuil], [meilleure_fom], color=COULEUR_PULLS, s=80, zorder=5, label=f"Optimum : {meilleur_seuil} (FOM = {meilleure_fom:.1f})")
    # Ajuste un peu les bornes verticales pour bien voir la variation de la FOM
    plt.ylim(min(liste_fom) * 0.99, max(liste_fom) * 1.01)
    plt.title(f"Optimisation de la sélection : FOM pour {nom_variable}{suffixe_fichier}")
    plt.xlabel(f"Seuil sur {nom_variable}")
    plt.ylabel("FOM = S / √(S + B)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Enregistre le graphique et ferme la figure pour libérer la mémoire
    plt.savefig(f"{dossier}/FOM_{nom_variable}{suffixe_fichier}.png")
    plt.close()

    return meilleur_seuil, meilleure_fom