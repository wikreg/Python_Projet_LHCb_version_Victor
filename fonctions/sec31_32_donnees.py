import os
import uproot
import matplotlib.pyplot as plt

# Palette et configuration graphique commune a tout le projet
COULEUR_FOND = "#0d1117"
COULEUR_CADRE = "#161b22"
COULEUR_SIGNAL = "#38bdf8"
COULEUR_POINTS = "#7dd3fc"
COULEUR_FIT = "#c084fc"
COULEUR_BRUIT = "#fb7185"
COULEUR_PULLS = "#2dd4bf"

def configurer_style(dossier="histograms_plots"):
    os.makedirs(dossier, exist_ok=True)
    plt.style.use("dark_background")
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["figure.facecolor"] = COULEUR_FOND
    plt.rcParams["axes.facecolor"] = COULEUR_CADRE
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["grid.alpha"] = 0.25


def charge_donnees(chemin_fichier, branches=None):
    fichier = uproot.open(chemin_fichier)
    arbre = fichier["DecayTree"]
    return arbre.arrays(branches, library="pd")


def dessine_branche(df, noms_branches, nbins=169, echelle_log=True, dossier="histograms_plots"):
    for nom in noms_branches:
        if nom == "D0_MINIPCHI2":
            plage = (0, 50)
        elif nom == "D0_MINIP":
            plage = (0, 2)
        elif nom == "D0_DIRA_OWNPV":
            plage = (0.99, 1.0)
        else:
            plage = None

        plt.figure(figsize=(8, 5))
        plt.hist(df[nom], bins=nbins, range=plage, histtype="stepfilled", color=COULEUR_SIGNAL, alpha=0.25)
        plt.hist(df[nom], bins=nbins, range=plage, histtype="step", color=COULEUR_SIGNAL, linewidth=1.5)

        if echelle_log and nom != "D0_MM":
            plt.yscale("log")
            plt.ylabel("Nombre d'évènements (échelle log)")
        else:
            plt.ylim(bottom=0)
            plt.ylabel("Nombre d'évènements")

        plt.title(f"Distribution de {nom}")
        plt.xlabel(nom)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{dossier}/hist_{nom}.png")
        plt.close()


def dessine_branche_multiple(liste_dfs, liste_noms_dfs, noms_branches, nbins=169, normaliser=True, echelle_log=True, dossier="histograms_plots"):
    if normaliser:
        unite_y = "Densité de probabilité"
    else:
        unite_y = "Nombre d'évènements"

    if echelle_log:
        unite_y = unite_y + " (échelle log)"

    couleurs = [COULEUR_SIGNAL, COULEUR_BRUIT]

    for nom in noms_branches:
        if nom == "D0_MINIPCHI2":
            plage = (0, 50)
        elif nom == "D0_DIRA_OWNPV":
            plage = (0.99, 1.0)
        else:
            plage = None

        plt.figure(figsize=(8, 5))
        for i in range(len(liste_dfs)):
            plt.hist(liste_dfs[i][nom], bins=nbins, range=plage, density=normaliser, histtype="stepfilled", color=couleurs[i], alpha=0.2)
            plt.hist(liste_dfs[i][nom], bins=nbins, range=plage, density=normaliser, histtype="step", color=couleurs[i], linewidth=1.7, label=liste_noms_dfs[i])

        if echelle_log:
            plt.yscale("log")

        plt.title(f"Comparaison : {nom}")
        plt.xlabel(nom)
        plt.ylabel(unite_y)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{dossier}/superposition_{nom}.png")
        plt.close()