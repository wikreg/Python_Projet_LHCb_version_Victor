import os
import uproot
import matplotlib.pyplot as plt

# Palette pour que les graphiques soient sexy !
# Déclaration de constantes, code hexadécimal RRVVBB
COULEUR_FOND = "#0d1117"
COULEUR_CADRE = "#161b22"
COULEUR_SIGNAL = "#38bdf8"
COULEUR_POINTS = "#7dd3fc"
COULEUR_FIT = "#c084fc"
COULEUR_BRUIT = "#fb7185"
COULEUR_PULLS = "#2dd4bf"

# Configuration / style pour les graphiques
def configurer_style(dossier="histograms_plots"):
    # Création du dossier, sauf s'il existe déjà
    os.makedirs(dossier, exist_ok=True)
    # Thème sombre pour matplotlib
    plt.style.use("dark_background")

    # Les rcParams, c'est le dictionnaire global de réglage par défaut :
    # la configuration va s'appliquer à tous les graphiques.

    # Résolution de l'image en dots per inch
    plt.rcParams["figure.dpi"] = 200
    # Remplace la couleur de l'arrière-plan général
    plt.rcParams["figure.facecolor"] = COULEUR_FOND
    # Remplace la couleur de l'arrière-plan des graphiques
    plt.rcParams["axes.facecolor"] = COULEUR_CADRE
    # On garde que les axes en bas et à gauche
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    # Change l'opacité de la grille
    plt.rcParams["grid.alpha"] = 0.25

# Fonction qui prend le fichier root, extrait les données, puis les convertit en dataframes
# chemin_fichier -> le chemin vers le fichier root
# branches -> si on ne met rien, bah ça charge toutes les colonnes du fichier
def charge_donnees(chemin_fichier, branches=None):
    # Ouverture du fichier root
    fichier = uproot.open(chemin_fichier)
    # Sélection de l'arbre du fichier
    arbre = fichier["DecayTree"]
    # Lecture des données des branches et conversion immédiate en dataframe pandas
    return arbre.arrays(branches, library="pd")

# Fonction qui crée des histogrammes des données extraites du fichier root
# df -> le dataframe pandas
# noms_branches -> liste des noms de colonnes à tracer
# nbins -> le nombre de barres
# echelle_log -> bool pour savoir si on veut une échelle log ou non
# dossier -> le dossier d'enregistrement des graphiques
def dessine_branche(df, noms_branches, nbins=169, echelle_log=True, dossier="histograms_plots"):

    # Boucle de construction de graphique pour chaque branche
    # Choix manuel de plages pour une lecture plus sympathique des endroits importants de certains graphiques
    for nom in noms_branches:
        if nom == "D0_MINIPCHI2":
            plage = (0, 50)
        elif nom == "D0_MINIP":
            plage = (0, 2)
        elif nom == "D0_DIRA_OWNPV":
            plage = (0.99, 1.0)
        else:
            plage = None

        # Création d'une nouvelle figure : largeur 8 pouces, hauteur 5 pouces
        plt.figure(figsize=(8, 5))
        # Trace l'histogramme rempli légèrement
        plt.hist(df[nom], bins=nbins, range=plage, histtype="stepfilled", color=COULEUR_SIGNAL, alpha=0.25)
        # Trace le contour du même histogramme avec une épaisseur pour que ce soit joli
        plt.hist(df[nom], bins=nbins, range=plage, histtype="step", color=COULEUR_SIGNAL, linewidth=1.5)

        # Si échelle log, bah on met en log, voilà
        if echelle_log and nom != "D0_MM":
            plt.yscale("log")
            plt.ylabel("Nombre d'évènements (échelle log)")
        # Sinon bah non
        else:
            plt.ylim(bottom=0)
            plt.ylabel("Nombre d'évènements")

        # Titre, nom de la variable
        plt.title(f"Distribution de {nom}")
        plt.xlabel(nom)
        plt.grid(True)
        # Ajustement automatique des labels
        plt.tight_layout()
        # Enregistre un png de l'histogramme dans le dossier choisi
        plt.savefig(f"{dossier}/hist_{nom}.png")
        # Ferme la figure en mémoire, important
        plt.close()

# Fonction qui trace plusieurs histogrammes sur la même figure de matplotlib
# liste_dfs -> une liste de dataframes (branches)
# liste_noms_dfs -> liste des noms pour chaque dataframe, c'est pour la légende
# noms_branches -> liste des variables à tracer
# normaliser -> bool : si oui, l'histogramme en question est normalisé pour que l'aire totale vaille 1 (permet de comparer les formes), très cool !
def dessine_branche_multiple(liste_dfs, liste_noms_dfs, noms_branches, nbins=169, normaliser=True, echelle_log=True, dossier="histograms_plots"):

    # Choix de texte d'axe
    if normaliser:
        unite_y = "Densité de probabilité"
    else:
        unite_y = "Nombre d'évènements"

    if echelle_log:
        unite_y = unite_y + " (échelle log)"

    # Tableau pour le choix de couleur
    couleurs = [COULEUR_SIGNAL, COULEUR_BRUIT]

    # Choix manuel de plages pour une lecture plus sympathique des endroits importants de certains graphiques
    for nom in noms_branches:
        if nom == "D0_MINIPCHI2":
            plage = (0, 50)
        elif nom == "D0_DIRA_OWNPV":
            plage = (0.99, 1.0)
        else:
            plage = None

        # On trace chaque dataset (indice i) sur la même figure
        plt.figure(figsize=(8, 5))
        for i in range(len(liste_dfs)):
            # Sélectionne la colonne nom du i-ème dataframe
            # Trace l'histogramme rempli légèrement
            plt.hist(liste_dfs[i][nom], bins=nbins, range=plage, density=normaliser, histtype="stepfilled", color=couleurs[i], alpha=0.2)
            # Trace le contour du même histogramme avec une épaisseur pour que ce soit joli
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