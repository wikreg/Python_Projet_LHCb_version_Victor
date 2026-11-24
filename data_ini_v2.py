import uproot
import numpy as np
import pandas as pd
import zfit
import matplotlib.pyplot as plt
from hepstats.splot import compute_sweights

# ==========================================
# PALETTE DE COULEURS (THEME SOMBRE SIMPLE)
# ==========================================

COULEUR_FOND = "#0d1117"      # fond global sombre
COULEUR_CADRE = "#161b22"     # fond interieur des graphiques
COULEUR_SIGNAL = "#38bdf8"    # bleu cyan (histogrammes et signal)
COULEUR_POINTS = "#7dd3fc"    # bleu clair (points de donnees)
COULEUR_FIT = "#c084fc"       # violet (courbes de fit et FOM)
COULEUR_BRUIT = "#fb7185"     # rose corail (bruit de fond)
COULEUR_PULLS = "#2dd4bf"     # turquoise (pulls et optimum FOM)

# configuration simple du theme sombre sur tous les graphiques
plt.style.use("dark_background")
plt.rcParams["figure.dpi"] = 200
plt.rcParams["figure.facecolor"] = COULEUR_FOND
plt.rcParams["axes.facecolor"] = COULEUR_CADRE
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["grid.alpha"] = 0.25

# variable pour stocker le nom du dossier dans lequel je vais sauvegarder les differents histograms
dossier_sortie = "histograms_plots"


# ==========================================
# FONCTIONS SECTIONS 3.1 & 3.2
# ==========================================

def charge_donnees(chemin_fichier, branches=None):
    # --- THEORIE & FORMAT DES DONNEES ---
    # En physique des particules (LHCb), les donnees sont stockees dans des fichiers .root
    # sous forme d'arbres (TTree). Chaque ligne est un evenement (une collision)
    # et chaque colonne (branche) est une grandeur physique mesuree (masse, impulsion, etc.).
    #
    # --- FONCTIONNEMENT DU CODE ---
    # 1. uproot.open() ouvre le fichier ROOT directement en Python.
    # 2. fichier["DecayTree"] recupere l'arbre contenant les candidats D0 -> K+ pi-.
    # 3. .arrays(branches, library="pd") convertit les donnees en tableau DataFrame de pandas.

    fichier = uproot.open(chemin_fichier)
    arbre = fichier["DecayTree"]
    tableau = arbre.arrays(branches, library="pd")
    return tableau


def dessine_branche(df, noms_branches, nbins=169, echelle_log=True, dossier="histograms_plots"):
    # --- THEORIE ---
    # Un histogramme decoupe l'intervalle d'une variable en "nbins" barres et compte
    # combien d'evenements tombent dans chaque barre. L'echelle log sur Y permet de voir
    # les queues de distribution meme quand il y a un grand pic pres de 0 ou 1.
    #
    # --- FONCTIONNEMENT DU CODE ---
    # On parcourt chaque colonne de "noms_branches" avec une boucle for.
    # On zoome sur la zone utile pour D0_MINIPCHI2, D0_MINIP et D0_DIRA_OWNPV,
    # et on garde l'echelle lineaire partant de 0 pour la masse D0_MM.

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
    # --- THEORIE ---
    # Pour choisir les coupures, on compare la distribution d'une variable pour le signal (pic)
    # et pour le bruit de fond (bandes laterales). On normalise les histogrammes (density=True,
    # aire totale = 1) pour comparer leurs formes sur la meme echelle.
    #
    # --- FONCTIONNEMENT DU CODE ---
    # Pour chaque variable, on ouvre une figure et on superpose les histogrammes de "liste_dfs".

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
            df_actuel = liste_dfs[i]
            label_actuel = liste_noms_dfs[i]
            couleur_actuelle = couleurs[i]

            plt.hist(df_actuel[nom], bins=nbins, range=plage, density=normaliser, histtype="stepfilled", color=couleur_actuelle, alpha=0.2)
            plt.hist(df_actuel[nom], bins=nbins, range=plage, density=normaliser, histtype="step", color=couleur_actuelle, linewidth=1.7, label=label_actuel)

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


# ==========================================
# FONCTIONS SECTIONS 3.3 & 3.6
# ==========================================

def fit_donnees(donnees, obs, model, poids=None):
    # --- THEORIE STATISTIQUE (MAXIMUM DE VRAISEMBLANCE NON-BINNE ETENDU) ---
    # zfit evalue la densite de probabilite f(x_i) directement sur chaque evenement x_i
    # et minimise la log-vraisemblance negative -ln(L) avec l'algorithme Minuit.
    # L'option "Extended" permet d'ajuster en meme temps le nombre d'evenements (N_sig, N_bkg).
    # En section 3.6, "poids" permet de ponderer chaque evenement par son sWeight w_i.
    #
    # --- FONCTIONNEMENT DU CODE ---
    # Convertit les donnees en format zfit, lance Minuit et calcule les erreurs avec .hesse().

    data = zfit.Data(data=donnees.to_numpy(), obs=obs, weights=poids)
    nll = zfit.loss.ExtendedUnbinnedNLL(model=model, data=data)
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(nll).update_params()
    result.hesse()
    return result


# fit de la masse du D0 avec une gaussienne et une droite
def fit_masse(df, branche="D0_MM", min_fit=1816, max_fit=1914, nom_param=""):
    # --- THEORIE PHYSIQUE ---
    # Les vrais D0 forment un pic gaussien autour de mu (~1865 MeV/c^2) et d'ecart-type sigma.
    # Le bruit de fond combinatoire varie lentement et se modelise par un polynome de Chebyshev
    # d'ordre 1 (une droite de pente c1).
    #
    # --- FONCTIONNEMENT DU CODE ---
    # 1. Filtre les donnees entre min_fit et max_fit et declare les 5 parametres libres.
    # 2. Construit la somme Gauss + Chebyshev (SumPDF) et lance fit_donnees().
    # 3. Calcule S et B sous le pic (fenetre [mu - 3*sigma, mu + 3*sigma]) pour la FOM du 3.4.

    masque_plage = (df[branche] >= min_fit) & (df[branche] <= max_fit)
    donnees_masse = df[masque_plage][branche]
    n_total = len(donnees_masse)

    obs = zfit.Space(branche, min_fit, max_fit)

    mu = zfit.Parameter(f"mu{nom_param}", 1865.0, 1850.0, 1880.0)
    sigma = zfit.Parameter(f"sigma{nom_param}", 8.0, 1.0, 25.0)
    c1 = zfit.Parameter(f"c1{nom_param}", 0.0, -1.0, 1.0)

    n_sig = zfit.Parameter(f"n_sig{nom_param}", 0.5 * n_total, 0, n_total * 1.5)
    n_bkg = zfit.Parameter(f"n_bkg{nom_param}", 0.5 * n_total, 0, n_total * 1.5)

    sig_model = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma, extended=n_sig)
    bkg_model = zfit.pdf.Chebyshev(obs=obs, coeffs=[c1], extended=n_bkg)
    model = zfit.pdf.SumPDF([sig_model, bkg_model])

    result = fit_donnees(donnees_masse, obs, model)

    # calcul de S et B sous le pic (entre mu - 3*sigma et mu + 3*sigma)
    params = result.params
    mu_val = params[mu]["value"]
    sigma_val = params[sigma]["value"]
    S = params[n_sig]["value"]
    n_bkg_total = params[n_bkg]["value"]

    fenetre_pic = zfit.Space(branche, mu_val - 3 * sigma_val, mu_val + 3 * sigma_val)
    fraction_bruit_sous_pic = bkg_model.integrate(limits=fenetre_pic).numpy()[0]
    B = n_bkg_total * fraction_bruit_sous_pic

    return result, model, sig_model, bkg_model, donnees_masse, mu, sigma, n_sig, n_bkg, S, B


# fonction demandee en 3.3.3 et 3.3.4 : trace les donnees, le fit, les pulls et calcule le chi2
def trace_fit(donnees_masse, model, bkg_model, mu_val, mu_err, n_sig_val, n_bkg_val, min_fit=1816, max_fit=1914, nbins=100):
    # --- THEORIE STATISTIQUE (PULLS ET CHI2 REDUIT) ---
    # Dans chaque barre i, le pull vaut : pull_i = (N_modele - N_donnees) / sqrt(N_donnees).
    # Un bon fit donne des pulls aleatoires autour de 0 (entre -2 et +2) et un chi2 reduit
    # chi2 / ndf proche de 1 (avec ndf = nombre de barres - nombre de parametres libres).

    largeur_bin = (max_fit - min_fit) / nbins
    comptes, bords = np.histogram(donnees_masse, bins=nbins, range=(min_fit, max_fit))

    centres = []
    erreurs_data = []
    for i in range(nbins):
        milieu = (bords[i] + bords[i + 1]) / 2
        centres.append(milieu)
        if comptes[i] > 0:
            erreurs_data.append(np.sqrt(comptes[i]))
        else:
            erreurs_data.append(1.0)

    centres = np.array(centres)
    erreurs_data = np.array(erreurs_data)

    x_lisse = np.linspace(min_fit, max_fit, 500)
    y_total_lisse = model.pdf(x_lisse).numpy() * (n_sig_val + n_bkg_val) * largeur_bin
    y_bkg_lisse = bkg_model.pdf(x_lisse).numpy() * n_bkg_val * largeur_bin

    # 1. Graphique du fit de masse
    plt.figure(figsize=(8.5, 5))
    plt.fill_between(x_lisse, 0, y_bkg_lisse, color=COULEUR_BRUIT, alpha=0.15)
    plt.fill_between(x_lisse, y_bkg_lisse, y_total_lisse, color=COULEUR_FIT, alpha=0.15)
    plt.errorbar(centres, comptes, yerr=erreurs_data, fmt="o", color=COULEUR_POINTS, markersize=3.8, label="Données sélectionnées")
    plt.plot(x_lisse, y_total_lisse, color=COULEUR_FIT, linewidth=2.4, label=f"Fit zfit (M = {mu_val:.2f} ± {mu_err:.2f} MeV/c²)")
    plt.plot(x_lisse, y_bkg_lisse, color=COULEUR_BRUIT, linestyle="--", linewidth=1.8, label="Bruit de fond (Chebyshev)")
    plt.ylim(bottom=0)
    plt.title("Fit étendu (zfit) de la masse du D⁰")
    plt.xlabel("Masse D⁰ [MeV/c²]")
    plt.ylabel(f"Évènements / {largeur_bin:.2f} MeV/c²")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{dossier_sortie}/fit_masse_D0.png")
    plt.close()

    # 2. Calcul des pulls et du chi2 reduit
    prediction_bins = model.pdf(centres).numpy() * (n_sig_val + n_bkg_val) * largeur_bin
    pulls = []
    somme_pulls_carre = 0.0

    for i in range(nbins):
        pull_i = (prediction_bins[i] - comptes[i]) / erreurs_data[i]
        pulls.append(pull_i)
        somme_pulls_carre = somme_pulls_carre + pull_i ** 2

    plt.figure(figsize=(8.5, 3.8))
    plt.axhspan(-2, 2, color=COULEUR_PULLS, alpha=0.08)
    plt.bar(centres, pulls, width=largeur_bin * 0.85, color=COULEUR_PULLS)
    plt.axhline(0, color="white", linewidth=1)
    plt.axhline(2, color=COULEUR_BRUIT, linestyle="--", linewidth=1)
    plt.axhline(-2, color=COULEUR_BRUIT, linestyle="--", linewidth=1)
    plt.title("Pulls du fit de masse du D⁰")
    plt.xlabel("Masse D⁰ [MeV/c²]")
    plt.ylabel("(Modèle - Données) / σ")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{dossier_sortie}/pulls_masse_D0.png")
    plt.close()

    n_params_flottants = 5 # mu, sigma, c1, n_sig, n_bkg
    ndf = nbins - n_params_flottants
    chi2_reduit = somme_pulls_carre / ndf
    print("Chi2 réduit :", round(chi2_reduit, 3), "(pour", ndf, "degrés de liberté)")

    return comptes, centres, erreurs_data, largeur_bin, chi2_reduit


# ==========================================
# FONCTIONS SECTION 3.4
# ==========================================

def FOM_variable(df, nom_variable, liste_seuils, type_coupure="superieur", suffixe_fichier=""):
    # --- THEORIE STATISTIQUE (FIGURE DE MERITE) ---
    # L'incertitude statistique sur N = S + B evenements dans le pic vaut sqrt(S + B).
    # On cherche la coupure qui maximise FOM = S / sqrt(S + B).

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
    plt.savefig(f"{dossier_sortie}/FOM_{nom_variable}{suffixe_fichier}.png")
    plt.close()

    return meilleur_seuil, meilleure_fom


# ==========================================
# FONCTIONS SECTIONS 3.5 & 3.6
# ==========================================

def fit_duree_vie_simple(df, branche="D0_TAU", min_tau=0.0004, max_tau=0.004, nom_param="", colonne_poids=None):
    # --- THEORIE PHYSIQUE ---
    # Le D0 se desintegre selon exp(-t / tau). Dans zfit, la loi s'ecrit exp(lam * t),
    # donc tau = -1 / lam. On coupe sous 0.0004 ns pour eviter l'effet du trigger.
    # En section 3.6, si "colonne_poids" est donnee, on pondere chaque evenement par son sWeight.

    masque_tau = (df[branche] >= min_tau) & (df[branche] <= max_tau)
    donnees_tau = df[masque_tau][branche]

    if colonne_poids is not None:
        poids_tau = df[masque_tau][colonne_poids].to_numpy()
        n_total = np.sum(poids_tau)
    else:
        poids_tau = None
        n_total = len(donnees_tau)

    obs = zfit.Space(branche, min_tau, max_tau)

    lam = zfit.Parameter(f"lam{nom_param}", -2400.0, -15000.0, -100.0)
    n_evts = zfit.Parameter(f"n_evts{nom_param}", n_total, 0, abs(n_total) * 2.0)

    model = zfit.pdf.Exponential(obs=obs, lam=lam, extended=n_evts)
    result = fit_donnees(donnees_tau, obs, model, poids=poids_tau)

    return result, model, donnees_tau, lam, n_evts


def fit_duree_vie_complet(df, lam_bkg_fixe, branche="D0_TAU", min_tau=0.0004, max_tau=0.004):
    # --- THEORIE PHYSIQUE ---
    # Somme de deux exponentielles : signal (pente libre lam_sig) + bruit de fond
    # (pente lam_bkg bloquee a la valeur mesuree dans les bandes laterales).

    masque_tau = (df[branche] >= min_tau) & (df[branche] <= max_tau)
    donnees_tau = df[masque_tau][branche]
    n_total = len(donnees_tau)

    obs = zfit.Space(branche, min_tau, max_tau)

    lam_sig = zfit.Parameter("lam_sig_final", -2400.0, -15000.0, -100.0)
    lam_bkg = zfit.Parameter("lam_bkg_final", lam_bkg_fixe, floating=False)

    n_sig = zfit.Parameter("n_sig_tau", 0.8 * n_total, 0, n_total * 1.5)
    n_bkg = zfit.Parameter("n_bkg_tau", 0.2 * n_total, 0, n_total * 1.5)

    sig_model = zfit.pdf.Exponential(obs=obs, lam=lam_sig, extended=n_sig)
    bkg_model = zfit.pdf.Exponential(obs=obs, lam=lam_bkg, extended=n_bkg)
    model = zfit.pdf.SumPDF([sig_model, bkg_model])

    result = fit_donnees(donnees_tau, obs, model)

    return result, model, bkg_model, donnees_tau, lam_sig, n_sig, n_bkg


# ==========================================
# FONCTION SECTION 3.7
# ==========================================

# modele de masse ameliore : Gaussienne simple (signal) + Parabole negative (bruit)
def fit_masse_parabole_bkg(df, branche="D0_MM", min_fit=1816, max_fit=1914):
    # --- THEORIE ---
    # On garde le signal modelise par une Gaussienne simple (comme en 3.3),
    # mais on remplace la droite du bruit par une parabole tournee vers le bas
    # (polynome de Chebyshev d'ordre 2 avec c2 < 0) pour suivre la courbure du fond.

    masque_plage = (df[branche] >= min_fit) & (df[branche] <= max_fit)
    donnees_masse = df[masque_plage][branche]
    n_total = len(donnees_masse)

    obs = zfit.Space(branche, min_fit, max_fit)

    # Signal : meme gaussienne simple qu'en 3.3
    mu = zfit.Parameter("mu_parab", 1865.0, 1850.0, 1880.0)
    sigma = zfit.Parameter("sigma_parab", 8.0, 1.0, 25.0)

    # Bruit de fond : polynome de Chebyshev d'ordre 2 = parabole (c2 <= 0 pour une parabole negative)
    c1 = zfit.Parameter("c1_parab", -0.1, -0.5, 0.5)
    c2 = zfit.Parameter("c2_parab", -0.05, -0.5, 0.0)

    n_sig = zfit.Parameter("n_sig_parab", 0.7 * n_total, 0, n_total * 1.5)
    n_bkg = zfit.Parameter("n_bkg_parab", 0.3 * n_total, 0, n_total * 1.5)

    sig_model = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma, extended=n_sig)
    bkg_model = zfit.pdf.Chebyshev(obs=obs, coeffs=[c1, c2], extended=n_bkg)

    model = zfit.pdf.SumPDF([sig_model, bkg_model])
    result = fit_donnees(donnees_masse, obs, model)

    return result, model, bkg_model, donnees_masse, mu, sigma, c1, c2, n_sig, n_bkg


# ==========================================
# EXECUTION SECTIONS 3.1 & 3.2
# ==========================================

df_total = charge_donnees("data/MasterclassData_2012_all.root")

liste_branches = list(df_total.columns)
print("Branches disponibles :", liste_branches)

dessine_branche(df_total, liste_branches, echelle_log=True)

cond_kaon = df_total["Kplus_ProbNNk"] > 0.95
cond_pion = df_total["piminus_ProbNNpi"] > 0.8
masque_sel = cond_kaon & cond_pion
df_selectionne = df_total[masque_sel]

plt.figure(figsize=(8, 5))
plt.hist(df_total["D0_MM"], bins=120, range=(1815, 1915), density=True, histtype="stepfilled", color=COULEUR_BRUIT, alpha=0.2)
plt.hist(df_total["D0_MM"], bins=120, range=(1815, 1915), density=True, histtype="step", color=COULEUR_BRUIT, linewidth=1.6, label="Avant sélection")
plt.hist(df_selectionne["D0_MM"], bins=120, range=(1815, 1915), density=True, histtype="stepfilled", color=COULEUR_SIGNAL, alpha=0.2)
plt.hist(df_selectionne["D0_MM"], bins=120, range=(1815, 1915), density=True, histtype="step", color=COULEUR_SIGNAL, linewidth=1.6, label="Après coupures ProbNN")
plt.title("Effet des coupures ProbNN sur le pic de masse du D⁰")
plt.xlabel("Masse D0 [MeV/c²]")
plt.ylabel("Densité de probabilité")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{dossier_sortie}/comparaison_selection.png")
plt.close()

cond_pic = (df_total["D0_MM"] >= 1850) & (df_total["D0_MM"] <= 1882)
df_signal_region = df_total[cond_pic]

cond_bande_gauche = (df_total["D0_MM"] >= 1814) & (df_total["D0_MM"] <= 1849)
cond_bande_droite = (df_total["D0_MM"] >= 1883) & (df_total["D0_MM"] <= 1914)
df_bruit_region = df_total[cond_bande_gauche | cond_bande_droite]

tableaux_a_comparer = [df_signal_region, df_bruit_region]
noms_courbes = ["Région Pic (Signal dominant)", "Bandes Latérales (Bruit)"]
variables_a_tracer = ["D0_DIRA_OWNPV", "D0_MINIPCHI2", "Kplus_ProbNNk", "piminus_ProbNNpi"]

dessine_branche_multiple(tableaux_a_comparer, noms_courbes, variables_a_tracer, echelle_log=True)


# ==========================================
# EXECUTION SECTION 3.3
# ==========================================

cond_dira = df_total["D0_DIRA_OWNPV"] > 0.9998
cond_ip = df_total["D0_MINIPCHI2"] < 9
masque_complet = cond_kaon & cond_pion & cond_dira & cond_ip
df_propre = df_total[masque_complet]

result, model, sig_model, bkg_model, donnees_masse, mu, sigma, n_sig, n_bkg, S, B = fit_masse(df_propre)
print(result)

params = result.params
mu_val = params[mu]["value"]
mu_err = params[mu]["hesse"]["error"]
n_sig_val = params[n_sig]["value"]
n_bkg_val = params[n_bkg]["value"]

print("Masse du D0 mesurée :", round(mu_val, 2), "+/-", round(mu_err, 2), "MeV/c²")
print("Nombre de signal :", int(n_sig_val), "| Nombre de bruit :", int(n_bkg_val))

comptes, centres, erreurs_data, largeur_bin, chi2_reduit = trace_fit(
    donnees_masse, model, bkg_model, mu_val, mu_err, n_sig_val, n_bkg_val
)


# ==========================================
# EXECUTION SECTION 3.4
# ==========================================

seuils_dira = [0.9990, 0.9993, 0.9995, 0.9997, 0.9998, 0.9999, 0.99995]
opt_dira, fom_dira = FOM_variable(df_selectionne, "D0_DIRA_OWNPV", seuils_dira, type_coupure="superieur")
print("Meilleur seuil D0_DIRA_OWNPV (avec PID) :", opt_dira, "| FOM max :", round(fom_dira, 2))

df_apres_dira = df_selectionne[df_selectionne["D0_DIRA_OWNPV"] > opt_dira]
seuils_ip = [25, 20, 16, 12, 9, 6, 4]
opt_ip, fom_ip = FOM_variable(df_apres_dira, "D0_MINIPCHI2", seuils_ip, type_coupure="inferieur")
print("Meilleur seuil D0_MINIPCHI2 (avec PID) :", opt_ip, "| FOM max :", round(fom_ip, 2))

opt_dira_sans_pid, fom_sans_pid = FOM_variable(
    df_total, "D0_DIRA_OWNPV", seuils_dira, type_coupure="superieur", suffixe_fichier="_sans_PID"
)
print("Meilleur seuil D0_DIRA_OWNPV (SANS PID) :", opt_dira_sans_pid, "| FOM max :", round(fom_sans_pid, 2))


# ==========================================
# EXECUTION SECTION 3.5
# ==========================================

masque_optimal = cond_kaon & cond_pion & (df_total["D0_DIRA_OWNPV"] > opt_dira) & (df_total["D0_MINIPCHI2"] < opt_ip)
df_optimal = df_total[masque_optimal]

df_tau_pic = df_optimal[(df_optimal["D0_MM"] >= 1850) & (df_optimal["D0_MM"] <= 1882)]
df_tau_bruit = df_optimal[((df_optimal["D0_MM"] >= 1815) & (df_optimal["D0_MM"] <= 1835)) |
                          ((df_optimal["D0_MM"] >= 1895) & (df_optimal["D0_MM"] <= 1915))]

min_tau = 0.0004
max_tau = 0.004
nbins_tau = 60
largeur_bin_tau = (max_tau - min_tau) / nbins_tau

# --- 3.5.1 : fit avec 1 seule exponentielle sur le pic ---
res_simple, mod_simple, donnees_tau_pic, lam_simple, n_evts_simple = fit_duree_vie_simple(
    df_tau_pic, min_tau=min_tau, max_tau=max_tau, nom_param="_simple"
)

lam_s_val = res_simple.params[lam_simple]["value"]
lam_s_err = res_simple.params[lam_simple]["hesse"]["error"]
n_s_val = res_simple.params[n_evts_simple]["value"]

tau_simple_ps = (-1.0 / lam_s_val) * 1000.0
err_tau_simple_ps = (lam_s_err / (lam_s_val ** 2)) * 1000.0
print("\n[3.5.1] Duree de vie (1 exp) :", round(tau_simple_ps, 4), "+/-", round(err_tau_simple_ps, 4), "ps")

comptes_tau, bords_tau = np.histogram(donnees_tau_pic, bins=nbins_tau, range=(min_tau, max_tau))
centres_tau = []
erreurs_tau = []
for i in range(nbins_tau):
    centres_tau.append((bords_tau[i] + bords_tau[i + 1]) / 2)
    if comptes_tau[i] > 0:
        erreurs_tau.append(np.sqrt(comptes_tau[i]))
    else:
        erreurs_tau.append(1.0)
centres_tau = np.array(centres_tau)
erreurs_tau = np.array(erreurs_tau)

pred_simple = mod_simple.pdf(centres_tau).numpy() * n_s_val * largeur_bin_tau
somme_pulls_simple = 0.0
for i in range(nbins_tau):
    pull_i = (pred_simple[i] - comptes_tau[i]) / erreurs_tau[i]
    somme_pulls_simple = somme_pulls_simple + pull_i ** 2
print("[3.5.1] Chi2 reduit (1 exp) :", round(somme_pulls_simple / (nbins_tau - 2), 3))

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
plt.savefig(f"{dossier_sortie}/fit_tau_simple.png")
plt.close()

# --- 3.5.2 : pente du bruit sur les bandes laterales ---
res_bkg, mod_bkg_tau, donnees_tau_bkg, lam_bkg, n_evts_bkg = fit_duree_vie_simple(
    df_tau_bruit, min_tau=min_tau, max_tau=max_tau, nom_param="_bkg"
)
lam_bkg_val = res_bkg.params[lam_bkg]["value"]
print("[3.5.2] Pente lam du bruit de fond :", round(lam_bkg_val, 1), "ns^-1")

# --- 3.5.3 : fit final avec signal + bruit fixe ---
res_comp, mod_comp, bkg_comp, donnees_tau_pic, lam_sig, n_sig_tau, n_bkg_tau = fit_duree_vie_complet(
    df_tau_pic, lam_bkg_val, min_tau=min_tau, max_tau=max_tau
)

lam_sig_val = res_comp.params[lam_sig]["value"]
lam_sig_err = res_comp.params[lam_sig]["hesse"]["error"]
n_sig_t_val = res_comp.params[n_sig_tau]["value"]
n_bkg_t_val = res_comp.params[n_bkg_tau]["value"]

tau_final_ps = (-1.0 / lam_sig_val) * 1000.0
err_tau_final_ps = (lam_sig_err / (lam_sig_val ** 2)) * 1000.0
print("[3.5.3] Duree de vie finale (Signal + Bruit) :", round(tau_final_ps, 4), "+/-", round(err_tau_final_ps, 4), "ps")

pred_comp = mod_comp.pdf(centres_tau).numpy() * (n_sig_t_val + n_bkg_t_val) * largeur_bin_tau
somme_pulls_comp = 0.0
for i in range(nbins_tau):
    pull_i = (pred_comp[i] - comptes_tau[i]) / erreurs_tau[i]
    somme_pulls_comp = somme_pulls_comp + pull_i ** 2
print("[3.5.3] Chi2 reduit (Signal + Bruit) :", round(somme_pulls_comp / (nbins_tau - 3), 3))

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
plt.savefig(f"{dossier_sortie}/fit_tau_complet.png")
plt.close()


# ==========================================
# EXECUTION SECTION 3.6 (METHODE SPLOT)
# ==========================================

res_m_splot, mod_m_splot, sig_m_splot, bkg_m_splot, donnees_m_splot, mu_sp, sig_sp, n_sig_sp, n_bkg_sp, S_sp, B_sp = fit_masse(
    df_optimal, min_fit=1816, max_fit=1914, nom_param="_splot"
)

poids_splot = compute_sweights(mod_m_splot, donnees_m_splot.to_numpy())
w_sig = poids_splot[n_sig_sp]
w_bkg = poids_splot[n_bkg_sp]

print("\n[3.6.1] Somme des poids signal :", round(np.sum(w_sig), 1), "| n_sig fit :", round(res_m_splot.params[n_sig_sp]["value"], 1))
print("[3.6.1] Somme des poids bruit  :", round(np.sum(w_bkg), 1), "| n_bkg fit :", round(res_m_splot.params[n_bkg_sp]["value"], 1))

df_splot = df_optimal[(df_optimal["D0_MM"] >= 1816) & (df_optimal["D0_MM"] <= 1914)].copy()
df_splot["w_sig"] = w_sig
df_splot["w_bkg"] = w_bkg

masque_tau_sp = (df_splot["D0_TAU"] >= min_tau) & (df_splot["D0_TAU"] <= max_tau)
tau_vals_sp = df_splot[masque_tau_sp]["D0_TAU"].to_numpy()
w_sig_tau = df_splot[masque_tau_sp]["w_sig"].to_numpy()
w_bkg_tau = df_splot[masque_tau_sp]["w_bkg"].to_numpy()

comptes_sig_sp, bords_sp = np.histogram(tau_vals_sp, bins=nbins_tau, range=(min_tau, max_tau), weights=w_sig_tau)
somme_w2_sig, _ = np.histogram(tau_vals_sp, bins=nbins_tau, range=(min_tau, max_tau), weights=w_sig_tau ** 2)
erreurs_sig_sp = np.sqrt(somme_w2_sig)

comptes_bkg_sp, _ = np.histogram(tau_vals_sp, bins=nbins_tau, range=(min_tau, max_tau), weights=w_bkg_tau)
somme_w2_bkg, _ = np.histogram(tau_vals_sp, bins=nbins_tau, range=(min_tau, max_tau), weights=w_bkg_tau ** 2)
erreurs_bkg_sp = np.sqrt(somme_w2_bkg)

centres_sp = []
for i in range(nbins_tau):
    centres_sp.append((bords_sp[i] + bords_sp[i + 1]) / 2)
centres_sp = np.array(centres_sp)

plt.figure(figsize=(8, 5))
masque_sig_pos = comptes_sig_sp > 0
masque_bkg_pos = comptes_bkg_sp > 0
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
plt.savefig(f"{dossier_sortie}/splot_tau_signal_vs_bruit.png")
plt.close()

res_tau_sp, mod_tau_sp, donnees_tau_sp, lam_sp, n_evts_sp = fit_duree_vie_simple(
    df_splot, min_tau=min_tau, max_tau=max_tau, nom_param="_splot_tau", colonne_poids="w_sig"
)

lam_sp_val = res_tau_sp.params[lam_sp]["value"]
lam_sp_err = res_tau_sp.params[lam_sp]["hesse"]["error"]
n_sp_val = res_tau_sp.params[n_evts_sp]["value"]

tau_splot_ps = (-1.0 / lam_sp_val) * 1000.0
err_tau_splot_ps = (lam_sp_err / (lam_sp_val ** 2)) * 1000.0
print("[3.6.4] Duree de vie sPlot (Signal seul, 1 exp) :", round(tau_splot_ps, 4), "+/-", round(err_tau_splot_ps, 4), "ps")

pred_sp = mod_tau_sp.pdf(centres_sp).numpy() * n_sp_val * largeur_bin_tau
somme_pulls_sp = 0.0
for i in range(nbins_tau):
    if erreurs_sig_sp[i] > 0:
        err_i = erreurs_sig_sp[i]
    else:
        err_i = 1.0
    pull_i = (pred_sp[i] - comptes_sig_sp[i]) / err_i
    somme_pulls_sp = somme_pulls_sp + pull_i ** 2
print("[3.6.4] Chi2 reduit (sPlot 1 exp) :", round(somme_pulls_sp / (nbins_tau - 2), 3))

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
plt.savefig(f"{dossier_sortie}/fit_tau_splot.png")
plt.close()


# ==========================================
# EXECUTION SECTION 3.7 (POUR ALLER PLUS LOIN)
# ==========================================

# --- 3.7.1 : Signal Gaussienne simple + Bruit de fond Parabole negative ---
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
ndf_pb = nbins - 6 # 6 parametres libres : mu, sigma, c1, c2, n_sig, n_bkg
chi2_reduit_pb = somme_pulls_pb / ndf_pb

print("\n[3.7.1] Masse D0 (Gauss + Parabole) :", round(mu_pb_val, 2), "+/-", round(mu_pb_err, 2), "MeV/c²")
print("[3.7.1] Courbure parabole c2 :", round(c2_pb_val, 4))
print("[3.7.1] Chi2 reduit (Gauss + Parabole) :", round(chi2_reduit_pb, 3), "(contre", round(chi2_reduit, 3), "pour Gauss + droite)")

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
plt.savefig(f"{dossier_sortie}/fit_masse_parabole.png")
plt.close()

# --- 3.7.2 : Calcul manuel des sWeights sans hepstats (inversion directe 2x2) ---
m_vals = donnees_m_splot.to_numpy()
f_S = sig_m_splot.pdf(m_vals).numpy()
f_B = bkg_m_splot.pdf(m_vals).numpy()

N_S = res_m_splot.params[n_sig_sp]["value"]
N_B = res_m_splot.params[n_bkg_sp]["value"]
denom = N_S * f_S + N_B * f_B

M_SS = np.sum((f_S * f_S) / (denom ** 2))
M_SB = np.sum((f_S * f_B) / (denom ** 2))
M_BB = np.sum((f_B * f_B) / (denom ** 2))

determinant = M_SS * M_BB - M_SB ** 2
V_SS = M_BB / determinant
V_SB = -M_SB / determinant

w_sig_manuel = (V_SS * f_S + V_SB * f_B) / denom
ecart_max = np.max(np.abs(w_sig_manuel - w_sig))
print(f"[3.7.2] Ecart max entre sWeights manuels et hepstats : {ecart_max:.2e}")

# --- 3.7.3 : Soustraction des bandes laterales (Sideband Subtraction) ---
mu_sp_val = res_m_splot.params[mu_sp]["value"]
sig_sp_val = res_m_splot.params[sig_sp]["value"]

min_pic = mu_sp_val - 3 * sig_sp_val
max_pic = mu_sp_val + 3 * sig_sp_val
max_bg_gauche = mu_sp_val - 4 * sig_sp_val
min_bg_droite = mu_sp_val + 4 * sig_sp_val

int_bruit_pic = bkg_m_splot.integrate(limits=zfit.Space("D0_MM", min_pic, max_pic)).numpy()[0]
int_bruit_side = (
    bkg_m_splot.integrate(limits=zfit.Space("D0_MM", 1816, max_bg_gauche)).numpy()[0]
    + bkg_m_splot.integrate(limits=zfit.Space("D0_MM", min_bg_droite, 1914)).numpy()[0]
)
alpha_sb = int_bruit_pic / int_bruit_side

df_sb_pic = df_optimal[(df_optimal["D0_MM"] >= min_pic) & (df_optimal["D0_MM"] <= max_pic)]
df_sb_side = df_optimal[((df_optimal["D0_MM"] >= 1816) & (df_optimal["D0_MM"] <= max_bg_gauche)) |
                        ((df_optimal["D0_MM"] >= min_bg_droite) & (df_optimal["D0_MM"] <= 1914))]

h_pic, _ = np.histogram(df_sb_pic["D0_TAU"], bins=nbins_tau, range=(min_tau, max_tau))
h_side, _ = np.histogram(df_sb_side["D0_TAU"], bins=nbins_tau, range=(min_tau, max_tau))

h_sig_sb = h_pic - alpha_sb * h_side
err_sig_sb = np.sqrt(h_pic + (alpha_sb ** 2) * h_side)

plt.figure(figsize=(8, 5))
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
plt.savefig(f"{dossier_sortie}/comparaison_splot_vs_sideband.png")
plt.close()


# ==========================================
# REPONSES TEXTE
# ==========================================

# 3.1 Preambule

# We can see that the Measured Mass of the D0 is contained in the branch D0_MM
# the lifetime of the D0 is contained in the branch D0_TAU
# and the transverse momentum of each particle is contained in the branches D0_PT, Kplus_PT, piminus_PT
# the probability of a kaon candidate being identified as pion or a kaon is stored in the branches Kplus_ProbNNpi and Kplus_ProbNNk
# the probability of a pion candidate being identified as pion or a kaon is stored in the branches piminus_ProbNNpi and piminus_ProbNNk

# 3.2 Selection

# The easiest data from which we can 'extract' the signal by a simple selection with as little noise as possible is the probability histograms of the kaons and the pions because of the probability peaks that are easy to spot.
# I think the best easy way to seperate the data with the noise and the important data is to check the full histograms and find a value on the Y axis where i will 'cut' the histogram.
# It comes from the cos of the theta angle, if we considere the conservation of momentum the angle between the primary vertex and the secondary vertex should be close to 0 for hte real D particle. so we can put aside all of the traces that have an angle that is superior to 0.