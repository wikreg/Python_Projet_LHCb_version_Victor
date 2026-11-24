import numpy as np
import zfit
import matplotlib.pyplot as plt
from fonctions.sec31_32_donnees import COULEUR_POINTS, COULEUR_FIT, COULEUR_BRUIT, COULEUR_PULLS

# Fonction générique qui ajuste un modèle zfit sur des données (avec poids optionnels pour sPlot)
# donnees -> série pandas contenant les valeurs mesurées
# obs -> espace d'observation zfit (variable et bornes min/max)
# model -> densité de probabilité (PDF) zfit à ajuster
# poids -> tableau optionnel de poids par évènement (utilisé en section 3.6)
def fit_donnees(donnees, obs, model, poids=None):
    # Convertit les données en objet zfit avec les éventuels poids
    data = zfit.Data(data=donnees.to_numpy(), obs=obs, weights=poids)
    # Construit la vraisemblance négative étendue non-binnée (prend en compte le nombre total d'évènements)
    nll = zfit.loss.ExtendedUnbinnedNLL(model=model, data=data)
    # Initialise l'optimiseur Minuit et lance la minimisation
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(nll).update_params()
    # Calcule les incertitudes symétriques sur les paramètres avec la méthode de Hesse
    result.hesse()
    return result

# Fonction qui ajuste la masse invariante du D0 avec une gaussienne (signal) et une droite (bruit)
# df -> dataframe pandas
# branche -> nom de la colonne de masse ("D0_MM")
# min_fit, max_fit -> bornes en MeV/c^2 de la fenêtre d'ajustement
# nom_param -> suffixe ajouté aux noms des paramètres pour éviter les doublons si on fait plusieurs fits
def fit_masse(df, branche="D0_MM", min_fit=1816, max_fit=1914, nom_param=""):
    # Filtre les évènements dans l'intervalle de masse choisi
    masque_plage = (df[branche] >= min_fit) & (df[branche] <= max_fit)
    donnees_masse = df[masque_plage][branche]
    n_total = len(donnees_masse)

    # Définit l'espace de la variable de masse
    obs = zfit.Space(branche, min_fit, max_fit)

    # Déclaration des paramètres libres : valeur initiale, borne min, borne max
    mu = zfit.Parameter(f"mu{nom_param}", 1865.0, 1850.0, 1880.0)
    sigma = zfit.Parameter(f"sigma{nom_param}", 8.0, 1.0, 25.0)
    c1 = zfit.Parameter(f"c1{nom_param}", 0.0, -1.0, 1.0)

    # Paramètres étendus : nombre estimé d'évènements de signal et de bruit de fond
    n_sig = zfit.Parameter(f"n_sig{nom_param}", 0.5 * n_total, 0, n_total * 1.5)
    n_bkg = zfit.Parameter(f"n_bkg{nom_param}", 0.5 * n_total, 0, n_total * 1.5)

    # Modèle du signal : gaussienne ; modèle du bruit : polynôme de Chebyshev d'ordre 1 (droite)
    sig_model = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma, extended=n_sig)
    bkg_model = zfit.pdf.Chebyshev(obs=obs, coeffs=[c1], extended=n_bkg)
    # Somme des deux composantes pour obtenir le modèle total
    model = zfit.pdf.SumPDF([sig_model, bkg_model])

    # Lance l'ajustement
    result = fit_donnees(donnees_masse, obs, model)

    # Récupère les valeurs ajustées pour calculer S et B sous le pic
    params = result.params
    mu_val = params[mu]["value"]
    sigma_val = params[sigma]["value"]
    S = params[n_sig]["value"]
    n_bkg_total = params[n_bkg]["value"]

    # Intègre la fraction de bruit présente dans la fenêtre [mu - 3*sigma, mu + 3*sigma]
    fenetre_pic = zfit.Space(branche, mu_val - 3 * sigma_val, mu_val + 3 * sigma_val)
    fraction_bruit_sous_pic = bkg_model.integrate(limits=fenetre_pic).numpy()[0]
    B = n_bkg_total * fraction_bruit_sous_pic

    return result, model, sig_model, bkg_model, donnees_masse, mu, sigma, n_sig, n_bkg, S, B

# Fonction qui trace le fit de masse, les pulls et calcule le chi2 réduit
# donnees_masse -> valeurs de masse ajustées
# model, bkg_model -> modèles total et bruit de fond zfit
# mu_val, mu_err -> masse moyenne et son incertitude
# n_sig_val, n_bkg_val -> nombres ajustés d'évènements de signal et de bruit
# min_fit, max_fit, nbins -> plage et nombre de barres de l'histogramme
# dossier -> dossier d'enregistrement des figures
def trace_fit(donnees_masse, model, bkg_model, mu_val, mu_err, n_sig_val, n_bkg_val, min_fit=1816, max_fit=1914, nbins=100, dossier="histograms_plots"):
    # Calcule la largeur d'une barre et regroupe les données dans un histogramme
    largeur_bin = (max_fit - min_fit) / nbins
    comptes, bords = np.histogram(donnees_masse, bins=nbins, range=(min_fit, max_fit))

    # Calcule le centre de chaque barre et l'incertitude statistique (racine de N, ou 1 si barre vide)
    centres = []
    erreurs_data = []
    for i in range(nbins):
        centres.append((bords[i] + bords[i + 1]) / 2)
        if comptes[i] > 0:
            erreurs_data.append(np.sqrt(comptes[i]))
        else:
            erreurs_data.append(1.0)

    centres = np.array(centres)
    erreurs_data = np.array(erreurs_data)

    # Évalue les courbes lisses sur 500 points pour un tracé bien fluide
    x_lisse = np.linspace(min_fit, max_fit, 500)
    y_total_lisse = model.pdf(x_lisse).numpy() * (n_sig_val + n_bkg_val) * largeur_bin
    y_bkg_lisse = bkg_model.pdf(x_lisse).numpy() * n_bkg_val * largeur_bin

    # 1. Graphique de l'ajustement de masse
    plt.figure(figsize=(8.5, 5))
    # Remplit l'aire du bruit en rose et celle ajoutée par le signal en violet
    plt.fill_between(x_lisse, 0, y_bkg_lisse, color=COULEUR_BRUIT, alpha=0.15)
    plt.fill_between(x_lisse, y_bkg_lisse, y_total_lisse, color=COULEUR_FIT, alpha=0.15)
    # Trace les points de données avec leurs barres d'erreur
    plt.errorbar(centres, comptes, yerr=erreurs_data, fmt="o", color=COULEUR_POINTS, markersize=3.8, label="Données sélectionnées")
    # Trace la courbe totale du fit et la composante du bruit en pointillés
    plt.plot(x_lisse, y_total_lisse, color=COULEUR_FIT, linewidth=2.4, label=f"Fit zfit (M = {mu_val:.2f} ± {mu_err:.2f} MeV/c²)")
    plt.plot(x_lisse, y_bkg_lisse, color=COULEUR_BRUIT, linestyle="--", linewidth=1.8, label="Bruit de fond (Chebyshev)")
    plt.ylim(bottom=0)
    plt.title("Fit étendu (zfit) de la masse du D⁰")
    plt.xlabel("Masse D⁰ [MeV/c²]")
    plt.ylabel(f"Évènements / {largeur_bin:.2f} MeV/c²")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{dossier}/fit_masse_D0.png")
    plt.close()

    # 2. Calcul des pulls et du chi2 réduit
    prediction_bins = model.pdf(centres).numpy() * (n_sig_val + n_bkg_val) * largeur_bin
    pulls = []
    somme_pulls_carre = 0.0

    for i in range(nbins):
        # Écart normalisé entre le modèle et les données dans chaque barre
        pull_i = (prediction_bins[i] - comptes[i]) / erreurs_data[i]
        pulls.append(pull_i)
        somme_pulls_carre = somme_pulls_carre + pull_i ** 2

    plt.figure(figsize=(8.5, 3.8))
    # Zone repère entre -2 et +2 écarts-types
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
    plt.savefig(f"{dossier}/pulls_masse_D0.png")
    plt.close()

    # Nombre de degrés de liberté : barres moins les 5 paramètres flottants (mu, sigma, c1, n_sig, n_bkg)
    ndf = nbins - 5
    chi2_reduit = somme_pulls_carre / ndf
    return comptes, centres, erreurs_data, largeur_bin, chi2_reduit