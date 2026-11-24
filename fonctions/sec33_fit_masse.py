import numpy as np
import zfit
import matplotlib.pyplot as plt
from fonctions.sec31_32_donnees import COULEUR_POINTS, COULEUR_FIT, COULEUR_BRUIT, COULEUR_PULLS


def fit_donnees(donnees, obs, model, poids=None):
    data = zfit.Data(data=donnees.to_numpy(), obs=obs, weights=poids)
    nll = zfit.loss.ExtendedUnbinnedNLL(model=model, data=data)
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(nll).update_params()
    result.hesse()
    return result


def fit_masse(df, branche="D0_MM", min_fit=1816, max_fit=1914, nom_param=""):
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

    params = result.params
    mu_val = params[mu]["value"]
    sigma_val = params[sigma]["value"]
    S = params[n_sig]["value"]
    n_bkg_total = params[n_bkg]["value"]

    fenetre_pic = zfit.Space(branche, mu_val - 3 * sigma_val, mu_val + 3 * sigma_val)
    fraction_bruit_sous_pic = bkg_model.integrate(limits=fenetre_pic).numpy()[0]
    B = n_bkg_total * fraction_bruit_sous_pic

    return result, model, sig_model, bkg_model, donnees_masse, mu, sigma, n_sig, n_bkg, S, B


def trace_fit(donnees_masse, model, bkg_model, mu_val, mu_err, n_sig_val, n_bkg_val, min_fit=1816, max_fit=1914, nbins=100, dossier="histograms_plots"):
    largeur_bin = (max_fit - min_fit) / nbins
    comptes, bords = np.histogram(donnees_masse, bins=nbins, range=(min_fit, max_fit))

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

    x_lisse = np.linspace(min_fit, max_fit, 500)
    y_total_lisse = model.pdf(x_lisse).numpy() * (n_sig_val + n_bkg_val) * largeur_bin
    y_bkg_lisse = bkg_model.pdf(x_lisse).numpy() * n_bkg_val * largeur_bin

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
    plt.savefig(f"{dossier}/fit_masse_D0.png")
    plt.close()

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
    plt.savefig(f"{dossier}/pulls_masse_D0.png")
    plt.close()

    ndf = nbins - 5
    chi2_reduit = somme_pulls_carre / ndf
    return comptes, centres, erreurs_data, largeur_bin, chi2_reduit