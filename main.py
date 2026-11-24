from sections import run_31_32, run_33, run_34, run_35, run_36, run_37

FICHIER_DONNEES = "data/MasterclassData_2012_all.root"
DOSSIER_SORTIE = "histograms_plots"

# 1. Chargement et exploration (Sections 3.1 et 3.2)
df_total, df_selectionne = run_31_32.executer(FICHIER_DONNEES, DOSSIER_SORTIE)

# 2. Fit de masse initial et pulls (Section 3.3)
df_propre, comptes, centres, erreurs_data, largeur_bin, chi2_reduit = run_33.executer(df_total, DOSSIER_SORTIE)

# 3. Optimisation des coupures par FOM (Section 3.4)
df_optimal, opt_dira, opt_ip = run_34.executer(df_total, df_selectionne, DOSSIER_SORTIE)

# 4. Extraction de la durée de vie par fit exponentiel (Section 3.5)
run_35.executer(df_optimal, DOSSIER_SORTIE)

# 5. Extraction de la durée de vie par sPlot (Section 3.6)
run_36.executer(df_optimal, DOSSIER_SORTIE)

# 6. Modèles avancés et vérifications (Section 3.7)
run_37.executer(df_propre, df_optimal, comptes, centres, erreurs_data, largeur_bin, chi2_reduit, DOSSIER_SORTIE)