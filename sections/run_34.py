from fonctions.sec31_32_donnees import configurer_style, charge_donnees
from fonctions.sec34_fom import FOM_variable

# Script exécutable pour la section 3.4 : optimisation des coupures DIRA et MINIPCHI2 par Figure de Mérite (FOM)
# df_total, df_selectionne -> DataFrames optionnels (rechargés et filtrés si le script est lancé seul)
# dossier -> dossier où sauvegarder les courbes FOM
def executer(df_total=None, df_selectionne=None, dossier="histograms_plots"):
    # Initialise le style sombre et crée le dossier de sortie si besoin
    configurer_style(dossier)
    # Charge les données et applique les coupures PID si le script est exécuté directement
    if df_total is None:
        df_total = charge_donnees("data/MasterclassData_2012_all.root")
        df_selectionne = df_total[(df_total["Kplus_ProbNNk"] > 0.95) & (df_total["piminus_ProbNNpi"] > 0.8)]

    # Teste plusieurs seuils sur D0_DIRA_OWNPV après sélection PID pour trouver celui qui maximise la FOM
    seuils_dira = [0.9990, 0.9993, 0.9995, 0.9997, 0.9998, 0.9999, 0.99995]
    opt_dira, fom_dira = FOM_variable(df_selectionne, "D0_DIRA_OWNPV", seuils_dira, type_coupure="superieur", dossier=dossier)
    print(f"\n[3.4] Meilleur seuil D0_DIRA_OWNPV (avec PID) : {opt_dira} | FOM max : {fom_dira:.2f}")

    # Garde les évènements passant le seuil DIRA optimal, puis optimise D0_MINIPCHI2 (coupure vers le bas)
    df_apres_dira = df_selectionne[df_selectionne["D0_DIRA_OWNPV"] > opt_dira]
    seuils_ip = [25, 20, 16, 12, 9, 6, 4]
    opt_ip, fom_ip = FOM_variable(df_apres_dira, "D0_MINIPCHI2", seuils_ip, type_coupure="inferieur", dossier=dossier)
    print(f"[3.4] Meilleur seuil D0_MINIPCHI2 (avec PID) : {opt_ip} | FOM max : {fom_ip:.2f}")

    # Teste aussi l'optimisation de D0_DIRA_OWNPV sans les coupures PID pour comparer
    opt_dira_sans_pid, fom_sans_pid = FOM_variable(
        df_total, "D0_DIRA_OWNPV", seuils_dira, type_coupure="superieur", suffixe_fichier="_sans_PID", dossier=dossier
    )
    print(f"[3.4] Meilleur seuil D0_DIRA_OWNPV (SANS PID) : {opt_dira_sans_pid} | FOM max : {fom_sans_pid:.2f}")

    # Construit le DataFrame final avec toutes les coupures optimisées
    masque_opt = (
        (df_total["Kplus_ProbNNk"] > 0.95) &
        (df_total["piminus_ProbNNpi"] > 0.8) &
        (df_total["D0_DIRA_OWNPV"] > opt_dira) &
        (df_total["D0_MINIPCHI2"] < opt_ip)
    )
    df_optimal = df_total[masque_opt]
    return df_optimal, opt_dira, opt_ip

# Permet d'exécuter cette section directement (ex. python -m sections.run_34)
if __name__ == "__main__":
    executer()