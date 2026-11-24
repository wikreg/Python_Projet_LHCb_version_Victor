import matplotlib.pyplot as plt
from fonctions.sec31_32_donnees import (
    COULEUR_SIGNAL, COULEUR_BRUIT, configurer_style,
    charge_donnees, dessine_branche, dessine_branche_multiple
)

def executer(chemin_fichier="data/MasterclassData_2012_all.root", dossier="histograms_plots"):
    configurer_style(dossier)
    df_total = charge_donnees(chemin_fichier)

    liste_branches = list(df_total.columns)
    print("[3.1] Branches disponibles :", liste_branches)
    dessine_branche(df_total, liste_branches, echelle_log=True, dossier=dossier)

    cond_kaon = df_total["Kplus_ProbNNk"] > 0.95
    cond_pion = df_total["piminus_ProbNNpi"] > 0.8
    df_selectionne = df_total[cond_kaon & cond_pion]

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
    plt.savefig(f"{dossier}/comparaison_selection.png")
    plt.close()

    cond_pic = (df_total["D0_MM"] >= 1850) & (df_total["D0_MM"] <= 1882)
    df_signal_region = df_total[cond_pic]

    cond_bande_gauche = (df_total["D0_MM"] >= 1814) & (df_total["D0_MM"] <= 1849)
    cond_bande_droite = (df_total["D0_MM"] >= 1883) & (df_total["D0_MM"] <= 1914)
    df_bruit_region = df_total[cond_bande_gauche | cond_bande_droite]

    dessine_branche_multiple(
        [df_signal_region, df_bruit_region],
        ["Région Pic (Signal dominant)", "Bandes Latérales (Bruit)"],
        ["D0_DIRA_OWNPV", "D0_MINIPCHI2", "Kplus_ProbNNk", "piminus_ProbNNpi"],
        echelle_log=True, dossier=dossier
    )
    print("[3.2] Graphiques exploratoires enregistrés.")
    return df_total, df_selectionne

if __name__ == "__main__":
    executer()