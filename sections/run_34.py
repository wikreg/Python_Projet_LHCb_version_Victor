from fonctions.sec31_32_donnees import configurer_style, charge_donnees
from fonctions.sec34_fom import FOM_variable

def executer(df_total=None, df_selectionne=None, dossier="histograms_plots"):
    configurer_style(dossier)
    if df_total is None:
        df_total = charge_donnees("data/MasterclassData_2012_all.root")
        df_selectionne = df_total[(df_total["Kplus_ProbNNk"] > 0.95) & (df_total["piminus_ProbNNpi"] > 0.8)]

    seuils_dira = [0.9990, 0.9993, 0.9995, 0.9997, 0.9998, 0.9999, 0.99995]
    opt_dira, fom_dira = FOM_variable(df_selectionne, "D0_DIRA_OWNPV", seuils_dira, type_coupure="superieur", dossier=dossier)
    print(f"\n[3.4] Meilleur seuil D0_DIRA_OWNPV (avec PID) : {opt_dira} | FOM max : {fom_dira:.2f}")

    df_apres_dira = df_selectionne[df_selectionne["D0_DIRA_OWNPV"] > opt_dira]
    seuils_ip = [25, 20, 16, 12, 9, 6, 4]
    opt_ip, fom_ip = FOM_variable(df_apres_dira, "D0_MINIPCHI2", seuils_ip, type_coupure="inferieur", dossier=dossier)
    print(f"[3.4] Meilleur seuil D0_MINIPCHI2 (avec PID) : {opt_ip} | FOM max : {fom_ip:.2f}")

    opt_dira_sans_pid, fom_sans_pid = FOM_variable(
        df_total, "D0_DIRA_OWNPV", seuils_dira, type_coupure="superieur", suffixe_fichier="_sans_PID", dossier=dossier
    )
    print(f"[3.4] Meilleur seuil D0_DIRA_OWNPV (SANS PID) : {opt_dira_sans_pid} | FOM max : {fom_sans_pid:.2f}")

    masque_opt = (
        (df_total["Kplus_ProbNNk"] > 0.95) &
        (df_total["piminus_ProbNNpi"] > 0.8) &
        (df_total["D0_DIRA_OWNPV"] > opt_dira) &
        (df_total["D0_MINIPCHI2"] < opt_ip)
    )
    df_optimal = df_total[masque_opt]
    return df_optimal, opt_dira, opt_ip

if __name__ == "__main__":
    executer()