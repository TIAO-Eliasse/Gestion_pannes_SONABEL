"""
Génération d'une base de données de séries temporelles multivariées
simulant le comportement de 50 transformateurs électriques (poteaux H61
et postes cabines) du réseau SONABEL sur l'année 2025, au pas horaire.

Objectif : produire un jeu de données réaliste pour un prototype de
maintenance prédictive (classification binaire "panne dans les 24h").

Auteur : script généré pour un prototype de concours d'innovation
         (SEMH-AES / SAMAO 2026).
"""

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# 0. Paramètres généraux
# ----------------------------------------------------------------------
# La graine aléatoire n'est fixée que dans le bloc d'exécution principal
# (voir bas de fichier) : un script qui importe ce module (par exemple
# generate_data_for_model.py) doit rester maître de sa propre graine.
N_POSTES = 50
START_DATE = "2025-01-01"
END_DATE = "2025-12-31 23:00:00"
FREQ = "h"  # pas horaire

date_range = pd.date_range(start=START_DATE, end=END_DATE, freq=FREQ)
N_HOURS = len(date_range)  # 8760 heures

# ----------------------------------------------------------------------
# 1. Référentiel géographique et technique de Ouagadougou (Kadiogo)
# ----------------------------------------------------------------------
# Quelques arrondissements et quartiers réels de Ouagadougou, utilisés
# pour peupler les variables géographiques fixes de chaque poste.
QUARTIERS = [
    ("Arrondissement 1", "Zagtouli"),
    ("Arrondissement 1", "Kossodo"),
    ("Arrondissement 3", "Patte d'Oie"),
    ("Arrondissement 3", "Gounghin"),
    ("Arrondissement 4", "Wemtenga"),
    ("Arrondissement 4", "Kilwin"),
    ("Arrondissement 5", "Ouaga 2000"),
    ("Arrondissement 5", "Bassinko"),
    ("Arrondissement 6", "Tanghin"),
    ("Arrondissement 12", "Cissin"),
]

TYPES_INSTAL = ["H61", "H61", "H61", "Cabine"]  # H61 majoritaire

# Positions centrales approximatives des quartiers, à but ILLUSTRATIF
# uniquement (non vérifiées comme géoréférencement officiel) — utilisées
# pour placer les postes sur une carte de démonstration et simuler des
# effets de proximité géographique entre postes voisins.
QUARTIER_CENTRES = {
    "Zagtouli": (12.3505, -1.5825),
    "Kossodo": (12.4102, -1.4780),
    "Patte d'Oie": (12.3670, -1.5310),
    "Gounghin": (12.3660, -1.5430),
    "Wemtenga": (12.3790, -1.4990),
    "Kilwin": (12.3620, -1.5130),
    "Ouaga 2000": (12.3280, -1.4850),
    "Bassinko": (12.4250, -1.5100),
    "Tanghin": (12.4050, -1.5250),
    "Cissin": (12.3350, -1.5400),
}
JITTER_DEG = 0.0018  # ≈ 200 m de dispersion autour du centre du quartier


def build_static_postes(n_postes: int) -> pd.DataFrame:
    """Construit le référentiel statique des n_postes transformateurs."""
    records = []
    for i in range(1, n_postes + 1):
        arrondissement, quartier = QUARTIERS[np.random.randint(len(QUARTIERS))]
        prefixe = quartier.upper().replace(" ", "-").replace("'", "")[:6]
        id_poste = f"{prefixe}-{i:02d}"
        lat_centre, lon_centre = QUARTIER_CENTRES[quartier]
        records.append(
            {
                "ID_Poste": id_poste,
                "Region": "Kadiogo",
                "Commune": "Ouagadougou",
                "Arrondissement": arrondissement,
                "Quartier": quartier,
                "Age_Annees": np.random.randint(5, 31),  # 5 à 30 ans inclus
                "Type_Instal": np.random.choice(TYPES_INSTAL),
                "Latitude": round(lat_centre + np.random.uniform(-JITTER_DEG, JITTER_DEG), 6),
                "Longitude": round(lon_centre + np.random.uniform(-JITTER_DEG, JITTER_DEG), 6),
            }
        )
    return pd.DataFrame(records)


# ----------------------------------------------------------------------
# 2. Fonctions de simulation des séries dynamiques
# ----------------------------------------------------------------------
def simulate_temperature_externe(dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Température extérieure (°C) avec :
      - cycle journalier (creux ~4h, pic ~14h)
      - cycle annuel type Burkina Faso (pic mars-mai > 40°C,
        saison des pluies plus fraîche juillet-septembre,
        harmattan plus frais décembre-janvier)
      - bruit gaussien
    """
    hour = dates.hour.values
    day_of_year = dates.dayofyear.values

    # Cycle journalier : amplitude ~8°C autour d'une moyenne mobile annuelle,
    # minimum vers 4h, maximum vers 14h -> décalage de phase.
    cycle_journalier = 8 * np.sin(2 * np.pi * (hour - 8) / 24)

    # Cycle annuel : température moyenne de base qui varie selon la saison.
    # On calibre pour que mars-mai (jours ~60-150) culmine au-delà de 40°C,
    # et que juillet-septembre (saison des pluies) soit plus frais.
    base_annuelle = 30 + 8 * np.sin(2 * np.pi * (day_of_year - 75) / 365)

    bruit = np.random.normal(0, 1.2, size=len(dates))

    temp = base_annuelle + cycle_journalier + bruit
    return temp


def simulate_charge(dates: pd.DatetimeIndex, base_charge: float) -> np.ndarray:
    """
    Charge électrique (Ampères) avec pic de soirée (18h-23h,
    climatisation + éclairage domestique), creux nocturne, et bruit gaussien.
    """
    hour = dates.hour.values

    # Profil horaire normalisé (0 à 1) : creux la nuit, pic en soirée.
    # Construit comme une somme de gaussiennes centrées sur les heures de
    # pointe (12h "petit pic" repas + 20h "grand pic" soirée).
    pic_soir = np.exp(-((hour - 20) ** 2) / (2 * 3.5 ** 2))
    pic_midi = 0.35 * np.exp(-((hour - 12.5) ** 2) / (2 * 2.5 ** 2))
    profil = 0.25 + 0.6 * pic_soir + pic_midi
    profil = np.clip(profil, 0.15, None)

    bruit = np.random.normal(0, base_charge * 0.06, size=len(dates))
    charge = base_charge * profil + bruit
    return np.clip(charge, 0, None)


def simulate_temp_huile(temp_ext: np.ndarray, charge: np.ndarray) -> np.ndarray:
    """
    Température de l'huile du transformateur (°C), modélisée comme une
    fonction retardée (lissage exponentiel) de la température extérieure
    et de la charge : l'huile chauffe avec un temps de réponse thermique,
    d'où l'usage d'un filtre exponentiel plutôt qu'une dépendance instantanée.
    """
    n = len(temp_ext)
    temp_huile = np.zeros(n)

    # Contribution instantanée cible : température ambiante + échauffement
    # proportionnel au carré de la charge relative (pertes Joule ~ I²).
    charge_norm = charge / (np.max(charge) + 1e-6)
    cible = temp_ext + 35 * (charge_norm ** 1.5)

    alpha = 0.15  # coefficient d'inertie thermique (retard)
    temp_huile[0] = cible[0]
    for t in range(1, n):
        temp_huile[t] = temp_huile[t - 1] + alpha * (cible[t] - temp_huile[t - 1])

    bruit = np.random.normal(0, 0.5, size=n)
    return temp_huile + bruit


# ----------------------------------------------------------------------
# 3. Simulation complète, poste par poste (logique de panne séquentielle)
# ----------------------------------------------------------------------
# Valeurs par défaut = logique physique originale du prototype. Un appelant
# (ex. generate_data_for_model.py) peut passer des seuils différents sans
# dupliquer cette fonction, pour obtenir un jeu de données avec davantage
# de cas positifs à des fins de modélisation/démonstration.
SEUIL_CHARGE_CRITIQUE_RATIO = 0.85  # 85% de la charge de base = "critique"
SEUIL_TEMP_PANNE = 40.0             # °C
CONSEC_HEURES_REQUISES = 3          # heures consécutives en régime critique
PROBA_PANNE_SI_CONDITIONS = 0.80    # probabilité de bascule en panne


def simulate_one_poste(
    id_poste: str,
    age: int,
    dates: pd.DatetimeIndex,
    seuil_charge_critique_ratio: float = SEUIL_CHARGE_CRITIQUE_RATIO,
    seuil_temp_panne: float = SEUIL_TEMP_PANNE,
    consec_heures_requises: int = CONSEC_HEURES_REQUISES,
    proba_panne_si_conditions: float = PROBA_PANNE_SI_CONDITIONS,
) -> pd.DataFrame:
    """Simule les séries dynamiques et la logique de panne pour un poste."""
    n = len(dates)

    # Charge de base propre à chaque poste (variabilité inter-transfo).
    base_charge = np.random.uniform(60, 140)

    temp_ext = simulate_temperature_externe(dates)
    charge = simulate_charge(dates, base_charge)
    temp_huile = simulate_temp_huile(temp_ext, charge)

    panne_reelle = np.zeros(n, dtype=int)
    # Poste_Actif = 1 tant que l'équipement fonctionne normalement, 0 après
    # la panne. Sert à exclure les lignes post-panne (charge/temp figées à
    # 0, triviales à classer) d'une évaluation honnête du modèle.
    poste_actif = np.ones(n, dtype=int)
    poste_deja_en_panne = False

    seuil_charge_critique = base_charge * seuil_charge_critique_ratio

    # Compteur d'heures consécutives en régime "critique"
    compteur_consecutif = 0

    for t in range(n):
        if poste_deja_en_panne:
            # Après la panne : charge et température d'huile chutent à 0
            charge[t] = 0.0
            temp_huile[t] = 0.0
            poste_actif[t] = 0
            continue

        condition_age = age > 15
        condition_temp = temp_ext[t] > seuil_temp_panne
        condition_charge = charge[t] > seuil_charge_critique

        if condition_age and condition_temp and condition_charge:
            compteur_consecutif += 1
        else:
            compteur_consecutif = 0

        if compteur_consecutif >= consec_heures_requises:
            if np.random.random() < proba_panne_si_conditions:
                panne_reelle[t] = 1
                poste_deja_en_panne = True

    df = pd.DataFrame(
        {
            "Date_Heure": dates,
            "ID_Poste": id_poste,
            "Temp_Ext_C": np.round(temp_ext, 2),
            "Charge_Amperes": np.round(charge, 2),
            "Temp_Huile_C": np.round(temp_huile, 2),
            "Panne_Reelle": panne_reelle,
            "Poste_Actif": poste_actif,
        }
    )
    return df


def build_full_dataset(
    postes_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    **seuils,
) -> pd.DataFrame:
    """
    Assemble le dataset complet (dynamique + statique + cible + lags) pour
    un référentiel de postes donné. `**seuils` est transmis tel quel à
    simulate_one_poste (seuil_charge_critique_ratio, seuil_temp_panne,
    consec_heures_requises, proba_panne_si_conditions) — omettre un seuil
    revient à garder sa valeur par défaut.
    """
    frames = [
        simulate_one_poste(row["ID_Poste"], row["Age_Annees"], dates, **seuils)
        for _, row in postes_df.iterrows()
    ]
    dynamic_df = pd.concat(frames, ignore_index=True)

    df = dynamic_df.merge(postes_df, on="ID_Poste", how="left")
    colonnes_ordre = [
        "Date_Heure", "ID_Poste", "Region", "Commune", "Arrondissement", "Quartier",
        "Latitude", "Longitude",
        "Age_Annees", "Type_Instal",
        "Temp_Ext_C", "Charge_Amperes", "Temp_Huile_C", "Panne_Reelle", "Poste_Actif",
    ]
    df = df[colonnes_ordre]
    df = df.sort_values(["ID_Poste", "Date_Heure"]).reset_index(drop=True)

    # Variable cible : Panne_Dans_24h (fenêtre glissante rétroactive)
    df["Panne_Dans_24h"] = 0
    for id_poste, grp in df.groupby("ID_Poste"):
        idx_pannes = grp.index[grp["Panne_Reelle"] == 1]
        for idx_panne in idx_pannes:
            pos_panne = grp.index.get_loc(idx_panne)
            pos_debut = max(0, pos_panne - 24)
            indices_fenetre = grp.index[pos_debut:pos_panne]  # 24h précédentes
            df.loc[indices_fenetre, "Panne_Dans_24h"] = 1

    # Variables retardées (lag features) pour le Machine Learning
    df["Moyenne_Mobile_Charge_6h"] = (
        df.groupby("ID_Poste")["Charge_Amperes"]
        .transform(lambda s: s.rolling(window=6, min_periods=1).mean())
        .round(2)
    )
    df["Gradient_Temp_3h"] = (
        df.groupby("ID_Poste")["Temp_Huile_C"]
        .transform(lambda s: s.diff(periods=3))
        .fillna(0)
        .round(2)
    )

    # Charge relative à la ligne de base du poste : la règle de panne
    # dépend de la charge EN PROPORTION du base_charge propre à chaque
    # poste, jamais exposé tel quel — sans cette feature, le modèle et
    # SHAP ne voient que la charge absolue, qui n'a pas le même sens de
    # risque d'un poste à l'autre (fuite indirecte non exploitable).
    # On approxime la ligne de base par la moyenne de charge du poste sur
    # les seules heures actives (hors période post-panne à 0).
    base_estimee = (
        df[df["Poste_Actif"] == 1].groupby("ID_Poste")["Charge_Amperes"].transform("mean")
    )
    df["Charge_Ratio_Base"] = np.nan
    df.loc[df["Poste_Actif"] == 1, "Charge_Ratio_Base"] = (
        df.loc[df["Poste_Actif"] == 1, "Charge_Amperes"] / base_estimee
    ).round(3)
    df["Charge_Ratio_Base"] = df["Charge_Ratio_Base"].fillna(0)

    df["Type_Instal_Num"] = (df["Type_Instal"] == "H61").astype(int)
    return df

# ----------------------------------------------------------------------
# 7. Exécution principale — ne s'exécute QUE si ce fichier est lancé
# directement (python generate_sonabel_timeseries.py), jamais lors d'un
# `import` depuis un autre script (ex. generate_data_for_model.py), pour
# ne jamais écraser silencieusement sonabel_timeseries_24h.csv.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    postes_df = build_static_postes(N_POSTES)
    df = build_full_dataset(postes_df, date_range)

    OUTPUT_PATH = "sonabel_timeseries_24h.csv"
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Fichier exporté : {OUTPUT_PATH}")
    print(f"Dimensions du dataset : {df.shape[0]} lignes x {df.shape[1]} colonnes")
    print(f"Nombre de postes simulés : {df['ID_Poste'].nunique()}")
    print(f"Nombre total de pannes réelles : {df['Panne_Reelle'].sum()}")
    print(f"Nombre de lignes en pré-alerte (Panne_Dans_24h=1) : {df['Panne_Dans_24h'].sum()}")
    print(df.head(10).to_string(index=False))
