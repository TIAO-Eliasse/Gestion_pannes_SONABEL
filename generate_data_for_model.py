"""
Régénère le dataset avec des seuils de panne assouplis par rapport à la
logique physique originale (generate_sonabel_timeseries.py), afin d'obtenir
suffisamment de cas positifs pour entraîner et démontrer le classifieur.

Réutilise simulate_one_poste / build_full_dataset du module d'origine —
aucune duplication de la logique de simulation.
"""

import numpy as np
from generate_sonabel_timeseries import (
    build_static_postes, build_full_dataset, date_range,
)

np.random.seed(7)
N_POSTES = 50

# Seuils assouplis pour la démonstration (voir README pour la justification).
SEUILS_DEMO = dict(
    seuil_charge_critique_ratio=0.70,
    seuil_temp_panne=38.0,
    consec_heures_requises=2,
    proba_panne_si_conditions=0.85,
)

postes_df = build_static_postes(N_POSTES)
df = build_full_dataset(postes_df, date_range, **SEUILS_DEMO)

df.to_csv("sonabel_model_dataset.csv", index=False)
print("Pannes réelles:", df["Panne_Reelle"].sum())
print("Lignes Panne_Dans_24h=1:", df["Panne_Dans_24h"].sum(), "/", len(df))
print("Lignes actives (hors post-panne):", (df["Poste_Actif"] == 1).sum(), "/", len(df))
print(df["ID_Poste"].nunique(), "postes")
