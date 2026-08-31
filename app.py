# """
# Dispositif de maintenance prédictive SONABEL — application Streamlit.

# Cinq onglets :
#   1. Accueil — présentation du dispositif
#   2. Tableau de bord — vue d'ensemble, filtres, graphique des postes à risque
#   3. Explicabilité — détail d'un poste + SHAP
#   4. Collaboration humain-IA — validation des alertes, journal des décisionsF
#   5. Agent IA — assistant conversationnel (API Anthropic)

# Lancement local :
#     streamlit run app.py

# Prérequis : définir ANTHROPIC_API_KEY (variable d'environnement ou
# .streamlit/secrets.toml) pour activer l'agent conversationnel — le
# reste de l'application fonctionne sans clé.
# """

# import os
# from datetime import datetime

# import joblib
# import numpy as np
# import pandas as pd
# import shap
# import streamlit as st
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# import plotly.express as px

# # ----------------------------------------------------------------------
# # Configuration générale
# # ----------------------------------------------------------------------
# st.set_page_config(
#     page_title="SONABEL — Maintenance prédictive",
#     page_icon=None,
#     layout="wide",
# )

# DATA_PATH = "sonabel_model_dataset.csv"
# MODEL_PATH = "sonabel_model.joblib"
# FEEDBACK_LOG_PATH = "feedback_log.csv"
# SNAPSHOT_DATE = "2025-03-01 14:00:00"

# FEATURES = [
#     "Temp_Ext_C", "Charge_Amperes", "Temp_Huile_C", "Age_Annees",
#     "Type_Instal_Num", "Moyenne_Mobile_Charge_6h", "Gradient_Temp_3h",
#     "Charge_Ratio_Base",
# ]
# FEATURE_LABELS = {
#     "Temp_Ext_C": "Température extérieure",
#     "Charge_Amperes": "Charge électrique",
#     "Temp_Huile_C": "Température de l'huile",
#     "Age_Annees": "Âge de l'équipement",
#     "Type_Instal_Num": "Type d'installation (H61)",
#     "Moyenne_Mobile_Charge_6h": "Moyenne mobile de charge (6h)",
#     "Gradient_Temp_3h": "Gradient de température huile (3h)",
#     "Charge_Ratio_Base": "Charge relative à la ligne de base du poste",
# }

# # Palette — identité visuelle du dispositif
# INK = "#1A2027"
# MUTED = "#6B7684"
# BORDER = "#E4E7EC"
# BG_PAGE = "#F7F8FA"
# ACCENT = "#0E7C86"       # teal électrique — accent principal
# ACCENT_DARK = "#0A5C64"
# CRITIQUE = "#B23A2E"
# SURVEILLANCE = "#B98900"
# NORMAL_C = "#2E7D4F"
# CRITIQUE_BG = "#FBEAE7"
# SURVEILLANCE_BG = "#FBF3DB"
# NORMAL_BG = "#E8F3EC"

# # ----------------------------------------------------------------------
# # Style global — pas d'emoji, palette disciplinée, en-tête façon OMOA
# # ----------------------------------------------------------------------
# st.markdown(f"""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

# html, body, [class*="css"] {{
#     font-family: 'Inter', -apple-system, sans-serif;
# }}
# .stApp {{ background: {BG_PAGE}; }}
# #MainMenu, footer, header {{ visibility: hidden; }}
# .block-container {{ padding-top: 1.2rem; max-width: 1200px; }}

# /* En-tête */
# .topbar {{
#     display: flex; align-items: center; justify-content: space-between;
#     padding: 14px 26px; background: white; border: 1px solid {BORDER};
#     border-radius: 10px; margin-bottom: 22px;
# }}
# .topbar-brand {{ display: flex; align-items: center; gap: 12px; }}
# .topbar-mark {{ width: 30px; height: 30px; border-radius: 6px; background: {ACCENT}; }}
# .topbar-title {{ font-size: 16px; font-weight: 700; color: {INK}; line-height: 1.1; }}
# .topbar-sub {{ font-size: 11.5px; color: {MUTED}; }}
# .topbar-tag {{
#     font-size: 11px; font-weight: 600; color: {ACCENT_DARK};
#     background: #E4F2F1; padding: 5px 12px; border-radius: 20px;
# }}

# /* Onglets */
# .stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {BORDER}; }}
# .stTabs [data-baseweb="tab"] {{
#     height: 42px; font-weight: 600; font-size: 14px; color: {MUTED};
#     padding: 0 4px;
# }}
# .stTabs [aria-selected="true"] {{ color: {ACCENT_DARK} !important; }}

# /* Cartes métriques */
# .kpi-card {{
#     background: white; border: 1px solid {BORDER}; border-radius: 10px;
#     padding: 16px 18px;
# }}
# .kpi-label {{ font-size: 12.5px; color: {MUTED}; font-weight: 500; margin-bottom: 6px; }}
# .kpi-value {{ font-size: 30px; font-weight: 700; color: {INK}; line-height: 1; }}

# /* Hero (accueil) — fond sombre, halo dégradé + texture de points,
#    dans l'esprit "plateforme tech" (inspiré, pas copié) */
# .hero {{
#     position: relative; overflow: hidden;
#     border-radius: 14px; padding: 54px 44px; margin-bottom: 22px;
#     background: #0B1414;
#     color: white;
# }}
# .hero::before {{
#     content: ""; position: absolute; inset: 0;
#     background-image:
#         radial-gradient(circle at 18% 20%, rgba(20,150,158,0.55) 0%, transparent 42%),
#         radial-gradient(circle at 82% 15%, rgba(185,137,0,0.30) 0%, transparent 38%),
#         radial-gradient(circle at 60% 90%, rgba(14,124,134,0.35) 0%, transparent 45%);
#     filter: blur(6px);
# }}
# .hero::after {{
#     content: ""; position: absolute; inset: 0;
#     background-image: radial-gradient(rgba(255,255,255,0.16) 1px, transparent 1px);
#     background-size: 24px 24px;
#     mask-image: radial-gradient(ellipse at center, black 0%, transparent 75%);
# }}
# .hero > * {{ position: relative; z-index: 1; }}
# .hero h1 {{ font-size: 32px; font-weight: 700; margin: 0 0 10px 0; }}
# .hero p {{ font-size: 15.5px; line-height: 1.65; color: #D7E4E3; max-width: 700px; margin: 0; }}

# .feature-card {{
#     background: white; border: 1px solid {BORDER}; border-radius: 10px;
#     padding: 18px 20px; height: 100%;
# }}
# .feature-card h4 {{ font-size: 14.5px; font-weight: 700; color: {INK}; margin: 0 0 8px 0; }}
# .feature-card p {{ font-size: 13px; color: {MUTED}; line-height: 1.6; margin: 0; }}
# .feature-index {{
#     display: inline-block; font-size: 11px; font-weight: 700; color: {ACCENT_DARK};
#     background: #E4F2F1; border-radius: 6px; padding: 3px 8px; margin-bottom: 10px;
# }}

# .disclaimer {{
#     border: 1px solid {BORDER}; border-left: 3px solid {ACCENT}; border-radius: 6px;
#     padding: 12px 16px; font-size: 12.5px; color: {MUTED}; background: white;
# }}

# /* Badges de statut (sans emoji) */
# .badge {{
#     display: inline-flex; align-items: center; gap: 6px;
#     font-size: 12px; font-weight: 600; padding: 3px 10px; border-radius: 20px;
# }}
# .dot {{ width: 7px; height: 7px; border-radius: 50%; display: inline-block; }}

# /* Ligne poste (tableau de bord) */
# .poste-row {{
#     display: flex; justify-content: space-between; align-items: center;
#     padding: 10px 14px; border: 1px solid {BORDER}; border-radius: 8px;
#     background: white; margin-bottom: 6px;
# }}
# .poste-id {{ font-family: 'Inter', sans-serif; font-weight: 600; font-size: 13.5px; color: {INK}; }}
# .poste-meta {{ font-size: 12px; color: {MUTED}; }}
# </style>
# """, unsafe_allow_html=True)


# def badge_html(statut):
#     color = {"Critique": CRITIQUE, "Surveillance": SURVEILLANCE, "Normal": NORMAL_C}[statut]
#     bg = {"Critique": CRITIQUE_BG, "Surveillance": SURVEILLANCE_BG, "Normal": NORMAL_BG}[statut]
#     return (f'<span class="badge" style="color:{color};background:{bg};">'
#             f'<span class="dot" style="background:{color};"></span>{statut}</span>')


# # ----------------------------------------------------------------------
# # Chargement — données, modèle, explainer (mis en cache)
# # ----------------------------------------------------------------------
# @st.cache_resource
# def load_model():
#     return joblib.load(MODEL_PATH)


# @st.cache_resource
# def load_explainer(_model):
#     return shap.TreeExplainer(_model)


# @st.cache_data
# def load_snapshot():
#     df = pd.read_csv(DATA_PATH, parse_dates=["Date_Heure"])
#     snap = df[df["Date_Heure"] == SNAPSHOT_DATE].copy()
#     return snap.sort_values("ID_Poste").reset_index(drop=True)


# def load_feedback_log():
#     if os.path.exists(FEEDBACK_LOG_PATH):
#         return pd.read_csv(FEEDBACK_LOG_PATH)
#     return pd.DataFrame(columns=["horodatage", "poste", "action", "note"])


# def append_feedback(poste, action, note):
#     log = load_feedback_log()
#     new_row = pd.DataFrame([{
#         "horodatage": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         "poste": poste, "action": action, "note": note,
#     }])
#     pd.concat([log, new_row], ignore_index=True).to_csv(FEEDBACK_LOG_PATH, index=False)


# model = load_model()
# explainer = load_explainer(model)
# snapshot = load_snapshot()

# # Score de risque relatif (0-100) au sein du parc à cet instant, plutôt
# # que la probabilité brute — plus lisible et plus stable pour un tableau
# # de bord de supervision (voir README pour la justification).
# raw_proba = model.predict_proba(snapshot[FEATURES])[:, 1]
# p_min, p_max = raw_proba.min(), raw_proba.max()
# snapshot["risque_modele"] = np.round((raw_proba - p_min) / (p_max - p_min + 1e-9) * 100, 1)


# def statut_from_risque(r):
#     if r >= 70:
#         return "Critique"
#     if r >= 35:
#         return "Surveillance"
#     return "Normal"


# # ----------------------------------------------------------------------
# # Effet de proximité géographique : un poste critique peut indiquer un
# # stress localisé (même départ électrique, même vague de chaleur locale) —
# # les postes voisins dans un rayon de RAYON_IMPACT_M voient leur risque
# # relevé, avec une atténuation linéaire selon la distance. Positions
# # illustratives (voir generate_sonabel_timeseries.py), pas un GPS certifié.
# RAYON_IMPACT_M = 150
# BOOST_MAX = 25


# def distance_metres(lat1, lon1, lat2, lon2):
#     R = 6371000
#     phi1, phi2 = np.radians(lat1), np.radians(lat2)
#     dphi = np.radians(lat2 - lat1)
#     dlambda = np.radians(lon2 - lon1)
#     a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
#     return 2 * R * np.arcsin(np.sqrt(a))


# snapshot["statut_modele"] = snapshot["risque_modele"].apply(statut_from_risque)
# sources = snapshot[snapshot["statut_modele"] == "Critique"]

# boosts = np.zeros(len(snapshot))
# for idx, row in snapshot.iterrows():
#     for _, src in sources.iterrows():
#         if src["ID_Poste"] == row["ID_Poste"]:
#             continue
#         d = distance_metres(row["Latitude"], row["Longitude"], src["Latitude"], src["Longitude"])
#         if d <= RAYON_IMPACT_M:
#             boosts[idx] = max(boosts[idx], BOOST_MAX * (1 - d / RAYON_IMPACT_M))

# snapshot["risque"] = np.clip(snapshot["risque_modele"] + boosts, 0, 100).round(1)
# snapshot["proximite_alerte"] = boosts > 0
# snapshot["statut"] = snapshot["risque"].apply(statut_from_risque)
# snapshot = snapshot.sort_values("risque", ascending=False).reset_index(drop=True)


# def circle_points(lat, lon, radius_m, n=48):
#     lat_r = np.radians(lat)
#     d_lat = radius_m / 111320
#     d_lon = radius_m / (111320 * np.cos(lat_r) + 1e-9)
#     angles = np.linspace(0, 2 * np.pi, n)
#     return lat + d_lat * np.sin(angles), lon + d_lon * np.cos(angles)
# # ✅ REMPLACER LA FONCTION build_network_map() PAR CELLE-CI
# # ✅ REMPLACER LA FONCTION build_network_map() PAR CELLE-CI

# def build_network_map(df_map):
#     """Construit la carte avec gestion complète des erreurs et diagnostic."""
    
#     # ════════════════════════════════════════════════════════════
#     # ÉTAPE 1 : Vérification basique
#     # ════════════════════════════════════════════════════════════
#     if df_map is None or df_map.empty:
#         fig = go.Figure()
#         fig.add_annotation(
#             text="Aucun poste ne correspond aux filtres sélectionnés.",
#             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
#         )
#         fig.update_layout(height=460, mapbox_style="open-street-map")
#         return fig
    
#     # ════════════════════════════════════════════════════════════
#     # ÉTAPE 2 : Vérifier les colonnes essentielles
#     # ════════════════════════════════════════════════════════════
#     required_cols = ["Latitude", "Longitude", "ID_Poste", "statut"]
#     missing_cols = [c for c in required_cols if c not in df_map.columns]
#     if missing_cols:
#         st.error(f"❌ Colonnes manquantes : {', '.join(missing_cols)}")
#         st.write("Colonnes disponibles :", df_map.columns.tolist())
#         return go.Figure()
    
#     # ════════════════════════════════════════════════════════════
#     # ÉTAPE 3 : Préparer les données (nettoyer + convertir)
#     # ════════════════════════════════════════════════════════════
#     df_work = df_map.copy()
    
#     # Convertir lat/lon en numeric
#     df_work["Latitude"] = pd.to_numeric(df_work["Latitude"], errors='coerce')
#     df_work["Longitude"] = pd.to_numeric(df_work["Longitude"], errors='coerce')
    
#     # Supprimer les lignes avec NaN en lat/lon
#     initial_count = len(df_work)
#     df_work = df_work.dropna(subset=["Latitude", "Longitude"])
#     if len(df_work) < initial_count:
#         st.info(f"ℹ️ {initial_count - len(df_work)} postes ignorés (coordonnées invalides)")
    
#     if df_work.empty:
#         st.warning("⚠️ Aucun poste avec coordonnées géographiques valides après nettoyage.")
#         return go.Figure()
    
#     # ════════════════════════════════════════════════════════════
#     # ÉTAPE 4 : Valider et nettoyer la colonne "statut"
#     # ════════════════════════════════════════════════════════════
#     statuts_attendus = {"Critique", "Surveillance", "Normal"}
#     statuts_uniques = df_work["statut"].fillna("MANQUANT").unique()
#     statuts_invalides = set(statuts_uniques) - statuts_attendus
    
#     if "MANQUANT" in statuts_uniques:
#         st.warning(f"⚠️ {(df_work['statut'].isna()).sum()} postes ont un statut manquant")
#         df_work = df_work[df_work["statut"].notna()]
    
#     if statuts_invalides:
#         st.warning(f"⚠️ Statuts inattendus : {statuts_invalides}. Suppression…")
#         df_work = df_work[df_work["statut"].isin(statuts_attendus)]
    
#     if df_work.empty:
#         st.error("❌ Aucun poste avec un statut valide après nettoyage.")
#         return go.Figure()
    
#     # ════════════════════════════════════════════════════════════
#     # ÉTAPE 5 : Construire le color_discrete_map dynamiquement
#     # ════════════════════════════════════════════════════════════
#     color_map_full = {
#         "Critique": CRITIQUE, 
#         "Surveillance": SURVEILLANCE, 
#         "Normal": NORMAL_C
#     }
    
#     # Ne garder que les couleurs pour les statuts présents
#     color_map_used = {k: v for k, v in color_map_full.items() if k in df_work["statut"].values}
    
#     # ════════════════════════════════════════════════════════════
#     # ÉTAPE 6 : Créer la carte Plotly (using graph_objects pour compatibilité)
#     # ════════════════════════════════════════════════════════════
#     try:
#         fig = go.Figure()
        
#         # Créer une trace par statut (pour les couleurs discrètes)
#         for statut in color_map_used.keys():
#             df_statut = df_work[df_work["statut"] == statut]
#             if df_statut.empty:
#                 continue
            
#             hover_text = [
#                 f"<b>{row['ID_Poste']}</b><br>" +
#                 f"Quartier: {row.get('Quartier', 'N/A')}<br>" +
#                 f"Risque: {row.get('risque', 'N/A')}%"
#                 for _, row in df_statut.iterrows()
#             ]
            
#             fig.add_trace(go.Scattermapbox(
#                 lat=df_statut["Latitude"],
#                 lon=df_statut["Longitude"],
#                 mode="markers",
#                 marker=dict(
#                     size=13,
#                     opacity=0.8,
#                     color=color_map_used[statut]
#                 ),
#                 text=hover_text,
#                 hoverinfo="text",
#                 name=statut,
#                 showlegend=True,
#             ))
        
#     except Exception as e:
#         st.error(f"❌ Erreur lors de la création de la carte Plotly :")
#         st.code(str(e), language="python")
#         return go.Figure()
    
#     # (Les marqueurs sont déjà personnalisés dans la création des traces)
    
#     # ════════════════════════════════════════════════════════════
#     # ÉTAPE 8 : Ajouter les zones d'alerte (rayon d'impact)
#     # ════════════════════════════════════════════════════════════
#     try:
#         critiques = df_work[df_work["statut_modele"] == "Critique"]
#         for _, src in critiques.iterrows():
#             if pd.notna(src["Latitude"]) and pd.notna(src["Longitude"]):
#                 lats, lons = circle_points(
#                     float(src["Latitude"]), 
#                     float(src["Longitude"]), 
#                     RAYON_IMPACT_M
#                 )
#                 fig.add_trace(go.Scattermapbox(
#                     lat=lats, lon=lons, mode="lines",
#                     line=dict(color=CRITIQUE, width=1.5),
#                     fill="toself", fillcolor="rgba(178,58,46,0.12)",
#                     hoverinfo="skip", showlegend=False,
#                 ))
#     except KeyError:
#         st.warning("⚠️ Colonne 'statut_modele' manquante pour les zones d'alerte.")
#     except Exception as e:
#         st.warning(f"⚠️ Impossible d'ajouter les zones d'alerte : {str(e)}")
    
#     # ════════════════════════════════════════════════════════════
#     # ÉTAPE 9 : Finaliser le layout
#     # ════════════════════════════════════════════════════════════
#     fig.update_layout(
#         mapbox_style="open-street-map",
#         mapbox_zoom=11.5,
#         mapbox_center=dict(
#             lat=df_work["Latitude"].mean(),
#             lon=df_work["Longitude"].mean()
#         ),
#         height=460,
#         margin=dict(l=0, r=0, t=8, b=0),
#         legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
#         font=dict(family="Inter, sans-serif"),
#         showlegend=True,
#     )
    
#     return fig
# # ----------------------------------------------------------------------
# # SHAP — facteurs contributifs pour un poste
# # ----------------------------------------------------------------------
# def get_shap_row(poste_id):
#     row_idx = snapshot.index[snapshot["ID_Poste"] == poste_id][0]
#     row = snapshot.loc[[row_idx], FEATURES]
#     shap_values = explainer.shap_values(row)
#     if np.ndim(shap_values) == 3:
#         values = shap_values[0, :, 1]
#         base_value = explainer.expected_value[1]
#     else:
#         values = shap_values[0]
#         base_value = explainer.expected_value
#     return row.iloc[0], values, base_value

# # ✅ FONCTION CORRIGÉE : plot_shap_waterfall

# def plot_shap_waterfall(poste_id):
#     """
#     Crée un graphique SHAP waterfall pour un poste, avec gestion robuste de matplotlib.
#     Évite les problèmes de tight_layout() en Matplotlib 3.14+
#     """
#     row, values, base_value = get_shap_row(poste_id)
#     labels = [FEATURE_LABELS[f] for f in FEATURES]
#     exp = shap.Explanation(
#         values=values, 
#         base_values=base_value,
#         data=row.values, 
#         feature_names=labels
#     )
    
#     plt.rcParams["font.family"] = "sans-serif"
#     fig, ax = plt.subplots(figsize=(6.8, 3.6))
    
#     try:
#         # Essayer tight_layout avec gestion d'erreur
#         shap.plots.waterfall(exp, max_display=7, show=False)
#         try:
#             plt.tight_layout()
#         except ValueError as e:
#             # Si tight_layout échoue, utiliser subplots_adjust à la place
#             st.warning("⚠️ Ajustement automatique du layout (tight_layout indisponible)")
#             plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
#     except Exception as e:
#         st.error(f"❌ Erreur lors du rendu SHAP : {str(e)}")
#         plt.close(fig)
#         return None
    
#     return fig

# # ----------------------------------------------------------------------
# # Agent conversationnel (API Anthropic)
# # ----------------------------------------------------------------------
# def get_anthropic_client():
#     api_key = None
#     try:
#         api_key = st.secrets.get("ANTHROPIC_API_KEY")
#     except Exception:
#         pass
#     api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
#     if not api_key:
#         return None
#     import anthropic
#     return anthropic.Anthropic(api_key=api_key)


# def build_system_prompt():
#     lignes = [
#         f"- {r['ID_Poste']} ({r['Quartier']}, {r['Type_Instal']}, {r['Age_Annees']} ans) : "
#         f"risque {r['risque']}% — statut {r['statut']} — Temp_Ext={r['Temp_Ext_C']}°C, "
#         f"Charge={r['Charge_Amperes']}A, Temp_Huile={r['Temp_Huile_C']}°C"
#         for _, r in snapshot.iterrows()
#     ]
#     return f"""Tu es l'agent d'aide à la décision du dispositif de maintenance prédictive de SONABEL.
# Instantané simulé du réseau au {SNAPSHOT_DATE} :
# {chr(10).join(lignes)}

# Règles :
# - Réponds uniquement à partir de ces données. Pour un poste hors de cette liste, dis qu'il est hors du périmètre de cette démonstration.
# - Explique toujours par les facteurs concrets (température huile, charge, âge) et cite l'identifiant du poste.
# - Pour une priorisation, classe par risque décroissant.
# - Tu recommandes, tu ne déclenches jamais d'action : la décision finale revient au technicien.
# - Réponds en français, de façon concise et professionnelle."""


# def ask_agent(question, history):
#     client = get_anthropic_client()
#     if client is None:
#         return ("Agent indisponible : aucune clé ANTHROPIC_API_KEY n'est configurée "
#                 "(variable d'environnement ou .streamlit/secrets.toml).")
#     response = client.messages.create(
#         model="claude-sonnet-4-6", max_tokens=600,
#         system=build_system_prompt(), messages=history + [{"role": "user", "content": question}],
#     )
#     return "".join(b.text for b in response.content if b.type == "text")


# # ----------------------------------------------------------------------
# # En-tête
# # ----------------------------------------------------------------------
# st.markdown(f"""
# <div class="topbar">
#   <div class="topbar-brand">
#     <div class="topbar-mark"></div>
#     <div>
#       <div class="topbar-title">Maintenance prédictive — SONABEL</div>
#       <div class="topbar-sub">Concours d'Innovations Énergie et Hydrocarbures · SEMH-AES / SAMAO 2026</div>
#     </div>
#   </div>
#   <div class="topbar-tag">Démonstration — données simulées</div>
# </div>
# """, unsafe_allow_html=True)

# tab_accueil, tab_dashboard, tab_explicabilite, tab_collab, tab_agent = st.tabs(
#     ["Accueil", "Tableau de bord", "Explicabilité", "Collaboration humain-IA", "Agent IA"]
# )

# if "poste_selectionne" not in st.session_state:
#     st.session_state["poste_selectionne"] = snapshot.iloc[0]["ID_Poste"]

# # ----------------------------------------------------------------------
# # Onglet 1 — Accueil
# # ----------------------------------------------------------------------
# with tab_accueil:
#     st.markdown("""
#     <div class="hero">
#       <h1>Anticiper les pannes avant qu'elles n'arrivent</h1>
#       <p>Un dispositif de maintenance prédictive pour le réseau de transformateurs de SONABEL :
#       un modèle prédit le risque de panne à 24h, explique chaque alerte poste par poste,
#       et laisse la décision finale au technicien. Conçu pour le Concours d'Innovations
#       Énergie et Hydrocarbures (SEMH-AES / SAMAO 2026).</p>
#     </div>
#     """, unsafe_allow_html=True)

#     c1, c2, c3 = st.columns(3)
#     with c1:
#         st.markdown("""
#         <div class="feature-card">
#           <div class="feature-index">01</div>
#           <h4>Prédiction</h4>
#           <p>Un modèle de classification entraîné sur la télémétrie horaire de 50 postes
#           (température huile, charge, âge, type d'installation) estime le risque de panne
#           dans les 24 heures.</p>
#         </div>
#         """, unsafe_allow_html=True)
#     with c2:
#         st.markdown("""
#         <div class="feature-card">
#           <div class="feature-index">02</div>
#           <h4>Explicabilité</h4>
#           <p>Chaque alerte est décomposée par SHAP : quels facteurs — température, charge,
#           âge de l'équipement — poussent le risque à la hausse pour ce poste précis.</p>
#         </div>
#         """, unsafe_allow_html=True)
#     with c3:
#         st.markdown("""
#         <div class="feature-card">
#           <div class="feature-index">03</div>
#           <h4>Collaboration humain-IA</h4>
#           <p>Le technicien confirme, reporte ou écarte chaque alerte, avec un agent conversationnel
#           pour approfondir. Le système recommande, il ne décide jamais seul.</p>
#         </div>
#         """, unsafe_allow_html=True)

#     st.write("")
#     st.markdown(f"""
#     <div class="disclaimer">
#       Ce dispositif s'appuie sur des données de télémétrie <strong>simulées</strong>
#       (température extérieure, charge électrique, âge des équipements) autour de
#       Ouagadougou (Kadiogo), calibrées pour être cohérentes avec le climat et le
#       comportement du réseau — il ne s'agit pas d'une connexion au réseau réel de SONABEL.
#       L'instantané affiché correspond au {SNAPSHOT_DATE}. Les positions cartographiques
#       des postes sont également illustratives, pour situer visuellement les zones
#       concernées — elles ne constituent pas un géoréférencement officiel.
#     </div>
#     """, unsafe_allow_html=True)

# # ----------------------------------------------------------------------
# # Onglet 2 — Tableau de bord
# # ----------------------------------------------------------------------
# with tab_dashboard:
#     k1, k2, k3, k4 = st.columns(4)
#     for col, label, value in [
#         (k1, "Postes suivis", len(snapshot)),
#         (k2, "Critiques", int((snapshot["statut"] == "Critique").sum())),
#         (k3, "Sous surveillance", int((snapshot["statut"] == "Surveillance").sum())),
#         (k4, "Normaux", int((snapshot["statut"] == "Normal").sum())),
#     ]:
#         col.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
#                       f'<div class="kpi-value">{value}</div></div>', unsafe_allow_html=True)

#     st.write("")
#     f1, f2, f3 = st.columns(3)
#     arrondissements = sorted(snapshot["Arrondissement"].unique())
#     sel_arr = f1.multiselect("Arrondissement", arrondissements, default=[])
#     filtered_for_quartier = snapshot[snapshot["Arrondissement"].isin(sel_arr)] if sel_arr else snapshot
#     quartiers = sorted(filtered_for_quartier["Quartier"].unique())
#     sel_quartier = f2.multiselect("Quartier", quartiers, default=[])
#     sel_type = f3.multiselect("Type d'installation", sorted(snapshot["Type_Instal"].unique()), default=[])

#     filtered = snapshot.copy()
#     if sel_arr:
#         filtered = filtered[filtered["Arrondissement"].isin(sel_arr)]
#     if sel_quartier:
#         filtered = filtered[filtered["Quartier"].isin(sel_quartier)]
#     if sel_type:
#         filtered = filtered[filtered["Type_Instal"].isin(sel_type)]

#     st.write("")
#     n_dispo = len(filtered)
#     if n_dispo <= 1:
#         top_n = n_dispo
#         st.caption(f"{n_dispo} poste correspond à ces filtres." if n_dispo == 1 else "Aucun poste ne correspond à ces filtres.")
#     else:
#         slider_min = 1
#         slider_max = min(30, n_dispo)
#         slider_default = min(12, slider_max)
#         top_n = st.slider("Nombre de postes affichés dans le classement", slider_min, slider_max, slider_default)
#     top = filtered.sort_values("risque", ascending=False).head(top_n).iloc[::-1] if top_n else filtered.iloc[::-1]

#     colors = [{"Critique": CRITIQUE, "Surveillance": SURVEILLANCE, "Normal": NORMAL_C}[s] for s in top["statut"]]
#     fig = go.Figure(go.Bar(
#         x=top["risque"], y=top["ID_Poste"], orientation="h",
#         marker=dict(color=colors),
#         text=[f"{v:.0f}" for v in top["risque"]], textposition="outside",
#         hovertext=[f"{r.ID_Poste} — {r.Quartier}<br>Risque {r.risque}% — {r.statut}" for r in top.itertuples()],
#         hoverinfo="text",
#     ))
#     fig.update_layout(
#         title="Postes classés par risque relatif",
#         xaxis_title="Score de risque (0-100, relatif au parc)",
#         plot_bgcolor="white", paper_bgcolor="white",
#         font=dict(family="Inter, sans-serif", color=INK, size=12),
#         margin=dict(l=10, r=30, t=50, b=40), height=max(340, 26 * len(top)),
#         xaxis=dict(range=[0, 105], gridcolor=BORDER),
#         yaxis=dict(gridcolor=BORDER),
#     )
#     st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

#     st.write("")
#     st.markdown("**Carte du réseau**")
#     n_critiques = int((filtered["statut_modele"] == "Critique").sum())
#     if n_critiques:
#         st.caption(
#             f"Zone d'alerte : les postes situés à moins de {RAYON_IMPACT_M} m d'un poste critique "
#             "voient leur risque relevé (stress localisé probable — même départ électrique, "
#             "même vague de chaleur). Positions illustratives, non géoréférencées avec précision."
#         )
#     else:
#         st.caption("Positions illustratives, non géoréférencées avec précision.")
#     st.plotly_chart(build_network_map(filtered), use_container_width=True, config={"displaylogo": False})

#     st.write("")
#     st.markdown("**Détail des postes filtrés**")
#     table = filtered[["ID_Poste", "Quartier", "Arrondissement", "Type_Instal", "Age_Annees", "risque", "statut"]]
#     table = table.rename(columns={
#         "ID_Poste": "Poste", "Arrondissement": "Arrondissement", "Type_Instal": "Type",
#         "Age_Annees": "Âge (ans)", "risque": "Risque (%)",
#     })
#     st.dataframe(table, use_container_width=True, hide_index=True)
#     st.download_button(
#         "Télécharger ce classement (CSV)",
#         data=table.to_csv(index=False).encode("utf-8-sig"),
#         file_name="sonabel_postes_risque.csv", mime="text/csv",
#     )

# # ----------------------------------------------------------------------
# # Onglet 3 — Explicabilité
# # ----------------------------------------------------------------------
# with tab_explicabilite:
#     options = snapshot["ID_Poste"].tolist()
#     idx_default = options.index(st.session_state["poste_selectionne"]) if st.session_state["poste_selectionne"] in options else 0
#     poste_id = st.selectbox("Poste à analyser", options, index=idx_default,
#                              format_func=lambda pid: f"{pid} — {snapshot.loc[snapshot['ID_Poste']==pid,'Quartier'].values[0]}")
#     st.session_state["poste_selectionne"] = poste_id
#     poste_row = snapshot[snapshot["ID_Poste"] == poste_id].iloc[0]

#     st.markdown(badge_html(poste_row["statut"]), unsafe_allow_html=True)
#     st.markdown(f"#### {poste_id} — {poste_row['Quartier']}")
#     if poste_row["proximite_alerte"]:
#         st.caption(
#             f"Score de risque affiché : {poste_row['risque']}% "
#             f"(dont {poste_row['risque_modele']}% du modèle ML + effet de proximité géographique "
#             f"d'un poste critique voisin, à moins de {RAYON_IMPACT_M} m). "
#             "L'explicabilité SHAP ci-dessous porte sur la part expliquée par le modèle."
#         )
#     else:
#         st.caption(f"Score de risque relatif (modèle ML) : {poste_row['risque']}% — aucun effet de proximité détecté.")

#     m1, m2, m3, m4 = st.columns(4)
#     m1.metric("Temp. extérieure", f"{poste_row['Temp_Ext_C']} °C")
#     m2.metric("Charge", f"{poste_row['Charge_Amperes']} A")
#     m3.metric("Temp. huile", f"{poste_row['Temp_Huile_C']} °C")
#     m4.metric("Âge", f"{poste_row['Age_Annees']} ans")

#     st.markdown("**Facteurs contributifs (SHAP)**")
#     fig_shap = plot_shap_waterfall(poste_id)
#     st.pyplot(fig_shap, use_container_width=True)
#     plt.close(fig_shap)

# # ----------------------------------------------------------------------
# # Onglet 4 — Collaboration humain-IA
# # ----------------------------------------------------------------------
# with tab_collab:
#     options = snapshot["ID_Poste"].tolist()
#     idx_default = options.index(st.session_state["poste_selectionne"]) if st.session_state["poste_selectionne"] in options else 0
#     poste_id = st.selectbox("Poste concerné", options, index=idx_default, key="collab_select",
#                              format_func=lambda pid: f"{pid} — {snapshot.loc[snapshot['ID_Poste']==pid,'Quartier'].values[0]}")
#     st.session_state["poste_selectionne"] = poste_id
#     poste_row = snapshot[snapshot["ID_Poste"] == poste_id].iloc[0]

#     st.markdown(badge_html(poste_row["statut"]), unsafe_allow_html=True)
#     st.caption(f"Risque {poste_row['risque']}% — décision du technicien à journaliser")

#     note = st.text_input("Commentaire (optionnel)", key=f"note_{poste_id}")
#     b1, b2, b3 = st.columns(3)
#     if b1.button("Confirmer l'intervention", key=f"confirm_{poste_id}", use_container_width=True):
#         append_feedback(poste_id, "Intervention confirmée", note)
#         st.success(f"Intervention confirmée sur {poste_id} et journalisée.")
#     if b2.button("Reporter / surveiller", key=f"defer_{poste_id}", use_container_width=True):
#         append_feedback(poste_id, "Reporté sous surveillance", note)
#         st.info(f"{poste_id} placé sous surveillance renforcée.")
#     if b3.button("Ignorer (faux positif)", key=f"ignore_{poste_id}", use_container_width=True):
#         append_feedback(poste_id, "Ignoré (faux positif)", note)
#         st.warning(f"Alerte sur {poste_id} marquée comme faux positif.")

#     st.write("")
#     st.markdown("**Journal des décisions**")
#     log = load_feedback_log()
#     st.dataframe(log, use_container_width=True, hide_index=True)
#     if len(log):
#         st.download_button(
#             "Télécharger le journal (CSV)",
#             data=log.to_csv(index=False).encode("utf-8-sig"),
#             file_name="sonabel_journal_decisions.csv", mime="text/csv",
#         )

# # ----------------------------------------------------------------------
# # Onglet 5 — Agent IA
# # ----------------------------------------------------------------------
# with tab_agent:
#     st.caption("Posez une question sur un poste, une comparaison ou une priorité d'intervention.")

#     if "chat_history" not in st.session_state:
#         st.session_state["chat_history"] = []

#     for msg in st.session_state["chat_history"]:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])

#     question = st.chat_input("Posez votre question…")
#     if question:
#         st.session_state["chat_history"].append({"role": "user", "content": question})
#         with st.chat_message("user"):
#             st.markdown(question)
#         with st.chat_message("assistant"):
#             with st.spinner("L'agent analyse les données…"):
#                 reponse = ask_agent(question, st.session_state["chat_history"][:-1])
#             st.markdown(reponse)
#         st.session_state["chat_history"].append({"role": "assistant", "content": reponse})





"""
Dispositif de maintenance prédictive SONABEL — application Streamlit.

Cinq onglets :
  1. Accueil — présentation du dispositif
  2. Tableau de bord — vue d'ensemble, filtres, graphique des postes à risque
  3. Explicabilité — détail d'un poste + SHAP
  4. Collaboration humain-IA — validation des alertes, journal des décisions
  5. Agent IA — assistant conversationnel (API Anthropic)

Lancement local :
    streamlit run app.py

Prérequis : définir ANTHROPIC_API_KEY (variable d'environnement ou
.streamlit/secrets.toml) pour activer l'agent conversationnel — le
reste de l'application fonctionne sans clé.
"""

import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# ✅ IMPORTS FOLIUM AJOUTÉS
import folium
from folium import plugins
from streamlit_folium import st_folium

# ----------------------------------------------------------------------
# Configuration générale
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="SONABEL — Maintenance prédictive",
    page_icon=None,
    layout="wide",
)

DATA_PATH = "sonabel_model_dataset.csv"
MODEL_PATH = "sonabel_model.joblib"
FEEDBACK_LOG_PATH = "feedback_log.csv"
SNAPSHOT_DATE = "2025-03-01 14:00:00"

FEATURES = [
    "Temp_Ext_C", "Charge_Amperes", "Temp_Huile_C", "Age_Annees",
    "Type_Instal_Num", "Moyenne_Mobile_Charge_6h", "Gradient_Temp_3h",
    "Charge_Ratio_Base",
]
FEATURE_LABELS = {
    "Temp_Ext_C": "Température extérieure",
    "Charge_Amperes": "Charge électrique",
    "Temp_Huile_C": "Température de l'huile",
    "Age_Annees": "Âge de l'équipement",
    "Type_Instal_Num": "Type d'installation (H61)",
    "Moyenne_Mobile_Charge_6h": "Moyenne mobile de charge (6h)",
    "Gradient_Temp_3h": "Gradient de température huile (3h)",
    "Charge_Ratio_Base": "Charge relative à la ligne de base du poste",
}

# Palette — identité visuelle du dispositif
INK = "#1A2027"
MUTED = "#6B7684"
BORDER = "#E4E7EC"
BG_PAGE = "#F7F8FA"
ACCENT = "#0E7C86"       # teal électrique — accent principal
ACCENT_DARK = "#0A5C64"
CRITIQUE = "#B23A2E"
SURVEILLANCE = "#B98900"
NORMAL_C = "#2E7D4F"
CRITIQUE_BG = "#FBEAE7"
SURVEILLANCE_BG = "#FBF3DB"
NORMAL_BG = "#E8F3EC"

# ----------------------------------------------------------------------
# Style global — pas d'emoji, palette disciplinée, en-tête façon OMOA
# ----------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, sans-serif;
}}
.stApp {{ background: {BG_PAGE}; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1.2rem; max-width: 1200px; }}

/* En-tête */
.topbar {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 26px; background: white; border: 1px solid {BORDER};
    border-radius: 10px; margin-bottom: 22px;
}}
.topbar-brand {{ display: flex; align-items: center; gap: 12px; }}
.topbar-mark {{ width: 30px; height: 30px; border-radius: 6px; background: {ACCENT}; }}
.topbar-title {{ font-size: 16px; font-weight: 700; color: {INK}; line-height: 1.1; }}
.topbar-sub {{ font-size: 11.5px; color: {MUTED}; }}
.topbar-tag {{
    font-size: 11px; font-weight: 600; color: {ACCENT_DARK};
    background: #E4F2F1; padding: 5px 12px; border-radius: 20px;
}}

/* Onglets */
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{
    height: 42px; font-weight: 600; font-size: 14px; color: {MUTED};
    padding: 0 4px;
}}
.stTabs [aria-selected="true"] {{ color: {ACCENT_DARK} !important; }}

/* Cartes métriques */
.kpi-card {{
    background: white; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 16px 18px;
}}
.kpi-label {{ font-size: 12.5px; color: {MUTED}; font-weight: 500; margin-bottom: 6px; }}
.kpi-value {{ font-size: 30px; font-weight: 700; color: {INK}; line-height: 1; }}

/* Hero (accueil) — fond sombre, halo dégradé + texture de points,
   dans l'esprit "plateforme tech" (inspiré, pas copié) */
.hero {{
    position: relative; overflow: hidden;
    border-radius: 14px; padding: 54px 44px; margin-bottom: 22px;
    background: #0B1414;
    color: white;
}}
.hero::before {{
    content: ""; position: absolute; inset: 0;
    background-image:
        radial-gradient(circle at 18% 20%, rgba(20,150,158,0.55) 0%, transparent 42%),
        radial-gradient(circle at 82% 15%, rgba(185,137,0,0.30) 0%, transparent 38%),
        radial-gradient(circle at 60% 90%, rgba(14,124,134,0.35) 0%, transparent 45%);
    filter: blur(6px);
}}
.hero::after {{
    content: ""; position: absolute; inset: 0;
    background-image: radial-gradient(rgba(255,255,255,0.16) 1px, transparent 1px);
    background-size: 24px 24px;
    mask-image: radial-gradient(ellipse at center, black 0%, transparent 75%);
}}
.hero > * {{ position: relative; z-index: 1; }}
.hero h1 {{ font-size: 32px; font-weight: 700; margin: 0 0 10px 0; }}
.hero p {{ font-size: 15.5px; line-height: 1.65; color: #D7E4E3; max-width: 700px; margin: 0; }}

.feature-card {{
    background: white; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 18px 20px; height: 100%;
}}
.feature-card h4 {{ font-size: 14.5px; font-weight: 700; color: {INK}; margin: 0 0 8px 0; }}
.feature-card p {{ font-size: 13px; color: {MUTED}; line-height: 1.6; margin: 0; }}
.feature-index {{
    display: inline-block; font-size: 11px; font-weight: 700; color: {ACCENT_DARK};
    background: #E4F2F1; border-radius: 6px; padding: 3px 8px; margin-bottom: 10px;
}}

.disclaimer {{
    border: 1px solid {BORDER}; border-left: 3px solid {ACCENT}; border-radius: 6px;
    padding: 12px 16px; font-size: 12.5px; color: {MUTED}; background: white;
}}

/* Badges de statut (sans emoji) */
.badge {{
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 12px; font-weight: 600; padding: 3px 10px; border-radius: 20px;
}}
.dot {{ width: 7px; height: 7px; border-radius: 50%; display: inline-block; }}

/* Ligne poste (tableau de bord) */
.poste-row {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 14px; border: 1px solid {BORDER}; border-radius: 8px;
    background: white; margin-bottom: 6px;
}}
.poste-id {{ font-family: 'Inter', sans-serif; font-weight: 600; font-size: 13.5px; color: {INK}; }}
.poste-meta {{ font-size: 12px; color: {MUTED}; }}
</style>
""", unsafe_allow_html=True)


def badge_html(statut):
    color = {"Critique": CRITIQUE, "Surveillance": SURVEILLANCE, "Normal": NORMAL_C}[statut]
    bg = {"Critique": CRITIQUE_BG, "Surveillance": SURVEILLANCE_BG, "Normal": NORMAL_BG}[statut]
    return (f'<span class="badge" style="color:{color};background:{bg};">'
            f'<span class="dot" style="background:{color};"></span>{statut}</span>')


# ----------------------------------------------------------------------
# Chargement — données, modèle, explainer (mis en cache)
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


@st.cache_data
def load_snapshot():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date_Heure"])
    snap = df[df["Date_Heure"] == SNAPSHOT_DATE].copy()
    return snap.sort_values("ID_Poste").reset_index(drop=True)


def load_feedback_log():
    if os.path.exists(FEEDBACK_LOG_PATH):
        return pd.read_csv(FEEDBACK_LOG_PATH)
    return pd.DataFrame(columns=["horodatage", "poste", "action", "note"])


def append_feedback(poste, action, note):
    log = load_feedback_log()
    new_row = pd.DataFrame([{
        "horodatage": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "poste": poste, "action": action, "note": note,
    }])
    pd.concat([log, new_row], ignore_index=True).to_csv(FEEDBACK_LOG_PATH, index=False)


model = load_model()
explainer = load_explainer(model)
snapshot = load_snapshot()

# Score de risque relatif (0-100) au sein du parc à cet instant, plutôt
# que la probabilité brute — plus lisible et plus stable pour un tableau
# de bord de supervision (voir README pour la justification).
raw_proba = model.predict_proba(snapshot[FEATURES])[:, 1]
p_min, p_max = raw_proba.min(), raw_proba.max()
snapshot["risque_modele"] = np.round((raw_proba - p_min) / (p_max - p_min + 1e-9) * 100, 1)


def statut_from_risque(r):
    if r >= 70:
        return "Critique"
    if r >= 35:
        return "Surveillance"
    return "Normal"


# ----------------------------------------------------------------------
# Effet de proximité géographique : un poste critique peut indiquer un
# stress localisé (même départ électrique, même vague de chaleur locale) —
# les postes voisins dans un rayon de RAYON_IMPACT_M voient leur risque
# relevé, avec une atténuation linéaire selon la distance. Positions
# illustratives (voir generate_sonabel_timeseries.py), pas un GPS certifié.
RAYON_IMPACT_M = 150
BOOST_MAX = 25


def distance_metres(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


snapshot["statut_modele"] = snapshot["risque_modele"].apply(statut_from_risque)
sources = snapshot[snapshot["statut_modele"] == "Critique"]

boosts = np.zeros(len(snapshot))
for idx, row in snapshot.iterrows():
    for _, src in sources.iterrows():
        if src["ID_Poste"] == row["ID_Poste"]:
            continue
        d = distance_metres(row["Latitude"], row["Longitude"], src["Latitude"], src["Longitude"])
        if d <= RAYON_IMPACT_M:
            boosts[idx] = max(boosts[idx], BOOST_MAX * (1 - d / RAYON_IMPACT_M))

snapshot["risque"] = np.clip(snapshot["risque_modele"] + boosts, 0, 100).round(1)
snapshot["proximite_alerte"] = boosts > 0
snapshot["statut"] = snapshot["risque"].apply(statut_from_risque)
snapshot = snapshot.sort_values("risque", ascending=False).reset_index(drop=True)


def circle_points(lat, lon, radius_m, n=48):
    lat_r = np.radians(lat)
    d_lat = radius_m / 111320
    d_lon = radius_m / (111320 * np.cos(lat_r) + 1e-9)
    angles = np.linspace(0, 2 * np.pi, n)
    return lat + d_lat * np.sin(angles), lon + d_lon * np.cos(angles)


# ✅ NOUVELLE FONCTION : build_network_map avec FOLIUM
def build_network_map(df_map):
    """
    Construit la carte réseau avec Folium (compatible avec TOUTES les versions Plotly).
    Folium est plus robuste que Plotly Mapbox sur Streamlit Cloud.
    """
    
    # Vérification 1 : dataframe vide
    if df_map is None or df_map.empty:
        st.warning("Aucun poste ne correspond aux filtres sélectionnés.")
        return None
    
    # Vérification 2 : colonnes essentielles
    required_cols = ["Latitude", "Longitude", "ID_Poste", "statut"]
    missing_cols = [c for c in required_cols if c not in df_map.columns]
    if missing_cols:
        st.error(f"❌ Colonnes manquantes : {', '.join(missing_cols)}")
        return None
    
    # Vérification 3 : valeurs numériques valides
    df_work = df_map.copy()
    df_work["Latitude"] = pd.to_numeric(df_work["Latitude"], errors='coerce')
    df_work["Longitude"] = pd.to_numeric(df_work["Longitude"], errors='coerce')
    df_work = df_work.dropna(subset=["Latitude", "Longitude"])
    
    if df_work.empty:
        st.warning("⚠️ Aucun poste avec coordonnées géographiques valides.")
        return None
    
    # Définir les couleurs
    color_map = {
        "Critique": "red",
        "Surveillance": "orange",
        "Normal": "green"
    }
    
    # Nettoyer les statuts invalides
    statuts_attendus = {"Critique", "Surveillance", "Normal"}
    df_work = df_work[df_work["statut"].isin(statuts_attendus)]
    
    if df_work.empty:
        st.error("❌ Aucun poste avec un statut valide.")
        return None
    
    # Créer la carte centrée sur les données
    center_lat = df_work["Latitude"].mean()
    center_lon = df_work["Longitude"].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles="OpenStreetMap"
    )
    
    # Ajouter les marqueurs pour chaque poste
    for _, row in df_work.iterrows():
        statut = row["statut"]
        icon_color = color_map.get(statut, "gray")
        
        # Icône selon le statut
        if statut == "Critique":
            icon_prefix = "fa-exclamation-triangle"
        elif statut == "Surveillance":
            icon_prefix = "fa-exclamation-circle"
        else:
            icon_prefix = "fa-check-circle"
        
        # Popup avec infos
        popup_text = f"""
        <b>{row['ID_Poste']}</b><br>
        Quartier: {row.get('Quartier', 'N/A')}<br>
        Risque: {row.get('risque', 'N/A'):.1f}%<br>
        Statut: {statut}
        """
        
        folium.Marker(
            location=[row["Latitude"], row["Longitude"]],
            popup=folium.Popup(popup_text, max_width=250),
            icon=folium.Icon(color=icon_color, icon=icon_prefix, prefix="fa"),
            tooltip=row["ID_Poste"]
        ).add_to(m)
    
    # Ajouter les zones d'alerte (rayon d'impact) autour des postes critiques
    critiques = df_work[df_work["statut"] == "Critique"]
    
    for _, src in critiques.iterrows():
        if pd.notna(src["Latitude"]) and pd.notna(src["Longitude"]):
            folium.Circle(
                location=[src["Latitude"], src["Longitude"]],
                radius=RAYON_IMPACT_M,
                color="#B23A2E",
                fill=True,
                fillColor="#B23A2E",
                fillOpacity=0.12,
                weight=1.5,
                popup=f"Zone d'alerte : {src['ID_Poste']}",
                tooltip=f"Rayon d'impact {RAYON_IMPACT_M}m"
            ).add_to(m)
    
    # Ajouter une légende
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 200px; height: auto;
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px; border-radius: 5px;">
    <p style="margin: 0 0 10px 0;"><b>Légende</b></p>
    <p style="margin: 5px 0;"><i class="fa fa-exclamation-triangle" style="color:red"></i> Critique</p>
    <p style="margin: 5px 0;"><i class="fa fa-exclamation-circle" style="color:orange"></i> Surveillance</p>
    <p style="margin: 5px 0;"><i class="fa fa-check-circle" style="color:green"></i> Normal</p>
    <p style="margin: 10px 0 0 0; font-size: 11px; color: gray;">
    Zones roses = rayon d'impact (150m)
    </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m


# ✅ FONCTION CORRIGÉE : plot_shap_waterfall
def plot_shap_waterfall(poste_id):
    """
    Crée un graphique SHAP waterfall pour un poste, avec gestion robuste de matplotlib.
    Évite les problèmes de tight_layout() en Matplotlib 3.14+
    """
    row, values, base_value = get_shap_row(poste_id)
    labels = [FEATURE_LABELS[f] for f in FEATURES]
    exp = shap.Explanation(
        values=values, 
        base_values=base_value,
        data=row.values, 
        feature_names=labels
    )
    
    plt.rcParams["font.family"] = "sans-serif"
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    
    try:
        shap.plots.waterfall(exp, max_display=7, show=False)
        try:
            plt.tight_layout()
        except ValueError:
            # Si tight_layout échoue, utiliser subplots_adjust
            plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    except Exception as e:
        st.error(f"❌ Erreur SHAP : {str(e)}")
        plt.close(fig)
        return None
    
    return fig


# ✅ FONCTION POUR RÉCUPÉRER LES DONNÉES SHAP
def get_shap_row(poste_id):
    row_idx = snapshot.index[snapshot["ID_Poste"] == poste_id][0]
    row = snapshot.loc[[row_idx], FEATURES]
    shap_values = explainer.shap_values(row)
    if np.ndim(shap_values) == 3:
        values = shap_values[0, :, 1]
        base_value = explainer.expected_value[1]
    else:
        values = shap_values[0]
        base_value = explainer.expected_value
    return row.iloc[0], values, base_value


# ----------------------------------------------------------------------
# Agent conversationnel (API Anthropic)
# ----------------------------------------------------------------------
def get_anthropic_client():
    api_key = None
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        pass
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    import anthropic
    return anthropic.Anthropic(api_key=api_key)


def build_system_prompt():
    lignes = [
        f"- {r['ID_Poste']} ({r['Quartier']}, {r['Type_Instal']}, {r['Age_Annees']} ans) : "
        f"risque {r['risque']}% — statut {r['statut']} — Temp_Ext={r['Temp_Ext_C']}°C, "
        f"Charge={r['Charge_Amperes']}A, Temp_Huile={r['Temp_Huile_C']}°C"
        for _, r in snapshot.iterrows()
    ]
    return f"""Tu es l'agent d'aide à la décision du dispositif de maintenance prédictive de SONABEL.
Instantané simulé du réseau au {SNAPSHOT_DATE} :
{chr(10).join(lignes)}

Règles :
- Réponds uniquement à partir de ces données. Pour un poste hors de cette liste, dis qu'il est hors du périmètre de cette démonstration.
- Explique toujours par les facteurs concrets (température huile, charge, âge) et cite l'identifiant du poste.
- Pour une priorisation, classe par risque décroissant.
- Tu recommandes, tu ne déclenches jamais d'action : la décision finale revient au technicien.
- Réponds en français, de façon concise et professionnelle."""


def ask_agent(question, history):
    client = get_anthropic_client()
    if client is None:
        return ("Agent indisponible : aucune clé ANTHROPIC_API_KEY n'est configurée "
                "(variable d'environnement ou .streamlit/secrets.toml).")
    response = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=600,
        system=build_system_prompt(), messages=history + [{"role": "user", "content": question}],
    )
    return "".join(b.text for b in response.content if b.type == "text")


# ----------------------------------------------------------------------
# En-tête
# ----------------------------------------------------------------------
st.markdown(f"""
<div class="topbar">
  <div class="topbar-brand">
    <div class="topbar-mark"></div>
    <div>
      <div class="topbar-title">Maintenance prédictive — SONABEL</div>
      <div class="topbar-sub">Concours d'Innovations Énergie et Hydrocarbures · SEMH-AES / SAMAO 2026</div>
    </div>
  </div>
  <div class="topbar-tag">Démonstration — données simulées</div>
</div>
""", unsafe_allow_html=True)

tab_accueil, tab_dashboard, tab_explicabilite, tab_collab, tab_agent = st.tabs(
    ["Accueil", "Tableau de bord", "Explicabilité", "Collaboration humain-IA", "Agent IA"]
)

if "poste_selectionne" not in st.session_state:
    st.session_state["poste_selectionne"] = snapshot.iloc[0]["ID_Poste"]

# ----------------------------------------------------------------------
# Onglet 1 — Accueil
# ----------------------------------------------------------------------
with tab_accueil:
    st.markdown("""
    <div class="hero">
      <h1>Anticiper les pannes avant qu'elles n'arrivent</h1>
      <p>Un dispositif de maintenance prédictive pour le réseau de transformateurs de SONABEL :
      un modèle prédit le risque de panne à 24h, explique chaque alerte poste par poste,
      et laisse la décision finale au technicien. Conçu pour le Concours d'Innovations
      Énergie et Hydrocarbures (SEMH-AES / SAMAO 2026).</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="feature-card">
          <div class="feature-index">01</div>
          <h4>Prédiction</h4>
          <p>Un modèle de classification entraîné sur la télémétrie horaire de 50 postes
          (température huile, charge, âge, type d'installation) estime le risque de panne
          dans les 24 heures.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="feature-card">
          <div class="feature-index">02</div>
          <h4>Explicabilité</h4>
          <p>Chaque alerte est décomposée par SHAP : quels facteurs — température, charge,
          âge de l'équipement — poussent le risque à la hausse pour ce poste précis.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="feature-card">
          <div class="feature-index">03</div>
          <h4>Collaboration humain-IA</h4>
          <p>Le technicien confirme, reporte ou écarte chaque alerte, avec un agent conversationnel
          pour approfondir. Le système recommande, il ne décide jamais seul.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown(f"""
    <div class="disclaimer">
      Ce dispositif s'appuie sur des données de télémétrie <strong>simulées</strong>
      (température extérieure, charge électrique, âge des équipements) autour de
      Ouagadougou (Kadiogo), calibrées pour être cohérentes avec le climat et le
      comportement du réseau — il ne s'agit pas d'une connexion au réseau réel de SONABEL.
      L'instantané affiché correspond au {SNAPSHOT_DATE}. Les positions cartographiques
      des postes sont également illustratives, pour situer visuellement les zones
      concernées — elles ne constituent pas un géoréférencement officiel.
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Onglet 2 — Tableau de bord
# ----------------------------------------------------------------------
with tab_dashboard:
    k1, k2, k3, k4 = st.columns(4)
    for col, label, value in [
        (k1, "Postes suivis", len(snapshot)),
        (k2, "Critiques", int((snapshot["statut"] == "Critique").sum())),
        (k3, "Sous surveillance", int((snapshot["statut"] == "Surveillance").sum())),
        (k4, "Normaux", int((snapshot["statut"] == "Normal").sum())),
    ]:
        col.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                      f'<div class="kpi-value">{value}</div></div>', unsafe_allow_html=True)

    st.write("")
    f1, f2, f3 = st.columns(3)
    arrondissements = sorted(snapshot["Arrondissement"].unique())
    sel_arr = f1.multiselect("Arrondissement", arrondissements, default=[])
    filtered_for_quartier = snapshot[snapshot["Arrondissement"].isin(sel_arr)] if sel_arr else snapshot
    quartiers = sorted(filtered_for_quartier["Quartier"].unique())
    sel_quartier = f2.multiselect("Quartier", quartiers, default=[])
    sel_type = f3.multiselect("Type d'installation", sorted(snapshot["Type_Instal"].unique()), default=[])

    filtered = snapshot.copy()
    if sel_arr:
        filtered = filtered[filtered["Arrondissement"].isin(sel_arr)]
    if sel_quartier:
        filtered = filtered[filtered["Quartier"].isin(sel_quartier)]
    if sel_type:
        filtered = filtered[filtered["Type_Instal"].isin(sel_type)]

    st.write("")
    n_dispo = len(filtered)
    if n_dispo <= 1:
        top_n = n_dispo
        st.caption(f"{n_dispo} poste correspond à ces filtres." if n_dispo == 1 else "Aucun poste ne correspond à ces filtres.")
    else:
        slider_min = 1
        slider_max = min(30, n_dispo)
        slider_default = min(12, slider_max)
        top_n = st.slider("Nombre de postes affichés dans le classement", slider_min, slider_max, slider_default)
    top = filtered.sort_values("risque", ascending=False).head(top_n).iloc[::-1] if top_n else filtered.iloc[::-1]

    colors = [{"Critique": CRITIQUE, "Surveillance": SURVEILLANCE, "Normal": NORMAL_C}[s] for s in top["statut"]]
    fig = go.Figure(go.Bar(
        x=top["risque"], y=top["ID_Poste"], orientation="h",
        marker=dict(color=colors),
        text=[f"{v:.0f}" for v in top["risque"]], textposition="outside",
        hovertext=[f"{r.ID_Poste} — {r.Quartier}<br>Risque {r.risque}% — {r.statut}" for r in top.itertuples()],
        hoverinfo="text",
    ))
    fig.update_layout(
        title="Postes classés par risque relatif",
        xaxis_title="Score de risque (0-100, relatif au parc)",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", color=INK, size=12),
        margin=dict(l=10, r=30, t=50, b=40), height=max(340, 26 * len(top)),
        xaxis=dict(range=[0, 105], gridcolor=BORDER),
        yaxis=dict(gridcolor=BORDER),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    st.write("")
    st.markdown("**Carte du réseau**")
    n_critiques = int((filtered["statut_modele"] == "Critique").sum())
    if n_critiques:
        st.caption(
            f"Zone d'alerte : les postes situés à moins de {RAYON_IMPACT_M} m d'un poste critique "
            "voient leur risque relevé (stress localisé probable — même départ électrique, "
            "même vague de chaleur). Positions illustratives, non géoréférencées avec précision."
        )
    else:
        st.caption("Positions illustratives, non géoréférencées avec précision.")
    
    # ✅ AFFICHAGE FOLIUM AULIEU DE PLOTLY
    m = build_network_map(filtered)
    if m is not None:
        st_folium(m, width=1200, height=460)

    st.write("")
    st.markdown("**Détail des postes filtrés**")
    table = filtered[["ID_Poste", "Quartier", "Arrondissement", "Type_Instal", "Age_Annees", "risque", "statut"]]
    table = table.rename(columns={
        "ID_Poste": "Poste", "Arrondissement": "Arrondissement", "Type_Instal": "Type",
        "Age_Annees": "Âge (ans)", "risque": "Risque (%)",
    })
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button(
        "Télécharger ce classement (CSV)",
        data=table.to_csv(index=False).encode("utf-8-sig"),
        file_name="sonabel_postes_risque.csv", mime="text/csv",
    )

# ----------------------------------------------------------------------
# Onglet 3 — Explicabilité
# ----------------------------------------------------------------------
with tab_explicabilite:
    options = snapshot["ID_Poste"].tolist()
    idx_default = options.index(st.session_state["poste_selectionne"]) if st.session_state["poste_selectionne"] in options else 0
    poste_id = st.selectbox("Poste à analyser", options, index=idx_default,
                             format_func=lambda pid: f"{pid} — {snapshot.loc[snapshot['ID_Poste']==pid,'Quartier'].values[0]}")
    st.session_state["poste_selectionne"] = poste_id
    poste_row = snapshot[snapshot["ID_Poste"] == poste_id].iloc[0]

    st.markdown(badge_html(poste_row["statut"]), unsafe_allow_html=True)
    st.markdown(f"#### {poste_id} — {poste_row['Quartier']}")
    if poste_row["proximite_alerte"]:
        st.caption(
            f"Score de risque affiché : {poste_row['risque']}% "
            f"(dont {poste_row['risque_modele']}% du modèle ML + effet de proximité géographique "
            f"d'un poste critique voisin, à moins de {RAYON_IMPACT_M} m). "
            "L'explicabilité SHAP ci-dessous porte sur la part expliquée par le modèle."
        )
    else:
        st.caption(f"Score de risque relatif (modèle ML) : {poste_row['risque']}% — aucun effet de proximité détecté.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Temp. extérieure", f"{poste_row['Temp_Ext_C']} °C")
    m2.metric("Charge", f"{poste_row['Charge_Amperes']} A")
    m3.metric("Temp. huile", f"{poste_row['Temp_Huile_C']} °C")
    m4.metric("Âge", f"{poste_row['Age_Annees']} ans")

    st.markdown("**Facteurs contributifs (SHAP)**")
    fig_shap = plot_shap_waterfall(poste_id)
    if fig_shap is not None:
        st.pyplot(fig_shap, use_container_width=True)
        plt.close(fig_shap)

# ----------------------------------------------------------------------
# Onglet 4 — Collaboration humain-IA
# ----------------------------------------------------------------------
with tab_collab:
    options = snapshot["ID_Poste"].tolist()
    idx_default = options.index(st.session_state["poste_selectionne"]) if st.session_state["poste_selectionne"] in options else 0
    poste_id = st.selectbox("Poste concerné", options, index=idx_default, key="collab_select",
                             format_func=lambda pid: f"{pid} — {snapshot.loc[snapshot['ID_Poste']==pid,'Quartier'].values[0]}")
    st.session_state["poste_selectionne"] = poste_id
    poste_row = snapshot[snapshot["ID_Poste"] == poste_id].iloc[0]

    st.markdown(badge_html(poste_row["statut"]), unsafe_allow_html=True)
    st.caption(f"Risque {poste_row['risque']}% — décision du technicien à journaliser")

    note = st.text_input("Commentaire (optionnel)", key=f"note_{poste_id}")
    b1, b2, b3 = st.columns(3)
    if b1.button("Confirmer l'intervention", key=f"confirm_{poste_id}", use_container_width=True):
        append_feedback(poste_id, "Intervention confirmée", note)
        st.success(f"Intervention confirmée sur {poste_id} et journalisée.")
    if b2.button("Reporter / surveiller", key=f"defer_{poste_id}", use_container_width=True):
        append_feedback(poste_id, "Reporté sous surveillance", note)
        st.info(f"{poste_id} placé sous surveillance renforcée.")
    if b3.button("Ignorer (faux positif)", key=f"ignore_{poste_id}", use_container_width=True):
        append_feedback(poste_id, "Ignoré (faux positif)", note)
        st.warning(f"Alerte sur {poste_id} marquée comme faux positif.")

    st.write("")
    st.markdown("**Journal des décisions**")
    log = load_feedback_log()
    st.dataframe(log, use_container_width=True, hide_index=True)
    if len(log):
        st.download_button(
            "Télécharger le journal (CSV)",
            data=log.to_csv(index=False).encode("utf-8-sig"),
            file_name="sonabel_journal_decisions.csv", mime="text/csv",
        )

# ----------------------------------------------------------------------
# Onglet 5 — Agent IA
# ----------------------------------------------------------------------
with tab_agent:
    st.caption("Posez une question sur un poste, une comparaison ou une priorité d'intervention.")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Posez votre question…")
    if question:
        st.session_state["chat_history"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("L'agent analyse les données…"):
                reponse = ask_agent(question, st.session_state["chat_history"][:-1])
            st.markdown(reponse)
        st.session_state["chat_history"].append({"role": "assistant", "content": reponse})
