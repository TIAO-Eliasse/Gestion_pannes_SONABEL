# Dispositif de maintenance prédictive SONABEL — application Streamlit

Application de démonstration pour le Concours d'Innovations Énergie et
Hydrocarbures (SEMH-AES / SAMAO 2026). Elle réunit :

- un modèle de classification (RandomForest) entraîné sur des données
  de télémétrie simulées, prédisant `Panne_Dans_24h` ;
- l'explicabilité du modèle via SHAP (waterfall plot par poste) ;
- une couche de collaboration humain-IA (confirmer / reporter / ignorer
  une alerte, avec journal des décisions) ;
- un agent conversationnel (API Anthropic) pour interroger le système
  en langage naturel.

## Fichiers

| Fichier | Rôle |
|---|---|
| `generate_sonabel_timeseries.py` | Génère la base de séries temporelles originale (respecte le prompt initial). |
| `generate_data_for_model.py` | Variante avec seuils de panne assouplis, pour obtenir assez de cas positifs à modéliser. |
| `train_and_shap_test.py` | Entraîne le modèle et sauvegarde `sonabel_model.joblib`. |
| `sonabel_model_dataset.csv` | Jeu de données utilisé par l'application. |
| `sonabel_model.joblib` | Modèle entraîné, chargé par l'application. |
| `app.py` | L'application Streamlit elle-même. |
| `requirements.txt` | Dépendances Python. |

## Lancer en local

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="votre_clé_api"   # optionnel : active l'agent conversationnel
streamlit run app.py
```

L'application s'ouvre automatiquement dans votre navigateur
(`http://localhost:8501`). Sans clé API, tout le dispositif fonctionne
normalement (modèle, SHAP, collaboration humain-IA) sauf la section
« Agent d'aide à la décision », qui affichera un message l'indiquant.

## Déployer gratuitement (Streamlit Community Cloud)

1. Poussez ce dossier dans un dépôt GitHub (public ou privé).
2. Allez sur [share.streamlit.io](https://share.streamlit.io), connectez
   votre compte GitHub, et sélectionnez le dépôt + `app.py` comme fichier
   principal.
3. Dans les paramètres de l'app (« Secrets »), ajoutez :
   ```toml
   ANTHROPIC_API_KEY = "votre_clé_api"
   ```
4. Déployez — vous obtenez une URL publique (`https://xxx.streamlit.app`)
   à montrer directement au jury, ou à intégrer dans votre dossier de
   candidature.

## Notes pour la présentation au jury

- Le jeu de données est **simulé** (télémétrie synthétique cohérente
  avec le climat du Burkina et le comportement électrique attendu) —
  précisez-le clairement, c'est un prototype, pas une connexion au
  réseau réel de SONABEL.
- Le modèle est volontairement entraîné avec des seuils de panne
  assouplis par rapport à la logique physique initiale, pour obtenir
  suffisamment de cas positifs à expliquer et à démontrer. Une version
  en production s'appuierait sur un historique réel de pannes.
- L'agent ne déclenche jamais d'action automatique : il répond à des
  questions et propose des priorités, la décision reste au technicien
  — c'est le point à souligner pour le critère « collaboration
  humain-IA ».
