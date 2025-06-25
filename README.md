#  Projet SMART - Assistant Confort et Consommation Énergétique

Une application web intelligente pour prédire la consommation énergétique et recommander des températures intérieures optimales basées sur les conditions météorologiques et les préférences de confort.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://projetsmart.streamlit.app/)

##  Aperçu

Cette application utilise des algorithmes d'apprentissage automatique pour :
- **Prédire la consommation énergétique** en fonction des conditions météorologiques
- **Recommander des températures intérieures** optimales selon votre profil de confort
- **Analyser les données historiques** avec des visualisations interactives
- **Intégrer des données météorologiques en temps réel** via l'API Open-Meteo

##  Fonctionnalités

###  Prédiction Intelligente
- **Segmentation par profil** : Low, Medium, High consumption
- **Algorithme KNN** pour la prédiction de consommation
- **Extra Trees Regressor** pour la recommandation de température
- **Données météo en temps réel** ou saisie manuelle

###  Visualisations Avancées
- Graphiques interactifs avec **Plotly**
- Analyses statistiques détaillées
- Évolution temporelle des consommations
- Distribution par segments de confort

###  Intégration Météo
- API Open-Meteo pour les données en temps réel
- Température extérieure et ensoleillement
- Prévisions automatiques pour votre localisation

##  Technologies Utilisées

- **[Streamlit](https://streamlit.io)** - Framework web pour applications Python
- **scikit-learn** - Algorithmes d'apprentissage automatique
- **Pandas & NumPy** - Manipulation et analyse de données  
- **Plotly & Matplotlib** - Visualisations interactives
- **Requests** - Intégration API météo
- **Pillow** - Traitement d'images

##  Guide d'Utilisation

### 1. **Sélection des Données Météo**
- Choisissez entre l'API météo automatique ou la saisie manuelle
- Les données incluent température extérieure et ensoleillement

### 2. **Configuration des Paramètres**
- **Mode de confort** : Low, Medium, ou High
- **Niveau de similarité** : Ajustez la précision des prédictions
- **Jour de la semaine** : Impact sur les habitudes de consommation

### 3. **Résultats**
- **Consommation estimée** : Prédiction en kWh
- **Température recommandée** : Consigne optimale en °C
- **Visualisations** : Graphiques détaillés et analyses

##  Architecture du Projet

```
Projet_SMART/
├── app_confort_prediction.py    # Application principale
├── requirements.txt             # Dépendances Python
├── df_daily.pickle             # Données d'entraînement
├── Centrale Med.png            # Logo
├── LICENSE                     # Licence MIT
└── README.md                  # Documentation
```

##  Modèles d'IA Utilisés

### 1. **K-Nearest Neighbors (KNN)**
- Prédiction de consommation par similarité
- Segmentation par profils de confort
- Normalisation des features avec StandardScaler

### 2. **Extra Trees Regressor**
- Recommandation de température intérieure
- Ensemble d'arbres de décision
- Features : météo + consommation + jour

##  Données et Features

### Variables d'Entrée
- **Température extérieure** (°C)
- **Ensoleillement** (W/m²)
- **Jour de la semaine** (0-6)
- **Mode de confort** (Low/Medium/High)

### Variables de Sortie
- **Consommation énergétique** (kWh)
- **Température intérieure recommandée** (°C)

##  API Météo

L'application utilise l'[API Open-Meteo](https://open-meteo.com/) pour récupérer :
- Température actuelle
- Rayonnement solaire direct
- Prévisions en temps réel

Coordonnées par défaut : Nice, France (43.7°N, 7.25°E)

##  Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

##  Support

Pour toute question ou problème :
- Ouvrez une [issue](https://github.com/Fatimouden/Projet_SMART/issues)
- Contactez l'équipe de développement
