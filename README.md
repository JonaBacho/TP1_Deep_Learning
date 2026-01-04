# 🧠 MNIST MLOps : Pipeline de Deep Learning End-to-End

[![MLflow](https://img.shields.io/badge/MLflow-2.10.2-blue.svg)](https://mlflow.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Ce projet implémente un pipeline MLOps complet pour la classification de chiffres manuscrits (MNIST). Il couvre l'entraînement, le suivi des expériences, le registre de modèles avec **MLflow**, le déploiement via **Flask** dans un conteneur **Docker**, et l'automatisation **CI/CD** avec GitHub Actions.

## 🚀 Fonctionnalités Clés

*   **Entraînement Automatisé** : Script d'entraînement avec gestion des hyperparamètres et export vers MLflow.
*   **Tracking & Registry** : Suivi des métriques (Accuracy, F1-Score) et gestion des versions de modèles avec MLflow.
*   **Promotion Intelligente** : Logique de promotion automatique du modèle en "Production" basée sur des seuils de performance (Accuracy > 95% par défaut).
*   **API REST** : Service de prédiction performant utilisant Flask et Gunicorn.
*   **Conteneurisation** : Image Docker optimisée basée sur Python Slim pour le déploiement en production.
*   **CI/CD robuste** : 
    *   **Workflow d'Entraînement** : Entraîne, enregistre et auto-promeut les modèles.
    *   **Workflow de Déploiement** : Build Docker, test de santé et déploiement automatique sur serveur distant via SSH.

---

## 📁 Structure du Projet

```text
├── config/                 # Configuration centralisée (MLflow, S3/MinIO)
├── src/                    # Code source principal
│   ├── app.py              # API Flask (Service de prédiction)
│   ├── train_model.py      # Script d'entraînement et logging MLflow
│   ├── auto_promote.py     # Logique de promotion basée sur les seuils
│   └── promote_model.py    # Utilitaire de gestion manuelle des stages
├── tests/                  # Tests unitaires (API, chargement modèle)
├── .github/workflows/      # Pipelines CI/CD (GitHub Actions)
├── Dockerfile              # Configuration de l'image de production
├── requirements.txt        # Dépendances Python
└── .env.example            # Template des variables d'environnement
```

## 🛠️ Configuration et Installation
1. Prérequis
 - Python 3.10+

 - Docker & Docker Compose

 - Un accès à un serveur MLflow (distant ou local)

2. Installation locale

```Bash
# Cloner le projet
git clone <votre-repo-url>
cd TP1_Deep_Learning

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

3. Variables d'environnement
Copiez le fichier .env.example en .env et remplissez vos accès :

```Bash
cp .env.example .env
Assurez-vous de bien configurer MLFLOW_TRACKING_URI et les accès S3 pour les artefacts.

🔌 Utilisation
Entraînement d'un modèle
Lancement de l'entraînement avec enregistrement automatique dans MLflow :
```

```Bash
python -m src.train_model
Lancement de l'API (Développement)
```

```Bash
python -m src.app
L'API sera accessible sur http://localhost:5000.
```

Déploiement avec Docker
```Bash
# Build de l'image
docker build -t mnist-mlflow-app .

# Lancement du conteneur
docker run -p 5000:5000 --env-file .env mnist-mlflow-app
```

## 🛣️ API Endpoints
Méthode	Endpoint	Description
GET	/health	État de santé et version du modèle chargé.
GET	/model/info	Informations détaillées sur le modèle en production.
POST	/predict	Prédit un chiffre à partir d'un array JSON (784 pixels).
POST	/model/reload	Force le rechargement du modèle depuis MLflow.
Exemple de requête /predict :


```JSON
{
  "image": [0, 0, 0.5, 0.8, ..., 0] 
}
```

## 🤖 Pipeline CI/CD
Le projet utilise deux workflows principaux :

1. MLflow Train and Register (train_register.yml) :

 - S'exécute sur push ou workflow_dispatch.

 - Entraîne le modèle avec TensorFlow.

 - Compare les performances avec le modèle actuel en Production.

 - Promeut le modèle si les critères (MIN_ACCURACY) sont dépassés.

2. Deploy Application (deploy.yml) :

 - Build l'image Docker et la pousse sur GHCR (GitHub Container Registry).

 - Déploie automatiquement sur le serveur de production via SSH.

 - Vérifie la santé du déploiement (/health) après le redémarrage.

## 🧪 Tests
Les tests sont automatisés et vérifient l'API ainsi que la robustesse du chargement de modèle :

```Bash
pytest tests/test_model.py
```

## ⚖️ Licence
Distribué sous la licence MIT. Voir LICENSE pour plus d'informations.