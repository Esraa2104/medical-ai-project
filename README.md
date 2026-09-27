# Détection de pneumonie à partir de radiographies thoraciques

Projet académique de Machine Learning et Computer Vision : classification d'images
radiographiques du thorax en deux classes (**NORMAL** / **PNEUMONIA**), avec un pipeline
complet allant du dataset jusqu'à une interface web fonctionnelle.

> ⚠️ **Avertissement** : ce projet est strictement pédagogique. Le modèle n'a pas été
> validé cliniquement et ne constitue en aucun cas un outil de diagnostic médical.
> Il ne remplace pas l'avis d'un professionnel de santé.

## Aperçu

| | |
|---|---|
| **Dataset** | Chest X-Ray Images (Pneumonia) — Kaggle |
| **Lien du dataset** | https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia |
| **Modèle retenu** | MobileNetV2 (Transfer Learning) avec pondération des classes |
| **Accuracy (test)** | 84.9% |
| **Backend** | FastAPI |
| **Frontend** | Angular |

## Architecture du projet

```
medical-ai-project/
├── data/                  # Dataset (non versionné, voir "Installation")
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── validation/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── test/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── processed/         # Images prétraitées (.npy)
├── notebooks/
│   ├── 01_exploration.ipynb      # Exploration du dataset
│   ├── 02_preprocessing.ipynb    # Prétraitement des images
│   ├── 03_training.ipynb         # Entraînement des 3 modèles
│   └── 04_evaluation.ipynb       # Évaluation comparative
├── src/
│   ├── prepare_dataset.py        # Split train/validation/test
│   └── predict.py                # Logique de prédiction (utilisée par l'API)
├── models/                        # Modèles entraînés (.keras)
│   ├── model_v1_simple_cnn.keras
│   ├── model_v2_mobilenet_tl.keras
│   └── model_v3_mobilenet_weighted.keras
├── api/
│   └── main.py                    # API FastAPI
├── frontend/                      # Application Angular
├── requirements.txt
└── README.md
```

## Prérequis

- Python 3.12 (recommandé — les versions trop récentes de Python posent des
  problèmes de compatibilité avec TensorFlow)
- Node.js 18+ et Angular CLI
- ~2 Go d'espace disque (dataset + modèles)
- Un compte Kaggle (gratuit) pour télécharger le dataset

## Installation

### 1. Créer l'environnement Python

Depuis la racine du projet :

```bash
python -m venv .venv
```

Activation sous Windows (PowerShell) :

```bash
.venv\Scripts\Activate.ps1
```

Activation sous macOS/Linux :

```bash
source .venv/bin/activate
```

Puis installer les dépendances :

```bash
pip install -r requirements.txt
```

### 2. Récupérer le dataset

1. Télécharger le dataset "Chest X-Ray Images (Pneumonia)" depuis Kaggle :
   https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Extraire l'archive téléchargée.
3. Ouvrir `src/prepare_dataset.py` et adapter la variable `SOURCE` pour qu'elle pointe
   vers le dossier extrait (celui qui contient les sous-dossiers `train/` et `test/`).
4. Lancer le script de préparation :

```bash
python src/prepare_dataset.py
```

Ce script répartit les images en trois ensembles dans `data/` :
- `train/` (85% du train Kaggle d'origine)
- `validation/` (15% du train Kaggle d'origine)
- `test/` (le test set Kaggle, copié tel quel)

### 3. Installer le frontend

```bash
cd frontend
npm install
```

## Utilisation

### Reproduire l'entraînement (optionnel)

Les modèles entraînés sont déjà fournis dans `models/`. Pour ré-entraîner depuis zéro,
exécuter les notebooks dans l'ordre depuis le dossier `notebooks/` :

```bash
jupyter notebook
```

1. `01_exploration.ipynb` — statistiques du dataset, vérification des images,
   détection d'images corrompues, distribution des classes.
2. `02_preprocessing.ipynb` — resize (224x224), conversion en niveaux de gris,
   normalisation, sauvegarde en fichiers `.npy` dans `data/processed/`.
3. `03_training.ipynb` — entraîne 3 modèles successifs :
   - un CNN simple entraîné from scratch,
   - MobileNetV2 en Transfer Learning,
   - MobileNetV2 avec pondération des classes (`class_weight`).
4. `04_evaluation.ipynb` — évalue les 3 modèles sur le test set, matrices de
   confusion, precision/recall/F1-score, comparaison finale.

Les modèles entraînés sont sauvegardés automatiquement dans `models/`.

### Lancer l'application complète

Deux serveurs doivent tourner simultanément, dans deux terminaux séparés.

**Terminal 1 — API (FastAPI) :**

```bash
.venv\Scripts\Activate.ps1
uvicorn api.main:app --reload
```

L'API est disponible sur `http://127.0.0.1:8000`.
Documentation interactive (Swagger) : `http://127.0.0.1:8000/docs`.

**Terminal 2 — Frontend (Angular) :**

```bash
cd frontend
ng serve
```

L'interface est disponible sur `http://localhost:4200`.

### Endpoint de l'API

`POST /predict`

Requête : un fichier image (`multipart/form-data`, champ `file`).

Réponse JSON :

```json
{
  "prediction": "PNEUMONIA",
  "probability_normal": 0.1239,
  "probability_pneumonia": 0.8761,
  "model_name": "model_v3_mobilenet_weighted.keras"
}
```

### Tester la prédiction en ligne de commande (sans l'API)

```bash
python src/predict.py
```

Ce script teste directement la fonction de prédiction sur une image d'exemple du
dataset, sans passer par l'API.

## Résultats

Trois modèles ont été entraînés et comparés sur le test set (données jamais vues
pendant l'entraînement ni la validation) :

| Modèle                              | Accuracy | Recall NORMAL | Recall PNEUMONIA |
|--------------------------------------|----------|----------------|--------------------|
| CNN simple (from scratch)            | 78.2%    | 44%            | 99%                |
| MobileNetV2 (Transfer Learning)      | 84.5%    | 59%            | 100%               |
| MobileNetV2 pondéré (modèle retenu)  | 84.9%    | 62%            | 99%                |

Un écart significatif a été observé entre les performances en validation (~97% pour
les trois modèles) et en test (78 à 85%). L'analyse détaillée (voir
`04_evaluation.ipynb`) a permis d'identifier que cet écart provient majoritairement
d'un **distribution shift** entre les ensembles train et test propre à ce dataset
Kaggle, plutôt que d'un simple surapprentissage — un cas concret illustrant
l'importance de ne pas se fier uniquement à l'accuracy en validation, et de toujours
challenger un dataset avant de conclure sur la performance d'un modèle.

D'un point de vue clinique, le biais observé reste orienté du côté le moins risqué :
les faux négatifs (pneumonie non détectée) sont très rares sur les trois modèles,
au prix d'un taux de faux positifs plus élevé sur la classe NORMAL.

## Limites

- Dataset limité à une seule source, qualité et provenance des images variables.
- Distribution shift entre train et test non résolu.
- Déséquilibre des classes (3294 PNEUMONIA contre 1140 NORMAL en train) seulement
  partiellement compensé par la pondération des classes.
- Modèle non validé cliniquement — usage strictement académique et expérimental.
- Résultat du modèle non équivalent à un diagnostic médical.

## Améliorations possibles

- Grad-CAM pour visualiser les zones de l'image qui influencent la prédiction.
- Classification à 3 classes (NORMAL / pneumonie bactérienne / pneumonie virale),
  information déjà présente dans les noms de fichiers du dataset.
- Recherche d'hyperparamètres, data augmentation ciblée sur la classe NORMAL.
- Fine-tuning des dernières couches de MobileNetV2 (au lieu de les garder gelées).
- Déploiement (Docker, hébergement cloud).
- Authentification de l'API, historique des analyses, tableau de bord de suivi.

## Technologies utilisées

**Intelligence artificielle**
Python, NumPy, Pandas, Matplotlib, Seaborn, Pillow, OpenCV, scikit-learn,
TensorFlow / Keras

**Backend**
FastAPI, Uvicorn

**Frontend**
Angular

**Environnement**
Jupyter Notebook, environnement virtuel Python (venv)

## Auteur

Projet réalisé dans un cadre académique, à des fins d'apprentissage du pipeline
complet d'un projet d'intelligence artificielle appliqué à la Computer Vision :
dataset, prétraitement, Machine Learning, Deep Learning, CNN, Transfer Learning,
évaluation critique, API REST et interface utilisateur.