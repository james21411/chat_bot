# Guide d'Intégration : Reconnaissance de Nourriture + Chat

Ce guide explique comment intégrer le modèle de reconnaissance de nourriture africaine et maghrébine avec fonctionnalité de chat Gemini à votre projet chatbot existant.

## Entraînement du Modèle

Si les fichiers de modèle n'existent pas, exécutez la commande suivante pour entraîner le modèle :
```bash
python improved_model_training.py
```

## Prérequis

- Python 3.11+
- pip
- Accès à une clé API Google Gemini (pour le mode chat)

## Fichiers à Copier

Copiez les fichiers suivants depuis le projet source vers votre projet cible :

### Code Core
- `serveur_simple_final.py` : Serveur FastAPI principal avec endpoints de prédiction et chat
- `improved_model_training.py` : Classe du modèle CNN entraîné
- `dataset_loader.py` : Classes pour le chargement et traitement des données
- `gemini_integration.py` : Intégration avec l'API Gemini pour le chat

### Données et Modèles
- `data_set_images/` : Dataset des images de nourriture (dossier complet)
- `models/` : Modèles PyTorch entraînés (dossier complet)

### Configuration
- `requirements.txt` : Liste des dépendances Python

## Installation des Dépendances

### Option 1 : Environnement Virtuel Python (venv)
```bash
# Créer l'environnement virtuel
python -m venv food_chat_env

# Activer l'environnement
# Sur Windows :
food_chat_env\Scripts\activate
# Sur Linux/Mac :
source food_chat_env/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

### Option 2 : Nouvel Environnement Conda
```bash
# Créer un nouvel environnement
conda create -n food_chat python=3.11
conda activate food_chat

# Installer les dépendances
pip install -r requirements.txt
```

### Option 3 : Environnement Existant
```bash
# Dans votre environnement existant
pip install -r requirements.txt
```

### Dépendances Principales
- torch : Framework de deep learning
- torchvision : Outils pour la vision par ordinateur
- fastapi : Framework web API
- uvicorn : Serveur ASGI
- pillow : Traitement d'images
- google-generativeai : API Gemini
- pydantic : Validation de données

## Configuration

### Clé API Gemini
Modifiez `serveur_simple_final.py` ligne ~131 :
```python
api_key = "VOTRE_CLE_API_GEMINI"
```

### Chemins des Modèles
Assurez-vous que les chemins dans `serveur_simple_final.py` correspondent :
```python
best_model_path = 'models/food_classifier_12_dishes.pth'
info_path = 'models/food_classifier_12_dishes_info.json'
```

## Intégration dans Votre Chatbot

### 1. Import des Modules
```python
from improved_model_training import ImprovedFoodClassifier
from dataset_loader import FoodDataset
from gemini_integration import GeminiTextProcessor
from serveur_simple_final import predict_image, generate_image_text_response
```

### 2. Initialisation
```python
# Charger le modèle (adapter load_best_model)
model_manager = ImprovedFoodClassifier(num_classes)
# Charger les poids
checkpoint = torch.load('models/food_classifier_12_dishes.pth')
model_manager.load_state_dict(checkpoint['model_state_dict'])
model_manager.eval()

# Initialiser Gemini
gemini_processor = GeminiTextProcessor("VOTRE_CLE_API")

# Charger le dataset pour les mappings
dataset = FoodDataset('data_set_images')
```

### 3. Utilisation des Fonctions

#### Prédiction d'Image
```python
# image_data : bytes de l'image
result = predict_image(image_data)
print(f"Plat identifié : {result['predicted_class']}")
```

#### Chat Texte-Uniquement
```python
response = generate_text_response("Quelle est la recette du ndolè ?")
print(response['response'])
```

#### Chat Image + Texte
```python
# image_data : bytes de l'image
# user_text : question de l'utilisateur
result = generate_image_text_response(image_data, user_text)
print(result['response'])
```

### 4. Adaptation à Votre Architecture

Modifiez les fonctions selon votre architecture :
- Remplacez FastAPI par votre framework (Flask, Django, etc.)
- Adaptez les modèles de réponse Pydantic à vos besoins
- Intégrez la logique dans vos handlers de messages

## Classes Alimentaires Supportées

Le modèle reconnaît 11 plats :
- attieke
- egusi_soup
- eru
- ewedu_soup
- fufu
- jollof_rice
- kedjenou
- koki
- ndolè
- saka_saka
- thieboudienne

## Déploiement

### Avec Docker
Utilisez le `Dockerfile` et `docker-compose.yml` fournis :
```bash
docker-compose up
```

### Serveur Standalone
```bash
python serveur_simple_final.py
```

## Dépannage

- **Erreur de chargement modèle** : Vérifiez les chemins des fichiers .pth et .json
- **API Gemini** : Vérifiez votre clé API et quota
- **CUDA** : Le modèle fonctionne sur CPU par défaut, ajoutez `.to('cuda')` pour GPU
- **Mémoire** : Les images sont redimensionnées à 224x224 pour optimiser la mémoire

## Performance

- **Précision** : ~90%+ sur les classes entraînées
- **Temps de réponse** : ~2-3 secondes par prédiction
- **Taille modèle** : ~100MB
- **Dataset** : 84 images d'entraînement

## Extension

Pour ajouter de nouveaux plats :
1. Ajoutez des images dans `data_set_images/`
2. Réentraînez le modèle avec `improved_model_training.py`
3. Mettez à jour les métadonnées dans les fichiers JSON