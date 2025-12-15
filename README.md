# Reconnaissance de Nourriture Africaine + Chatbot

Un système complet de reconnaissance d'images de plats africains et maghrébins avec fonctionnalité de chatbot alimenté par Gemini AI.

## 🚀 Fonctionnalités

- **Reconnaissance d'images** : Identifie 11 plats traditionnels africains avec précision
- **Chatbot intelligent** : Réponses contextuelles sur les plats via Gemini AI
- **Interface web** : Interface simple pour tester les fonctionnalités
- **API REST** : Endpoints pour intégration dans d'autres applications

## 📋 Plats Supportés

- Attieke • Egusi Soup • Eru • Ewedu Soup • Fufu
- Jollof Rice • Kedjenou • Koki • Ndolè • Saka Saka • Thieboudienne

## 🛠️ Installation Rapide

### Prérequis
- Python 3.11+
- pip

### Installation
```bash
# Cloner ou copier les fichiers
# Créer l'environnement virtuel
python -m venv food_chat_env
source food_chat_env/bin/activate  # Linux/Mac
# food_chat_env\Scripts\activate   # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## 🚀 Démarrage

### Serveur Complet
```bash
python serveur_simple_final.py
```

### Avec Docker
```bash
docker-compose up
```

L'API sera disponible sur `http://localhost:8000`

## 📖 Utilisation

### Interface Web
Accédez à `http://localhost:8000/interface` pour une interface simple de test.

### API Endpoints

#### Prédiction d'Image
```bash
POST /predict
Content-Type: multipart/form-data

file: [image_file]
```

#### Chat Texte
```bash
POST /chat/text
{
  "text": "Quelle est la recette du ndolè ?"
}
```

#### Chat Image + Texte
```bash
POST /chat/image-text
{
  "image": "[base64_image]",
  "text": "Qu'est-ce que c'est comme plat ?"
}
```

## 📁 Structure du Projet

```
├── serveur_simple_final.py      # Serveur FastAPI principal
├── improved_model_training.py   # Modèle CNN entraîné
├── dataset_loader.py           # Chargement des données
├── gemini_integration.py       # Intégration Gemini AI
├── interface_simple_finale.html # Interface web
├── requirements.txt            # Dépendances Python
├── data_set_images/            # Dataset d'images
├── models/                     # Modèles entraînés
├── backup_servers/             # Fichiers de sauvegarde
└── integration.md              # Guide d'intégration
```

## 🔧 Configuration

### Clé API Gemini
Modifiez `serveur_simple_final.py` :
```python
api_key = "VOTRE_CLE_API_GEMINI"
```

### Chemins des Modèles
Assurez-vous que `models/food_classifier_12_dishes.pth` et `models/food_classifier_12_dishes_info.json` existent.

### Entraînement du Modèle
Si les fichiers de modèle n'existent pas, exécutez la commande suivante pour entraîner le modèle :
```bash
python improved_model_training.py
```

## 📊 Performance

- **Précision** : 90%+ sur les classes entraînées
- **Temps de réponse** : 2-3 secondes par prédiction
- **Support GPU** : Automatique si CUDA disponible

## 🔗 Intégration

Pour intégrer ce système dans votre propre projet, consultez `integration.md` pour un guide détaillé.

## 📝 Licence

Ce projet est destiné à des fins éducatives et de recherche sur la cuisine africaine.

## 🤝 Contribution

Les contributions pour ajouter de nouveaux plats ou améliorer les performances sont les bienvenues !

## 🆘 Support

En cas de problème :
1. Vérifiez les logs du serveur
2. Assurez-vous que tous les fichiers sont présents
3. Vérifiez votre clé API Gemini
4. Consultez `integration.md` pour le dépannage# chatbot
# my_chatbot
# my_chatbot
# chat_bot
