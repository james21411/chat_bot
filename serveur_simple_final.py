#!/usr/bin/env python3

"""
Serveur simple qui utilise le modèle le plus entraîné + mode texte-uniquement Gemini
Interface directe et simple comme avant + fonctionnalité chatbot texte + mode image+texte
"""

import os
import json
import torch
from PIL import Image
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
import io
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Import simple + Gemini
from improved_model_training import ImprovedFoodClassifier
from dataset_loader import FoodDataset
from gemini_integration import GeminiTextProcessor
from torchvision import transforms

# Modèles de réponse
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    food_info: Optional[Dict[str, Any]] = None
    top_3_predictions: List[Dict[str, Any]] = []
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gemini_available: bool
    timestamp: str

class ChatTextRequest(BaseModel):
    text: str

class ChatImageTextRequest(BaseModel):
    image: str  # base64 image
    text: str

class ChatResponse(BaseModel):
    mode: str
    response: str
    timestamp: str
    prediction: Optional[Dict[str, Any]] = None

# Variables globales
model_manager = None
dataset = None
gemini_processor = None

def load_best_model():
    """Charger uniquement le meilleur modèle (le plus entraîné)"""
    global model_manager, dataset

    print("🤖 Chargement du meilleur modèle...")

    # Le modèle principal est généralement le plus entraîné
    best_model_path = 'models/food_classifier_12_dishes.pth'
    info_path = 'models/food_classifier_12_dishes_info.json'

    if not os.path.exists(best_model_path):
        print(f"❌ Modèle non trouvé: {best_model_path}")
        return False

    try:
        # Charger les informations du modèle d'abord
        if os.path.exists(info_path):
            with open(info_path, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
                saved_idx_to_class = {int(k): v for k, v in model_info['idx_to_class'].items()}
                num_classes = model_info['num_classes']
                print(f"📄 Informations du modèle chargées: {num_classes} classes")
        else:
            print(f"❌ Fichier info non trouvé: {info_path}")
            return False

        # Créer un dataset factice juste pour la compatibilité
        dataset = FoodDataset('data_set_images', transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))

        # Remplacer le mapping par celui sauvegardé lors de l'entraînement
        dataset.idx_to_class = saved_idx_to_class

        print(f"📊 Mapping des classes corrigé selon l'entraînement")
        print(f"🎯 Classes: {list(saved_idx_to_class.values())}")

        # Charger le modèle principal (le plus entraîné)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Créer le modèle avec la même architecture que l'entraînement
        model_manager = ImprovedFoodClassifier(num_classes).to(device)

        # Charger les poids
        checkpoint = torch.load(best_model_path, map_location=device)
        model_manager.load_state_dict(checkpoint['model_state_dict'])
        model_manager.eval()

        print(f"✅ Modèle chargé sur device: {device}")

        print(f"✅ Meilleur modèle chargé avec succès!")
        print(f"🎯 Modèle utilisé: {best_model_path}")
        return True

    except Exception as e:
        print(f"❌ Erreur chargement modèle: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_gemini():
    """Charger Gemini pour le mode texte-uniquement"""
    global gemini_processor
    
    try:
        # Clé API configurée
        api_key = "AIzaSyDpRzkWeVssJiL_CcDSxQGrl2oLCPA7Kek"
        gemini_processor = GeminiTextProcessor(api_key)
        print("✅ Gemini initialisé avec succès!")
        return True
    except Exception as e:
        print(f"❌ Erreur initialisation Gemini: {e}")
        return False

def preprocess_image(image_data: bytes) -> Optional[torch.Tensor]:
    """Prétraiter l'image"""
    try:
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image)
    except Exception as e:
        print(f"Erreur traitement image: {e}")
        return None

def predict_image(image_data: bytes) -> Dict[str, Any]:
    """Prédire la nourriture avec le meilleur modèle"""
    if not model_manager or not dataset:
        return {"error": "Modèle non chargé"}

    image_tensor = preprocess_image(image_data)
    if image_tensor is None:
        return {"error": "Impossible de traiter l'image"}

    try:
        device = next(model_manager.parameters()).device
        print(f"DEBUG: Running prediction on device {device}")
        print(f"DEBUG: Input tensor shape: {image_tensor.shape}")

        model_manager.eval()
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).to(device)
            print(f"DEBUG: Unsqueezed tensor shape: {image_tensor.shape}")

            outputs = model_manager(image_tensor)
            print(f"DEBUG: Model outputs shape: {outputs.shape}")

            probabilities = torch.softmax(outputs, dim=1)
            print(f"DEBUG: Probabilities shape: {probabilities.shape}")

            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_idx = predicted_idx.item()  # Convert tensor to Python int
            confidence = confidence.item()
            probabilities = probabilities.cpu().numpy()[0]

            print(f"DEBUG: Predicted index {predicted_idx} with confidence {confidence:.3f}")
            print(f"DEBUG: Predicted index type: {type(predicted_idx)}")
            print(f"DEBUG: Class name: {dataset.idx_to_class.get(predicted_idx, 'Unknown')}")

        # Fix the type issue - ensure we use the integer key directly
        predicted_idx_int = int(predicted_idx)  # Ensure Python int
        class_name = dataset.idx_to_class.get(predicted_idx_int, f"Unknown_Class_{predicted_idx_int}")
        class_info = dataset.get_class_info(predicted_idx_int)
        
        # Top 3 prédictions avec priorité aux plats camerounais
        top_3_indices = np.argsort(probabilities)[::-1][:3]
        top_3_predictions = []

        # Plats camerounais à prioriser
        cameroonian_dishes = ['ndolè', 'saka_saka', 'koki', 'eru']

        for idx in top_3_indices:
            idx_int = int(idx)  # Ensure Python int
            pred_class_name = dataset.idx_to_class.get(idx_int, f"Unknown_Class_{idx_int}")
            pred_info = dataset.get_class_info(idx_int)
            top_3_predictions.append({
                'class_name': pred_class_name,
                'confidence': float(probabilities[idx]),
                'food_info': pred_info,
                'is_cameroonian': pred_class_name in cameroonian_dishes
            })

        # Réorganiser pour mettre les plats camerounais en premier
        cameroonian_predictions = [p for p in top_3_predictions if p['is_cameroonian']]
        other_predictions = [p for p in top_3_predictions if not p['is_cameroonian']]

        # Combiner : plats camerounais d'abord, puis les autres
        top_3_predictions = cameroonian_predictions + other_predictions
        
        return {
            'predicted_class': class_name,
            'confidence': float(confidence),
            'food_info': class_info,
            'top_3_predictions': top_3_predictions,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": f"Erreur prédiction: {str(e)}"}

def generate_text_response(user_text: str) -> Dict[str, Any]:
    """Générer une réponse texte avec Gemini"""
    if not gemini_processor:
        return {"error": "Gemini non disponible"}
    
    try:
        gemini_response = gemini_processor.generate_response(user_text)
        
        return {
            'mode': 'text_only',
            'response': f"💬 **Réponse :**\n{gemini_response}",
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'mode': 'text_only',
            'response': f"❌ Désolé, erreur Gemini: {str(e)}",
            'timestamp': datetime.now().isoformat()
        }

def generate_image_text_response(image_data: bytes, user_text: str) -> Dict[str, Any]:
    """Générer une réponse en combinant analyse d'image + Gemini"""
    # D'abord analyser l'image
    image_result = predict_image(image_data)
    
    if "error" in image_result:
        # Si l'image ne peut pas être analysée, tomber en mode texte-uniquement
        return generate_text_response(user_text)
    
    # Préparer le contexte pour Gemini avec les informations de l'image
    food_info = image_result.get('food_info')
    confidence = image_result['confidence'] * 100

    origin = food_info.get('origin', 'Inconnue') if food_info else 'Inconnue'
    description = food_info.get('description', 'Aucune description disponible') if food_info else 'Aucune description disponible'

    context_info = f"""
Plat identifié dans l'image:
- Nom: {image_result.get('predicted_class', 'Inconnu').replace('_', ' ')}
- Origine: {origin}
- Confiance: {confidence:.1f}%
- Description: {description}
"""
    
    prompt = f"""Tu es un expert culinaire spécialisé dans les plats africains et maghrébins.

{context_info}

Question de l'utilisateur: {user_text}

Réponds de manière informative en utilisant les informations sur le plat identifié ET répond à sa question."""
    
    try:
        gemini_response = gemini_processor.generate_response(prompt)
        
        # Formater la réponse combinée
        food_name = image_result.get('predicted_class', 'Inconnu').replace('_', ' ')
        cam_flag = "🇨🇲 " if image_result.get('predicted_class') in ['ndolè', 'saka_saka', 'koki', 'eru'] else ""

        response_text = f"🍽️ J'ai analysé votre image ({confidence:.1f}% de confiance) et voici ce que je pense :\n\n"
        response_text += f"**Plat identifié : {cam_flag}{food_name}**\n\n"
        response_text += f"**Réponse à votre question :**\n{gemini_response}\n"

        # Ajouter alternatives si confiance faible
        if confidence < 70:
            response_text += "\n🤔 **Autres possibilités :**\n"
            for i, pred in enumerate(image_result['top_3_predictions'][1:3], 2):
                if pred['confidence'] > 0.1:
                    food_name_alt = pred['class_name'].replace('_', ' ')
                    flag = "🇨🇲 " if pred.get('is_cameroonian', False) else ""
                    response_text += f"  {i}. {flag}{food_name_alt} ({pred['confidence']*100:.1f}%)\n"
        
        return {
            'mode': 'image_and_text',
            'response': response_text,
            'timestamp': datetime.now().isoformat(),
            'prediction': image_result
        }
    except Exception as e:
        return {
            'mode': 'image_and_text',
            'response': f"❌ Désolé, erreur lors de la génération: {str(e)}",
            'timestamp': datetime.now().isoformat(),
            'prediction': image_result
        }

def decode_base64_image(base64_string: str) -> Optional[bytes]:
    """Décoder une image base64"""
    try:
        # Supprimer le préfixe data URL si présent
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Nettoyer la chaîne base64 (enlever espaces et caractères invalides)
        base64_string = base64_string.strip()
        
        image_bytes = base64.b64decode(base64_string)
        return image_bytes
    except Exception as e:
        print(f"Erreur décodage base64: {e}")
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire de cycle de vie de l'application"""
    print("🚀 Initialisation du serveur...")

    # Charger le modèle
    if load_best_model():
        print("✅ Modèle alimentaire chargé")
    else:
        print("❌ Échec chargement modèle")

    # Charger Gemini
    if load_gemini():
        print("✅ Gemini chargé")
    else:
        print("❌ Échec chargement Gemini")

    yield

    # Code de nettoyage si nécessaire
    print("🛑 Arrêt du serveur...")

# Créer l'app FastAPI
app = FastAPI(title="Reconnaissance de Nourriture + Chat", version="3.0.0", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=Dict[str, Any])
async def root():
    return {
        "message": "API de Reconnaissance de Nourriture + Chat",
        "version": "3.0.0",
        "status": "Prêt pour reconnaissance d'images et chat",
        "endpoints": {
            "predict": "POST /predict - Reconnaissance d'image",
            "chat_text": "POST /chat/text - Chat texte-uniquement avec Gemini",
            "chat_image_text": "POST /chat/image-text - Image + texte combinés"
        },
        "interface_simple": "/interface - Interface web simple pour tester",
        "docs": "/docs - Documentation Swagger"
    }

@app.get("/interface")
async def interface_simple():
    """Servir l'interface web simple"""
    return FileResponse("interface_simple.html", media_type="text/html")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if model_manager else "model_not_loaded",
        model_loaded=model_manager is not None,
        gemini_available=gemini_processor is not None,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_food(file: UploadFile = File(...)):
    """Prédiction simple avec le meilleur modèle"""
    
    # Vérifier le type de fichier
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    file_extension = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Type de fichier non supporté. Types acceptés: {', '.join(allowed_extensions)}"
        )
    
    # Lire l'image
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Fichier trop volumineux. Taille maximum: 10MB"
        )
    
    # Faire la prédiction
    prediction_result = predict_image(contents)
    
    if "error" in prediction_result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=prediction_result["error"]
        )
    
    return PredictionResponse(**prediction_result)

@app.post("/chat/text", response_model=ChatResponse)
async def chat_text(request: ChatTextRequest):
    """Chat texte-uniquement avec Gemini"""
    
    if not gemini_processor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service Gemini non disponible"
        )
    
    result = generate_text_response(request.text)
    
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["error"]
        )
    
    return ChatResponse(**result)

@app.post("/chat/image-text", response_model=ChatResponse)
async def chat_image_text(request: ChatImageTextRequest):
    """Chat avec image + texte combinés"""
    
    if not gemini_processor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service Gemini non disponible"
        )
    
    # Décoder l'image base64
    image_data = decode_base64_image(request.image)
    if image_data is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image base64 invalide"
        )
    
    result = generate_image_text_response(image_data, request.text)
    
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["error"]
        )
    
    return ChatResponse(**result)

@app.get("/classes")
async def get_classes():
    """Obtenir la liste des classes disponibles"""
    if not dataset:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    classes_info = []
    for class_name, idx in dataset.class_to_idx.items():
        class_info = dataset.get_class_info(idx)
        classes_info.append({
            "class_name": class_name,
            "index": idx,
            "food_info": class_info
        })
    
    return {
        "total_classes": len(classes_info),
        "classes": classes_info
    }

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🍽️  RECONNAISSANCE + CHAT - SERVEUR FINAL")
    print("=" * 60)
    print("URL: http://localhost:8000")
    print("Documentation: http://localhost:8000/docs")
    print("Endpoints:")
    print("  - POST /predict (reconnaissance d'image)")
    print("  - POST /chat/text (texte-uniquement)")
    print("  - POST /chat/image-text (image + texte)")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)