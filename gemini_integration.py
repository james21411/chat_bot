import google.generativeai as genai
import os
from typing import Optional, Dict, Any

class GeminiTextProcessor:
    def __init__(self, api_key: str):
        """Initialize Gemini API client"""
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            print(f"✅ Gemini initialisé avec succès")
        except Exception as e:
            print(f"❌ Erreur initialisation Gemini: {e}")
            self.model = None
    
    def generate_response(self, user_message: str, food_context: Optional[Dict[str, Any]] = None) -> str:
        """Generate response using Gemini API"""
        if self.model is None:
            return "Désolé, le service Gemini n'est pas disponible."
        
        try:
            # Create enhanced prompt with food context if available
            if food_context:
                context_info = f"""
Contexte alimentaire détecté:
- Plat identifié: {food_context.get('name', 'Inconnu')}
- Origine: {food_context.get('origin', 'Inconnue')}
- Confiance: {food_context.get('confidence', 0):.1%}
- Description: {food_context.get('description', 'Aucune description disponible')}
"""
                prompt = f"""Tu es un expert culinaire spécialisé dans les plats africains et maghrébins.

{context_info}

Question de l'utilisateur: {user_message}

Réponds de manière informative, culturelle et passionnée sur ce plat spécifique."""
            else:
                prompt = f"""Tu es un expert culinaire spécialisé dans les plats africains et maghrébins. 
Tu connais l'histoire, la culture, les ingrédients, les techniques de préparation et les variations régionales de ces plats.

Réponds de manière informative, culturelle et passionnée sur la cuisine africaine et maghrébine.

Question de l'utilisateur: {user_message}"""
            
            # Generate response
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            print(f"❌ Erreur génération Gemini: {e}")
            return f"Désolé, j'ai rencontré un problème technique avec Gemini: {str(e)}"
    
    def chat_response(self, message: str, food_prediction: Optional[Dict[str, Any]] = None) -> str:
        """Generate a complete chat response"""
        # If we have food context, provide more detailed responses
        if food_prediction:
            food_info = food_prediction.get('food_info', {})
            context = {
                'name': food_info.get('name', food_prediction.get('predicted_class', '')),
                'origin': food_info.get('origin', ''),
                'confidence': food_prediction.get('confidence', 0),
                'description': food_info.get('description', '')
            }
            return self.generate_response(message, context)
        else:
            return self.generate_response(message)
    
    def test_connection(self) -> str:
        """Test the Gemini connection"""
        try:
            if self.model is None:
                return "Gemini non initialisé"
            
            response = self.model.generate_content("Test de connexion")
            return f"Gemini fonctionne! Réponse: {response.text[:50]}..."
        except Exception as e:
            return f"Erreur Gemini: {str(e)}"

if __name__ == "__main__":
    # Test the Gemini integration
    api_key = "AIzaSyDpRzkWeVssJiL_CcDSxQGrl2oLCPA7Kek"
    processor = GeminiTextProcessor(api_key)
    
    print("Test de connexion:", processor.test_connection())
    
    # Test basic response
    response = processor.generate_response("Parle-moi du jollof rice")
    print("Test response:", response[:100])
    
    # Test with food context
    food_context = {
        'name': 'Jollof Rice',
        'origin': 'Afrique de l\'Ouest',
        'confidence': 0.95,
        'description': 'Riz épicé traditionnel d\'Afrique de l\'Ouest'
    }
    response_with_context = processor.generate_response("Quelle est l\'histoire de ce plat ?", food_context)
    print("Response with context:", response_with_context[:100])
