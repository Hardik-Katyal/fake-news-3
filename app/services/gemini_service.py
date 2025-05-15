import google.generativeai as genai
import os
from typing import Optional, Dict
import logging
import json
from dotenv import load_dotenv
import asyncio

logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            logger.error("Missing GEMINI_API_KEY")
            raise ValueError("API key not configured. Set GEMINI_API_KEY in .env")
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
        #self.min_confidence = 0.95

    def test_model(self):
        try:
            response = self.model.generate_content("Is 2+2=5?")
            return "4" in response.text
        except Exception as e:
            logger.error("API test failed", exc_info=True)
            return False


    async def explain_prediction(self, content: str, prediction: str, confidence: float) -> str:
        """Get model behavior explanation from Gemini"""
        try:
            prompt = f"""Analyze this news content and ML model prediction:
            Content: {content[:1500]}
            Prediction: {prediction} (Confidence: {confidence:.0%})
            
            Explain 3 potential linguistic patterns that might have led to this prediction.
            Focus on: 
            - Emotional language
            - Unusual claims patterns
            - Source reliability indicators
            - Confidence score interpretation
            - Model output reliability interpretation
            
            Use this format:
            1. [Pattern Type]: Brief explanation (max 15 words)
            2. [Pattern Type]: Brief explanation
            3. [Pattern Type]: Brief explanation
            4. [Pattern Type]: Brief explanation
            
            Keep response under 120 words. No markdown."""
            
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            return response.text
            
        except Exception as e:
            logger.error(f"Explanation failed: {str(e)}", exc_info=True)
            return "Explanation unavailable: Service error"            
        except asyncio.TimeoutError:
            logger.warning("Gemini API timeout")
            return "Explanation unavailable: API timeout"

    async def verify_content(self, title: str, text: str) -> Optional[Dict]:
        try:
            prompt = f"""Analyze this news content for factual accuracy:
            Title: {title}
            Content: {text[:2000]}
            
            Respond STRICTLY in this JSON format:
            {{
                "verdict": "true/false/unverified",
                "confidence": 0-1,
                "inaccurate_phrases": ["list", "of", "phrases"],
                "related_articles": [
                    {{
                        "title": "...",
                        "url": "...",
                        "source": "..."
                    }}
                ]
            }}"""
            
            response = self.model.generate_content(prompt)
            return self._parse_response(response.text)
        except Exception as e:
            logger.error("Content verification failed", exc_info=True)
            return None

    def _parse_response(self, response_text: str) -> Dict:
        try:
            json_str = response_text.strip().replace('```json', '').replace('```', '')
            return json.loads(json_str)
        except Exception as e:
            logger.error("Response parsing failed", exc_info=True)
            return {
                "verdict": "unverified",
                "confidence": 0,
                "inaccurate_phrases": [],
                "related_articles": []
            }