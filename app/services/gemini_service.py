import google.generativeai as genai
import os
from typing import Optional, Dict
import logging
import json  # Safer alternative to eval()

logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.0-pro')
        self.min_confidence = 0.95

    def test_model(self):
        """Test if Gemini API is working"""
        try:
            response = self.model.generate_content("Is 2+2=5?")
            print(f"Model Response: {response.text}")
            return True
        except Exception as e:
            logger.error(f"Model test failed: {str(e)}")
            return False

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
            
            response = await self.model.generate_content_async(prompt)
            return self._parse_response(response.text)
        except Exception as e:
            logger.error(f"Gemini verification failed: {str(e)}")
            return None

    def _parse_response(self, response_text: str) -> Dict:
        try:
            # Safer JSON parsing
            json_str = response_text.strip().replace('```json', '').replace('```', '')
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Failed to parse Gemini response: {str(e)}")
            return {
                "verdict": "unverified",
                "confidence": 0,
                "inaccurate_phrases": [],
                "related_articles": []
            }