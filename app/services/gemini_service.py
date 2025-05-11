import google.generativeai as genai
import os
from typing import Optional, Dict
import logging
import json
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            logger.error("Missing GEMINI_API_KEY")
            raise ValueError(
                "API key not configured. "
                "Set GEMINI_API_KEY in .env or environment variables"
            )
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
        self.min_confidence = 0.95

    def test_model(self):
        try:
            response = self.model.generate_content("Is 2+2=5?")
            logger.info(f"API test response: {response.text}")
            return True
        except Exception as e:
            logger.error("API test failed", exc_info=True)
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