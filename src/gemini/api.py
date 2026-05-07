"""
Wrapper for Google Gemini API.
"""
import google.generativeai as genai
import os

class GeminiClient:
    def __init__(self, api_key: str = None):
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if key:
            genai.configure(api_key=key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')

    def generate(self, prompt: str, **kwargs) -> str:
        """Generates text from the given prompt."""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {e}"
