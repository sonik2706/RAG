"""
config.py

Loads configuration settings from environment variables:
- OpenAI API key
- LLM model 
- embedding model
- model provider 
"""

import os
from dotenv import load_dotenv


class Config:
    API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    MODEL_PROVIDER = os.getenv("MODEL_PROVIDER")
