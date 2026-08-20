import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Demo mode
# True  = Test the Agentic RAG workflow without using Gemini API
# False = Use Gemini API normally
DEMO_MODE = os.getenv("DEMO_MODE", "True").lower() == "true"

# Project paths
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

DOCUMENTS_DIR = os.path.join(
    DATA_DIR,
    "documents"
)

LOGS_DIR = os.path.join(
    BASE_DIR,
    "logs"
)