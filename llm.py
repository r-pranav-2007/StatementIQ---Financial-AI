"""
llm.py - the ONLY place in StatementIQ that talks to a language model.

Every tool that needs an LLM imports from here, so swapping the model
(Claude, OpenAI, a local model, ...) means editing just this file.

Right now this is a STUB: no real model is called yet. Each function
returns a safe default so the rest of the project works without an API key.
"""
import os

from dotenv import load_dotenv

# Load keys from a .env file in the project root (if it exists).
load_dotenv()

# MODEL PROVIDER: currently Google Gemini. To swap providers, change the key
# name below and the code inside the functions in this file - nothing else.
API_KEY_ENV = "GEMINI_API_KEY"


def has_api_key() -> bool:
    """Return True if GEMINI_API_KEY is set in .env or the environment."""
    return bool(os.environ.get(API_KEY_ENV))


def classify_batch(merchant_names: list[str], categories: list[str]) -> dict[str, str]:
    """Map each merchant name to one of `categories`.

    STUB: always answers "Other". A real version will send all names in one
    prompt and return e.g. {"Raj Textiles": "Shopping", ...}.
    """
    return {name: "Other" for name in merchant_names}
