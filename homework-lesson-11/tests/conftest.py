import sys
import os

# Дозволяє тестам імпортувати модулі з батьківської директорії (agents/, config.py, tools.py тощо)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from deepeval.models import GeminiModel

from config import settings


@pytest.fixture(scope="session")
def eval_model():
    """Модель-суддя для GEval та вбудованих метрик DeepEval. Окрема від моделі самих агентів,
    хоча тут використовуємо ту саму (gemini-3.5-flash-lite) з міркувань економії квоти."""
    return GeminiModel(model=settings.model_name, api_key=settings.google_api_key)