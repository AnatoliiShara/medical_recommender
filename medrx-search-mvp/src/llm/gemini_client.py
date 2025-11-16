# src/llm/gemini_client.py
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, Optional

import google.generativeai as genai


GEMINI_DEFAULT_MODEL = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-pro")


class GeminiClientError(RuntimeError):
    """Помилка при роботі з Gemini API."""


def _configure() -> None:
    """Ініціалізує google.generativeai з API-ключем з env."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise GeminiClientError(
            "GEMINI_API_KEY не встановлено. "
            "Будь ласка, export GEMINI_API_KEY=... перед запуском."
        )
    genai.configure(api_key=api_key)


@lru_cache(maxsize=4)
def get_model(
    model_name: str = GEMINI_DEFAULT_MODEL,
    response_mime_type: Optional[str] = None,
) -> "genai.GenerativeModel":
    """
    Повертає кешований екземпляр Gemini-моделі.

    response_mime_type:
      - None -> звичайний текст
      - 'application/json' -> просимо Gemini повертати чистий JSON
    """
    _configure()

    generation_config: Dict[str, Any] = {}
    if response_mime_type:
        generation_config["response_mime_type"] = response_mime_type

    return genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config or None,
    )


def generate_json(
    prompt: str,
    model_name: str = GEMINI_DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    Надсилає промпт до Gemini і очікує СТРОГИЙ JSON у відповіді.

    Повертає:
        dict із розпарсеним JSON.

    Кидає:
        GeminiClientError якщо відповідь порожня або JSON не парситься.
    """
    model = get_model(model_name=model_name, response_mime_type="application/json")
    try:
        response = model.generate_content(prompt)
    except Exception as exc:  # noqa: BLE001
        raise GeminiClientError(f"Помилка виклику Gemini: {exc!r}") from exc

    text = getattr(response, "text", None)
    if not text:
        raise GeminiClientError("Порожня відповідь від Gemini (response.text is empty).")

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:  # noqa: B904
        snippet = text[:500].replace("\n", "\\n")
        raise GeminiClientError(
            f"Не вдалося розпарсити JSON-відповідь від Gemini: {e}. "
            f"Фрагмент відповіді: {snippet}"
        ) from e
