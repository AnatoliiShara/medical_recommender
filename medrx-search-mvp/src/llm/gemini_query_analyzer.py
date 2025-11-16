# src/llm/gemini_query_analyzer.py
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

from llm.gemini_client import GeminiClientError, generate_json


@dataclass
class ClinicalFrame:
    """
    Структурований опис запиту користувача (не діагноз, а тріаж-рамка).

    ⚠️ Це НЕ є медичним висновком чи рецептом.
    """

    original_query: str
    primary_complaint: str
    symptoms: List[str]
    possible_conditions: List[str]
    age_group: str          # infant | child | adult | elderly | pregnant | unknown
    severity: str           # mild | moderate | severe | unknown
    urgency: str            # self_care | routine | urgent | emergency | unknown
    red_flags: List[str]
    notes: str

    @classmethod
    def from_llm_payload(cls, payload: Dict[str, Any]) -> "ClinicalFrame":
        """Створює ClinicalFrame із "сирого" JSON, з дефолтами."""
        def _list(key: str) -> List[str]:
            value = payload.get(key, [])
            if isinstance(value, list):
                return [str(x) for x in value]
            if value is None:
                return []
            return [str(value)]

        def _str(key: str, default: str = "unknown") -> str:
            value = payload.get(key, default)
            return str(value) if value is not None else default

        return cls(
            original_query=_str("original_query", ""),
            primary_complaint=_str("primary_complaint", ""),
            symptoms=_list("symptoms"),
            possible_conditions=_list("possible_conditions"),
            age_group=_str("age_group", "unknown"),
            severity=_str("severity", "unknown"),
            urgency=_str("urgency", "unknown"),
            red_flags=_list("red_flags"),
            notes=_str("notes", ""),
        )


def build_prompt(user_query: str) -> str:
    """
    Будує бі-лінгвальний промпт для витягу clinical_frame.

    Ми спеціально просимо:
    - STRICT JSON (через response_mime_type='application/json' і схему),
    - без прямих рекомендацій щодо лікування/дозування.
    """
    template = """
You are an assistant that performs *clinical triage style* analysis of a user's
free-text query about symptoms and medicines.

Your task:
- READ the user's query in Ukrainian (possibly with slang).
- UNDERSTAND the clinical situation on a high level.
- EXTRACT a structured JSON object describing the situation.

IMPORTANT SAFETY INSTRUCTIONS:
- You are NOT a doctor and you must NOT provide diagnoses or treatment plans.
- Do NOT recommend specific drugs, dosages or treatment regimens.
- Instead, you only:
  * identify the main complaint and symptoms,
  * highlight potential red-flag signs,
  * estimate severity and urgency level in very coarse categories.

Return STRICTLY ONE JSON object with the following schema:

{{
  "original_query": string,
  "primary_complaint": string,
  "symptoms": [string, ...],
  "possible_conditions": [string, ...],  // hypotheses, NOT final diagnoses
  "age_group": string,   // one of: "infant", "child", "adult", "elderly", "pregnant", "unknown"
  "severity": string,    // one of: "mild", "moderate", "severe", "unknown"
  "urgency": string,     // one of: "self_care", "routine", "urgent", "emergency", "unknown"
  "red_flags": [string, ...],
  "notes": string
}}

Guidelines:

- Language:
  * Use Ukrainian as the primary language.
  * You MAY add English medical terms in parentheses (e.g., "запалення товстого кишківника (colitis)").

- "possible_conditions":
  * Should contain HIGH-LEVEL hypotheses only (e.g. "інфекційна діарея", "бактеріальна інфекція кишечника").
  * Do NOT "upgrade" them to confirmed diagnoses.

- "red_flags":
  * Include alarming features like:
    - кров у калі / сечі
    - висока температура > 38.5°C тривалий час
    - сильний біль у грудях, задишка
    - ознаки сильного зневоднення
    - судоми, втрата свідомості
  * If there are clear red flags, set "urgency" to "urgent" or "emergency" accordingly.

- If information is insufficient, use "unknown" and explain missing pieces in "notes".

-------------------------
USER QUERY (Ukrainian, possibly with slang):

"{user_query}"
"""
    return template.format(user_query=user_query.replace('"', '\\"'))


def analyze_query(user_query: str) -> ClinicalFrame:
    """
    Основна функція: бере сирий текст запиту, повертає ClinicalFrame.
    """
    prompt = build_prompt(user_query)
    raw = generate_json(prompt)
    # Якщо Gemini сам не підставив original_query – гарантуємо його.
    raw.setdefault("original_query", user_query)
    return ClinicalFrame.from_llm_payload(raw)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "LLM-аналіз медичного запиту (Gemini) -> clinical_frame "
            "(тільки для розробницьких цілей, НЕ для пацієнтів)."
        )
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        help="Текст запиту користувача (якщо не вказано — зчитується з stdin).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Вивести JSON з відступами.",
    )
    args = parser.parse_args(argv)

    if args.query:
        query = args.query.strip()
    else:
        # дозволяє: echo "..." | PYTHONPATH=src python -m llm.gemini_query_analyzer
        query = sys.stdin.read().strip()

    if not query:
        print("Порожній запит. Використай --query або передай текст у stdin.", file=sys.stderr)
        return 1

    try:
        frame = analyze_query(query)
    except GeminiClientError as e:
        print(f"[ERROR] GeminiClientError: {e}", file=sys.stderr)
        return 2

    payload = asdict(frame)
    if args.pretty:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
