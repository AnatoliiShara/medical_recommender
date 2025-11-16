# src/llm/gemini_reranker.py
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

try:
    # Класичний SDK Gemini
    import google.generativeai as genai  # type: ignore[import]
except ImportError as exc:  # pragma: no cover
    genai = None

from llm.types import ClinicalFrame, DrugCandidate, GeminiCandidateScore


DEFAULT_MODEL = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-pro")


# ---------------- HELPERS FOR GEMINI ---------------- #

def _ensure_genai() -> None:
    if genai is None:
        raise SystemExit(
            "Модуль 'google-generativeai' не знайдено.\n"
            "Встанови його в активному venv:\n"
            "  pip install google-generativeai"
        )


def _generate_json_with_gemini(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """
    Один виклик до Gemini, який повертає ЧИСТИЙ JSON (dict).
    Для gemini-2.5-pro:
    - system_prompt йде в system_instruction при створенні моделі
    - generate_content отримує тільки user-повідомлення (без role='system')
    - якщо JSON обрізаний (truncated) -> пробуємо його полагодити
    """
    _ensure_genai()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY не встановлено.\n"
            "Експортуй ключ у цьому шеллі:\n"
            "  export GEMINI_API_KEY='...'"
        )

    genai.configure(api_key=api_key)

    model_name = DEFAULT_MODEL  # 'gemini-2.5-pro' у твоєму випадку
    model = genai.GenerativeModel(
        model_name,
        system_instruction=system_prompt,
    )

    response = model.generate_content(
        [user_prompt],
        generation_config={
            "temperature": 0.1,
            "top_p": 0.9,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
        },
    )

    text = getattr(response, "text", None)
    if not text and getattr(response, "candidates", None):
        cand = response.candidates[0]
        if cand and getattr(cand, "content", None):
            parts = getattr(cand.content, "parts", None) or []
            if parts and getattr(parts[0], "text", None):
                text = parts[0].text

    if not text:
        raise RuntimeError("Порожня відповідь від Gemini (немає тексту).")

    text = text.strip()

    # Якщо LLM обгорнув JSON в ```json ... ```
    if text.startswith("```"):
        lines = []
        for line in text.splitlines():
            if line.strip().startswith("```"):
                continue
            lines.append(line)
        text = "\n".join(lines).strip()

    # 1-а спроба: звичайний json.loads
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise RuntimeError(f"Очікував JSON-об'єкт (dict), отримав: {type(data)}")
        return data
    except json.JSONDecodeError as e:
        # 2-а спроба: JSON частково обрізаний -> відрізаємо все після останньої '}'
        last_brace = text.rfind("}")
        if last_brace != -1:
            truncated = text[: last_brace + 1]
            try:
                data = json.loads(truncated)
                if not isinstance(data, dict):
                    raise RuntimeError(
                        f"Очікував JSON-об'єкт (dict) після truncate, отримав: {type(data)}"
                    )
                # Корисно залогувати, що це був "repaired JSON"
                print(
                    "[WARN] Gemini JSON був обрізаний, використовую відновлену частину "
                    "(до останньої '}')."
                )
                return data
            except json.JSONDecodeError:
                # якщо навіть після truncate все зле – падаємо як раніше
                pass

        # якщо дійшли сюди – все дуже погано, кидаємо помилку з debug-текстом
        raise RuntimeError(
            f"Не вдалося розпарсити JSON від Gemini: {e}\nТекст:\n{text}"
        ) from e




# ---------------- PROMPT ---------------- #

SYSTEM_PROMPT = (
    "Ти клінічний фармацевт та експерт з доказової медицини.\n"
    "ТВОЄ ЗАВДАННЯ: аналізувати симптоми пацієнта та перелік ЛІКІВ, "
    "і повертати СТРУКТУРОВАНУ оцінку препаратів за релевантністю та безпекою.\n"
    "Ти НЕ ставиш остаточний діагноз і НЕ призначаєш лікування, "
    "а лише оцінюєш, які препарати БІЛЬШЕ чи МЕНШЕ ПІДХОДЯТЬ.\n"
    "Ти базуєшся на принципах evidence-based medicine.\n"
    "Відповідь ПОВИННА бути у форматі ЧИСТОГО JSON, без зайвого тексту."
)


def build_rerank_user_prompt(query: str, candidates: List[DrugCandidate]) -> str:
    """
    User-повідомлення для Gemini:
    - опис задачі
    - сирий запит користувача
    - список кандидатів
    - сувора JSON-схема відповіді
    """
    lines: List[str] = []

    lines.append(
        "Тобі дано:\n"
        "1) Неструктурований запит користувача українською мовою.\n"
        "2) Список кандидатів-препаратів з коротким описом.\n\n"
        "Треба:\n"
        "A) Проаналізувати запит та побудувати clinical_frame.\n"
        "B) Оцінити КОЖЕН препарат за релевантністю та БЕЗПЕКОЮ.\n\n"
    )

    # A. сирий запит
    lines.append("=== USER QUERY (UA) ===")
    lines.append(query.strip())
    lines.append("")

    # B. кандидати
    lines.append("=== DRUG CANDIDATES ===")
    for idx, cand in enumerate(candidates, start=1):
        desc = cand.short_description_uk()
        lines.append(f"{idx}. [drug_id={cand.drug_id}] {desc}")
    lines.append("")

    # clinical_frame
    lines.append("Спочатку побудуй об'єкт clinical_frame зі структурою:")
    lines.append(
        """
clinical_frame = {
  "original_query": <рядок, вихідний запит>,
  "primary_complaint": <короткий опис головної скарги>,
  "symptoms": [<список симптомів>],
  "possible_conditions": [<список можливих станів/хвороб (EN або лат.)>],
  "age_group": <"adult"|"child"|"elderly"|"unknown">,
  "severity": <"mild"|"moderate"|"severe"|"unknown">,
  "urgency": <"routine"|"urgent"|"emergency"|"unknown">,
  "red_flags": [<список тривожних ознак>],
  "notes": <короткі примітки, якщо треба>
}
"""
    )

    # candidates
    lines.append(
        "Потім створи масив candidates, де КОЖЕН препарат оцінений за схемою:\n"
        """
candidates = [
  {
    "drug_id": "<drug_id з вхідних даних, БЕЗ ЗМІНИ>",
    "gemini_score": <число від 0 до 10>,
    "safety_label": "<one of: 'ok', 'caution', 'contraindicated'>",
    "line_of_therapy": "<one of: 'first_line', 'second_line', 'avoid'>",
    "explanation_uk": "<1–2 речення українською, чому така оцінка>",
    "explanation_en": "<коротке пояснення англійською, optional>",
    "safety_reasons": "<якщо 'caution' або 'contraindicated' – чому саме>"
  },
  ...
]
"""
    )

    # правила безпеки
    lines.append(
        "КЛЮЧОВІ ПРАВИЛА БЕЗПЕКИ (приклади):\n"
        "- Якщо в запиті згадується кров у калі, ЛОПЕРАМІД та інші протидіарейні, що блокують перистальтику, "
        "мають бути позначені як 'contraindicated' або максимум 'caution'.\n"
        "- При тяжкій діареї та частих випорожненнях пріоритет мають засоби для пероральної регідратації.\n"
        "- Якщо препарат загалом не відповідає показанням – 'gemini_score' ближче до 0 та 'line_of_therapy'='avoid'.\n"
        "- Якщо препарат є препаратом ПЕРШОЇ ЛІНІЇ для даної ситуації, без явних протипоказань – "
        "дай високий бал (8–10) і 'line_of_therapy'='first_line'.\n"
        "- Не вигадуй протипоказань, які явно не випливають з опису препарату.\n"
    )

    # JSON вимога
    lines.append(
        "ФІНАЛЬНА ВІДПОВІДЬ ПОВИННА БУТИ СУВОРО У ВИГЛЯДІ JSON:\n"
        """
{
  "clinical_frame": { ... },
  "candidates": [ ... ]
}
"""
        "Не додавай жодних коментарів до або після JSON."
    )

    return "\n".join(lines)


# ---------------- CORE LOGIC ---------------- #

def gemini_rerank(
    query: str,
    candidates: List[DrugCandidate],
) -> Tuple[ClinicalFrame, List[GeminiCandidateScore]]:
    """
    Один виклик до Gemini:
    - аналізує запит -> clinical_frame
    - оцінює кожен препарат -> GeminiCandidateScore
    """
    user_prompt = build_rerank_user_prompt(query, candidates)
    raw = _generate_json_with_gemini(SYSTEM_PROMPT, user_prompt)

    if "clinical_frame" not in raw or "candidates" not in raw:
        raise RuntimeError(f"JSON від Gemini без потрібних ключів: {raw.keys()}")

    clinical = ClinicalFrame.from_dict(raw["clinical_frame"])

    scored: List[GeminiCandidateScore] = []
    for item in raw.get("candidates", []):
        try:
            scored.append(GeminiCandidateScore.from_dict(item))
        except Exception as e:
            print(f"[WARN] Failed to parse candidate item {item!r}: {e}", flush=True)
            continue

    return clinical, scored


# ---------------- DEMO / CLI ---------------- #

def demo_diarrhea_case(pretty: bool = True) -> None:
    """
    Демонстраційний сценарій:
    - запит про діарею з кров'ю
    - 3 препарати: лоперамід, регідрон, смекта
    """
    query = "у мене діарея зеленого кольору з кров'ю вже 7 разів сьогодні"

    candidates = [
        DrugCandidate(
            drug_id="loperamide_demo",
            brand_name="Лоперамід",
            inn_name="Loperamide",
            indications=(
                "Симптоматичне лікування гострої та хронічної діареї без крові в калі "
                "та без високої температури."
            ),
            contraindications=(
                "Гостра дизентерія з кров'янистими випорожненнями і високою температурою; "
                "псевдомембранозний коліт; бактеріальний ентероколіт; вік до 6 років."
            ),
            dosage_form="капсули",
            ce_score=0.85,
        ),
        DrugCandidate(
            drug_id="rehydron_demo",
            brand_name="Регідрон",
            inn_name="Oral rehydration salts",
            indications=(
                "Пероральна регідратація при гострій діареї, профілактика та лікування "
                "зневоднення легкої та середньої тяжкості."
            ),
            contraindications=(
                "Тяжка дегідратація, що потребує внутрішньовенної регідратації; "
                "виражена гіперкаліємія; деякі порушення функції нирок."
            ),
            dosage_form="порошок для приготування розчину",
            ce_score=0.9,
        ),
        DrugCandidate(
            drug_id="smecta_demo",
            brand_name="Смекта",
            inn_name="Diosmectite",
            indications=(
                "Симптоматичне лікування гострої та хронічної діареї, "
                "у тому числі інфекційної, у складі комплексної терапії."
            ),
            contraindications=(
                "Кишкова непрохідність; важкі хронічні запори; "
                "підвищена чутливість до компонентів препарату."
            ),
            dosage_form="суспензія / порошок для приготування суспензії",
            ce_score=0.7,
        ),
    ]

    clinical, scored = gemini_rerank(query, candidates)
    out = {
        "clinical_frame": clinical.to_dict(),
        "candidates": [c.to_dict() for c in scored],
    }

    if pretty:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(out, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gemini LLM-rerank + safety demo (поки на демо-кейсах)."
    )
    parser.add_argument(
        "--query",
        type=str,
        default="у мене діарея зеленого кольору з кров'ю вже 7 разів сьогодні",
        help="Запит користувача українською (якщо не --demo).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Якщо вказано, ігнорує --query і запускає вбудований демо-сценарій.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Гарне форматування JSON (indent=2).",
    )

    args = parser.parse_args()

    if args.demo:
        demo_diarrhea_case(pretty=args.pretty)
    else:
        query = args.query

        candidates = [
            DrugCandidate(
                drug_id="loperamide_demo",
                brand_name="Лоперамід",
                inn_name="Loperamide",
                indications=(
                    "Симптоматичне лікування гострої та хронічної діареї без крові в калі "
                    "та без високої температури."
                ),
                contraindications=(
                    "Гостра дизентерія з кров'янистими випорожненнями і високою температурою; "
                    "псевдомембранозний коліт; бактеріальний ентероколіт; вік до 6 років."
                ),
                dosage_form="капсули",
                ce_score=0.85,
            ),
            DrugCandidate(
                drug_id="rehydron_demo",
                brand_name="Регідрон",
                inn_name="Oral rehydration salts",
                indications=(
                    "Пероральна регідратація при гострій діареї, профілактика та лікування "
                    "зневоднення легкої та середньої тяжкості."
                ),
                contraindications=(
                    "Тяжка дегідратація, що потребує внутрішньовенної регідратації; "
                    "виражена гіперкаліємія; деякі порушення функції нирок."
                ),
                dosage_form="порошок для приготування розчину",
                ce_score=0.9,
            ),
            DrugCandidate(
                drug_id="smecta_demo",
                brand_name="Смекта",
                inn_name="Diosmectite",
                indications=(
                    "Симптоматичне лікування гострої та хронічної діареї, "
                    "у тому числі інфекційної, у складі комплексної терапії."
                ),
                contraindications=(
                    "Кишкова непрохідність; важкі хронічні запори; "
                    "підвищена чутливість до компонентів препарату."
                ),
                dosage_form="суспензія / порошок для приготування суспензії",
                ce_score=0.7,
            ),
        ]

        clinical, scored = gemini_rerank(query, candidates)
        out = {
            "clinical_frame": clinical.to_dict(),
            "candidates": [c.to_dict() for c in scored],
        }

        if args.pretty:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
