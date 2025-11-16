# src/llm/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------- ClinicalFrame ---------------- #

@dataclass
class ClinicalFrame:
    """
    Структурований аналіз запиту користувача (те, що зараз повертає gemini_query_analyzer).
    """
    original_query: str
    primary_complaint: str
    symptoms: List[str] = field(default_factory=list)
    possible_conditions: List[str] = field(default_factory=list)
    age_group: str = "unknown"        # e.g. "adult", "child", "elderly", "unknown"
    severity: str = "unknown"         # e.g. "mild", "moderate", "severe"
    urgency: str = "routine"          # e.g. "routine", "urgent", "emergency"
    red_flags: List[str] = field(default_factory=list)
    notes: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClinicalFrame":
        return cls(
            original_query=data.get("original_query", "") or "",
            primary_complaint=data.get("primary_complaint", "") or "",
            symptoms=list(data.get("symptoms", []) or []),
            possible_conditions=list(data.get("possible_conditions", []) or []),
            age_group=data.get("age_group", "unknown") or "unknown",
            severity=data.get("severity", "unknown") or "unknown",
            urgency=data.get("urgency", "routine") or "routine",
            red_flags=list(data.get("red_flags", []) or []),
            notes=data.get("notes", "") or "",
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "primary_complaint": self.primary_complaint,
            "symptoms": self.symptoms,
            "possible_conditions": self.possible_conditions,
            "age_group": self.age_group,
            "severity": self.severity,
            "urgency": self.urgency,
            "red_flags": self.red_flags,
            "notes": self.notes,
        }


# ---------------- DrugCandidate ---------------- #

@dataclass
class DrugCandidate:
    """
    Один кандидат-препарат із нашого пайплайна (після BM25 + dense + CE).
    ЦЕ ЩЕ НЕ прив'язано до конкретних полів compendium — це узагальнена структура.
    """
    drug_id: str                      # внутрішній id (brand_id / doc_id / whatever)
    brand_name: str                   # комерційна назва
    inn_name: Optional[str]           # МНН (міжнародна непатентована назва)
    indications: str                  # основні показання
    contraindications: str            # ключові протипоказання (особливо red-flag'и)
    dosage_form: Optional[str] = None # форма випуску, якщо є
    ce_score: float = 0.0             # score від CrossEncoder або hybrid-пайплайна

    def short_description_uk(self) -> str:
        """
        Дуже стисла summary для prompt'а (укр + англ/латинь, якщо є).
        Тут критично: показання + протипоказання.
        """
        inn_part = f" (МНН: {self.inn_name})" if self.inn_name else ""
        form_part = f", форма: {self.dosage_form}" if self.dosage_form else ""
        return (
            f"Препарат: {self.brand_name}{inn_part}{form_part}. "
            f"Показання: {self.indications}. "
            f"Протипоказання: {self.contraindications}."
        )


# ---------------- GeminiCandidateScore ---------------- #

@dataclass
class GeminiCandidateScore:
    """
    Результат оцінки одного препарату з боку Gemini.
    """
    drug_id: str
    gemini_score: float               # 0–10
    safety_label: str                 # "ok" | "caution" | "contraindicated"
    line_of_therapy: str              # "first_line" | "second_line" | "avoid"
    explanation_uk: str               # коротке пояснення українською
    explanation_en: Optional[str] = None
    safety_reasons: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeminiCandidateScore":
        return cls(
            drug_id=data.get("drug_id", ""),
            gemini_score=float(data.get("gemini_score", 0.0)),
            safety_label=data.get("safety_label", "ok"),
            line_of_therapy=data.get("line_of_therapy", "second_line"),
            explanation_uk=data.get("explanation_uk", "") or "",
            explanation_en=data.get("explanation_en"),
            safety_reasons=data.get("safety_reasons"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drug_id": self.drug_id,
            "gemini_score": self.gemini_score,
            "safety_label": self.safety_label,
            "line_of_therapy": self.line_of_therapy,
            "explanation_uk": self.explanation_uk,
            "explanation_en": self.explanation_en,
            "safety_reasons": self.safety_reasons,
        }
