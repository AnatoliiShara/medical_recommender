# src/safety/medical_safery_filter.py
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

@dataclass
class UserProfile:
    age: Optional[int] = None
    chronic_conditions: List[str] = field(default_factory=list)
    pregnancy: bool = False
    breastfeeding: bool = False
    allergies: List[str] = field(default_factory=list)

class MedicalSafetyFilter:
    def __init__(self, dedup_per_category: bool = True):
        self.dedup_per_category = dedup_per_category

        # Критичні протипоказання
        self.critical_patterns = {
            'age_restrictions': [
                r'діти до (\d+) років?',
                r'не застосовувати у дітей',
                r'протипоказано дітям',
                r'віком до (\d+)',
                r'особам старше (\d+) років',
                r'старше (\d+) років',
            ],
            'pregnancy_restrictions': [
                r'вагітн\w+',
                r'період вагітності',
                r'протипоказано при вагітності',
                r'не застосовувати вагітним'
            ],
            'breastfeeding_restrictions': [
                r'годуван\w+ грудд(ю|і)',
                r'лактац\w+',
                r'період лактації'
            ],
            'chronic_conditions': [
                r'серцева недостатність',
                r'печінкова недостатність',
                r'ниркова недостатність',
                r'цукровий діабет',
                r'артеріальна гіпертензія',
                r'бронхіальна астма'
            ],
            'allergy_hypersensitivity': [
                r'гіперчутлив\w+',
                r'алергічн\w+ реакці\w+',
                r'алергія на [\w\s-]+'
            ]
        }

        # Вагові коефіцієнти ризику
        self.risk_weights = {
            'age_restrictions': 10,
            'pregnancy_restrictions': 10,
            'breastfeeding_restrictions': 8,
            'chronic_conditions': 6,
            'allergy_hypersensitivity': 9
        }

    def assess_drug_safety(self, contraindications: str, user_profile: Optional[UserProfile] = None) -> Dict:
        safety_report = {
            'risk_score': 0,
            'risk_level': 'LOW',
            'warnings': [],
            'critical_warnings': [],
            'safe_to_use': True
        }
        if not contraindications:
            return safety_report

        text = contraindications.lower()
        categories_already_counted = set()

        for category, patterns in self.critical_patterns.items():
            category_hit = False
            for pattern in patterns:
                for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                    warning_info = {
                        'category': category,
                        'matched_text': match.group(0),
                        'weight': self.risk_weights[category]
                    }
                    conflict = user_profile and self._check_user_conflict(category, match, user_profile)

                    # Якщо треба — рахуємо ризик максимум 1 раз на категорію
                    if conflict:
                        if not self.dedup_per_category or category not in categories_already_counted:
                            safety_report['risk_score'] += self.risk_weights[category]
                            categories_already_counted.add(category)
                        safety_report['critical_warnings'].append(warning_info)
                        if self.risk_weights[category] >= 8:
                            safety_report['safe_to_use'] = False
                        category_hit = True
                    else:
                        safety_report['warnings'].append(warning_info)
                        category_hit = True

                # якщо вже було співпадіння цієї категорії і dedup_per_category вкл — можна не шукати далі
                if category_hit and self.dedup_per_category:
                    break

        # Глобальний рівень ризику
        if safety_report['risk_score'] >= 15:
            safety_report['risk_level'] = 'CRITICAL'
        elif safety_report['risk_score'] >= 8:
            safety_report['risk_level'] = 'HIGH'
        elif safety_report['risk_score'] >= 3:
            safety_report['risk_level'] = 'MEDIUM'

        if safety_report['risk_level'] == 'CRITICAL':
            safety_report['safe_to_use'] = False

        return safety_report

    def _check_user_conflict(self, category: str, match: re.Match, profile: UserProfile) -> bool:
        if not profile:
            return False

        mtxt = match.group(0).lower()

        if category == 'age_restrictions':
            # підтримка і "до N", і "старше N"
            mnum = re.search(r'(\d+)', mtxt)
            if mnum and profile.age is not None:
                n = int(mnum.group(1))
                if 'до' in mtxt:
                    return profile.age < n
                if 'старше' in mtxt or 'осіб старше' in mtxt or 'особам старше' in mtxt:
                    return profile.age > n
                # загальний випадок — трактуємо як "до n"
                return profile.age < n

        elif category == 'pregnancy_restrictions':
            return bool(profile.pregnancy)

        elif category == 'breastfeeding_restrictions':
            return bool(profile.breastfeeding)

        elif category == 'chronic_conditions':
            if not profile.chronic_conditions:
                return False
            return any(cond.lower() in mtxt for cond in profile.chronic_conditions)

        elif category == 'allergy_hypersensitivity':
            # якщо в профілі є "allergies", спробуємо знайти згадку конкретного алергену
            if profile.allergies:
                return any(a.lower() in mtxt for a in profile.allergies)
            # інакше вважаємо, що гіперчутливість сама по собі не 100% конфлікт без конкретики
            return False

        return False

    def filter_safe_drugs(self, drug_results: List[Dict], user_profile: Optional[UserProfile] = None) -> List[Dict]:
        filtered = []
        for drug in drug_results:
            contraindications = drug.get('contraindications', '') or ''
            report = self.assess_drug_safety(contraindications, user_profile)
            drug['safety_report'] = report
            drug['risk_score'] = report['risk_score']
            drug['risk_level'] = report['risk_level']
            # включаємо всі, окрім явних CRITICAL/unsafe
            if report['safe_to_use'] or report['risk_level'] in ['LOW', 'MEDIUM']:
                filtered.append(drug)
        filtered.sort(key=lambda x: x.get('risk_score', 0))
        return filtered
