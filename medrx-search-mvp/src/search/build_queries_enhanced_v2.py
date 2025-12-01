#!/usr/bin/env python
import argparse
import copy
import json
import os


# ----------------------------
#  СИНТЕТИЧНІ ДІАГНОСТИЧНІ ЗАПИТИ
# ----------------------------

SYNTHETIC_TEXTS = [
    # 1. Діарея / пронос – базовий симптом, OTC
    "У мене третій день рідкий водянистий пронос без температури, що можна приймати з аптечних препаратів?",
    # 2. Змішана мова, після антибіотиків, кров у калі
    "Понос зелёного цвета с прожилками крови после антибиотиков – какие лекарства от диареи есть в аптеке для взрослого?",
    # 3. Дитяча діарея + регідратація
    "Дитина 2 роки, пронос і блювання, температура 38.5 – які оральні регідратаційні розчини типу регідрону продаються в аптеці?",
    # 4. Хронічний запор
    "Запор вже тиждень, тверді калові маси, біль при дефекації – які проносні краще при хронічному запорі?",
    # 5. Метеоризм / здуття
    "М'який стілець раз на два дні, здуття та гази – які препарати від метеоризму та для нормалізації мікрофлори є в аптеці?",
    # 6. Сухий кашель, можлива алергія
    "Сухий кашель вночі, без температури, підозра на алергію – які безрецептурні протикашльові підходять дорослому?",
    # 7. Мокрий кашель / муколітики
    "Мокрий кашель з жовто-зеленою мокротою, температура 37.8 – які муколітики та відхаркувальні є в аптеці?",
    # 8. Стенозуючий / гавкаючий кашель у дитини
    "Гавкаючий кашель і утруднене дихання в дитини 3 роки – які ліки для полегшення дихання дозволені дітям?",
    # 9. Ангіна / біль у горлі, висока температура
    "Біль у горлі, температура 39 і наліт на мигдаликах – які спреї та таблетки від горла можна купити в аптеці?",
    # 10. Легка ГРВІ / симптоматичні засоби
    "ГРВІ: нежить, слабкість, температура 37.5 – які комплексні порошки від застуди є в українських аптеках?",
    # 11. Сезонна алергія, неседативні антигістамінні
    "Сезонна алергія, закладеність носа, чхання – які антигістамінні краще не викликають сонливості?",
    # 12. Гіпертонія + МНН
    "Високий тиск 160/100, періодично приймаю лозартан – які ще препарати від гіпертонії існують за МНН в аптеках?",
    # 13. Діабет + застуда
    "Цукровий діабет 2 типу, приймаю метформін – які препарати від застуди безпечніші при діабеті?",
    # 14. Ниркова недостатність + знеболювальні
    "Хронічна хвороба нирок, креатинін підвищений – які знеболювальні краще уникати, а які відносно безпечніші?",
    # 15. Вагітність + застуда
    "Вагітність 12 тижнів, нежить та кашель – які ліки від застуди дозволені вагітним?",
    # 16. Лактація + біль
    "Годую грудьми і маю сильний головний біль – які знеболювальні сумісні з лактацією?",
    # 17. Алергія на пеніцилін + антибіотик при ангіні
    "Алергія на пеніцилін – які антибіотики при ангіні можна розглядати як альтернативу?",
    # 18. Залізодефіцитна анемія
    "В аналізі крові впав гемоглобін, лікар сказав залізодефіцитна анемія – які препарати заліза краще засвоюються?",
    # 19. Печія / рефлюкс
    "Сильна печія після їжі, відрижка кислим – які препарати від печії та рефлюксу є в аптеці?",
    # 20. Біль у шлунку після НПЗП
    "Гострий біль у шлунку після прийому ібупрофену – які гастропротектори чи інші засоби захищають слизову шлунка?",
    # 21. Тривога / сон, безрецептурні седативні
    "Часті панічні атаки, тривога, поганий сон – які безрецептурні заспокійливі та фітопрепарати існують?",
    # 22. Грибок нігтів
    "Підозра на грибок нігтів стопи – які протигрибкові засоби місцевої дії є в продажу?",
    # 23. Висип після нового препарату
    "Висип після прийому нового препарату – які протиалергічні та мазі від висипу можна купити без рецепта?",
    # 24. Хронічний біль у спині, НПЗП
    "Хронічний біль у спині, іноді приймаю ібупрофен – які альтернативні НПЗП або комбіновані засоби є?",
    # 25. Варфарин + кровотечі
    "Я вже приймаю варфарин – які знеболювальні безпечніші з урахуванням ризику кровотеч?",
    # 26. Дитяча лихоманка
    "Дитина 5 років, температура 39, погано збивається – які форми жарознижувальних зручніші для дітей?",
    # 27. Морська хвороба у дитини
    "Потрібні ліки від морської хвороби для дитини 8 років – які препарати є в українських аптеках?",
    # 28. Порушення сну
    "Поганий сон, складно заснути – які безрецептурні препарати мелатоніну та фітозасоби є в аптеках?",
    # 29. Аналоги бренду за МНН
    "Шукаю дешевший аналог брендових крапель у ніс на ксилометазоліні – які є аналоги за МНН?",
    # 30. Подорож + профілактика діареї
    "Які препарати від діареї та зневоднення краще взяти з собою у подорож до Єгипту?"
]


# ----------------------------
#  СЕРВІСНІ ФУНКЦІЇ
# ----------------------------

def load_json_or_jsonl(path):
    """Повертає (список_запитів, формат), де формат = 'json' або 'jsonl'."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # пробуємо як JSON-масив
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data, "json"
    except Exception:
        pass

    # fallback: JSONL
    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items, "jsonl"


def detect_keys(sample):
    """Автовизначення ключів для ID та тексту запиту."""
    id_candidates = ["qid", "id", "query_id"]
    text_candidates = ["query", "original_query", "text", "question"]

    id_key = next((k for k in id_candidates if k in sample), None)
    text_key = next((k for k in text_candidates if k in sample), None)

    if text_key is None:
        raise ValueError(
            f"Не знайшов поле з текстом запиту у прикладі: доступні ключі {list(sample.keys())}"
        )

    return id_key, text_key


def get_query_id(obj, id_key, text_key):
    """ID запиту – або явний ID, або сам текст."""
    if id_key and id_key in obj:
        return str(obj[id_key])
    return str(obj[text_key])


def load_processed_ids(processed_path, id_key, text_key):
    """
    Зчитує ID уже оброблених запитів з JSONL результатів Gemini.
    Підтримує структуру з clinical_frame.
    """
    processed = set()
    if not processed_path or not os.path.exists(processed_path):
        return processed

    with open(processed_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            pid = None

            # 1) пробуємо явне поле в корені
            if id_key and id_key in obj:
                pid = obj[id_key]

            cf = obj.get("clinical_frame", {}) or {}

            # 2) пробуємо id_key всередині clinical_frame
            if pid is None and id_key and id_key in cf:
                pid = cf[id_key]

            # 3) падіння до тексту запиту в clinical_frame
            if pid is None and text_key in cf:
                pid = cf[text_key]

            # 4) останній fallback – стандартні імена полів
            if pid is None:
                for k in ["original_query", "query", "text", "question"]:
                    if k in obj:
                        pid = obj[k]
                        break

            if pid is None:
                continue

            processed.add(str(pid))

    return processed


def make_synth_records(template, id_key, text_key, start_idx=1, max_count=None):
    """Створює записи для синтетичних запитів на базі першого шаблону."""
    records = []
    counter = start_idx
    for txt in SYNTHETIC_TEXTS:
        rec = copy.deepcopy(template)
        if id_key:
            rec[id_key] = f"syn_{counter:03d}"
        rec[text_key] = txt
        records.append(rec)
        counter += 1
        if max_count is not None and len(records) >= max_count:
            break
    return records


# ----------------------------
#  MAIN
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--orig",
        required=True,
        help="Шлях до queries_medrx_ua.enhanced.json(.jsonl)",
    )
    ap.add_argument(
        "--processed",
        required=True,
        help="JSONL з результатами попереднього запуску Gemini "
             "(наприклад gemini_priors_enhanced.jsonl або gemini_reranked.jsonl)",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Куди писати queries_medrx_ua.enhanced_2.json(.jsonl)",
    )
    ap.add_argument(
        "--max_queries",
        type=int,
        default=50,
        help="Максимальна кількість запитів у новому файлі (за замовчуванням 50).",
    )
    args = ap.parse_args()

    queries, fmt = load_json_or_jsonl(args.orig)
    if not queries:
        raise SystemExit("У вихідному файлі немає жодного запиту")

    id_key, text_key = detect_keys(queries[0])
    print(f"[INFO] Визначив ключі: id_key={id_key}, text_key={text_key}")

    processed_ids = load_processed_ids(args.processed, id_key, text_key)
    print(f"[INFO] Завантажено {len(processed_ids)} вже оброблених запитів Gemini")

    # відфільтровуємо необроблені
    unprocessed = []
    for q in queries:
        qid = get_query_id(q, id_key, text_key)
        if qid not in processed_ids:
            unprocessed.append(q)
    print(f"[INFO] Необроблених запитів знайдено: {len(unprocessed)}")

    selected = []
    for q in unprocessed:
        if len(selected) >= args.max_queries:
            break
        selected.append(q)

    remaining_slots = args.max_queries - len(selected)
    print(
        f"[INFO] Беремо {len(selected)} оригінальних необроблених запитів, "
        f"залишилось слотів для синтетичних: {remaining_slots}"
    )

    if remaining_slots > 0:
        template = queries[0]
        synth_records = make_synth_records(
            template,
            id_key,
            text_key,
            start_idx=1,
            max_count=remaining_slots,
        )
        selected.extend(synth_records)
        print(f"[INFO] Додано {len(synth_records)} синтетичних запитів")

    # записуємо у тому ж форматі, що й оригінал
    with open(args.output, "w", encoding="utf-8") as f:
        if fmt == "json":
            json.dump(selected, f, ensure_ascii=False, indent=2)
        else:
            for obj in selected:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(
        f"[DONE] Записано {len(selected)} запитів у {args.output} (формат {fmt})"
    )


if __name__ == "__main__":
    main()
