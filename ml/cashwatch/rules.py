"""Resolve verified local references only for a trained semantic prediction."""
import copy
import json
from pathlib import Path


def action_explanation(type_id, confidence, start_sec, end_sec):
    catalog = json.loads(Path(__file__).with_suffix(".json").read_text(encoding="utf-8"))
    item = catalog["types"].get(type_id)
    if item is None:
        raise ValueError("Unknown semantic type in checkpoint")
    description = (f"Модель предполагает: {item['observed_action_template']} "
        f"Фрагмент для визуальной проверки: {start_sec:.1f}–{end_sec:.1f} с. "
        f"{item['review_reason']} Это описание предсказанного класса; детали необходимо сверить с кадрами.")
    return {"type": item["label"], "type_id": type_id, "type_confidence": float(confidence),
        "description": description, "description_source": "class_template",
        "rule_source": copy.deepcopy(item["rule_source"]),
        "rule_context": {key: copy.deepcopy(item[key]) for key in ("applicability_required", "limitations", "source_status")}}
