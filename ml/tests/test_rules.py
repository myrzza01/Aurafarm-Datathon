import json
from pathlib import Path

import pytest

from cashwatch.rules import action_explanation
from cashwatch.reviewed_data import ACTION_TYPES


def test_all_human_classes_have_explicit_conditional_or_unmapped_references():
    catalog = json.loads((Path(__file__).parents[1] / "cashwatch" / "rules.json").read_text(encoding="utf-8"))
    assert set(catalog["types"]) == set(ACTION_TYPES)
    for key, rule in catalog["types"].items():
        explanation = action_explanation(key, .65, 10, 12)
        assert explanation["type_id"] == key
        assert explanation["description_source"] == "class_template"
        assert "предполагает" in explanation["description"]
        if rule["source_status"] == "verified_conditional":
            assert explanation["rule_source"]["status"] == "provided"
            assert rule["rule_ids"] and explanation["rule_context"]["applicability_required"]
        else:
            assert explanation["rule_source"]["status"] == "unmapped"
            assert explanation["rule_source"]["reason"]
    with pytest.raises(ValueError):
        action_explanation("invented", .99, 1, 2)
