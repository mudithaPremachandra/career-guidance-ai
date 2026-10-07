"""Engine A (30%): forward-chaining rule-based reasoning over the data/rules.json knowledge base."""

import json
import os
from typing import Any, Dict, List, Tuple

from src.knowledge_base import CAREER_UNIVERSE
from src.paths import DATA_DIR
from src.profile import StudentProfile


# The IF-THEN knowledge base lives in rules.json so rules can be audited and edited without code changes.
RULE_OPERATORS = {
    "gte": lambda a, b: a >= b,
    "gt": lambda a, b: a > b,
    "lte": lambda a, b: a <= b,
    "lt": lambda a, b: a < b,
    "eq": lambda a, b: a == b,
}


def load_rule_base() -> Dict[str, Any]:
    """Loads the IF-THEN career rule base from rules.json."""
    json_path = os.path.join(DATA_DIR, "rules.json")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


RULE_BASE = load_rule_base()


def evaluate_rule_condition(cond: Dict[str, Any], profile: StudentProfile, unified: Dict[str, float]) -> bool:
    """Recursively evaluates one rules.json condition against a student profile."""
    if "always" in cond:
        return bool(cond["always"])
    if "all" in cond:
        return all(evaluate_rule_condition(c, profile, unified) for c in cond["all"])
    if "any" in cond:
        return any(evaluate_rule_condition(c, profile, unified) for c in cond["any"])
    if "elective" in cond:
        return cond["elective"] in profile.electives

    if "skill" in cond:
        value = unified.get(cond["skill"], 0.0)
    elif "profile" in cond:
        value = getattr(profile, cond["profile"])
    else:
        raise ValueError(f"Unknown rule condition: {cond}")

    ops = [op for op in RULE_OPERATORS if op in cond]
    if len(ops) != 1:
        raise ValueError(f"Rule condition needs exactly one operator {list(RULE_OPERATORS)}: {cond}")
    return RULE_OPERATORS[ops[0]](value, cond[ops[0]])


def evaluate_rule_engine(profile: StudentProfile, career_name: str) -> Tuple[float, List[str]]:
    """Forward-chains the rules.json knowledge base and returns a normalized score plus the firing trace."""
    rules_fired: List[str] = []
    points = 0.0
    max_possible = float(RULE_BASE.get("max_points", 100.0))

    unified = profile.get_unified_skill_dict()
    career_info = CAREER_UNIVERSE[career_name]

    # Common rules first, then the career's own rules; an exclusive_group behaves as an IF / ELSE IF chain.
    fired_groups = set()
    for rule in RULE_BASE["common_rules"] + RULE_BASE["career_rules"].get(career_name, []):
        group = rule.get("exclusive_group")
        if group in fired_groups:
            continue
        if evaluate_rule_condition(rule["if"], profile, unified):
            points += float(rule["points"])
            rules_fired.append(f"Rule [{rule['id']}]: {rule['text']} (+{rule['points']:g}).")
            if group:
                fired_groups.add(group)

    # Job-to-Soft-Skill Archetype Requirement Mapping (+20% Weight Bonus)
    soft_rule = RULE_BASE["soft_skill_rule"]
    required_softs = career_info.get("required_soft_skills", [])
    weight_bonus = career_info.get("weight_bonus", soft_rule.get("default_weight_bonus", 0.20))
    matched_softs = [s for s in required_softs if s in profile.soft_skills]

    if required_softs:
        soft_ratio = len(matched_softs) / len(required_softs)
        soft_points = soft_ratio * (weight_bonus * 100.0)
        points += soft_points
        if matched_softs:
            rules_fired.append(
                f"Rule [{soft_rule['id']}]: Archetype '{career_info['archetype']}' "
                f"matched ({len(matched_softs)}/{len(required_softs)}) required soft skills: {', '.join(matched_softs[:2])} (+{soft_points:.1f})."
            )

    # Work style alignment rule
    style_rule = RULE_BASE["work_style_rule"]
    if profile.work_style in career_info["target_workstyles"]:
        points += float(style_rule["points"])
        rules_fired.append(f"Rule [{style_rule['id']}]: Work style preference '{profile.work_style}' aligns with role requirements (+{style_rule['points']:g}).")

    normalized_score = min(max(points / max_possible, 0.0), 1.0)
    return normalized_score, rules_fired
