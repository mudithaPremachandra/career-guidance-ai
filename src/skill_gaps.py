"""Skill gap analysis against the target career's benchmark."""

from typing import Any, Dict, List

from src.knowledge_base import CAREER_UNIVERSE, SKILL_LABELS
from src.profile import StudentProfile


def calculate_skill_gaps(profile: StudentProfile, top_career: str) -> List[Dict[str, Any]]:
    """Calculates missing competencies (Target Requisites minus Current Proficiency)."""
    career_info = CAREER_UNIVERSE[top_career]
    bench = career_info["benchmark_skills"]
    unified = profile.get_unified_skill_dict()
    gaps = []

    # 1. Technical Skill Gaps
    for skill_key, req_val in bench.items():
        curr_val = unified.get(skill_key, 0)
        if curr_val < req_val:
            deficit = req_val - curr_val
            scale = 100.0 if req_val > 5 else 5.0
            norm_deficit = deficit / scale

            if norm_deficit >= 0.25:
                urgency = "High Urgency"
                badge_class = "pf-badge-high"
            elif norm_deficit >= 0.12:
                urgency = "Medium Priority"
                badge_class = "pf-badge-med"
            else:
                urgency = "Low / Minor"
                badge_class = "pf-badge-low"

            gaps.append({
                "skill_key": skill_key,
                "skill_name": SKILL_LABELS.get(skill_key, skill_key),
                "current": f"{curr_val:g}/{scale:g}",
                "target": f"{req_val:g}/{scale:g}",
                "deficit": round(deficit, 1),
                "gap_pct": round(norm_deficit * 100.0, 1),
                "urgency": urgency,
                "badge_class": badge_class,
            })

    # 2. Critical Soft Skill Gaps
    req_softs = career_info.get("required_soft_skills", [])
    for rs in req_softs:
        if rs not in profile.soft_skills:
            gaps.append({
                "skill_key": f"SOFT_{rs}",
                "skill_name": f"Soft Skill: {rs}",
                "current": "Missing",
                "target": "Recommended",
                "deficit": 18.0,
                "gap_pct": 18.0,
                "urgency": "Medium Priority",
                "badge_class": "pf-badge-med",
            })

    # Module grades (0-100) and proficiencies (1-5) only compare as a share of their own scale
    gaps.sort(key=lambda x: x["gap_pct"], reverse=True)
    return gaps
