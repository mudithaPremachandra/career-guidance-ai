"""Hybrid score fusion of the three engines and confidence bands."""

from typing import Any, Dict, List

from src.fuzzy_engine import evaluate_fuzzy_suitability
from src.knowledge_base import CAREER_UNIVERSE
from src.ml_engine import evaluate_ml_model
from src.profile import StudentProfile
from src.rule_engine import evaluate_rule_engine


HIGH_FIT_THRESHOLD = 0.63

VIABLE_FIT_THRESHOLD = 0.46


def run_hybrid_inference(profile: StudentProfile) -> List[Dict[str, Any]]:
    """Combines Rule-based (30%), Fuzzy logic (30%), and ML probabilities (40%)."""
    ml_probs = evaluate_ml_model(profile)
    results = []

    for career_name, career_info in CAREER_UNIVERSE.items():
        rule_score, rules_fired = evaluate_rule_engine(profile, career_name)
        fuzzy_score = evaluate_fuzzy_suitability(profile, career_name)
        ml_score = ml_probs.get(career_name, 0.1)

        # Weighted Score Fusion
        final_score = (0.30 * rule_score) + (0.30 * fuzzy_score) + (0.40 * ml_score)
        final_score = min(max(final_score, 0.0), 0.99)

        if final_score >= HIGH_FIT_THRESHOLD:
            confidence = "High (⚡ Strong Fit)"
            conf_color = "#34D399"
        elif final_score >= VIABLE_FIT_THRESHOLD:
            confidence = "Moderate (⚡ Viable Path)"
            conf_color = "#00F0FF"
        else:
            confidence = "Exploring (⚡ Emerging)"
            conf_color = "#FBBF24"

        # Calculate soft skill match breakdown
        req_softs = career_info.get("required_soft_skills", [])
        matched_softs = [s for s in req_softs if s in profile.soft_skills]
        missing_softs = [s for s in req_softs if s not in profile.soft_skills]

        results.append({
            "career": career_name,
            "title": career_info["title"],
            "archetype": career_info["archetype"],
            "role_dynamic": career_info["role_dynamic"],
            "category": career_info["category"],
            "description": career_info["description"],
            "icon": career_info["icon"],
            "final_score": final_score,
            "match_pct": round(final_score * 100, 1),
            "confidence": confidence,
            "conf_color": conf_color,
            "rule_score": round(rule_score * 100, 1),
            "fuzzy_score": round(fuzzy_score * 100, 1),
            "ml_score": round(ml_score * 100, 1),
            "rules_fired": rules_fired,
            "matched_soft_skills": matched_softs,
            "missing_soft_skills": missing_softs,
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results
