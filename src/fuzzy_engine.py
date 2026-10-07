"""Engine B (30%): Mamdani fuzzy inference system built with scikit-fuzzy."""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from src.knowledge_base import CAREER_UNIVERSE
from src.profile import StudentProfile


def build_fuzzy_career_system() -> ctrl.ControlSystem:
    """Builds the Mamdani fuzzy inference system (scikit-fuzzy): 4 fuzzified inputs -> career suitability."""
    gpa = ctrl.Antecedent(np.linspace(0.0, 4.0, 401), "gpa")
    gpa["low"] = fuzz.trapmf(gpa.universe, [0.0, 0.0, 2.4, 2.9])
    gpa["medium"] = fuzz.trimf(gpa.universe, [2.6, 3.2, 3.6])
    gpa["high"] = fuzz.trapmf(gpa.universe, [3.3, 3.7, 4.0, 4.0])

    # Average ratio of the student's skills to the career benchmark (each ratio capped at 1.2)
    tech = ctrl.Antecedent(np.linspace(0.0, 1.5, 151), "tech_mastery")
    tech["weak"] = fuzz.trapmf(tech.universe, [0.0, 0.0, 0.5, 0.7])
    tech["moderate"] = fuzz.trimf(tech.universe, [0.6, 0.85, 1.05])
    tech["strong"] = fuzz.trapmf(tech.universe, [0.9, 1.1, 1.5, 1.5])

    # Share of the career archetype's required soft skills the student has
    soft = ctrl.Antecedent(np.linspace(0.0, 1.0, 101), "soft_match")
    soft["low"] = fuzz.trapmf(soft.universe, [0.0, 0.0, 0.25, 0.45])
    soft["moderate"] = fuzz.trimf(soft.universe, [0.25, 0.5, 0.75])
    soft["high"] = fuzz.trapmf(soft.universe, [0.5, 0.75, 1.0, 1.0])

    # Work-style preference is crisp (1 = preferred style for the career), so its sets are complementary
    style = ctrl.Antecedent(np.linspace(0.0, 1.0, 101), "style_fit")
    style["mismatch"] = fuzz.trimf(style.universe, [0.0, 0.0, 1.0])
    style["match"] = fuzz.trimf(style.universe, [0.0, 1.0, 1.0])

    suit = ctrl.Consequent(np.linspace(0.0, 1.0, 101), "suitability", defuzzify_method="centroid")
    suit["poor"] = fuzz.trapmf(suit.universe, [0.0, 0.0, 0.2, 0.4])
    suit["fair"] = fuzz.trimf(suit.universe, [0.25, 0.45, 0.65])
    suit["good"] = fuzz.trimf(suit.universe, [0.55, 0.72, 0.88])
    suit["excellent"] = fuzz.trapmf(suit.universe, [0.8, 0.92, 1.0, 1.0])

    # Every tech_mastery term has at least one rule that always fires, so the output is always defined.
    rules = [
        ctrl.Rule(tech["strong"] & gpa["high"] & soft["high"], suit["excellent"], label="F1"),
        ctrl.Rule(tech["strong"] & soft["high"], suit["excellent"], label="F2"),
        ctrl.Rule(tech["strong"] & (soft["moderate"] | soft["low"]), suit["good"], label="F3"),
        ctrl.Rule(tech["moderate"] & (gpa["medium"] | gpa["high"]) & soft["moderate"], suit["good"], label="F4"),
        ctrl.Rule(tech["moderate"] & soft["high"], suit["good"], label="F5"),
        ctrl.Rule(tech["moderate"] & style["match"], suit["good"], label="F6"),
        ctrl.Rule(tech["moderate"] & (style["mismatch"] | soft["low"]), suit["fair"], label="F7"),
        ctrl.Rule(tech["weak"] & soft["high"] & style["match"], suit["fair"], label="F8"),
        ctrl.Rule(tech["weak"] & gpa["low"] & soft["low"], suit["poor"], label="F9"),
        ctrl.Rule(tech["weak"], suit["poor"], label="F10"),
    ]
    return ctrl.ControlSystem(rules)


FUZZY_CAREER_SYSTEM = build_fuzzy_career_system()


def evaluate_fuzzy_suitability(profile: StudentProfile, career_name: str) -> float:
    """Mamdani Fuzzy Inference (fuzzify -> min/max rule evaluation -> centroid defuzzification)."""
    career_info = CAREER_UNIVERSE[career_name]

    # Compute Technical Mastery Degree
    unified = profile.get_unified_skill_dict()
    skill_ratios = []
    for skill_k, req_val in career_info["benchmark_skills"].items():
        curr_val = unified.get(skill_k, 0)
        ratio = min(curr_val / req_val, 1.2) if req_val > 0 else 1.0
        skill_ratios.append(ratio)
    avg_skill_ratio = float(np.mean(skill_ratios)) if skill_ratios else 0.7

    # Soft Skill Archetype Overlap
    req_softs = career_info.get("required_soft_skills", [])
    matched_softs = [s for s in req_softs if s in profile.soft_skills]
    soft_ratio = len(matched_softs) / len(req_softs) if req_softs else 0.5

    sim = ctrl.ControlSystemSimulation(FUZZY_CAREER_SYSTEM)
    sim.input["gpa"] = float(np.clip(profile.gpa, 0.0, 4.0))
    sim.input["tech_mastery"] = float(np.clip(avg_skill_ratio, 0.0, 1.5))
    sim.input["soft_match"] = float(soft_ratio)
    sim.input["style_fit"] = 1.0 if profile.work_style in career_info["target_workstyles"] else 0.0
    sim.compute()
    crisp_suitability = float(sim.output["suitability"])

    # Academic Year Readiness scales the defuzzified output (final-years are closer to job-ready)
    year_map = {"1st Year": 0.4, "2nd Year": 0.65, "3rd Year": 0.85, "4th Year": 1.0}
    year_factor = year_map.get(profile.year, 0.75)

    final_fuzzy_score = crisp_suitability * (0.85 + 0.15 * year_factor)
    return min(max(final_fuzzy_score, 0.0), 1.0)
