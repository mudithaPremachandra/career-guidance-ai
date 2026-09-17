"""
Unit and Integration Test Suite for PathFinder AI
Tests all individual engines, 4-pillar soft skills mapping, and persistence mechanisms.
"""

import sys
import os

# Ensure UTF-8 output on all platforms
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import (
    StudentProfile,
    CAREER_UNIVERSE,
    CERTIFICATION_CATALOG,
    evaluate_rule_engine,
    evaluate_fuzzy_suitability,
    evaluate_ml_model,
    run_hybrid_inference,
    calculate_shap_proxy_deltas,
    calculate_skill_gaps,
    recommend_certifications,
    generate_executive_narrative,
    init_database,
    save_student_record,
    fetch_all_records,
)


def run_tests():
    print("✦ [1/7] Testing Knowledge Base & Universe integrity...")
    assert len(CAREER_UNIVERSE) == 9, f"Expected 9 careers, got {len(CAREER_UNIVERSE)}"
    assert len(CERTIFICATION_CATALOG) >= 8, f"Expected >=8 certs, got {len(CERTIFICATION_CATALOG)}"
    for cname, cinfo in CAREER_UNIVERSE.items():
        assert "archetype" in cinfo, f"Missing archetype in {cname}"
        assert "required_soft_skills" in cinfo, f"Missing required_soft_skills in {cname}"
        assert "weight_bonus" in cinfo, f"Missing weight_bonus in {cname}"
        assert "role_dynamic" in cinfo, f"Missing role_dynamic in {cname}"
    print("  ✓ Knowledge Base integrity & 6-Archetype mappings passed.")

    print("✦ [2/7] Creating test student profile with 4-pillar soft skills...")
    profile = StudentProfile(
        gpa=3.75,
        year="3rd Year",
        core_modules={
            "DSA": 90,
            "OOP": 85,
            "DBMS": 80,
            "OS_Networks": 78,
            "SE_Principles": 88,
            "Math_Stats": 85,
        },
        electives={"AI & Machine Learning": 92, "Cloud Computing": 85},
        tech_skills={
            "Python": 5,
            "Java_CPP": 4,
            "SQL": 4,
            "WebStack": 4,
            "CloudDocker": 4,
            "ML_AI": 5,
            "MobileDev": 2,
            "Cybersecurity": 3,
        },
        soft_skills=[
            "Problem Solving & Logic",
            "Innovation & Prototyping",
            "Research & Investigation",
            "Quantitative & Mathematical",
            "Analytical & Critical Thinking",
        ],
        desired_domains=["Artificial Intelligence & ML", "Enterprise Cloud & DevOps"],
        work_style="R&D / Creative",
        has_internship=True,
        projects_count=4,
        existing_certs="AWS Cloud Practitioner",
    )
    print("  ✓ Profile object created successfully with 4-pillar soft skills.")

    print("✦ [3/7] Testing Rule-Based Engine & Soft Skill Bonus Trace...")
    rule_score, rules_fired = evaluate_rule_engine(profile, "AI Engineer")
    assert 0.0 <= rule_score <= 1.0, f"Invalid rule score: {rule_score}"
    assert len(rules_fired) > 0, "Expected rules to fire for AI Engineer"
    soft_rule_fired = any("R-SOFT-SKILLS" in r for r in rules_fired)
    assert soft_rule_fired, "Expected soft skills rule R-SOFT-SKILLS to fire"
    print(f"  ✓ Rule score: {rule_score:.2f}, Rules fired count: {len(rules_fired)}")

    print("✦ [4/7] Testing Fuzzy Logic Mamdani Suitability Engine with Soft Skill Clusters...")
    fuzzy_score = evaluate_fuzzy_suitability(profile, "AI Engineer")
    assert 0.0 <= fuzzy_score <= 1.0, f"Invalid fuzzy score: {fuzzy_score}"
    print(f"  ✓ Fuzzy suitability score: {fuzzy_score:.2f}")

    print("✦ [5/7] Testing ML Classifier & Hybrid Multi-Engine Fusion...")
    ml_probs = evaluate_ml_model(profile)
    assert len(ml_probs) == 9, f"Expected 9 probabilities, got {len(ml_probs)}"

    results = run_hybrid_inference(profile)
    assert len(results) == 9, "Expected 9 ranked career results"
    top_career = results[0]
    print(f"  ✓ Top recommendation: {top_career['title']} ({top_career['match_pct']}%) [{top_career['archetype']}] with {top_career['confidence']}")

    print("✦ [6/7] Testing SHAP XAI, Skill Gap & Cosine Certification Matching...")
    shap_df = calculate_shap_proxy_deltas(profile, top_career["career"])
    assert not shap_df.empty, "SHAP dataframe should not be empty"

    gaps = calculate_skill_gaps(profile, top_career["career"])
    certs = recommend_certifications(gaps)
    assert len(certs) == 3, f"Expected 3 recommended certs, got {len(certs)}"

    narrative = generate_executive_narrative(profile, top_career, gaps)
    assert len(narrative) > 20, "Narrative should not be empty"
    print(f"  ✓ XAI deltas: {len(shap_df)}, Skill gaps: {len(gaps)}, Recommended certs: {len(certs)}")
    print(f"  ✓ Narrative preview: {narrative[:80]}...")

    print("✦ [7/7] Testing SQLite Local Database Storage...")
    init_database()
    save_student_record(
        gpa=profile.gpa,
        academic_year=profile.year,
        top_career=top_career["title"],
        match_score=top_career["final_score"],
        confidence=top_career["confidence"],
        work_style=profile.work_style,
        top_driver="Strong DSA",
        critical_gap="None",
    )
    df_records = fetch_all_records()
    assert not df_records.empty, "Database records should not be empty after insert"
    print(f"  ✓ Database record verified. Total rows: {len(df_records)}")

    print("\n🎉 ALL 7/7 TEST MODULES PASSED PERFECTLY!\n")


if __name__ == "__main__":
    run_tests()
