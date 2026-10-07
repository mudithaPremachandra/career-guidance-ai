"""
Unit and Integration Test Suite for PathFinder AI
Tests all individual engines, 4-pillar soft skills mapping, and persistence mechanisms.
"""

import sys
import os
import tempfile

# Ensure UTF-8 output on all platforms
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Make the repository root importable (this file lives in tests/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use a throwaway database by default, so test runs never touch or migrate data/career_records.db
TEST_DB_DIR = tempfile.TemporaryDirectory()
os.environ["CAREER_DB_FILE"] = os.path.join(TEST_DB_DIR.name, "import_records.db")

import src.database as database
import src.guidance as guidance_module
from src.database import init_database, save_student_record, fetch_all_records, fetch_student_history
from src.explainability import calculate_shap_contributions
from src.fusion import run_hybrid_inference
from src.fuzzy_engine import evaluate_fuzzy_suitability
from src.guidance import generate_executive_narrative, generate_guidance
from src.knowledge_base import CAREER_UNIVERSE, CERTIFICATION_CATALOG
from src.ml_engine import evaluate_ml_model, generate_benchmark_dataset, train_custom_classifier, compare_classifiers
from src.profile import StudentProfile
from src.progress import compare_assessments, profile_to_dict, parse_held_certifications
from src.recommender import recommend_certifications
from src.rule_engine import RULE_BASE, evaluate_rule_engine
from src.skill_gaps import calculate_skill_gaps
from src.train import train_and_save


def run_tests():
    print("✦ [1/11] Testing Knowledge Base & Universe integrity...")
    assert len(CAREER_UNIVERSE) == 9, f"Expected 9 careers, got {len(CAREER_UNIVERSE)}"
    assert len(CERTIFICATION_CATALOG) >= 8, f"Expected >=8 certs, got {len(CERTIFICATION_CATALOG)}"
    for cname, cinfo in CAREER_UNIVERSE.items():
        assert "archetype" in cinfo, f"Missing archetype in {cname}"
        assert "required_soft_skills" in cinfo, f"Missing required_soft_skills in {cname}"
        assert "weight_bonus" in cinfo, f"Missing weight_bonus in {cname}"
        assert "role_dynamic" in cinfo, f"Missing role_dynamic in {cname}"
    assert set(RULE_BASE["career_rules"]) == set(CAREER_UNIVERSE), "rules.json must define rules for every career"
    print("  ✓ Knowledge Base integrity & 5-Archetype mappings passed.")

    print("✦ [2/11] Creating test student profile with 4-pillar soft skills...")
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

    print("✦ [3/11] Testing Rule-Based Engine & Soft Skill Bonus Trace...")
    rule_score, rules_fired = evaluate_rule_engine(profile, "AI Engineer")
    assert 0.0 <= rule_score <= 1.0, f"Invalid rule score: {rule_score}"
    assert len(rules_fired) > 0, "Expected rules to fire for AI Engineer"
    soft_rule_fired = any("R-SOFT-SKILLS" in r for r in rules_fired)
    assert soft_rule_fired, "Expected soft skills rule R-SOFT-SKILLS to fire"
    print(f"  ✓ Rule score: {rule_score:.2f}, Rules fired count: {len(rules_fired)}")

    print("✦ [4/11] Testing Fuzzy Logic Mamdani Suitability Engine with Soft Skill Clusters...")
    fuzzy_score = evaluate_fuzzy_suitability(profile, "AI Engineer")
    assert 0.0 <= fuzzy_score <= 1.0, f"Invalid fuzzy score: {fuzzy_score}"
    perfect = StudentProfile(4.0, profile.year, profile.core_modules, profile.electives, profile.tech_skills,
                             CAREER_UNIVERSE["AI Engineer"]["required_soft_skills"], [], profile.work_style, True, 4, "")
    assert evaluate_fuzzy_suitability(perfect, "AI Engineer") >= fuzzy_score, "A perfect GPA/soft-skill match must not score lower"
    print(f"  ✓ Fuzzy suitability score: {fuzzy_score:.2f}")

    print("✦ [5/11] Testing ML Classifier & Hybrid Multi-Engine Fusion...")
    ml_probs = evaluate_ml_model(profile)
    assert len(ml_probs) == 9, f"Expected 9 probabilities, got {len(ml_probs)}"

    results = run_hybrid_inference(profile)
    assert len(results) == 9, "Expected 9 ranked career results"
    top_career = results[0]
    print(f"  ✓ Top recommendation: {top_career['title']} ({top_career['match_pct']}%) [{top_career['archetype']}] with {top_career['confidence']}")

    print("✦ [6/11] Testing SHAP XAI, Skill Gap & Cosine Certification Matching...")
    shap_df = calculate_shap_contributions(profile, top_career["career"])
    assert not shap_df.empty, "SHAP dataframe should not be empty"
    assert shap_df.attrs["method"] == "shap", "Expected real SHAP TreeExplainer values, got the proxy fallback"
    raw_prob = ml_probs[top_career["career"]]
    assert abs(shap_df.attrs["prediction"] - raw_prob) < 1e-4, "SHAP base + contributions must equal the model probability"

    gaps = calculate_skill_gaps(profile, top_career["career"])
    certs = recommend_certifications(gaps)
    assert len(certs) == 3, f"Expected 3 recommended certs, got {len(certs)}"

    narrative = generate_executive_narrative(profile, top_career, gaps)
    assert len(narrative) > 20, "Narrative should not be empty"
    print(f"  ✓ XAI deltas: {len(shap_df)}, Skill gaps: {len(gaps)}, Recommended certs: {len(certs)}")
    print(f"  ✓ Narrative preview: {narrative[:80]}...")

    print("✦ [7/11] Testing Decision Tree baseline vs Random Forest...")
    df = generate_benchmark_dataset(samples_per_class=20)
    dt = train_custom_classifier(df, max_depth=8, model_type="decision_tree")
    assert "error" not in dt and type(dt["model"]).__name__ == "DecisionTreeClassifier", dt.get("error")
    comparison = compare_classifiers(df, n_estimators=30, max_depth=8)
    assert list(comparison["Model"]) == ["Decision Tree", "Random Forest"]
    print("  ✓ " + " | ".join(f"{r.Model}: test {r._3:.1%}, CV {r._4:.1%}" for r in comparison.itertuples()))

    print("✦ [8/11] Testing SQLite Local Database Storage...")
    # Use a throwaway database so test runs never add rows to the real career_records.db
    real_db = database.DB_FILE
    tmp_dir = tempfile.TemporaryDirectory()
    database.DB_FILE = os.path.join(tmp_dir.name, "test_records.db")
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
    assert len(df_records) == 1, f"Expected exactly the 1 test row, got {len(df_records)}"
    database.DB_FILE = real_db
    tmp_dir.cleanup()
    print(f"  ✓ Database record verified in a temporary database. Total rows: {len(df_records)}")

    print("✦ [9/11] Testing Learning Pathway & Narration Guardrails...")
    certs = recommend_certifications(gaps, target_career=top_career["career"], student_year=profile.year)
    # Simulate a machine with no Gemini key, whatever secrets.toml or the environment contains
    real_setting = guidance_module.get_app_setting
    guidance_module.get_app_setting = lambda name, default="": default
    guidance = generate_guidance(profile, top_career, gaps, certs)
    guidance_module.get_app_setting = real_setting
    assert guidance["source"] == "template" and "not configured" in guidance["note"]
    pathway = guidance["pathway"]
    assert pathway and [s["id"] for s in pathway] == [f"S{i + 1}" for i in range(len(pathway))]
    assert any(s["kind"] == "Project" for s in pathway), "Pathway should include a portfolio project"
    recommended = {c["title"] for c in certs}
    for s in pathway:
        if s["kind"] == "Certification":
            assert any(t in s["title"] for t in recommended), f"Pathway introduced an unrecommended cert: {s['title']}"

    weak = dict(top_career, final_score=0.30, match_pct=30.0)
    weak_text = generate_executive_narrative(profile, weak, gaps)
    assert "exceptional" not in weak_text and "top tier" not in weak_text and "emerging" in weak_text
    assert "Math_Stats" not in generate_executive_narrative(profile, top_career, gaps), "Narrative must use readable skill names"

    def narration(summary, steps=None):
        return lambda facts: {"summary": summary, "steps": steps if steps is not None else [{"id": s["id"], "text": s["detail"]} for s in facts["pathway"]]}

    ok = generate_guidance(profile, top_career, gaps, certs, narrator=narration(f"You are a {top_career['match_pct']}% match for {top_career['title']}."))
    assert ok["source"] == "gemini", ok["note"]
    unrecommended = next(c["title"] for c in CERTIFICATION_CATALOG if c["title"] not in recommended)
    rejections = {
        "invented number": narration("You are a 99.9% match."),
        "unrecommended cert": narration(f"Consider the {unrecommended}."),
        "invented provider": narration("Try an Oracle course." if "Oracle" not in str(certs) else "Try a Salesforce course."),
        "dropped step": narration("Good fit.", steps=[]),
    }
    for name, fake in rejections.items():
        res = generate_guidance(profile, top_career, gaps, certs, narrator=fake)
        assert res["source"] == "template" and "rejected" in res["note"], f"{name} was not rejected: {res['note']}"
    injected = generate_guidance(profile, top_career, gaps, certs, narrator=narration("<script>x</script> Good fit."))
    assert "<script>" not in injected["narrative"], "LLM output must be HTML-escaped"

    def failing(facts):
        raise TimeoutError("simulated timeout")
    failed = generate_guidance(profile, top_career, gaps, certs, narrator=failing)
    assert failed["source"] == "template" and "failed" in failed["note"]
    print(f"  ✓ Pathway steps: {len(pathway)}; valid narration accepted; {len(rejections)} invalid narrations rejected; failures fall back to template")

    print("✦ [10/11] Testing Adaptive Progress Tracking & Held Certifications...")
    import sqlite3
    tmp_dir = tempfile.TemporaryDirectory()
    database.DB_FILE = os.path.join(tmp_dir.name, "progress_records.db")
    # A database in the original schema (no progress columns) must migrate without losing rows
    conn = sqlite3.connect(database.DB_FILE)
    conn.execute("CREATE TABLE records (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, gpa REAL NOT NULL, "
                 "academic_year TEXT NOT NULL, top_career TEXT NOT NULL, match_score REAL NOT NULL, confidence TEXT NOT NULL, "
                 "work_style TEXT NOT NULL, top_driver TEXT NOT NULL, critical_gap TEXT NOT NULL)")
    conn.execute("INSERT INTO records (timestamp, gpa, academic_year, top_career, match_score, confidence, work_style, top_driver, critical_gap) "
                 "VALUES ('2026-09-01 10:00:00', 3.1, '2nd Year', 'Data Scientist', 61.0, 'Moderate', 'Technical Specialist', 'SQL', 'None')")
    conn.commit()
    conn.close()
    init_database()
    assert len(fetch_all_records()) == 1, "Migration must keep existing rows"

    def assess(p, sid):
        res = run_hybrid_inference(p)
        g = calculate_skill_gaps(p, res[0]["career"])
        prev = fetch_student_history(sid)
        progress = compare_assessments(prev[-1], p, res, g) if prev else None
        save_student_record(p.gpa, p.year, res[0]["title"], res[0]["final_score"], res[0]["confidence"], p.work_style, "x", "y",
                            student_id=sid, career_scores={r["career"]: r["match_pct"] for r in res},
                            profile=profile_to_dict(p), gaps=[x["skill_name"] for x in g])
        return res, progress

    weaker = StudentProfile(3.2, "2nd Year", dict(profile.core_modules, Math_Stats=70), {}, dict(profile.tech_skills, ML_AI=2, Python=3),
                            profile.soft_skills[:2], [], profile.work_style, False, 2, "")
    _, first = assess(weaker, " d/bit/24/0088 ")
    assert first is None, "First assessment has nothing to compare with"
    stronger = StudentProfile(3.6, "3rd Year", profile.core_modules, profile.electives, profile.tech_skills,
                              profile.soft_skills, [], profile.work_style, True, 4, "AWS Cloud Practitioner")
    res2, progress = assess(stronger, "D/BIT/24/0088")
    history = fetch_student_history("d/bit/24/0088")
    assert len(history) == 2, "IDs must match regardless of case and spacing"
    assert any(s["skill"] == "Machine Learning & AI" and s["improved"] for s in progress["skill_changes"])
    assert "Completed an internship" in progress["events"]
    assert any("AWS Certified Cloud Practitioner" in e for e in progress["events"]), progress["events"]
    assert progress["insights"], "Progress should explain how the recommendation adapted"
    assert len(fetch_all_records("D/BIT/24/0088")) == 2 and len(fetch_all_records()) == 3

    assert parse_held_certifications("AWS Cloud Practitioner, CS50x") == ["aws-cloud-practitioner"]
    assert parse_held_certifications("AWS") == [], "Ambiguous entries matching several certs must be ignored"
    cloud_gaps = calculate_skill_gaps(stronger, "Cloud Architect")
    without = recommend_certifications(cloud_gaps, "Cloud Architect", "1st Year")
    held = recommend_certifications(cloud_gaps, "Cloud Architect", "1st Year", held_cert_ids=("aws-cloud-practitioner",))
    assert "aws-cloud-practitioner" in [c["id"] for c in without], "Fixture should recommend the cert when it is not held"
    assert "aws-cloud-practitioner" not in [c["id"] for c in held], "Held certifications must not be recommended again"
    database.DB_FILE = real_db
    tmp_dir.cleanup()
    print(f"  ✓ Old schema migrated; 2 assessments tracked for one ID; {len(progress['skill_changes'])} skill changes, "
          f"{len(progress['events'])} profile events; held certs excluded from recommendations")

    print("✦ [11/11] Testing Saved Model Artifacts...")
    import json
    import joblib
    import numpy as np
    from src.ml_engine import get_default_calibrated_ml_classifier, preprocess_features
    out_dir = tempfile.TemporaryDirectory()
    metrics = train_and_save(models_dir=out_dir.name, data_dir=out_dir.name)
    saved_rf = joblib.load(os.path.join(out_dir.name, "random_forest.joblib"))
    saved_dt = joblib.load(os.path.join(out_dir.name, "decision_tree.joblib"))
    app_rf, _ = get_default_calibrated_ml_classifier()
    X, _, _ = preprocess_features(generate_benchmark_dataset(samples_per_class=10))
    assert np.array_equal(saved_rf.predict_proba(X), app_rf.predict_proba(X)), "Saved RF must be the exact model the app trains"
    assert type(saved_dt).__name__ == "DecisionTreeClassifier"
    with open(os.path.join(out_dir.name, "metrics.json"), encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk["models"]["random_forest"]["test_accuracy"] == metrics["models"]["random_forest"]["test_accuracy"]
    with open(os.path.join(out_dir.name, "benchmark_dataset.csv"), encoding="utf-8") as f:
        assert on_disk["dataset"]["rows"] == len(f.readlines()) - 1, "metrics.json row count must match the CSV"
    out_dir.cleanup()
    print(f"  ✓ Saved RF identical to the app's model; RF test {metrics['models']['random_forest']['test_accuracy']:.1%}, "
          f"DT test {metrics['models']['decision_tree']['test_accuracy']:.1%}; metrics.json and dataset CSV consistent")

    print("\n🎉 ALL 11/11 TEST MODULES PASSED PERFECTLY!\n")


if __name__ == "__main__":
    run_tests()
