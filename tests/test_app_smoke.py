"""
UI smoke test for PathFinder AI: runs app.py headlessly with Streamlit's AppTest and drives the main workflow.
Catches errors the engine tests cannot see, such as a widget or chart crashing on render.
"""

import os
import sys
import tempfile

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Throwaway database, and no Gemini key, so the run is offline and never touches data/career_records.db
TMP = tempfile.TemporaryDirectory()
os.environ["CAREER_DB_FILE"] = os.path.join(TMP.name, "smoke.db")
os.environ.pop("GEMINI_API_KEY", None)

from streamlit.testing.v1 import AppTest  # noqa: E402

import src.guidance  # noqa: E402

# Ignore any local secrets.toml so the smoke test never calls the Gemini API (the app shares this process)
src.guidance.get_app_setting = lambda name, default="": default


def fail_on_exception(at: AppTest, step: str) -> None:
    if at.exception:
        raise AssertionError(f"{step}: app raised {[e.value for e in at.exception]}")


def run_smoke_test() -> None:
    os.chdir(ROOT)
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=120)

    print("✦ [1/4] Rendering the app on first load...")
    at.run()
    fail_on_exception(at, "first load")
    assert len(at.tabs) >= 4, "Expected the four main tabs"
    print(f"  ✓ Rendered {len(at.tabs)} tabs without errors")

    print("✦ [2/4] Running an assessment with a student ID...")
    at.text_input(key="student_id_input").set_value("ci-smoke-student")
    run_button = next(b for b in at.button if "Run Multi-Engine Inference" in b.label)
    run_button.click()
    at.run()
    fail_on_exception(at, "first assessment")
    assert not any("pf-loading-overlay" in m.value for m in at.markdown), "Loading overlay was left on screen"
    assert at.session_state["main_tabs"] == "02 // Guidance & Roadmap", "Finished assessment should open the results tab"
    data = at.session_state["evaluation_data"]
    assert data["student_id"] == "CI-SMOKE-STUDENT" and data["progress"] is None
    print(f"  ✓ Top recommendation: {data['top_rec']['title']} ({data['top_rec']['match_pct']}%)")

    print("✦ [3/4] Re-running to trigger progress tracking (renders the progress chart in two tabs)...")
    at.text_input(key="history_filter").set_value("ci-smoke-student")
    next(b for b in at.button if "Run Multi-Engine Inference" in b.label).click()
    at.run()
    fail_on_exception(at, "second assessment")
    data = at.session_state["evaluation_data"]
    assert data["progress"] is not None and len(data["history"]) == 2, "Second run should compare with the first"
    assert data["guidance"]["source"] == "template", "Smoke test must run offline with template narration"
    print(f"  ✓ Progress computed across {len(data['history'])} assessments; guidance source: {data['guidance']['source']}")

    print("✦ [4/4] Training the Decision Tree and comparing models in the studio...")
    classifier = next(r for r in at.radio if r.label == "Classifier")
    classifier.set_value("decision_tree")
    next(b for b in at.button if "Train Model on Active Dataset" in b.label).click()
    at.run()
    fail_on_exception(at, "decision tree training")
    assert at.session_state["active_ml_type"] == "decision_tree"
    next(b for b in at.button if "Compare Decision Tree vs Random Forest" in b.label).click()
    at.run()
    fail_on_exception(at, "model comparison")
    comparison = at.session_state["model_comparison"]
    print("  ✓ " + " | ".join(f"{r.Model}: test {r._3:.1%}" for r in comparison.itertuples()))

    print("\n🎉 UI SMOKE TEST PASSED (4/4)\n")


if __name__ == "__main__":
    run_smoke_test()
