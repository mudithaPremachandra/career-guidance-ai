# PathFinder AI: Undergraduate AI Career Guidance System

A Streamlit application featuring hybrid multi-engine career matching for undergraduate students.

Live app: https://eai-career-guidance-ai.streamlit.app/

## 🚀 Features
- **4-Pillar Soft Skills & Work Strengths Assessment**: People, Ideas, Data, Execution.
- **9 Careers across 5 Archetypes**:
  - Engineering & Architecture: Software Engineer, AI Engineer, CAD-CAM Engineer
  - Data & Analytics: Data Scientist
  - Infrastructure, Systems & Security: Cloud Architect, Cybersecurity Specialist
  - Creative, UI/UX & Media: UX Designer, Game Developer
  - Management & Consulting: IT Business Analyst
- **Hybrid Multi-Engine Score Fusion**:
  - 30% Rule-Based Expert System over the `rules.json` knowledge base (with rule-firing trace)
  - 30% Fuzzy Logic Suitability Engine (scikit-fuzzy Mamdani inference, centroid defuzzification)
  - 40% Supervised Machine Learning Classifier (Random Forest, or a Decision Tree baseline)
- **Explainable AI (XAI)**: per-student SHAP TreeExplainer attributions for the ML engine.
- **Skill Gap Analysis**: gaps ranked by size relative to each skill's scale, with urgency levels.
- **Personalized Learning Pathway**: a phased plan built from the skill gaps, the recommended certifications, soft-skill activities and a career-specific portfolio project.
- **Guidance Narration**: template-based text generation by default. If a Gemini API key is configured, Gemini rewrites the same facts more naturally. Its output is checked against the facts (no new numbers, certifications or providers), and the app falls back to the template if a check fails.
- **Cosine-Similarity Industry Certifications**: recommendations from a curated catalogue of 19 certifications (`certifications.json`).
- **Dataset & ML Model Studio**: benchmark dataset generation, custom CSV upload, model retraining, Decision Tree vs Random Forest comparison, feature importances, and batch prediction.
- **SQLite History & Analytics**: built-in persistence (`career_records.db`) with an advisor history view.
- **Adaptive Progress Tracking**: enter an optional student ID and each assessment is saved against it. Re-running shows how the recommendation adapted: match changes per career, skills that improved, gaps closed, and a trend chart. Certifications the student already holds are recognised and no longer recommended. The History tab can be filtered by student ID.

## 📁 File Structure
```
career-guidance-ai/
│
├── app.py                   # Main Streamlit application
├── rules.json               # IF-THEN rule base for the rule engine
├── certifications.json      # Curated industry certification catalogue
├── career_records.db        # SQLite database for persistent logs
├── test_pipeline.py         # Verification and unit test script
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── .streamlit/
    └── config.toml          # Theme and server settings
```

## 🛠️ How to Run

1. Open a terminal in the project folder.

2. (Optional) Create and activate a virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\Activate
```

3. Install dependencies:
```powershell
pip install -r requirements.txt
```

4. Launch the application:
```powershell
streamlit run app.py
```

5. Run the tests (they use a temporary database, not `career_records.db`):
```powershell
python test_pipeline.py
```

## ✨ Optional: Gemini narration

The app works without this. To turn it on, create `.streamlit/secrets.toml` (it is git-ignored, so never commit your key):
```toml
GEMINI_API_KEY = "your-key-from-aistudio.google.com"
# Optional, defaults to gemini-3.5-flash-lite
GEMINI_MODEL = "gemini-3.5-flash-lite"
```
On Streamlit Community Cloud, paste the same lines into the app's **Settings → Secrets** instead. The page shows which narration was used under the AI Advisory summary.
