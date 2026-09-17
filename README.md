# PathFinder AI: Undergraduate AI Career Guidance System

A complete, high-tech cyber/AI Streamlit application featuring hybrid multi-engine career matching for undergraduate students.

## 🚀 Features
- **4-Pillar Soft Skills & Work Strengths Assessment**: People, Ideas, Data, Execution.
- **6-Archetype Career Requirement Mapping**: Software Engineer, Data Scientist / AI Engineer, Cybersecurity Analyst, Cloud Solutions Architect, Product / Project Manager, UI/UX & Frontend Specialist.
- **Hybrid Multi-Engine Score Fusion**:
  - 30% Rule-Based Expert System (with rule trace firing)
  - 30% Fuzzy Logic Suitability Engine (Mamdani inference & defuzzification)
  - 40% Supervised Machine Learning Classifier (Random Forest proxy)
- **Explainable AI (XAI)**: SHAP-style divergent feature attribution breakdown.
- **Skill Gap Analysis**: Urgent vs Moderate gaps with targeted mitigation steps.
- **Cosine-Similarity Industry Certifications**: Live recommendations from a curated database (certifications.json).
- **Dataset & ML Model Studio**: Benchmark dataset generation, custom CSV upload, on-the-fly model retraining, feature importances, and batch prediction.
- **SQLite History & Analytics**: Built-in persistence (career_records.db) with analytics dashboard.

## 📁 File Structure
`
D:\PathFinder_AI\
│
├── app.py                   # Main Streamlit application
├── certifications.json      # Curated industry certification database
├── career_records.db        # SQLite database for persistent logs
├── test_pipeline.py         # Verification and unit test script
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── .streamlit/
    └── config.toml          # Custom theme and server settings
`

## 🛠️ How to Run

1. Open PowerShell / Command Prompt and navigate to the project folder:
`powershell
cd D:\PathFinder_AI
`

2. (Optional) Create and activate a virtual environment:
`powershell
python -m venv venv
.\venv\Scripts\Activate
`

3. Install dependencies:
`powershell
pip install -r requirements.txt
`

4. Launch the application:
`powershell
streamlit run app.py
`
