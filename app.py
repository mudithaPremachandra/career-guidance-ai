"""
PathFinder AI: Technology-Themed Undergraduate AI Career Guidance System
-----------------------------------------------------------------------
Streamlit UI entry point. The engines below live in the src/ package and the data files in data/:
1. 4-Pillar Categorized Soft Skills & Work Strengths (People, Ideas, Data, Execution)
2. 5-Archetype Job-to-Soft-Skill Requirement Mapping with Weight Bonus
3. Reactive Profile Intake with real-time dynamic Elective & Module Sliders
4. Rule-Based Reasoning over the rules.json knowledge base, with rule-firing trace
5. Fuzzy Logic Suitability Engine (scikit-fuzzy Mamdani inference & centroid defuzzification)
6. Supervised Machine Learning Classifier (Random Forest or Decision Tree baseline / Retrainable on Custom Datasets)
7. Multi-Engine Score Fusion (30% Rule + 30% Fuzzy + 40% ML)
8. Explainable AI (SHAP TreeExplainer per-student attribution)
9. Skill Gap Analysis with Urgency Prioritization (Technical & Soft Skills)
10. Cosine-Similarity Industry Certification Recommender
11. Dataset & Model Studio (Upload Custom CSV / Load Benchmark / Train Model / Feature Importances / Batch Predictions)
12. Local SQLite persistence for history logging
13. Personalized learning pathway + template NLG, with optional verified Gemini narration
14. Adaptive progress tracking per student ID (match changes, closed gaps, held certifications)

Visual Identity: AI + Career Intelligence + Education + Personal Growth
"""

import datetime
import html

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.database import clear_all_records, fetch_all_records, fetch_student_history, init_database, normalize_student_id, save_student_record
from src.explainability import calculate_shap_contributions
from src.fusion import run_hybrid_inference
from src.guidance import PATHWAY_PHASES, generate_guidance
from src.knowledge_base import ALL_24_SOFT_SKILLS, ALL_DATA_SKILLS, ALL_EXEC_SKILLS, ALL_IDEAS_SKILLS, ALL_PEOPLE_SKILLS, CAREER_UNIVERSE
from src.ml_engine import MODEL_LABELS, compare_classifiers, generate_benchmark_dataset, train_custom_classifier
from src.profile import StudentProfile
from src.progress import build_progress_chart, cert_titles, compare_assessments, parse_held_certifications, profile_to_dict
from src.recommender import recommend_certifications
from src.skill_gaps import calculate_skill_gaps

# -----------------------------------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION & MODERN RESPONSIVE DESIGN SYSTEM
# -----------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="PathFinder AI — Intelligent Career Guidance Platform",
    page_icon="⚡",
    initial_sidebar_state="collapsed",
)

# Custom Modern AI SaaS / Glassmorphism Dark Theme
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700;800&display=swap');

    /* Global Dark Reset & Background */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #070B14 !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(0, 229, 255, 0.07) 0px, transparent 45%),
            radial-gradient(at 100% 0%, rgba(99, 102, 241, 0.08) 0px, transparent 45%),
            radial-gradient(at 50% 100%, rgba(14, 165, 233, 0.06) 0px, transparent 50%),
            linear-gradient(rgba(255, 255, 255, 0.015) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.015) 1px, transparent 1px) !important;
        background-size: 100% 100%, 100% 100%, 100% 100%, 48px 48px, 48px 48px !important;
        color: #F8FAFC !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Responsive Full-Width Container (min 94%, max 1500px) */
    .block-container {
        padding-top: clamp(1.5rem, 2vw, 2.5rem) !important;
        padding-bottom: clamp(3rem, 4vw, 5rem) !important;
        padding-left: clamp(1rem, 2.5vw, 3rem) !important;
        padding-right: clamp(1rem, 2.5vw, 3rem) !important;
        max-width: min(94%, 1500px) !important;
        margin: 0 auto !important;
    }

    /* Top Platform Header */
    .pf-hero-header {
        margin-bottom: clamp(1.4rem, 2.2vw, 2.2rem);
        padding: clamp(1.6rem, 2.4vw, 2.5rem);
        background: linear-gradient(135deg, rgba(16, 26, 48, 0.85) 0%, rgba(11, 18, 36, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.22);
        border-radius: 20px;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
        position: relative;
        overflow: hidden;
    }
    .pf-hero-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #00E5FF 0%, #6366F1 50%, #38BDF8 100%);
        box-shadow: 0 0 16px rgba(0, 229, 255, 0.6);
    }
    .pf-hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: clamp(0.72rem, 0.8vw, 0.82rem);
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #00E5FF;
        background: rgba(0, 229, 255, 0.1);
        border: 1px solid rgba(0, 229, 255, 0.35);
        box-shadow: 0 0 12px rgba(0, 229, 255, 0.15);
        padding: 0.3rem 0.9rem;
        border-radius: 9999px;
        margin-bottom: 0.75rem;
    }
    .pf-hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2rem, 3vw, 3.2rem);
        font-weight: 800;
        letter-spacing: -0.025em;
        background: linear-gradient(135deg, #FFFFFF 0%, #38BDF8 55%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.15;
    }
    .pf-hero-subtitle {
        font-size: clamp(0.95rem, 1.1vw, 1.2rem);
        color: #94A3B8;
        margin-top: 0.5rem;
        line-height: 1.5;
    }

    /* Stepper Navigation Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: clamp(0.5rem, 1vw, 1rem);
        background: rgba(14, 23, 44, 0.75);
        padding: 0.55rem 0.75rem;
        border-radius: 16px;
        border: 1px solid rgba(56, 189, 248, 0.2);
        backdrop-filter: blur(16px);
        margin-bottom: clamp(1.4rem, 2vw, 2rem);
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Space Grotesk', 'Inter', sans-serif;
        font-size: clamp(0.9rem, 1.05vw, 1.15rem);
        font-weight: 600;
        color: #94A3B8 !important;
        padding: clamp(0.6rem, 1vw, 0.85rem) clamp(1rem, 1.5vw, 1.6rem);
        border-radius: 12px;
        transition: all 0.22s ease-in-out;
        border: 1px solid transparent;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #38BDF8 !important;
        background: rgba(56, 189, 248, 0.08);
    }
    .stTabs [aria-selected="true"] {
        color: #00E5FF !important;
        background: linear-gradient(135deg, rgba(2, 132, 199, 0.3) 0%, rgba(99, 102, 241, 0.3) 100%) !important;
        border: 1px solid rgba(0, 229, 255, 0.5) !important;
        box-shadow: 0 0 18px rgba(0, 229, 255, 0.25) !important;
        font-weight: 700 !important;
    }

    /* Modern Glassmorphism Cards */
    .pf-card {
        background: linear-gradient(135deg, rgba(16, 26, 48, 0.8) 0%, rgba(12, 20, 38, 0.9) 100%);
        border: 1px solid rgba(56, 189, 248, 0.18);
        border-radius: 18px;
        padding: clamp(1.2rem, 1.8vw, 2.2rem);
        margin-bottom: clamp(1.2rem, 1.8vw, 1.8rem);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        transition: all 0.22s ease-in-out;
    }
    .pf-card:hover {
        border-color: rgba(56, 189, 248, 0.38);
        box-shadow: 0 14px 38px rgba(0, 229, 255, 0.08), inset 0 1px 0 rgba(56, 189, 248, 0.15);
    }
    .pf-card-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(1.15rem, 1.4vw, 1.6rem);
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.35rem;
        display: flex;
        align-items: center;
        gap: 0.65rem;
    }
    .pf-card-desc {
        font-size: clamp(0.85rem, 0.95vw, 1.05rem);
        color: #94A3B8;
        margin-bottom: 1.15rem;
        line-height: 1.5;
    }

    /* Metric & HUD Cards */
    .pf-stat-card {
        background: rgba(14, 23, 44, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: clamp(1rem, 1.4vw, 1.6rem);
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .pf-stat-card:hover {
        transform: translateY(-2px);
        border-color: #00E5FF;
        box-shadow: 0 8px 24px rgba(0, 229, 255, 0.1);
    }
    .pf-stat-label {
        font-size: clamp(0.75rem, 0.85vw, 0.9rem);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
        margin-bottom: 0.3rem;
    }
    .pf-stat-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(1.8rem, 2.5vw, 3rem);
        font-weight: 800;
        color: #00E5FF;
        letter-spacing: -0.02em;
    }
    .pf-stat-sub {
        font-size: clamp(0.75rem, 0.85vw, 0.95rem);
        color: #38BDF8;
        margin-top: 0.25rem;
    }

    /* Core Module Responsive Cards (Clean Score + Progress Bar) */
    .pf-module-card {
        background: rgba(14, 23, 44, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.18);
        border-radius: 16px;
        padding: clamp(1rem, 1.3vw, 1.4rem);
        margin-bottom: 0.85rem;
        transition: all 0.2s ease;
    }
    .pf-module-card:hover {
        border-color: rgba(0, 229, 255, 0.45);
        background: rgba(16, 26, 48, 0.85);
        box-shadow: 0 8px 20px rgba(0, 229, 255, 0.08);
    }
    .pf-module-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .pf-module-title-wrap {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .pf-module-icon {
        font-size: clamp(1.4rem, 1.8vw, 2rem);
    }
    .pf-module-title {
        font-weight: 700;
        font-size: clamp(0.92rem, 1.05vw, 1.2rem);
        color: #F8FAFC;
    }
    .pf-module-score-text {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(1.15rem, 1.4vw, 1.5rem);
        font-weight: 800;
        color: #00E5FF;
    }

    /* Modern Progress Bar Component */
    .pf-progress-track {
        width: 100%;
        height: 8px;
        background: rgba(30, 41, 59, 0.8);
        border-radius: 9999px;
        overflow: hidden;
        margin: 0.45rem 0 0.65rem 0;
    }
    .pf-progress-fill {
        height: 100%;
        border-radius: 9999px;
        background: linear-gradient(90deg, #0284C7, #00E5FF);
        transition: width 0.3s ease;
    }

    /* Persona Cards */
    .pf-persona-card {
        background: rgba(14, 23, 44, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: clamp(1.1rem, 1.5vw, 1.6rem);
        text-align: center;
        transition: all 0.22s ease-in-out;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .pf-persona-card:hover {
        border-color: #00E5FF;
        transform: translateY(-3px);
        box-shadow: 0 12px 28px rgba(0, 229, 255, 0.15);
    }
    .pf-persona-icon {
        font-size: clamp(2.2rem, 2.8vw, 3.2rem);
        margin-bottom: 0.5rem;
    }
    .pf-persona-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(1.05rem, 1.25vw, 1.4rem);
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.35rem;
    }
    .pf-persona-desc {
        font-size: clamp(0.8rem, 0.9vw, 0.95rem);
        color: #94A3B8;
        line-height: 1.45;
        margin-bottom: 0.9rem;
    }

    /* Snapshot Cards */
    .pf-snapshot-card {
        background: linear-gradient(135deg, rgba(16, 26, 48, 0.8) 0%, rgba(12, 20, 38, 0.9) 100%);
        border: 1px solid rgba(56, 189, 248, 0.22);
        border-radius: 16px;
        padding: clamp(1.1rem, 1.4vw, 1.6rem);
        margin-bottom: 0.6rem;
    }
    .pf-snapshot-title {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: clamp(0.92rem, 1.05vw, 1.15rem);
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.4rem;
    }

    /* Primary CTA Section */
    .pf-cta-card {
        background: linear-gradient(135deg, rgba(16, 30, 58, 0.9) 0%, rgba(11, 20, 42, 0.95) 100%);
        border: 1px solid #00E5FF;
        border-radius: 20px;
        padding: clamp(1.8rem, 2.5vw, 3rem);
        margin-top: clamp(1.5rem, 2.5vw, 2.8rem);
        box-shadow: 0 0 35px rgba(0, 229, 255, 0.2), inset 0 0 15px rgba(0, 229, 255, 0.05);
        text-align: center;
        position: relative;
    }
    .pf-cta-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(1.6rem, 2.4vw, 2.5rem);
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 0.45rem;
    }
    .pf-cta-sub {
        font-size: clamp(0.95rem, 1.1vw, 1.25rem);
        color: #94A3B8;
        margin-bottom: 1.4rem;
    }

    /* Hero Recommendation Card for Rank 1 */
    .pf-hero-card {
        background: linear-gradient(135deg, rgba(16, 30, 58, 0.9) 0%, rgba(11, 20, 42, 0.95) 100%);
        border: 1px solid #00E5FF;
        border-radius: 18px;
        padding: clamp(1.4rem, 2vw, 2.2rem);
        margin-bottom: 1.5rem;
        box-shadow: 0 0 30px rgba(0, 229, 255, 0.22), inset 0 0 15px rgba(0, 229, 255, 0.06);
        position: relative;
        overflow: hidden;
    }
    .pf-hero-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, #00E5FF 0%, #818CF8 100%);
        box-shadow: 0 0 14px #00E5FF;
    }

    /* Badges */
    .pf-badge {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: clamp(0.72rem, 0.8vw, 0.85rem);
        font-weight: 600;
        padding: 0.28rem 0.7rem;
        border-radius: 8px;
        letter-spacing: 0.03em;
    }
    .pf-badge-blue {
        background-color: rgba(14, 165, 233, 0.18);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.45);
        box-shadow: 0 0 8px rgba(56, 189, 248, 0.15);
    }
    .pf-badge-purple {
        background-color: rgba(168, 85, 247, 0.18);
        color: #C084FC;
        border: 1px solid rgba(192, 132, 252, 0.45);
        box-shadow: 0 0 8px rgba(192, 132, 252, 0.15);
    }
    .pf-badge-high {
        background-color: rgba(244, 63, 94, 0.18);
        color: #FB7185;
        border: 1px solid rgba(251, 113, 133, 0.45);
        box-shadow: 0 0 8px rgba(251, 113, 133, 0.15);
    }
    .pf-badge-med {
        background-color: rgba(245, 158, 11, 0.18);
        color: #FBBF24;
        border: 1px solid rgba(251, 191, 36, 0.45);
        box-shadow: 0 0 8px rgba(251, 191, 36, 0.15);
    }
    .pf-badge-low {
        background-color: rgba(16, 185, 129, 0.18);
        color: #34D399;
        border: 1px solid rgba(52, 211, 153, 0.45);
        box-shadow: 0 0 8px rgba(52, 211, 153, 0.15);
    }
    .pf-badge-slate {
        background-color: rgba(51, 65, 85, 0.35);
        color: #CBD5E1;
        border: 1px solid rgba(100, 116, 139, 0.4);
    }

    /* Executive AI Callout Box */
    .pf-callout {
        background: linear-gradient(135deg, rgba(16, 26, 48, 0.9) 0%, rgba(14, 22, 40, 0.95) 100%);
        border-left: 4px solid #00E5FF;
        border-radius: 0 14px 14px 0;
        padding: clamp(1.2rem, 1.8vw, 1.8rem);
        margin: 1.4rem 0;
        font-size: clamp(0.92rem, 1.05vw, 1.15rem);
        line-height: 1.65;
        color: #E2E8F0;
        box-shadow: 0 4px 24px rgba(0, 229, 255, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }

    /* Primary Gradient Glow Buttons */
    div.stButton > button[data-testid="stBaseButton-primary"],
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0284C7 0%, #4F46E5 100%) !important;
        color: #FFFFFF !important;
        font-family: 'Space Grotesk', 'Inter', sans-serif !important;
        font-size: clamp(0.92rem, 1.05vw, 1.15rem) !important;
        font-weight: 700 !important;
        border: 1px solid rgba(56, 189, 248, 0.5) !important;
        border-radius: 12px !important;
        padding: clamp(0.65rem, 0.9vw, 0.95rem) clamp(1.2rem, 1.8vw, 2rem) !important;
        transition: all 0.22s ease !important;
        box-shadow: 0 0 18px rgba(2, 132, 199, 0.35) !important;
    }
    div.stButton > button[data-testid="stBaseButton-primary"]:hover,
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #00E5FF 0%, #6366F1 100%) !important;
        color: #080C15 !important;
        font-weight: 800 !important;
        border-color: #00E5FF !important;
        box-shadow: 0 0 28px rgba(0, 229, 255, 0.65) !important;
        transform: translateY(-2px) !important;
    }

    /* Secondary Glassmorphism Buttons & Stepper */
    div.stButton > button[data-testid="stBaseButton-secondary"],
    div.stButton > button[kind="secondary"],
    div.stButton > button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]) {
        background: rgba(14, 23, 44, 0.75) !important;
        color: #CBD5E1 !important;
        font-family: 'Space Grotesk', 'Inter', sans-serif !important;
        font-size: clamp(0.88rem, 1vw, 1.05rem) !important;
        font-weight: 600 !important;
        border: 1px solid rgba(56, 189, 248, 0.22) !important;
        border-radius: 12px !important;
        padding: clamp(0.6rem, 0.85vw, 0.85rem) clamp(1rem, 1.5vw, 1.6rem) !important;
        transition: all 0.22s ease !important;
        backdrop-filter: blur(12px) !important;
    }
    div.stButton > button[data-testid="stBaseButton-secondary"]:hover,
    div.stButton > button[kind="secondary"]:hover,
    div.stButton > button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]):hover {
        border-color: #00E5FF !important;
        color: #00E5FF !important;
        background: rgba(16, 26, 48, 0.95) !important;
        box-shadow: 0 0 16px rgba(0, 229, 255, 0.2) !important;
        transform: translateY(-2px) !important;
    }

    /* Sliders */
    div[data-testid="stSlider"] {
        padding: 0.2rem 0 0.5rem 0 !important;
    }

    /* Inputs Overrides */
    label[data-testid="stWidgetLabel"] p {
        color: #E2E8F0 !important;
        font-weight: 600 !important;
        font-size: clamp(0.88rem, 0.95vw, 1.05rem) !important;
    }
    .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: #0B1120 !important;
        border-color: rgba(56, 189, 248, 0.3) !important;
        color: #F8FAFC !important;
        border-radius: 12px !important;
    }
    .stSelectbox div[data-baseweb="select"] > div:hover,
    .stMultiSelect div[data-baseweb="select"] > div:hover {
        border-color: #00E5FF !important;
        box-shadow: 0 0 12px rgba(0, 229, 255, 0.2) !important;
    }

    /* Multiselect Tag */
    span[data-baseweb="tag"] {
        background-color: rgba(14, 165, 233, 0.25) !important;
        border: 1px solid rgba(56, 189, 248, 0.5) !important;
        color: #38BDF8 !important;
        border-radius: 8px !important;
    }

    /* Expander */
    div[data-testid="stExpander"] {
        background-color: rgba(14, 23, 44, 0.65) !important;
        border: 1px solid rgba(56, 189, 248, 0.22) !important;
        border-radius: 16px !important;
    }
    div[data-testid="stExpander"] summary {
        color: #E2E8F0 !important;
        font-weight: 600 !important;
        font-size: clamp(0.95rem, 1.05vw, 1.2rem) !important;
    }

    /* Custom Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #070B14;
    }
    ::-webkit-scrollbar-thumb {
        background: #1E293B;
        border-radius: 4px;
        border: 1px solid rgba(56, 189, 248, 0.2);
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #00E5FF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


init_database()


# -----------------------------------------------------------------------------
# 8. STREAMLIT APPLICATION UI
# -----------------------------------------------------------------------------
def apply_preset(preset_key: str) -> None:
    """Applies preset student archetype parameters and immediately calibrates roadmap."""
    for s in ALL_24_SOFT_SKILLS:
        st.session_state[f"soft_{s}"] = False

    if preset_key == "ai_engineer":
        gpa, year = 3.82, "3rd Year"
        core = {"DSA": 92, "OOP": 88, "DBMS": 82, "OS_Networks": 78, "SE_Principles": 85, "Math_Stats": 94}
        elecs = ["AI & Machine Learning", "Data Mining & Big Data", "Cloud Computing"]
        tech = {"Python": 5, "Java_CPP": 3, "SQL": 4, "WebStack": 3, "CloudDocker": 4, "ML_AI": 5, "MobileDev": 2, "Cybersecurity": 2}
        softs = ["Problem Solving & Logic", "Innovation & Prototyping", "Research & Investigation", "Quantitative & Mathematical", "Analytical & Critical Thinking"]
        style = "R&D / Creative"
        domains = ["Artificial Intelligence & ML", "Enterprise Cloud & DevOps"]
        intern = True
        projects = 4
        certs = "AWS Cloud Practitioner, TensorFlow Developer"

    elif preset_key == "cloud_architect":
        gpa, year = 3.55, "4th Year"
        core = {"DSA": 80, "OOP": 82, "DBMS": 88, "OS_Networks": 94, "SE_Principles": 86, "Math_Stats": 72}
        elecs = ["Cloud Computing", "Network Security & Cyber Defense"]
        tech = {"Python": 4, "Java_CPP": 3, "SQL": 4, "WebStack": 3, "CloudDocker": 5, "ML_AI": 2, "MobileDev": 2, "Cybersecurity": 4}
        softs = ["Hardware & System Troubleshooting", "Planning & Task Organizing", "Working Under Pressure / Incidents", "Problem Solving & Logic"]
        style = "Technical Specialist"
        domains = ["Enterprise Cloud & DevOps", "Information Security & Defense"]
        intern = True
        projects = 5
        certs = "AWS Certified Cloud Practitioner, CKA"

    elif preset_key == "fullstack_engineer":
        gpa, year = 3.60, "3rd Year"
        core = {"DSA": 88, "OOP": 90, "DBMS": 85, "OS_Networks": 78, "SE_Principles": 92, "Math_Stats": 78}
        elecs = ["Cloud Computing", "Human-Computer Interaction (UI/UX)"]
        tech = {"Python": 4, "Java_CPP": 4, "SQL": 4, "WebStack": 5, "CloudDocker": 3, "ML_AI": 2, "MobileDev": 3, "Cybersecurity": 2}
        softs = ["Problem Solving & Logic", "Debugging & Root-Cause Analysis", "Detail-Oriented & Precision", "Hands-on Prototyping"]
        style = "Technical Specialist"
        domains = ["Full-Stack Web Engineering", "Enterprise Cloud & DevOps"]
        intern = True
        projects = 4
        certs = "freeCodeCamp Full-Stack Path, Meta Front-End"

    else:  # business_analyst
        gpa, year = 3.65, "3rd Year"
        core = {"DSA": 70, "OOP": 72, "DBMS": 86, "OS_Networks": 68, "SE_Principles": 94, "Math_Stats": 80}
        elecs = ["IT Project Management", "Human-Computer Interaction (UI/UX)"]
        tech = {"Python": 3, "Java_CPP": 2, "SQL": 5, "WebStack": 2, "CloudDocker": 2, "ML_AI": 2, "MobileDev": 1, "Cybersecurity": 2}
        softs = ["Team Leadership & Delegation", "Negotiation & Persuasion", "Planning & Task Organizing", "Communicating & Articulating", "Active Listening & Empathy"]
        style = "Consulting / Management"
        domains = ["Fintech & Data Analytics", "Digital Product Design (UX/UI)"]
        intern = True
        projects = 3
        certs = "Google Data Analytics Professional Certificate"

    # Set form state variables
    st.session_state["input_gpa"] = gpa
    st.session_state["input_year"] = year
    st.session_state["slide_dsa"] = core["DSA"]
    st.session_state["slide_oop"] = core["OOP"]
    st.session_state["slide_dbms"] = core["DBMS"]
    st.session_state["slide_os"] = core["OS_Networks"]
    st.session_state["slide_se"] = core["SE_Principles"]
    st.session_state["slide_math"] = core["Math_Stats"]
    st.session_state["selected_electives_input"] = elecs
    st.session_state["sl_py"] = tech["Python"]
    st.session_state["sl_jcpp"] = tech["Java_CPP"]
    st.session_state["sl_sql"] = tech["SQL"]
    st.session_state["sl_web"] = tech["WebStack"]
    st.session_state["sl_cloud"] = tech["CloudDocker"]
    st.session_state["sl_ml"] = tech["ML_AI"]
    st.session_state["sl_mob"] = tech["MobileDev"]
    st.session_state["sl_sec"] = tech["Cybersecurity"]
    st.session_state["work_style_input"] = style
    st.session_state["domains_input"] = domains
    st.session_state["intern_toggle"] = intern
    st.session_state["projects_input"] = projects
    st.session_state["certs_input"] = certs

    for s in softs:
        st.session_state[f"soft_{s}"] = True

    # Pre-calculate guidance data for Tab 2
    elec_dict = {e: 85 for e in elecs}
    prof = StudentProfile(
        gpa=gpa,
        year=year,
        core_modules=core,
        electives=elec_dict,
        tech_skills=tech,
        soft_skills=softs,
        desired_domains=domains,
        work_style=style,
        has_internship=intern,
        projects_count=projects,
        existing_certs=certs,
    )
    res = run_hybrid_inference(prof)
    top = res[0]
    sh = calculate_shap_contributions(prof, top["career"])
    gp = calculate_skill_gaps(prof, top["career"])
    rc = recommend_certifications(gp, target_career=top["career"], student_year=year, held_cert_ids=tuple(parse_held_certifications(certs)))
    gd = generate_guidance(prof, top, gp, rc)
    st.session_state["evaluation_data"] = {
        "profile": prof,
        "results": res,
        "top_rec": top,
        "shap_df": sh,
        "gaps": gp,
        "certs": rc,
        "narrative": gd["narrative"],
        "guidance": gd,
    }


# Ensure baseline evaluation data is present on first load
if "evaluation_data" not in st.session_state:
    def_gpa, def_year = 3.65, "3rd Year"
    def_core = {"DSA": 88, "OOP": 85, "DBMS": 82, "OS_Networks": 78, "SE_Principles": 88, "Math_Stats": 80}
    def_elecs = {"AI & Machine Learning": 88, "Cloud Computing": 82}
    def_tech = {"Python": 4, "Java_CPP": 3, "SQL": 4, "WebStack": 2, "CloudDocker": 3, "ML_AI": 4, "MobileDev": 2, "Cybersecurity": 2}
    def_softs = ["Problem Solving & Logic", "Analytical & Critical Thinking", "Detail-Oriented & Precision", "Innovation & Prototyping"]
    def_domains = ["Full-Stack Web Engineering", "Enterprise Cloud & DevOps"]
    def_style = "Technical Specialist"
    def_prof = StudentProfile(
        gpa=def_gpa,
        year=def_year,
        core_modules=def_core,
        electives=def_elecs,
        tech_skills=def_tech,
        soft_skills=def_softs,
        desired_domains=def_domains,
        work_style=def_style,
        has_internship=True,
        projects_count=3,
        existing_certs="AWS Cloud Practitioner",
    )
    def_res = run_hybrid_inference(def_prof)
    def_top = def_res[0]
    def_sh = calculate_shap_contributions(def_prof, def_top["career"])
    def_gp = calculate_skill_gaps(def_prof, def_top["career"])
    def_rc = recommend_certifications(def_gp, target_career=def_top["career"], student_year=def_year, held_cert_ids=tuple(parse_held_certifications(def_prof.existing_certs)))
    def_gd = generate_guidance(def_prof, def_top, def_gp, def_rc)
    st.session_state["evaluation_data"] = {
        "profile": def_prof,
        "results": def_res,
        "top_rec": def_top,
        "shap_df": def_sh,
        "gaps": def_gp,
        "certs": def_rc,
        "narrative": def_gd["narrative"],
        "guidance": def_gd,
    }


# Profile form defaults, keyed by widget key. Widgets read their value from session state only (no value= argument),
# so presets and the keep-alive below never trigger Streamlit's "default value also set via Session State" warning.
DEFAULT_CHECKED_SOFTS = {"Problem Solving & Logic", "Analytical & Critical Thinking", "Detail-Oriented & Precision"}
PROFILE_DEFAULTS = {
    "student_id_input": "",
    "input_gpa": 3.40,
    "input_year": "3rd Year",
    "slide_dsa": 82, "slide_oop": 78, "slide_dbms": 75, "slide_os": 72, "slide_se": 80, "slide_math": 76,
    "selected_electives_input": ["AI & Machine Learning", "Cloud Computing"],
    "sl_py": 4, "sl_jcpp": 3, "sl_sql": 4, "sl_web": 3, "sl_cloud": 3, "sl_ml": 4, "sl_mob": 2, "sl_sec": 2,
    "work_style_input": "Technical Specialist",
    "domains_input": ["Artificial Intelligence & ML", "Enterprise Cloud & DevOps"],
    "intern_toggle": False,
    "projects_input": 3,
    "certs_input": "",
    "history_filter": "",
    **{f"soft_{s}": s in DEFAULT_CHECKED_SOFTS for s in ALL_24_SOFT_SKILLS},
}
for _key, _default in PROFILE_DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default
# Only the active page is drawn, and Streamlit deletes the state of widgets that are not drawn in a run.
# Re-assigning every form key each run keeps the profile intact while another page is open.
for _key in list(st.session_state.keys()):
    if _key in PROFILE_DEFAULTS or _key.startswith("elec_"):
        st.session_state[_key] = st.session_state[_key]


# -----------------------------------------------------------------------------
# 9. STREAMLIT APPLICATION VIEW & STEPPER NAVIGATION
# -----------------------------------------------------------------------------

# 1. Global Platform Heading (At the very top of the page)
st.markdown(
    """
    <div class="pf-hero-header" style="margin-bottom: 1.25rem;">
        <span class="pf-hero-badge">⚡ AI-POWERED CAREER INTELLIGENCE</span>
        <h1 class="pf-hero-title">PATHFINDER AI</h1>
        <p class="pf-hero-subtitle">Discover your strengths. Understand your options. Build your path.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

NAV_PROFILE, NAV_ROADMAP, NAV_STUDIO, NAV_HISTORY = "① Profile", "② Career Roadmap", "③ Model Studio", "④ History"
if "active_nav_tab" not in st.session_state:
    st.session_state["active_nav_tab"] = NAV_PROFILE


def go_to(page: str) -> None:
    """Button callback: switch the active page (runs before the next script run, so no st.rerun needed)."""
    st.session_state["active_nav_tab"] = page


def nav_button(label: str, page: str, key: str, primary: bool = False) -> None:
    st.button(label, key=key, type="primary" if primary else "secondary", use_container_width=True, on_click=go_to, args=(page,))


# 2. Top Stepper Navigation Header (Right below heading)
nav_steps = [
    (NAV_PROFILE, "Profile Intake & Strengths", "👤"),
    (NAV_ROADMAP, "AI Career Guidance & Roadmap", "🧭"),
    (NAV_STUDIO, "Dataset & Model Studio", "🔬"),
    (NAV_HISTORY, "Evaluation History & Logs", "📋"),
]
for idx, (nav_col, (tab_id, tab_label, tab_icon)) in enumerate(zip(st.columns(4), nav_steps)):
    with nav_col:
        nav_button(f"{tab_icon} {tab_id} • {tab_label}", tab_id, key=f"nav_step_btn_{idx}", primary=st.session_state["active_nav_tab"] == tab_id)

current_tab = st.session_state["active_nav_tab"]


def bottom_nav(buttons: list) -> None:
    """Row of (label, page, key, primary) buttons under each page."""
    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(56, 189, 248, 0.2); margin: 2rem 0 1.2rem 0;'>", unsafe_allow_html=True)
    for col, (label, page, key, primary) in zip(st.columns(len(buttons)), buttons):
        with col:
            nav_button(label, page, key, primary)


# -----------------------------------------------------------------------------
# TAB 1: ① PROFILE (CAREER PROFILE INTAKE & COMPETENCIES)
# -----------------------------------------------------------------------------
if current_tab == NAV_PROFILE:
    # Preset Profiles Quick Loader
    with st.expander("⚡ Quick Load Sample Student Profiles (Optional)", expanded=False):
        qcol1, qcol2, qcol3, qcol4 = st.columns(4)
        if qcol1.button("🤖 Load AI Engineer Profile", use_container_width=True):
            apply_preset("ai_engineer")
            st.rerun()

        if qcol2.button("☁️ Load Cloud Architect Profile", use_container_width=True):
            apply_preset("cloud_architect")
            st.rerun()

        if qcol3.button("💻 Load Full-Stack Engineer Profile", use_container_width=True):
            apply_preset("fullstack_engineer")
            st.rerun()

        if qcol4.button("📈 Load IT Business Analyst Profile", use_container_width=True):
            apply_preset("business_analyst")
            st.rerun()

    in_student_id = st.text_input(
        "🪪 Student ID (optional, enables progress tracking)",
        placeholder="e.g. your registration number or a nickname",
        help="Use the same ID each time to compare assessments over time. Use a pseudonymous ID rather than your full name.",
        key="student_id_input",
    )

    # 1. Section: Academic Foundation (Direct Sliders with Icons & Marks)
    st.markdown(
        """
        <div class="pf-card">
            <div class="pf-card-title">🎓 Your Academic Foundation</div>
            <div class="pf-card-desc">Enter your university cumulative GPA standing and compulsory computer science foundation module marks (0–100).</div>
        """,
        unsafe_allow_html=True,
    )

    col_gpa, col_year = st.columns(2)

    with col_gpa:
        in_gpa = st.slider(
            "🎓 Cumulative GPA (0.00 – 4.00 Scale)",
            min_value=0.00,
            max_value=4.00,
            step=0.01,
            key="input_gpa",
            help="Your overall university cumulative GPA (0.00 to 4.00).",
        )

    with col_year:
        in_year = st.selectbox(
            "📅 Academic Year Level",
            ["1st Year", "2nd Year", "3rd Year", "4th Year"],
            help="Current year of university study.",
            key="input_year",
        )

    st.markdown(
        "<p style='font-size: clamp(0.92rem, 1.05vw, 1.15rem); font-weight: 700; color: #38BDF8; margin-top: 1.25rem; margin-bottom: 0.75rem;'>Compulsory Core CS Module Marks (0 – 100):</p>",
        unsafe_allow_html=True,
    )

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        m_dsa = st.slider(
            "🧠 1. Data Structures & Algorithms",
            min_value=0,
            max_value=100,
            key="slide_dsa",
            help="Algorithmic complexity, trees, graphs, sorting, and dynamic programming.",
        )
        m_dbms = st.slider(
            "🗄️ 3. Database Management Systems",
            min_value=0,
            max_value=100,
            key="slide_dbms",
            help="Relational modeling, SQL optimization, indexing, ACID transactions.",
        )
        m_se = st.slider(
            "⚙️ 5. Software Engineering Principles",
            min_value=0,
            max_value=100,
            key="slide_se",
            help="Design patterns, modular architecture, agile lifecycle, CI/CD.",
        )

    with col_m2:
        m_oop = st.slider(
            "💻 2. Object-Oriented Programming",
            min_value=0,
            max_value=100,
            key="slide_oop",
            help="Encapsulation, inheritance, polymorphism, design SOLID principles.",
        )
        m_os = st.slider(
            "🌐 4. Operating Systems & Networks",
            min_value=0,
            max_value=100,
            key="slide_os",
            help="Process scheduling, memory paging, socket networking, TCP/IP protocols.",
        )
        m_math = st.slider(
            "📐 6. Mathematics & Statistics",
            min_value=0,
            max_value=100,
            key="slide_math",
            help="Discrete math, linear algebra, calculus, probability & inferential statistics.",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Section 2: Specialized University Electives
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="pf-card">
            <div class="pf-card-title">📚 Specialized University Electives</div>
            <div class="pf-card-desc">Select your completed or current university elective courses and provide your performance score (0–100) for each.</div>
        """,
        unsafe_allow_html=True,
    )

    all_electives_catalog = [
        "AI & Machine Learning",
        "Data Mining & Big Data",
        "Human-Computer Interaction (UI/UX)",
        "Computer Graphics & Animation",
        "Game Engine Development",
        "CAD/CAM Principles",
        "Network Security & Cyber Defense",
        "Cloud Computing",
        "IT Project Management",
        "Technical Writing",
    ]

    selected_electives = st.multiselect(
        "Choose University Elective Modules",
        all_electives_catalog,
        key="selected_electives_input",
        help="Select elective modules you have completed or are currently taking.",
    )

    electives_dict = {}
    if selected_electives:
        st.markdown(
            f"<p style='font-size: 0.88rem; font-weight: 700; color: #38BDF8; margin-top: 0.95rem; margin-bottom: 0.5rem;'>Elective Module Performance Scores ({len(selected_electives)} Active):</p>",
            unsafe_allow_html=True,
        )
        e_cols = st.columns(2 if len(selected_electives) > 1 else 1)
        for i, elec_name in enumerate(selected_electives):
            with e_cols[i % len(e_cols)]:
                slider_key = f"elec_{elec_name.replace(' ', '_').replace('&', 'and')}"
                if slider_key not in st.session_state:
                    st.session_state[slider_key] = 80
                elec_score = st.slider(
                    f"📖 {elec_name}",
                    min_value=0,
                    max_value=100,
                    key=slider_key,
                    help=f"Your performance/mark in {elec_name} (0-100 scale).",
                )
                electives_dict[elec_name] = elec_score
    else:
        st.info("💡 No elective courses selected. You can select electives above to refine your specialized career matching.")

    st.markdown("</div>", unsafe_allow_html=True)

    # 5. Section: Hands-on Technical Proficiencies
    st.markdown(
        """
        <div class="pf-card">
            <div class="pf-card-title">💻 Hands-on Technical Proficiencies</div>
            <div class="pf-card-desc">Rate your practical hands-on experience across major software engineering, data, and infrastructure toolchains.</div>
        """,
        unsafe_allow_html=True,
    )

    lvl_map = {1: "Novice", 2: "Basic", 3: "Intermediate", 4: "Advanced", 5: "Expert"}

    def tech_slider(label: str, key: str) -> int:
        st.markdown(f"**{label}**")
        level = st.select_slider(label, options=[1, 2, 3, 4, 5], key=key, label_visibility="collapsed")
        st.caption(f"Level: `{lvl_map[level]}`")
        return level

    tc1, tc2, tc3, tc4 = st.columns(4)
    with tc1:
        t_python = tech_slider("🐍 Python", "sl_py")
        t_jcpp = tech_slider("☕ Java / C++", "sl_jcpp")
    with tc2:
        t_sql = tech_slider("🗄️ SQL / Databases", "sl_sql")
        t_web = tech_slider("🌐 Web Stack (React/Node)", "sl_web")
    with tc3:
        t_cloud = tech_slider("☁️ Cloud / Docker", "sl_cloud")
        t_ml = tech_slider("🧠 ML / AI Frameworks", "sl_ml")
    with tc4:
        t_mob = tech_slider("📱 Mobile Dev", "sl_mob")
        t_sec = tech_slider("🛡️ Cybersecurity", "sl_sec")

    st.markdown("</div>", unsafe_allow_html=True)

    # 6. Section: 4-Pillar Categorized Soft Skills & Work Strengths
    st.markdown(
        """
        <div class="pf-card">
            <div class="pf-card-title">🤝 Soft Skills & Work Strengths</div>
            <div class="pf-card-desc">Select the attributes that best describe your natural work habits across the 4 professional pillars.</div>
        """,
        unsafe_allow_html=True,
    )

    selected_soft_skills = []
    soft_pillars = [
        ("👥 People & Leadership", "#38BDF8", ALL_PEOPLE_SKILLS),
        ("💡 Ideas & Innovation", "#C084FC", ALL_IDEAS_SKILLS),
        ("📊 Data & Systems", "#34D399", ALL_DATA_SKILLS),
        ("🛠️ Execution & Practical", "#FBBF24", ALL_EXEC_SKILLS),
    ]
    for pillar_col, (pillar_title, pillar_color, pillar_skills) in zip(st.columns(4), soft_pillars):
        with pillar_col:
            st.markdown(f"<p style='font-size: 0.88rem; font-weight: 700; color: {pillar_color}; margin-bottom: 0.5rem;'>{pillar_title}</p>", unsafe_allow_html=True)
            for skill in pillar_skills:
                if st.checkbox(skill, key=f"soft_{skill}"):
                    selected_soft_skills.append(skill)

    st.markdown(
        f"<div style='margin-top: 0.75rem; font-size: 0.9rem; color: #94A3B8;'>"
        f"Selected Attributes: <b style='color: #00E5FF;'>{len(selected_soft_skills)} active soft skills</b> across 4 pillars."
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # 5. Section: Advanced Profile Preferences & Practical Experience
    st.markdown(
        """
        <div class="pf-card">
            <div class="pf-card-title">⚙️ Profile Preferences & Practical Experience</div>
            <div class="pf-card-desc">Configure your preferred career work style, target industry domains, and hands-on portfolio experience.</div>
        """,
        unsafe_allow_html=True,
    )

    adv_col1, adv_col2 = st.columns(2)

    with adv_col1:
        in_work_style = st.radio(
            "💼 Work Style Preference",
            ["Technical Specialist", "Consulting / Management", "R&D / Creative"],
            horizontal=True,
            key="work_style_input",
            help="Your preferred day-to-day role dynamic and technical focus.",
        )
        in_domains = st.multiselect(
            "🌐 Desired Industry Domains",
            [
                "Artificial Intelligence & ML",
                "Enterprise Cloud & DevOps",
                "Full-Stack Web Engineering",
                "Information Security & Defense",
                "Interactive Gaming & Graphics",
                "Digital Product Design (UX/UI)",
                "Fintech & Data Analytics",
                "Automated Manufacturing & Robotics",
            ],
            key="domains_input",
            help="Industries and software engineering subfields you are most interested in.",
        )

    with adv_col2:
        in_internship = st.toggle(
            "💼 Completed University / Industry Internship",
            key="intern_toggle",
            help="Toggle if you have completed an internship or formal work experience.",
        )
        in_projects = st.number_input(
            "🚀 Completed Technical Projects",
            min_value=0,
            max_value=20,
            key="projects_input",
            help="Number of major academic or personal projects in your portfolio.",
        )
        in_certs = st.text_input(
            "📜 Existing Certifications (comma separated)",
            placeholder="e.g. AWS Cloud Practitioner, CS50x",
            key="certs_input",
            help="List any credentials you already hold.",
        )
        held_ids = parse_held_certifications(in_certs)
        if held_ids:
            st.caption("✓ Recognised (won't be recommended again): " + ", ".join(cert_titles(held_ids)))

    st.markdown("</div>", unsafe_allow_html=True)

    # 6. Profile Snapshot Summary Gauge Cards
    avg_core = (m_dsa + m_oop + m_dbms + m_os + m_se + m_math) / 6.0
    avg_tech = ((t_python + t_jcpp + t_sql + t_web + t_cloud + t_ml + t_mob + t_sec) / 40.0) * 100.0
    soft_pct = min(100, int((len(selected_soft_skills) / 6.0) * 100))
    snapshots = [
        ("💻 Technical Skills", f"{int(avg_tech)}%", int(avg_tech), "#0284C7", "#00E5FF", "8 Hands-on Proficiencies"),
        ("🎓 Academic Performance", f"{int(avg_core)}%", int(avg_core), "#059669", "#34D399", f"GPA {in_gpa:.2f} • 6 Core Modules"),
        ("🤝 Soft Skills", f"{len(selected_soft_skills)} Active", soft_pct, "#7C3AED", "#C084FC", "Across 4 Pillars"),
        ("🚀 Projects / Experience", f"{in_projects} Projects", min(100, in_projects * 20), "#D97706", "#FBBF24", "Self-reported portfolio builds"),
    ]

    st.markdown(
        "<p style='font-size: clamp(0.85rem, 0.95vw, 1.05rem); font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #38BDF8; margin-bottom: 0.75rem;'>📊 YOUR PROFILE SNAPSHOT</p>",
        unsafe_allow_html=True,
    )
    for snap_col, (snap_title, snap_value, snap_pct, grad_from, grad_to, snap_sub) in zip(st.columns(4), snapshots):
        with snap_col:
            st.markdown(
                f"""
                <div class="pf-snapshot-card">
                    <div class="pf-snapshot-title">
                        <span>{snap_title}</span>
                        <span style="color: {grad_to}; font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem;">{snap_value}</span>
                    </div>
                    <div class="pf-progress-track">
                        <div class="pf-progress-fill" style="width: {snap_pct}%; background: linear-gradient(90deg, {grad_from}, {grad_to});"></div>
                    </div>
                    <div style="font-size: 0.78rem; color: #94A3B8;">{snap_sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 7. Primary Call to Action Section
    st.markdown(
        f"""
        <div class="pf-cta-card">
            <div class="pf-cta-title">✨ READY TO DISCOVER YOUR PATH?</div>
            <div class="pf-cta-sub">Our AI engines will score your profile against <b style="color: #00E5FF;">{len(CAREER_UNIVERSE)} career paths</b> and build your personalised roadmap.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    btn_submit = st.button("🚀 Analyze My Career & Generate Roadmap", type="primary", use_container_width=True)

    # Process Assessment
    if btn_submit:
        core_mods = {
            "DSA": m_dsa,
            "OOP": m_oop,
            "DBMS": m_dbms,
            "OS_Networks": m_os,
            "SE_Principles": m_se,
            "Math_Stats": m_math,
        }
        tech_ratings = {
            "Python": t_python,
            "Java_CPP": t_jcpp,
            "SQL": t_sql,
            "WebStack": t_web,
            "CloudDocker": t_cloud,
            "ML_AI": t_ml,
            "MobileDev": t_mob,
            "Cybersecurity": t_sec,
        }

        profile = StudentProfile(
            gpa=in_gpa,
            year=in_year,
            core_modules=core_mods,
            electives=electives_dict,
            tech_skills=tech_ratings,
            soft_skills=selected_soft_skills,
            desired_domains=in_domains,
            work_style=in_work_style,
            has_internship=in_internship,
            projects_count=in_projects,
            existing_certs=in_certs,
        )

        # Run Multi-Engine Inference
        inference_results = run_hybrid_inference(profile)
        top_rec = inference_results[0]

        # Calculate SHAP & Gaps
        shap_df = calculate_shap_contributions(profile, top_rec["career"])
        gaps = calculate_skill_gaps(profile, top_rec["career"])
        rec_certs = recommend_certifications(gaps, target_career=top_rec["career"], student_year=in_year, held_cert_ids=tuple(parse_held_certifications(in_certs)))
        guidance = generate_guidance(profile, top_rec, gaps, rec_certs)

        top_driver = shap_df[shap_df["Delta"] > 0].iloc[-1]["Feature"] if not shap_df[shap_df["Delta"] > 0].empty else "Academic Foundation"
        critical_gap = gaps[0]["skill_name"] if gaps else "None (Target Met)"

        # Compare with this student's previous assessment before saving the new one
        student_id = normalize_student_id(in_student_id)
        previous_runs = fetch_student_history(student_id)
        progress = compare_assessments(previous_runs[-1], profile, inference_results, gaps) if previous_runs else None

        # Save to SQLite
        save_student_record(
            gpa=in_gpa,
            academic_year=in_year,
            top_career=top_rec["title"],
            match_score=top_rec["final_score"],
            confidence=top_rec["confidence"],
            work_style=in_work_style,
            top_driver=top_driver,
            critical_gap=critical_gap,
            student_id=student_id,
            career_scores={r["career"]: r["match_pct"] for r in inference_results},
            profile=profile_to_dict(profile),
            gaps=[g["skill_name"] for g in gaps],
        )

        # Store in Session State
        st.session_state["evaluation_data"] = {
            "profile": profile,
            "results": inference_results,
            "top_rec": top_rec,
            "shap_df": shap_df,
            "gaps": gaps,
            "certs": rec_certs,
            "narrative": guidance["narrative"],
            "guidance": guidance,
            "student_id": student_id,
            "progress": progress,
            "history": fetch_student_history(student_id),
        }

        st.session_state["active_nav_tab"] = NAV_ROADMAP
        st.toast("⚡ Multi-Engine Evaluation complete! Redirecting to Career Roadmap...", icon="🚀")
        st.rerun()

    # Bottom Navigation for Tab 1
    bottom_nav([
        ("🔬 Explore Model Studio", NAV_STUDIO, "p_bot_studio", False),
        ("📋 View Evaluation History", NAV_HISTORY, "p_bot_hist", False),
        ("Next: View Career Roadmap →", NAV_ROADMAP, "p_bot_next", True),
    ])


# -----------------------------------------------------------------------------
# TAB 2: ② CAREER ROADMAP (AI RECOMMENDATIONS & GUIDANCE)
# -----------------------------------------------------------------------------
elif current_tab == NAV_ROADMAP:
    if "evaluation_data" not in st.session_state:
        st.info("⚡ No evaluation results available yet. Please complete and submit the **① Profile** form to generate your personalized career roadmap.")
        nav_button("← Go to Profile Intake Form", NAV_PROFILE, "roadmap_goto_prof", primary=True)
    else:
        eval_data = st.session_state["evaluation_data"]
        top_rec = eval_data["top_rec"]
        results = eval_data["results"]
        shap_df = eval_data["shap_df"]
        gaps = eval_data["gaps"]
        certs = eval_data["certs"]
        narrative = eval_data["narrative"]
        guidance = eval_data.get("guidance")

        # Active Model indicator badge
        active_dataset_name = st.session_state.get("active_dataset_name", "Standard University Benchmark Dataset")
        st.markdown(
            f"<div style='margin-bottom: 1.2rem; display: flex; gap: 0.6rem; align-items: center; flex-wrap: wrap;'>"
            f"<span class='pf-badge pf-badge-slate'>Model Ground Truth: <b style='color: #F8FAFC;'>{active_dataset_name}</b></span>"
            f"<span class='pf-badge pf-badge-blue'>Primary Archetype: <b style='color: #00E5FF;'>{top_rec.get('archetype', 'Engineering & Architecture')}</b></span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Top 3 Recommendations
        st.markdown("<p style='font-size: clamp(0.85rem, 0.95vw, 1.05rem); font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #38BDF8; margin-bottom: 0.75rem;'>🧭 Top AI Career Recommendations</p>", unsafe_allow_html=True)
        top3 = results[:3]

        col_top1, col_top2, col_top3 = st.columns(3)

        # Rank 1 Hero Card
        with col_top1:
            st.markdown(
                f"""
                <div class="pf-hero-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="pf-badge pf-badge-blue">Primary Match • Rank 1</span>
                        <span class="pf-badge pf-badge-purple">{top3[0].get('archetype', 'Archetype')}</span>
                    </div>
                    <h3 style="font-size: clamp(1.2rem, 1.5vw, 1.6rem); font-weight: 700; margin: 0.75rem 0 0.35rem 0; color: #F8FAFC;">
                        {top3[0]['icon']} {top3[0]['title']}
                    </h3>
                    <div style="font-size: clamp(2rem, 2.8vw, 3.4rem); font-weight: 800; color: #00E5FF; letter-spacing: -0.03em; text-shadow: 0 0 15px rgba(0, 229, 255, 0.45);">
                        {top3[0]['match_pct']}%
                    </div>
                    <div style="font-size: clamp(0.82rem, 0.9vw, 1rem); color: #94A3B8; margin-top: 0.35rem;">
                        Confidence: <b style="color: {top3[0]['conf_color']};">{top3[0]['confidence']}</b>
                    </div>
                    <div style="font-size: clamp(0.85rem, 0.92vw, 1.02rem); color: #CBD5E1; margin-top: 0.65rem; line-height: 1.5;">
                        {top3[0]['description']}
                    </div>
                    <div style="background: rgba(14, 23, 44, 0.65); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 10px; padding: 0.85rem 1.1rem; margin-top: 0.95rem; font-size: clamp(0.82rem, 0.9vw, 0.98rem); color: #CBD5E1;">
                        <b style="color: #38BDF8;">Role Dynamic:</b> {top3[0].get('role_dynamic', '')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Rank 2 Card
        with col_top2:
            st.markdown(
                f"""
                <div class="pf-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="pf-badge pf-badge-purple">Alternative Path • Rank 2</span>
                        <span class="pf-badge pf-badge-slate">{top3[1].get('archetype', 'Archetype')}</span>
                    </div>
                    <h3 style="font-size: clamp(1.1rem, 1.3vw, 1.45rem); font-weight: 700; margin: 0.75rem 0 0.35rem 0; color: #F8FAFC;">
                        {top3[1]['icon']} {top3[1]['title']}
                    </h3>
                    <div style="font-size: clamp(1.8rem, 2.3vw, 2.8rem); font-weight: 800; color: #C084FC; letter-spacing: -0.03em; text-shadow: 0 0 15px rgba(192, 132, 252, 0.35);">
                        {top3[1]['match_pct']}%
                    </div>
                    <div style="font-size: clamp(0.82rem, 0.9vw, 1rem); color: #94A3B8; margin-top: 0.35rem;">
                        Confidence: <b style="color: {top3[1]['conf_color']};">{top3[1]['confidence']}</b>
                    </div>
                    <div style="font-size: clamp(0.85rem, 0.92vw, 1.02rem); color: #CBD5E1; margin-top: 0.65rem; line-height: 1.5;">
                        {top3[1]['description']}
                    </div>
                    <div style="background: rgba(14, 23, 44, 0.65); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 10px; padding: 0.85rem 1.1rem; margin-top: 0.95rem; font-size: clamp(0.82rem, 0.9vw, 0.98rem); color: #CBD5E1;">
                        <b style="color: #C084FC;">Role Dynamic:</b> {top3[1].get('role_dynamic', '')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Rank 3 Card
        with col_top3:
            st.markdown(
                f"""
                <div class="pf-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="pf-badge pf-badge-med">Emerging Fit • Rank 3</span>
                        <span class="pf-badge pf-badge-slate">{top3[2].get('archetype', 'Archetype')}</span>
                    </div>
                    <h3 style="font-size: clamp(1.1rem, 1.3vw, 1.45rem); font-weight: 700; margin: 0.75rem 0 0.35rem 0; color: #F8FAFC;">
                        {top3[2]['icon']} {top3[2]['title']}
                    </h3>
                    <div style="font-size: clamp(1.8rem, 2.3vw, 2.8rem); font-weight: 800; color: #FBBF24; letter-spacing: -0.03em; text-shadow: 0 0 15px rgba(251, 191, 36, 0.35);">
                        {top3[2]['match_pct']}%
                    </div>
                    <div style="font-size: clamp(0.82rem, 0.9vw, 1rem); color: #94A3B8; margin-top: 0.35rem;">
                        Confidence: <b style="color: {top3[2]['conf_color']};">{top3[2]['confidence']}</b>
                    </div>
                    <div style="font-size: clamp(0.85rem, 0.92vw, 1.02rem); color: #CBD5E1; margin-top: 0.65rem; line-height: 1.5;">
                        {top3[2]['description']}
                    </div>
                    <div style="background: rgba(14, 23, 44, 0.65); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 10px; padding: 0.85rem 1.1rem; margin-top: 0.95rem; font-size: clamp(0.82rem, 0.9vw, 0.98rem); color: #CBD5E1;">
                        <b style="color: #FBBF24;">Role Dynamic:</b> {top3[2].get('role_dynamic', '')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Executive AI Summary Callout Box
        st.markdown(
            f"""
            <div class="pf-callout">
                <div style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #00E5FF; margin-bottom: 0.45rem; font-size: 0.88rem; letter-spacing: 0.05em; text-transform: uppercase;">
                    ⚡ AI Advisory Executive Summary
                </div>
                {narrative}
            </div>
            """,
            unsafe_allow_html=True,
        )
        if guidance and guidance.get("note"):
            st.caption(("✨ " if guidance["source"] == "gemini" else "📝 ") + guidance["note"])

        # Progress since this student's previous assessment (how the recommendations adapted)
        if eval_data.get("student_id"):
            progress = eval_data.get("progress")
            history = eval_data.get("history") or []
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='pf-card-title'>📈 Progress Since Your Last Assessment</div>", unsafe_allow_html=True)
            if not progress:
                st.info(
                    f"First assessment saved for student ID {eval_data['student_id']}. Run it again after you improve a skill, "
                    "finish a certification or change your interests to see how your recommendations adapt."
                )
            else:
                st.markdown(
                    f"<div class='pf-card-desc'>Compared with your assessment on {html.escape(progress['previous_timestamp'])} "
                    f"({len(history)} assessments saved for this ID).</div>",
                    unsafe_allow_html=True,
                )
                pc1, pc2, pc3 = st.columns(3)
                before = progress["top_before"]
                pc1.metric(
                    f"{top_rec['title']} match",
                    f"{progress['top_after']}%",
                    delta=f"{progress['top_after'] - before:+.1f} pts" if before is not None and progress["top_after"] != before else None,
                )
                pc2.metric("Previous top recommendation", progress["previous_top"])
                pc3.metric("Gaps closed", len(progress["closed_gaps"]) if progress["same_top"] else "n/a",
                           help="Only comparable when the top recommendation is unchanged.")
                for line in progress["insights"]:
                    st.markdown(f"- {line}")

                pcol1, pcol2 = st.columns(2)
                with pcol1:
                    st.markdown("**What changed in your profile**")
                    if not progress["events"] and not progress["skill_changes"]:
                        st.caption("No changes since the last assessment.")
                    for e in progress["events"]:
                        st.markdown(f"- {e}")
                    for s in progress["skill_changes"]:
                        st.markdown(f"- {'⬆️' if s['improved'] else '⬇️'} {s['skill']}: {s['before']} → {s['after']}")
                with pcol2:
                    st.markdown("**Match % across your assessments**")
                    st.plotly_chart(
                        build_progress_chart(history, [r["career"] for r in results[:3]]),
                        use_container_width=True,
                        config={"displayModeBar": False},
                        key="progress_chart_results",
                    )

        st.markdown("<br>", unsafe_allow_html=True)

        # Visual Analytics
        st.markdown(
            "<p style='font-size: clamp(0.85rem, 0.95vw, 1.05rem); font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #38BDF8; margin-bottom: 0.75rem;'>📊 Alignment Analytics & Competency Profiling</p>",
            unsafe_allow_html=True,
        )
        v_tab1, v_tab2 = st.tabs(["⚡ Alignment & Explainability (SHAP)", "🕸️ Competency Benchmark Radar"])

        with v_tab1:
            col_v1, col_v2 = st.columns(2)

            with col_v1:
                st.markdown("<div class='pf-card-title'>⚡ Career Alignment Distribution</div>", unsafe_allow_html=True)
                st.markdown("<div class='pf-card-desc'>Multi-engine match percentages across all evaluated careers.</div>", unsafe_allow_html=True)

                plot_df = pd.DataFrame(results).sort_values(by="final_score", ascending=True)

                fig_bar = go.Figure()
                fig_bar.add_trace(
                    go.Bar(
                        x=plot_df["match_pct"],
                        y=plot_df["title"],
                        orientation="h",
                        marker=dict(
                            color=plot_df["match_pct"].apply(lambda v: "#00E5FF" if v == max(plot_df["match_pct"]) else "#334155"),
                            line=dict(color="#38BDF8", width=plot_df["match_pct"].apply(lambda v: 1.5 if v == max(plot_df["match_pct"]) else 0)),
                        ),
                        text=plot_df["match_pct"].apply(lambda x: f"{x:.1f}%"),
                        textposition="outside",
                        textfont=dict(size=12, family="Inter", color="#F8FAFC"),
                    )
                )

                fig_bar.update_layout(
                    xaxis=dict(range=[0, 115], showgrid=False, showticklabels=False, zeroline=False),
                    yaxis=dict(showgrid=False, tickfont=dict(size=12, family="Inter", color="#E2E8F0")),
                    margin=dict(l=10, r=20, t=10, b=10),
                    height=340,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(14, 23, 44, 0.5)",
                )
                st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

            with col_v2:
                st.markdown("<div class='pf-card-title'>🔮 Explainability: Feature Contributions (SHAP)</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='pf-card-desc'>Factors driving (+) or penalizing (-) the primary match (<b style='color: #00E5FF;'>{top_rec['title']}</b>).</div>", unsafe_allow_html=True)
                if shap_df.attrs.get("method") == "shap":
                    shap_caption = (
                        f"SHAP TreeExplainer on the {shap_df.attrs['model']} engine: bars are percentage-point contributions "
                        f"to its raw {top_rec['title']} probability ({shap_df.attrs['base_value']*100:.1f}% average "
                        f"→ {shap_df.attrs['prediction']*100:.1f}% for this student). Top 7 of 20 features shown."
                    )
                else:
                    shap_caption = "SHAP unavailable for the active model, showing benchmark-delta approximation instead."

                fig_shap = go.Figure()
                fig_shap.add_trace(
                    go.Bar(
                        x=shap_df["Delta"],
                        y=shap_df["Feature"],
                        orientation="h",
                        marker=dict(color=shap_df["Color"], line=dict(width=0)),
                        text=shap_df["Delta"].apply(lambda d: f"+{d:.1f}%" if d > 0 else f"{d:.1f}%"),
                        textposition="outside",
                        textfont=dict(size=12, family="Inter", color="#F8FAFC"),
                    )
                )

                min_val = min(shap_df["Delta"].min() - 5, -10)
                max_val = max(shap_df["Delta"].max() + 8, 15)

                fig_shap.update_layout(
                    xaxis=dict(range=[min_val, max_val], showgrid=True, gridcolor="rgba(51, 65, 85, 0.4)", zeroline=True, zerolinecolor="#64748B"),
                    yaxis=dict(showgrid=False, tickfont=dict(size=12, family="Inter", color="#E2E8F0")),
                    margin=dict(l=10, r=20, t=10, b=10),
                    height=340,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(14, 23, 44, 0.5)",
                )
                st.plotly_chart(fig_shap, use_container_width=True, config={"displayModeBar": False})
                st.caption(shap_caption)

        with v_tab2:
            st.markdown(f"<div class='pf-card-title'>🕸️ Competency Benchmark Radar: Student vs. {top_rec['title']}</div>", unsafe_allow_html=True)
            st.markdown("<div class='pf-card-desc'>Multi-dimensional comparison across Core Module and Technical Proficiency dimensions (0–100 scale).</div>", unsafe_allow_html=True)

            benchmarks = CAREER_UNIVERSE[top_rec["career"]]["benchmark_skills"]
            unified = eval_data["profile"].get_unified_skill_dict()

            radar_labels = []
            student_norm = []
            target_norm = []

            key_skills_to_show = [
                ("DSA", "DSA"), ("OOP", "OOP"), ("DBMS", "DBMS"), ("OS_Networks", "Networks"),
                ("SE_Principles", "SE Principles"), ("Math_Stats", "Math/Stats"), ("Python", "Python"),
                ("SQL", "SQL"), ("WebStack", "Web Stack"), ("CloudDocker", "Cloud/Docker"), ("ML_AI", "ML/AI"),
                ("Cybersecurity", "Cybersecurity")
            ]

            for skey, slabel in key_skills_to_show:
                radar_labels.append(slabel)
                curr = unified.get(skey, 0)
                req = benchmarks.get(skey, 50 if skey in ["DSA", "OOP", "DBMS", "OS_Networks", "SE_Principles", "Math_Stats"] else 2.5)

                s_val = (curr / 5.0 * 100.0) if skey not in ["DSA", "OOP", "DBMS", "OS_Networks", "SE_Principles", "Math_Stats"] else curr
                t_val = (req / 5.0 * 100.0) if skey not in ["DSA", "OOP", "DBMS", "OS_Networks", "SE_Principles", "Math_Stats"] else req

                student_norm.append(min(100.0, max(0.0, float(s_val))))
                target_norm.append(min(100.0, max(0.0, float(t_val))))

            radar_labels.append(radar_labels[0])
            student_norm.append(student_norm[0])
            target_norm.append(target_norm[0])

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=student_norm,
                theta=radar_labels,
                fill='toself',
                name='Student Competency',
                line=dict(color='#00E5FF', width=2.5),
                fillcolor='rgba(0, 229, 255, 0.25)'
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=target_norm,
                theta=radar_labels,
                fill='toself',
                name=f'{top_rec["title"]} Requisite',
                line=dict(color='#C084FC', width=2, dash='dash'),
                fillcolor='rgba(192, 132, 252, 0.15)'
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], showticklabels=True, tickfont=dict(size=10, color="#94A3B8"), gridcolor="rgba(51, 65, 85, 0.5)"),
                    angularaxis=dict(tickfont=dict(size=12, family="Inter", color="#E2E8F0"), gridcolor="rgba(51, 65, 85, 0.5)")
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(14, 23, 44, 0.5)",
                legend=dict(font=dict(color="#CBD5E1", family="Inter"), orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                margin=dict(l=40, r=40, t=20, b=40),
                height=400,
            )
            st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

        # Soft Skill Archetype Match Breakdown
        matched_s = top_rec.get("matched_soft_skills", [])
        missing_s = top_rec.get("missing_soft_skills", [])

        matched_badges = "".join([f"<span class='pf-badge pf-badge-blue'>✓ {s}</span>" for s in matched_s])
        missing_badges = "".join([f"<span class='pf-badge pf-badge-high'>⚠ Missing: {s}</span>" for s in missing_s])
        all_badges = matched_badges + missing_badges
        if not all_badges:
            all_badges = "<span style='color: #94A3B8; font-size: 0.9rem;'>No archetype-specific soft skill requirements listed.</span>"

        st.markdown(
            f"<div class='pf-card'>"
            f"<div class='pf-card-title'>🧩 Soft Skill Archetype Match Breakdown ({top_rec.get('archetype', '')})</div>"
            f"<div style='margin-top: 0.75rem; display: flex; flex-wrap: wrap; gap: 0.55rem;'>"
            f"{all_badges}"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Symbolic Rule Trace Accordion
        with st.expander("🔍 View Transparent Rule-Based Firing Trace"):
            st.markdown(f"**Rules evaluated and triggered for {top_rec['title']}:**")
            for r in top_rec["rules_fired"]:
                st.markdown(f"- `{r}`")
            active_model_label = MODEL_LABELS.get(st.session_state.get("active_ml_type", "random_forest"), "Random Forest")
            st.caption(f"Score Fusion: 30% Rule Logic (rules.json) + 30% Mamdani Fuzzy System (scikit-fuzzy) + 40% {active_model_label} Probability.")

        st.markdown("<hr style='border: 0; border-top: 1px solid rgba(56, 189, 248, 0.2); margin: 1.8rem 0;'>", unsafe_allow_html=True)

        # 3. Two-Column Split: Skill Gap Audit & Cosine-Matched Certifications
        col_gap, col_cert = st.columns(2)

        with col_gap:
            st.markdown("<div class='pf-card-title'>🎯 Competency & Skill Gap Analysis</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='pf-card-desc'>Identified technical & soft competency requirements for <b style='color: #00E5FF;'>{top_rec['title']}</b>.</div>", unsafe_allow_html=True)

            if gaps:
                for g in gaps:
                    st.markdown(
                        f"<div style='background: rgba(14, 23, 44, 0.75); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 12px; padding: 0.95rem 1.2rem; margin-bottom: 0.7rem; display: flex; justify-content: space-between; align-items: center;'>"
                        f"<div>"
                        f"<div style='font-size: clamp(0.9rem, 1vw, 1.1rem); font-weight: 600; color: #F8FAFC;'>{g['skill_name']}</div>"
                        f"<div style='font-size: clamp(0.78rem, 0.85vw, 0.92rem); color: #94A3B8; margin-top: 0.2rem;'>"
                        f"Current: <b style='color: #CBD5E1;'>{g['current']}</b> • Target Requisite: <b style='color: #38BDF8;'>{g['target']}</b>"
                        f"</div>"
                        f"</div>"
                        f"<span class='pf-badge {g['badge_class']}'>{g['urgency']}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.success("⚡ Outstanding! No significant skill gaps detected against the target career benchmark.")

        with col_cert:
            st.markdown("<div class='pf-card-title'>🏆 Recommended Industry Certifications</div>", unsafe_allow_html=True)
            st.markdown("<div class='pf-card-desc'>Matched using Gap-Weighted Cosine Similarity against identified deficiency requirements.</div>", unsafe_allow_html=True)

            for cert in certs:
                with st.container(border=True):
                    col_title, col_score = st.columns([3, 1])
                    with col_title:
                        st.markdown(f"**{cert['title']}**")
                        st.caption(f"Issuer: {cert['issuer']} • Level: {cert['level']}")
                    with col_score:
                        st.markdown(f"`{cert['coverage_pct']}% Gap Coverage`")

                    st.write(cert['description'])

                    tag_html = " ".join([f"<span style='background:#1E293B; color:#38BDF8; padding:4px 10px; border-radius:6px; font-size:12px; margin-right:6px; margin-bottom:6px; display:inline-block; border:1px solid rgba(56, 189, 248, 0.25);'>{s}</span>" for s in cert['skills']])
                    st.markdown(tag_html, unsafe_allow_html=True)

                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        st.link_button("🌐 Official Exam & Syllabus", cert['official_url'], use_container_width=True)
                    with btn_col2:
                        st.link_button("🎓 Prepare on Course Platform", cert['prep_url'], use_container_width=True)

        # 4. Personalized Learning Pathway (sequenced from gaps, recommended certs and career projects)
        if guidance and guidance.get("pathway"):
            st.markdown("<hr style='border: 0; border-top: 1px solid rgba(56, 189, 248, 0.2); margin: 1.8rem 0;'>", unsafe_allow_html=True)
            st.markdown("<div class='pf-card-title'>🗺️ Personalized Learning Pathway</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='pf-card-desc'>A sequenced plan towards <b style='color: #00E5FF;'>{top_rec['title']}</b>: "
                "urgent gaps first, then supporting skills, then portfolio and experience.</div>",
                unsafe_allow_html=True,
            )
            kind_badges = {
                "Course": "pf-badge-blue", "Certification": "pf-badge-purple", "Activity": "pf-badge-med",
                "Project": "pf-badge-low", "Experience": "pf-badge-slate",
            }
            phases = [p for p in PATHWAY_PHASES if any(s["phase"] == p for s in guidance["pathway"])]
            for phase, col in zip(phases, st.columns(len(PATHWAY_PHASES))):
                with col:
                    st.markdown(f"**{phase}**")
                    for s in [s for s in guidance["pathway"] if s["phase"] == phase]:
                        st.markdown(
                            f"""
                            <div style="background: rgba(14, 23, 44, 0.75); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 12px; padding: 0.85rem 1rem; margin-bottom: 0.7rem;">
                                <span class="pf-badge {kind_badges.get(s['kind'], 'pf-badge-slate')}">{html.escape(s['kind'])}</span>
                                <div style="font-size: 0.88rem; font-weight: 600; color: #F8FAFC; margin-top: 0.45rem;">{html.escape(s['title'])}</div>
                                <div style="font-size: 0.8rem; color: #CBD5E1; margin-top: 0.25rem; line-height: 1.45;">{html.escape(s['text'])}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        # Bottom Navigation for Tab 2
        bottom_nav([
            ("← Back to Profile Intake", NAV_PROFILE, "r_bot_back", False),
            ("🔬 Dataset & Model Studio →", NAV_STUDIO, "r_bot_studio", False),
            ("📋 View Evaluation History →", NAV_HISTORY, "r_bot_hist", True),
        ])


# -----------------------------------------------------------------------------
# TAB 3: ③ MODEL STUDIO (DATASET INSPECTION & MODEL TRAINING)
# -----------------------------------------------------------------------------
elif current_tab == NAV_STUDIO:
    st.markdown(
        """
        <div class="pf-card">
            <div class="pf-card-title">🔬 Dataset & Model Studio</div>
            <div class="pf-card-desc">Inspect training datasets, upload custom student cohorts, and retrain the multi-class Machine Learning classifier.</div>
        """,
        unsafe_allow_html=True,
    )

    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.markdown("##### 📁 1. Select / Input Training Dataset")
        dataset_source = st.radio(
            "Dataset Source",
            ["Standard University Benchmark Dataset (450+ records)", "Upload Custom Dataset (CSV / Excel)"],
            index=0 if "uploaded_df" not in st.session_state else 1,
            key="dataset_source_radio",
        )

    with d_col2:
        st.markdown("##### 📥 2. Download Dataset Template")
        st.caption("Use this structured CSV template to prepare and upload your own student cohorts.")
        sample_template_df = generate_benchmark_dataset(samples_per_class=2)
        template_csv = sample_template_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download Sample Dataset Template (CSV)",
            data=template_csv,
            file_name="student_career_dataset_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Active DataFrame Resolution
    if dataset_source == "Upload Custom Dataset (CSV / Excel)":
        uploaded_file = st.file_uploader("Upload Student Dataset File (.csv or .xlsx)", type=["csv", "xlsx"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    active_df = pd.read_csv(uploaded_file)
                else:
                    active_df = pd.read_excel(uploaded_file)
                st.session_state["uploaded_df"] = active_df
                st.session_state["active_dataset_name"] = f"Custom Upload: {uploaded_file.name}"
                st.success(f"⚡ Successfully loaded `{uploaded_file.name}` with {len(active_df)} student records!")
            except Exception as e:
                st.error(f"Error parsing uploaded dataset: {e}")
                active_df = generate_benchmark_dataset(samples_per_class=45)
                st.session_state["active_dataset_name"] = "Standard University Benchmark Dataset"
        elif "uploaded_df" in st.session_state:
            active_df = st.session_state["uploaded_df"]
        else:
            st.info("⚡ No custom file uploaded yet. Using Standard Benchmark Dataset.")
            active_df = generate_benchmark_dataset(samples_per_class=45)
            st.session_state["active_dataset_name"] = "Standard University Benchmark Dataset"
    else:
        active_df = generate_benchmark_dataset(samples_per_class=45)
        st.session_state["active_dataset_name"] = "Standard University Benchmark Dataset"

    st.markdown("</div>", unsafe_allow_html=True)

    # Dataset Summary Metrics & Table Preview
    st.markdown("#### 📊 Active Dataset Overview & Statistics")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Student Records", len(active_df))
    with m2:
        gpa_col = active_df["GPA"] if "GPA" in active_df.columns else pd.Series([3.4])
        st.metric("Mean Cohort GPA", f"{gpa_col.mean():.2f}")
    with m3:
        target_col = "TargetCareer" if "TargetCareer" in active_df.columns else ("Career" if "Career" in active_df.columns else active_df.columns[-1])
        st.metric("Unique Career Classes", active_df[target_col].nunique() if target_col in active_df.columns else 9)
    with m4:
        st.metric("Feature Columns", len(active_df.columns))

    # Interactive Table Explorer
    with st.expander("🔍 Explore Full Active Dataset Table", expanded=True):
        st.dataframe(active_df, use_container_width=True, height=260)

    # Visual Distribution Charts
    c_v1, c_v2 = st.columns(2)
    with c_v1:
        st.markdown("##### Career Class Balance")
        if target_col in active_df.columns:
            class_counts = active_df[target_col].value_counts().reset_index()
            class_counts.columns = ["Career", "Count"]
            fig_class = px.bar(
                class_counts,
                x="Count",
                y="Career",
                orientation="h",
                color="Count",
                color_continuous_scale=[[0, "#0C4A6E"], [1, "#00E5FF"]],
            )
            fig_class.update_layout(
                height=280,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(14, 23, 44, 0.5)",
                coloraxis_showscale=False,
                xaxis=dict(tickfont=dict(color="#CBD5E1")),
                yaxis=dict(tickfont=dict(color="#CBD5E1")),
            )
            st.plotly_chart(fig_class, use_container_width=True)

    with c_v2:
        st.markdown("##### GPA Cohort Distribution")
        if "GPA" in active_df.columns:
            fig_gpa = px.histogram(
                active_df,
                x="GPA",
                nbins=20,
                color_discrete_sequence=["#00E5FF"],
            )
            fig_gpa.update_layout(
                height=280,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(14, 23, 44, 0.5)",
                xaxis=dict(tickfont=dict(color="#CBD5E1"), gridcolor="rgba(51, 65, 85, 0.4)"),
                yaxis=dict(tickfont=dict(color="#CBD5E1"), gridcolor="rgba(51, 65, 85, 0.4)"),
            )
            st.plotly_chart(fig_gpa, use_container_width=True)

    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(56, 189, 248, 0.2); margin: 1.8rem 0;'>", unsafe_allow_html=True)

    # Model Training Section
    st.markdown(
        """
        <div class="pf-card">
            <div class="pf-card-title">🚀 Train / Retrain Machine Learning Classifier</div>
            <div class="pf-card-desc">Fit a Random Forest ensemble or a single Decision Tree baseline on the active dataset to calibrate career predictions.</div>
        """,
        unsafe_allow_html=True,
    )

    model_choice = st.radio(
        "Classifier",
        ["random_forest", "decision_tree"],
        format_func=lambda m: MODEL_LABELS[m],
        horizontal=True,
    )
    tcol1, tcol2, tcol3 = st.columns(3)
    with tcol1:
        n_est = st.slider(
            "Number of Estimators (Trees)", 20, 200, 80, step=10,
            disabled=model_choice == "decision_tree", help="Random Forest only; a Decision Tree is a single tree.",
        )
    with tcol2:
        m_dep = st.slider("Maximum Tree Depth", 3, 20, 10)
    with tcol3:
        t_split = st.slider("Test Split Ratio", 0.1, 0.4, 0.2, step=0.05)

    bcol1, bcol2 = st.columns(2)
    with bcol1:
        train_clicked = st.button("⚡ Train Model on Active Dataset", type="primary", use_container_width=True)
    with bcol2:
        compare_clicked = st.button("⚖️ Compare Decision Tree vs Random Forest", use_container_width=True)

    if train_clicked:
        with st.spinner(f"Training {MODEL_LABELS[model_choice]} Classifier on dataset..."):
            train_results = train_custom_classifier(
                active_df, n_estimators=n_est, max_depth=m_dep, test_size=t_split, model_type=model_choice
            )

            if "error" in train_results:
                st.error(f"Training failed: {train_results['error']}")
            else:
                st.session_state["active_ml_model"] = train_results["model"]
                st.session_state["active_ml_careers"] = train_results["careers"]
                st.session_state["active_ml_type"] = train_results["model_type"]
                st.session_state["train_metrics"] = train_results
                st.success(f"⚡ {MODEL_LABELS[model_choice]} successfully trained and now drives the ML engine! Test Accuracy: **{train_results['test_acc']*100:.1f}%** | Train Accuracy: **{train_results['train_acc']*100:.1f}%**")

    if compare_clicked:
        with st.spinner("Training both classifiers with 5-fold cross-validation..."):
            try:
                st.session_state["model_comparison"] = compare_classifiers(
                    active_df, n_estimators=n_est, max_depth=m_dep, test_size=t_split
                )
            except Exception as e:
                st.error(f"Comparison failed: {e}")

    if "model_comparison" in st.session_state:
        st.markdown("##### ⚖️ Baseline Comparison (same split & folds)")
        st.dataframe(
            st.session_state["model_comparison"].style.format(
                {"Train Accuracy": "{:.1%}", "Test Accuracy": "{:.1%}", "CV Mean": "{:.1%}", "CV Std": "±{:.1%}"}
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption("Comparison only: the active ML engine changes only when you press Train.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Display Training Metrics if available
    if "train_metrics" in st.session_state:
        metrics = st.session_state["train_metrics"]
        st.markdown("##### 📈 Model Performance & Feature Importances")
        pm1, pm2, pm3 = st.columns(3)
        with pm1:
            st.metric("Test Set Accuracy", f"{metrics['test_acc']*100:.1f}%")
        with pm2:
            st.metric("Train Set Accuracy", f"{metrics['train_acc']*100:.1f}%")
        with pm3:
            st.metric("Training Samples", f"{metrics['train_samples']} train / {metrics['test_samples']} test")

        st.markdown("<br>", unsafe_allow_html=True)

        # Feature Importance Chart
        feat_df = metrics["feat_imp_df"]
        fig_feat = px.bar(
            feat_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"Global Feature Importances ({MODEL_LABELS.get(metrics.get('model_type'), 'Random Forest')} Gini Impurity)",
            color="Importance",
            color_continuous_scale=[[0, "#0C4A6E"], [1, "#00E5FF"]],
        )
        fig_feat.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(14, 23, 44, 0.5)",
            coloraxis_showscale=False,
            title_font=dict(color="#F8FAFC", family="Space Grotesk"),
            xaxis=dict(tickfont=dict(color="#CBD5E1"), gridcolor="rgba(51, 65, 85, 0.4)"),
            yaxis=dict(tickfont=dict(color="#CBD5E1")),
        )
        st.plotly_chart(fig_feat, use_container_width=True)

    # Bottom Navigation for Tab 3
    bottom_nav([
        ("← Back to Career Roadmap", NAV_ROADMAP, "s_bot_back", False),
        ("👤 Edit Student Profile", NAV_PROFILE, "s_bot_prof", False),
        ("📋 View Evaluation History →", NAV_HISTORY, "s_bot_next", True),
    ])


# -----------------------------------------------------------------------------
# TAB 4: ④ HISTORY (AUDIT TRAIL & LOG STORAGE)
# -----------------------------------------------------------------------------
elif current_tab == NAV_HISTORY:
    st.markdown(
        """
        <div class="pf-card-title">📋 Academic Advisor & Evaluation History</div>
        <div class="pf-card-desc">Structured audit log of student profile assessments persisted in local SQLite storage.</div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Records are kept in this app's SQLite file. On Streamlit Community Cloud that file is shared by every visitor "
        "and resets whenever the app restarts or is redeployed."
    )
    history_filter = st.text_input("Filter by Student ID", placeholder="Show every assessment for one student", key="history_filter")
    records_df = fetch_all_records(history_filter)

    student_history = fetch_student_history(history_filter)
    tracked = [h for h in student_history if h["career_scores"]]
    if len(tracked) >= 2:
        latest = tracked[-1]["career_scores"]
        top_keys = sorted(latest, key=latest.get, reverse=True)[:3]
        st.markdown(f"##### 📈 Trajectory for {normalize_student_id(history_filter)}")
        st.plotly_chart(build_progress_chart(tracked, top_keys), use_container_width=True, config={"displayModeBar": False}, key="progress_chart_history")

    if records_df.empty:
        st.info("⚡ No historical student records found in `career_records.db`. Submitting profiles in **① Profile** will automatically populate this database.")
    else:
        total_evals = len(records_df)
        avg_gpa = records_df["GPA"].mean()
        popular_role = records_df["Recommended Role"].mode()[0] if not records_df.empty else "N/A"

        m_c1, m_c2, m_c3 = st.columns(3)
        with m_c1:
            st.metric("Total Profile Assessments", f"{total_evals}")
        with m_c2:
            st.metric("Cohort Average GPA", f"{avg_gpa:.2f}")
        with m_c3:
            st.metric("Most Frequent Recommendation", f"{popular_role}")

        st.markdown("<br>", unsafe_allow_html=True)

        st.dataframe(
            records_df,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        col_act1, col_act2 = st.columns([1, 4])
        with col_act1:
            csv_data = records_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇ Export CSV Log",
                data=csv_data,
                file_name=f"career_records_{datetime.date.today()}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_act2:
            if st.button("🗑️ Clear History Database", help="Deletes all recorded logs from SQLite."):
                clear_all_records()
                st.rerun()

    # Bottom Navigation for Tab 4
    bottom_nav([
        ("← Back to Career Roadmap", NAV_ROADMAP, "h_bot_back", False),
        ("🔬 Open Model Studio", NAV_STUDIO, "h_bot_studio", False),
        ("🚀 Start New Profile Intake →", NAV_PROFILE, "h_bot_new", True),
    ])
