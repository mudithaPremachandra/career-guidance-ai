"""
PathFinder AI: Modern AI-Powered Career Intelligence Platform
-------------------------------------------------------------
A complete, standalone Streamlit application integrating:
1. 4-Pillar Categorized Soft Skills & Work Strengths (People, Ideas, Data, Execution)
2. 6-Archetype Job-to-Soft-Skill Requirement Mapping with Weight Bonus
3. Reactive Profile Intake with real-time dynamic Elective & Module Sliders
4. Rule-Based Reasoning with rule-firing trace
5. Fuzzy Logic Suitability Engine (Mamdani inference & defuzzification)
6. Supervised Machine Learning Classifier (Random Forest proxy / Retrainable on Custom Datasets)
7. Multi-Engine Score Fusion (30% Rule + 30% Fuzzy + 40% ML)
8. Explainable AI (SHAP proxy divergent attribution)
9. Skill Gap Analysis with Urgency Prioritization (Technical & Soft Skills)
10. Cosine-Similarity Industry Certification Recommender
11. Dataset & Model Studio (Upload Custom CSV / Load Benchmark / Train Model / Feature Importances / Batch Predictions)
12. Local SQLite persistence for history logging

Visual Identity: AI + Career Intelligence + Education + Personal Growth
"""

import datetime
import io
import json
import math
import os
import sqlite3
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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


# -----------------------------------------------------------------------------
# 2. LOCAL SQLITE DATABASE LAYER
# -----------------------------------------------------------------------------
DB_FILE = "career_records.db"


def init_database() -> None:
    """Initializes the SQLite database with the records table."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            gpa REAL NOT NULL,
            academic_year TEXT NOT NULL,
            top_career TEXT NOT NULL,
            match_score REAL NOT NULL,
            confidence TEXT NOT NULL,
            work_style TEXT NOT NULL,
            top_driver TEXT NOT NULL,
            critical_gap TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def save_student_record(
    gpa: float,
    academic_year: str,
    top_career: str,
    match_score: float,
    confidence: str,
    work_style: str,
    top_driver: str,
    critical_gap: str,
) -> None:
    """Saves a student profile evaluation result into SQLite."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        """
        INSERT INTO records (
            timestamp, gpa, academic_year, top_career, match_score, 
            confidence, work_style, top_driver, critical_gap
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            now_str,
            round(gpa, 2),
            academic_year,
            top_career,
            round(match_score * 100, 1),
            confidence,
            work_style,
            top_driver,
            critical_gap,
        ),
    )
    conn.commit()
    conn.close()


def fetch_all_records() -> pd.DataFrame:
    """Retrieves all past student submissions from SQLite."""
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql_query(
            "SELECT id AS 'ID', timestamp AS 'Timestamp', gpa AS 'GPA', "
            "academic_year AS 'Year', top_career AS 'Recommended Role', "
            "match_score AS 'Match %', confidence AS 'Confidence', "
            "work_style AS 'Work Style', top_driver AS 'Key Strength', "
            "critical_gap AS 'Primary Gap' FROM records ORDER BY id DESC",
            conn,
        )
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


def clear_all_records() -> None:
    """Clears history records from SQLite."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM records")
    conn.commit()
    conn.close()


init_database()


# -----------------------------------------------------------------------------
# 3. KNOWLEDGE BASE & 6-ARCHETYPE CAREER DEFINITIONS
# -----------------------------------------------------------------------------
CAREER_UNIVERSE = {
    "Software Engineer": {
        "title": "Software Engineer",
        "archetype": "Engineering & Architecture",
        "category": "Core Engineering",
        "description": "Architects scalable software systems, backend microservices, robust APIs, and enterprise applications.",
        "role_dynamic": "Requires sustained concentration on complex abstractions and anticipating edge cases.",
        "icon": "💻",
        "work_style_fit": "Technical Specialist",
        "required_soft_skills": [
            "Problem Solving & Logic",
            "Debugging & Root-Cause Analysis",
            "Detail-Oriented & Precision",
            "Innovation & Prototyping",
        ],
        "weight_bonus": 0.20,
        "benchmark_skills": {
            "DSA": 85,
            "OOP": 85,
            "DBMS": 75,
            "OS_Networks": 75,
            "SE_Principles": 85,
            "Math_Stats": 70,
            "Python": 3.5,
            "Java_CPP": 4.0,
            "SQL": 3.5,
            "WebStack": 4.0,
            "CloudDocker": 3.0,
            "ML_AI": 2.0,
            "MobileDev": 2.5,
            "Cybersecurity": 2.5,
        },
        "target_workstyles": ["Technical Specialist", "R&D / Creative"],
    },
    "Data Scientist": {
        "title": "Data Scientist",
        "archetype": "Data & Analytics",
        "category": "Analytics & Intelligence",
        "description": "Transforms high-dimensional datasets into predictive statistical models and strategic business insights.",
        "role_dynamic": "Translating messy raw data and business rules into actionable executive decisions.",
        "icon": "📊",
        "work_style_fit": "R&D / Creative",
        "required_soft_skills": [
            "Analytical & Critical Thinking",
            "Storytelling & Conceptualizing",
            "Detail-Oriented & Precision",
            "Quantitative & Mathematical",
            "Research & Investigation",
        ],
        "weight_bonus": 0.20,
        "benchmark_skills": {
            "DSA": 75,
            "OOP": 70,
            "DBMS": 85,
            "OS_Networks": 65,
            "SE_Principles": 70,
            "Math_Stats": 90,
            "Python": 4.5,
            "Java_CPP": 2.5,
            "SQL": 4.5,
            "WebStack": 2.0,
            "CloudDocker": 3.0,
            "ML_AI": 4.5,
            "MobileDev": 1.5,
            "Cybersecurity": 2.0,
        },
        "target_workstyles": ["R&D / Creative", "Consulting / Management"],
    },
    "AI Engineer": {
        "title": "AI Engineer",
        "archetype": "Engineering & Architecture",
        "category": "Artificial Intelligence",
        "description": "Develops deep learning architectures, neural networks, foundation model pipelines, and autonomous agent systems.",
        "role_dynamic": "Sustained concentration on multi-dimensional neural abstractions, model alignment, and anticipating failure modes.",
        "icon": "🧠",
        "work_style_fit": "R&D / Creative",
        "required_soft_skills": [
            "Problem Solving & Logic",
            "Innovation & Prototyping",
            "Research & Investigation",
            "Quantitative & Mathematical",
        ],
        "weight_bonus": 0.20,
        "benchmark_skills": {
            "DSA": 85,
            "OOP": 80,
            "DBMS": 75,
            "OS_Networks": 70,
            "SE_Principles": 80,
            "Math_Stats": 90,
            "Python": 5.0,
            "Java_CPP": 3.5,
            "SQL": 3.5,
            "WebStack": 2.5,
            "CloudDocker": 4.0,
            "ML_AI": 5.0,
            "MobileDev": 2.0,
            "Cybersecurity": 2.5,
        },
        "target_workstyles": ["R&D / Creative", "Technical Specialist"],
    },
    "Cloud Architect": {
        "title": "Cloud Architect",
        "archetype": "Infrastructure, Systems & Security",
        "category": "Cloud Infrastructure",
        "description": "Designs highly-available distributed systems, multi-cloud topologies, container orchestration, and CI/CD pipelines.",
        "role_dynamic": "Rapid diagnosis, fault-tolerant topology modeling, and incident escalation under pressure.",
        "icon": "☁️",
        "work_style_fit": "Technical Specialist",
        "required_soft_skills": [
            "Hardware & System Troubleshooting",
            "Planning & Task Organizing",
            "Working Under Pressure / Incidents",
            "Problem Solving & Logic",
        ],
        "weight_bonus": 0.20,
        "benchmark_skills": {
            "DSA": 70,
            "OOP": 70,
            "DBMS": 80,
            "OS_Networks": 90,
            "SE_Principles": 80,
            "Math_Stats": 65,
            "Python": 3.5,
            "Java_CPP": 3.0,
            "SQL": 3.5,
            "WebStack": 3.0,
            "CloudDocker": 5.0,
            "ML_AI": 2.0,
            "MobileDev": 2.0,
            "Cybersecurity": 4.0,
        },
        "target_workstyles": ["Technical Specialist", "Consulting / Management"],
    },
    "UX Designer": {
        "title": "UX Designer",
        "archetype": "Creative, UI/UX & Media",
        "category": "Product & Design",
        "description": "Conducts user research, cognitive ergonomics evaluation, and crafts design systems for delightful user journeys.",
        "role_dynamic": "Balancing aesthetic appeal with cognitive ergonomics, user empathy, and usability constraints.",
        "icon": "🎨",
        "work_style_fit": "R&D / Creative",
        "required_soft_skills": [
            "Active Listening & Empathy",
            "Creative & Visual Design",
            "Storytelling & Conceptualizing",
            "Innovation & Prototyping",
        ],
        "weight_bonus": 0.20,
        "benchmark_skills": {
            "DSA": 50,
            "OOP": 55,
            "DBMS": 50,
            "OS_Networks": 45,
            "SE_Principles": 80,
            "Math_Stats": 55,
            "Python": 2.0,
            "Java_CPP": 1.5,
            "SQL": 2.0,
            "WebStack": 4.0,
            "CloudDocker": 1.5,
            "ML_AI": 2.0,
            "MobileDev": 3.5,
            "Cybersecurity": 1.5,
        },
        "target_workstyles": ["R&D / Creative", "Consulting / Management"],
    },
    "IT Business Analyst": {
        "title": "IT Business Analyst",
        "archetype": "Management & Consulting",
        "category": "Business & Strategy",
        "description": "Synthesizes stakeholder requirements, functional specifications, Agile user stories, and digital transformation initiatives.",
        "role_dynamic": "Keeping cross-functional teams aligned, managing client expectations, and negotiating scope.",
        "icon": "📈",
        "work_style_fit": "Consulting / Management",
        "required_soft_skills": [
            "Communicating & Articulating",
            "Analytical & Critical Thinking",
            "Planning & Task Organizing",
            "Negotiation & Persuasion",
            "Active Listening & Empathy",
        ],
        "weight_bonus": 0.20,
        "benchmark_skills": {
            "DSA": 55,
            "OOP": 55,
            "DBMS": 75,
            "OS_Networks": 60,
            "SE_Principles": 90,
            "Math_Stats": 75,
            "Python": 2.5,
            "Java_CPP": 2.0,
            "SQL": 4.0,
            "WebStack": 2.5,
            "CloudDocker": 2.0,
            "ML_AI": 2.5,
            "MobileDev": 2.0,
            "Cybersecurity": 2.5,
        },
        "target_workstyles": ["Consulting / Management"],
    },
    "Game Developer": {
        "title": "Game Developer",
        "archetype": "Creative, UI/UX & Media",
        "category": "Interactive Media",
        "description": "Engineers real-time 2D/3D gameplay mechanics, custom graphics shaders, physics engines, and virtual simulations.",
        "role_dynamic": "Balancing visual aesthetic feel with rigorous real-time algorithmic physics constraints.",
        "icon": "🎮",
        "work_style_fit": "Technical Specialist",
        "required_soft_skills": [
            "Creative & Visual Design",
            "Problem Solving & Logic",
            "Hands-on Prototyping",
            "Storytelling & Conceptualizing",
        ],
        "weight_bonus": 0.20,
        "benchmark_skills": {
            "DSA": 85,
            "OOP": 90,
            "DBMS": 60,
            "OS_Networks": 70,
            "SE_Principles": 75,
            "Math_Stats": 85,
            "Python": 3.0,
            "Java_CPP": 5.0,
            "SQL": 2.5,
            "WebStack": 2.5,
            "CloudDocker": 2.5,
            "ML_AI": 2.5,
            "MobileDev": 3.5,
            "Cybersecurity": 2.0,
        },
        "target_workstyles": ["Technical Specialist", "R&D / Creative"],
    },
    "CAD-CAM Engineer": {
        "title": "CAD-CAM Engineer",
        "archetype": "Engineering & Architecture",
        "category": "Computational Manufacturing",
        "description": "Applies parametric geometric modeling, finite element analysis, and automated numerical manufacturing systems.",
        "role_dynamic": "High methodical precision, mathematical spatial reasoning, and manufacturing tolerances.",
        "icon": "📐",
        "work_style_fit": "Technical Specialist",
        "required_soft_skills": [
            "Detail-Oriented & Precision",
            "Problem Solving & Logic",
            "Hands-on Prototyping",
            "Quantitative & Mathematical",
        ],
        "weight_bonus": 0.20,
        "benchmark_skills": {
            "DSA": 70,
            "OOP": 70,
            "DBMS": 60,
            "OS_Networks": 60,
            "SE_Principles": 75,
            "Math_Stats": 85,
            "Python": 3.0,
            "Java_CPP": 3.5,
            "SQL": 2.5,
            "WebStack": 2.0,
            "CloudDocker": 2.0,
            "ML_AI": 2.5,
            "MobileDev": 1.5,
            "Cybersecurity": 2.0,
        },
        "target_workstyles": ["Technical Specialist"],
    },
    "Cybersecurity Specialist": {
        "title": "Cybersecurity Specialist",
        "archetype": "Infrastructure, Systems & Security",
        "category": "Information Security",
        "description": "Performs threat hunting, security auditing, vulnerability assessment, cryptography deployment, and incident defense.",
        "role_dynamic": "Rapid diagnosis, methodical protocol precision, and calm decision making during live security breaches.",
        "icon": "🛡️",
        "work_style_fit": "Technical Specialist",
        "required_soft_skills": [
            "Working Under Pressure / Incidents",
            "Debugging & Root-Cause Analysis",
            "Process & Protocol Adherence",
            "Analytical & Critical Thinking",
        ],
        "weight_bonus": 0.20,
        "benchmark_skills": {
            "DSA": 75,
            "OOP": 70,
            "DBMS": 75,
            "OS_Networks": 95,
            "SE_Principles": 75,
            "Math_Stats": 75,
            "Python": 4.0,
            "Java_CPP": 3.5,
            "SQL": 3.5,
            "WebStack": 3.0,
            "CloudDocker": 4.0,
            "ML_AI": 2.5,
            "MobileDev": 2.0,
            "Cybersecurity": 5.0,
        },
        "target_workstyles": ["Technical Specialist", "Consulting / Management"],
    },
}

# Industry Certification Catalogue with Skill Tags for Vector Matching
def load_certification_catalog() -> List[Dict[str, Any]]:
    """Loads certification database from certifications.json with fallback."""
    json_path = os.path.join(os.path.dirname(__file__), "certifications.json") if "__file__" in globals() else "certifications.json"
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return [
        {
            "id": "meta-frontend",
            "title": "Meta Front-End Developer Professional Certificate",
            "issuer": "Coursera / Meta",
            "level": "Beginner to Intermediate",
            "target_years": ["1st Year", "2nd Year", "3rd Year"],
            "domains": ["Software Engineering", "Web Development", "UI/UX Design"],
            "skills": ["Modern Web Stack", "Software Engineering Principles", "Mobile Development"],
            "skill_keys": ["WebStack", "SE_Principles", "MobileDev"],
            "vector": {"WebStack": 1.0, "SE_Principles": 0.7, "MobileDev": 0.5},
            "description": "Master React, JavaScript ES6+, HTML5/CSS3, responsive UI frameworks, and production-grade web application architecture.",
            "official_url": "https://www.coursera.org/professional-certificates/meta-front-end-developer",
            "prep_url": "https://www.metacareers.com/career-programs/certificates",
            "exam_guide_url": "https://www.metacareers.com/career-programs/certificates"
        },
        {
            "id": "aws-dev-assoc",
            "title": "AWS Certified Developer – Associate (DVA-C02)",
            "issuer": "Amazon Web Services",
            "level": "Intermediate",
            "target_years": ["2nd Year", "3rd Year", "4th Year"],
            "domains": ["Software Engineering", "Cloud & DevOps", "Web Development"],
            "skills": ["Modern Web Stack", "Cloud Architecture & Docker", "Software Engineering Principles"],
            "skill_keys": ["WebStack", "CloudDocker", "SE_Principles"],
            "vector": {"CloudDocker": 1.0, "WebStack": 0.85, "SE_Principles": 0.75},
            "description": "Validates cloud-native application development, serverless AWS Lambda, DynamoDB, API Gateways, container deployment, and CI/CD pipelines.",
            "official_url": "https://aws.amazon.com/certification/certified-developer-associate/",
            "prep_url": "https://aws.amazon.com/training/digital/",
            "exam_guide_url": "https://d1.awsstatic.com/training-and-certification/docs-dev-associate/AWS-Certified-Developer-Associate_Exam-Guide.pdf"
        },
        {
            "id": "aws-solutions-architect",
            "title": "AWS Certified Solutions Architect – Associate (SAA-C03)",
            "issuer": "Amazon Web Services",
            "level": "Intermediate to Advanced",
            "target_years": ["3rd Year", "4th Year"],
            "domains": ["Cloud & DevOps", "Infrastructure & Networking", "Software Engineering", "Cybersecurity & Defense"],
            "skills": ["Cloud Architecture & Docker", "OS & Computer Networks", "Cybersecurity Defense", "Software Engineering Principles"],
            "skill_keys": ["CloudDocker", "OS_Networks", "Cybersecurity", "SE_Principles"],
            "vector": {"CloudDocker": 1.0, "OS_Networks": 0.85, "Cybersecurity": 0.65, "SE_Principles": 0.65},
            "description": "Industry benchmark for designing resilient, high-performing, cost-optimized, and secure multi-tier distributed cloud architectures.",
            "official_url": "https://aws.amazon.com/certification/certified-solutions-architect-associate/",
            "prep_url": "https://aws.amazon.com/training/digital/",
            "exam_guide_url": "https://d1.awsstatic.com/training-and-certification/docs-sa-assoc/AWS-Certified-Solutions-Architect-Associate_Exam-Guide.pdf"
        },
        {
            "id": "tf-dev-cert",
            "title": "TensorFlow Developer Certificate",
            "issuer": "DeepLearning.AI / Google",
            "level": "Intermediate to Advanced",
            "target_years": ["2nd Year", "3rd Year", "4th Year"],
            "domains": ["Artificial Intelligence", "Data & Analytics", "Software Engineering"],
            "skills": ["Machine Learning & AI", "Python Programming", "Mathematics & Statistics"],
            "skill_keys": ["ML_AI", "Python", "Math_Stats"],
            "vector": {"ML_AI": 1.0, "Python": 0.95, "Math_Stats": 0.75},
            "description": "Validates practical deep learning architectures, CNNs, NLP sequence models, model optimization, and production inference.",
            "official_url": "https://www.deeplearning.ai/courses/tensorflow-developer-professional-certificate/",
            "prep_url": "https://www.tensorflow.org/certificate",
            "exam_guide_url": "https://www.tensorflow.org/extras/surveys/candidate_handbook.pdf"
        }
    ]

CERTIFICATION_CATALOG = load_certification_catalog()


# -----------------------------------------------------------------------------
# 4. STUDENT PROFILE DATA MODEL & UNIFIED SKILL EXTRACTOR
# -----------------------------------------------------------------------------
class StudentProfile:
    def __init__(
        self,
        gpa: float,
        year: str,
        core_modules: Dict[str, float],
        electives: Dict[str, float],
        tech_skills: Dict[str, int],
        soft_skills: List[str],
        desired_domains: List[str],
        work_style: str,
        has_internship: bool,
        projects_count: int,
        existing_certs: str,
    ):
        self.gpa = gpa
        self.year = year
        self.core_modules = core_modules
        self.electives = electives
        self.tech_skills = tech_skills
        self.soft_skills = soft_skills
        self.desired_domains = desired_domains
        self.work_style = work_style
        self.has_internship = has_internship
        self.projects_count = projects_count
        self.existing_certs = existing_certs

    def get_unified_skill_dict(self) -> Dict[str, float]:
        """Combines core modules, tech proficiencies (scaled 0-5), and electives into a unified map."""
        u: Dict[str, float] = {}
        for k, v in self.core_modules.items():
            u[k] = float(v)
        for k, v in self.tech_skills.items():
            u[k] = float(v)
        for k, v in self.electives.items():
            u[f"ELEC_{k}"] = float(v)
        return u


# -----------------------------------------------------------------------------
# 5. DATASET STUDIO: GENERATION, BENCHMARKS, RETRAINING
# -----------------------------------------------------------------------------
def generate_benchmark_dataset(samples_per_class: int = 50) -> pd.DataFrame:
    """Generates a rich benchmark training dataset covering all 9 career classes."""
    np.random.seed(42)
    rows = []
    careers = list(CAREER_UNIVERSE.keys())

    year_choices = ["1st Year", "2nd Year", "3rd Year", "4th Year"]
    style_choices = ["Technical Specialist", "Consulting / Management", "R&D / Creative"]

    for career in careers:
        bench = CAREER_UNIVERSE[career]["benchmark_skills"]
        pref_styles = CAREER_UNIVERSE[career]["target_workstyles"]
        req_softs = CAREER_UNIVERSE[career]["required_soft_skills"]

        for _ in range(samples_per_class):
            gpa = round(float(np.clip(np.random.normal(3.45, 0.28), 2.20, 4.00)), 2)
            year = np.random.choice(year_choices, p=[0.1, 0.25, 0.45, 0.20])
            dsa = int(np.clip(np.random.normal(bench["DSA"], 7), 40, 100))
            oop = int(np.clip(np.random.normal(bench["OOP"], 7), 40, 100))
            dbms = int(np.clip(np.random.normal(bench["DBMS"], 7), 40, 100))
            os_net = int(np.clip(np.random.normal(bench["OS_Networks"], 7), 40, 100))
            se = int(np.clip(np.random.normal(bench["SE_Principles"], 7), 40, 100))
            math_s = int(np.clip(np.random.normal(bench["Math_Stats"], 7), 40, 100))

            py = int(np.clip(np.random.normal(bench["Python"], 0.6), 1, 5))
            jcpp = int(np.clip(np.random.normal(bench["Java_CPP"], 0.6), 1, 5))
            sql = int(np.clip(np.random.normal(bench["SQL"], 0.6), 1, 5))
            web = int(np.clip(np.random.normal(bench["WebStack"], 0.6), 1, 5))
            cloud = int(np.clip(np.random.normal(bench["CloudDocker"], 0.6), 1, 5))
            ml = int(np.clip(np.random.normal(bench["ML_AI"], 0.6), 1, 5))
            mob = int(np.clip(np.random.normal(bench["MobileDev"], 0.6), 1, 5))
            sec = int(np.clip(np.random.normal(bench["Cybersecurity"], 0.6), 1, 5))

            intern = 1 if np.random.rand() > 0.45 else 0
            projects = int(np.clip(np.random.poisson(3.5), 1, 10))
            work_style = np.random.choice(pref_styles) if np.random.rand() > 0.2 else np.random.choice(style_choices)
            soft_count = int(np.clip(np.random.normal(len(req_softs) + 1, 1.2), 2, 8))

            rows.append({
                "StudentID": f"STD-{1000 + len(rows) + 1}",
                "GPA": gpa,
                "Year": year,
                "DSA": dsa,
                "OOP": oop,
                "DBMS": dbms,
                "OS_Networks": os_net,
                "SE_Principles": se,
                "Math_Stats": math_s,
                "Python": py,
                "Java_CPP": jcpp,
                "SQL": sql,
                "WebStack": web,
                "CloudDocker": cloud,
                "ML_AI": ml,
                "MobileDev": mob,
                "Cybersecurity": sec,
                "Internship": intern,
                "Projects": projects,
                "WorkStyle": work_style,
                "SoftSkillsCount": soft_count,
                "TargetCareer": career,
            })

    return pd.DataFrame(rows)


def preprocess_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Encodes a DataFrame into numerical X and y matrices."""
    year_map = {"1st Year": 0.25, "2nd Year": 0.5, "3rd Year": 0.75, "4th Year": 1.0}
    style_map = {"Technical Specialist": 1.0, "Consulting / Management": 0.5, "R&D / Creative": 0.8}

    X_list = []
    careers = list(CAREER_UNIVERSE.keys())
    career_to_idx = {c: i for i, c in enumerate(careers)}

    y_list = []

    for _, row in df.iterrows():
        gpa_norm = float(row.get("GPA", 3.0)) / 4.0
        year_norm = year_map.get(str(row.get("Year", "3rd Year")), 0.5)
        dsa = float(row.get("DSA", 70)) / 100.0
        oop = float(row.get("OOP", 70)) / 100.0
        dbms = float(row.get("DBMS", 70)) / 100.0
        os_net = float(row.get("OS_Networks", 70)) / 100.0
        se = float(row.get("SE_Principles", 70)) / 100.0
        math_s = float(row.get("Math_Stats", 70)) / 100.0

        py = float(row.get("Python", 3)) / 5.0
        jcpp = float(row.get("Java_CPP", 3)) / 5.0
        sql = float(row.get("SQL", 3)) / 5.0
        web = float(row.get("WebStack", 3)) / 5.0
        cloud = float(row.get("CloudDocker", 3)) / 5.0
        ml = float(row.get("ML_AI", 3)) / 5.0
        mob = float(row.get("MobileDev", 3)) / 5.0
        sec = float(row.get("Cybersecurity", 3)) / 5.0

        intern = float(row.get("Internship", 0))
        proj = min(float(row.get("Projects", 2)) / 5.0, 1.0)
        style = style_map.get(str(row.get("WorkStyle", "Technical Specialist")), 0.5)
        soft = min(float(row.get("SoftSkillsCount", 3)) / 6.0, 1.0)

        feat_row = [
            gpa_norm, year_norm, dsa, oop, dbms, os_net, se, math_s,
            py, jcpp, sql, web, cloud, ml, mob, sec,
            intern, proj, style, soft,
        ]
        X_list.append(feat_row)

        target = str(row.get("TargetCareer", "Software Engineer")).strip()
        idx = career_to_idx.get(target, 0)
        y_list.append(idx)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64), careers


def train_custom_classifier(
    df: pd.DataFrame, n_estimators: int = 100, max_depth: int = 10, test_size: float = 0.2
) -> Dict[str, Any]:
    """Trains a Random Forest classifier on the provided DataFrame and returns metrics."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split

        X, y, careers = preprocess_features(df)

        if len(X) < 10:
            return {"error": "Dataset must contain at least 10 student records for training."}

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) > 1 and min(np.bincount(y)) >= 2 else None
        )

        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        clf.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc = accuracy_score(y_test, clf.predict(X_test))

        feature_names = [
            "GPA", "Year", "DSA", "OOP", "DBMS", "OS & Networks",
            "SE Principles", "Math & Stats", "Python", "Java/C++",
            "SQL", "Web Stack", "Cloud/Docker", "ML/AI", "Mobile Dev",
            "Cybersecurity", "Internship", "Projects", "Work Style", "Soft Skills Affinity",
        ]

        importances = clf.feature_importances_
        feat_imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(
            by="Importance", ascending=True
        )

        return {
            "model": clf,
            "careers": careers,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "feat_imp_df": feat_imp_df,
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }
    except Exception as e:
        return {"error": str(e)}


# -----------------------------------------------------------------------------
# 6. INFERENCE ENGINES (RULE-BASED, FUZZY LOGIC, SUPERVISED ML)
# -----------------------------------------------------------------------------

# 6.1 ENGINE A: RULE-BASED REASONING (30% Weight)
def evaluate_rule_engine(profile: StudentProfile, career_name: str) -> Tuple[float, List[str]]:
    """Evaluates explicit academic prerequisite, gatekeeper rules, and soft-skill requirement bonus."""
    rules_fired: List[str] = []
    points = 0.0
    max_possible = 100.0

    unified = profile.get_unified_skill_dict()
    electives_set = set(profile.electives.keys())
    career_info = CAREER_UNIVERSE[career_name]

    # Baseline GPA rule
    if profile.gpa >= 3.5:
        points += 15.0
        rules_fired.append("Rule [R-GPA-HIGH]: Exceptional academic standing GPA >= 3.5 (+15).")
    elif profile.gpa >= 3.0:
        points += 10.0
        rules_fired.append("Rule [R-GPA-MED]: Strong academic standing GPA >= 3.0 (+10).")
    else:
        points += 5.0
        rules_fired.append("Rule [R-GPA-BASE]: Foundation academic standing (+5).")

    # Internship & Practical Experience Rule
    if profile.has_internship:
        points += 15.0
        rules_fired.append("Rule [R-EXP-INTERN]: Completed industry/university internship (+15).")
    if profile.projects_count >= 3:
        points += 10.0
        rules_fired.append("Rule [R-EXP-PROJ]: Robust technical project portfolio >= 3 (+10).")

    # Role-Specific Core Rules
    if career_name == "Software Engineer":
        if unified["DSA"] >= 75 and unified["OOP"] >= 75:
            points += 25.0
            rules_fired.append("Rule [R-SE-CORE]: High DSA & OOP core competency verified (+25).")
        if unified["SE_Principles"] >= 75:
            points += 15.0
            rules_fired.append("Rule [R-SE-ENG]: Software Engineering architecture principles verified (+15).")

    elif career_name == "Data Scientist":
        if unified["Math_Stats"] >= 80 and unified["DBMS"] >= 75:
            points += 25.0
            rules_fired.append("Rule [R-DS-MATH]: Advanced statistics and relational data foundations (+25).")
        if unified["Python"] >= 4 and unified["SQL"] >= 4:
            points += 15.0
            rules_fired.append("Rule [R-DS-TOOL]: High Python data science and SQL query proficiency (+15).")
        if "Data Mining & Big Data" in electives_set:
            points += 10.0
            rules_fired.append("Rule [R-DS-ELEC]: Specialized elective 'Data Mining & Big Data' completed (+10).")

    elif career_name == "AI Engineer":
        if unified["Math_Stats"] >= 80 and unified["DSA"] >= 80:
            points += 20.0
            rules_fired.append("Rule [R-AI-MATH]: Advanced algorithmic and mathematical foundations (+20).")
        if unified["ML_AI"] >= 4 and unified["Python"] >= 4:
            points += 20.0
            rules_fired.append("Rule [R-AI-FRAMEWORK]: Deep learning and PyTorch/TensorFlow expertise (+20).")
        if "AI & Machine Learning" in electives_set:
            points += 10.0
            rules_fired.append("Rule [R-AI-ELEC]: Specialized elective 'AI & Machine Learning' completed (+10).")

    elif career_name == "Cloud Architect":
        if unified["OS_Networks"] >= 80 and unified["CloudDocker"] >= 3:
            points += 25.0
            rules_fired.append("Rule [R-CLOUD-NET]: Strong OS networking, Linux, and containerization (+25).")
        if "Cloud Computing" in electives_set:
            points += 15.0
            rules_fired.append("Rule [R-CLOUD-ELEC]: Specialized elective 'Cloud Computing' completed (+15).")

    elif career_name == "UX Designer":
        if "Human-Computer Interaction (UI/UX)" in electives_set or unified["WebStack"] >= 3:
            points += 30.0
            rules_fired.append("Rule [R-UX-HCI]: HCI / UI Design coursework or front-end mastery (+30).")

    elif career_name == "IT Business Analyst":
        if "IT Project Management" in electives_set or unified["SE_Principles"] >= 80:
            points += 30.0
            rules_fired.append("Rule [R-BA-PM]: Agile PM coursework or strong SE process modeling (+30).")
        if unified["SQL"] >= 3 and unified["DBMS"] >= 70:
            points += 15.0
            rules_fired.append("Rule [R-BA-DATA]: Business data querying and requirements modeling (+15).")

    elif career_name == "Game Developer":
        if "Game Engine Development" in electives_set or "Computer Graphics & Animation" in electives_set:
            points += 30.0
            rules_fired.append("Rule [R-GAME-ELEC]: Specialized graphics or game engine electives (+30).")
        if unified["Java_CPP"] >= 4 and unified["DSA"] >= 75:
            points += 15.0
            rules_fired.append("Rule [R-GAME-CPP]: Robust C++ systems programming and real-time data structures (+15).")

    elif career_name == "CAD-CAM Engineer":
        if "CAD/CAM Principles" in electives_set:
            points += 30.0
            rules_fired.append("Rule [R-CAD-ELEC]: Specialized elective 'CAD/CAM Principles' verified (+30).")
        if unified["Math_Stats"] >= 75 and unified["SE_Principles"] >= 70:
            points += 15.0
            rules_fired.append("Rule [R-CAD-MATH]: Strong computational geometry and engineering principles (+15).")

    elif career_name == "Cybersecurity Specialist":
        if "Network Security & Cyber Defense" in electives_set or unified["Cybersecurity"] >= 4:
            points += 30.0
            rules_fired.append("Rule [R-SEC-CORE]: Cyber defense specialization or high technical security score (+30).")
        if unified["OS_Networks"] >= 80:
            points += 15.0
            rules_fired.append("Rule [R-SEC-NET]: Advanced mastery of TCP/IP, OSI protocols, and OS internals (+15).")

    # Job-to-Soft-Skill Archetype Requirement Mapping (+20% Weight Bonus)
    required_softs = career_info.get("required_soft_skills", [])
    weight_bonus = career_info.get("weight_bonus", 0.20)
    matched_softs = [s for s in required_softs if s in profile.soft_skills]

    if required_softs:
        soft_ratio = len(matched_softs) / len(required_softs)
        soft_points = soft_ratio * (weight_bonus * 100.0)
        points += soft_points
        if matched_softs:
            rules_fired.append(
                f"Rule [R-SOFT-SKILLS]: Archetype '{career_info['archetype']}' "
                f"matched ({len(matched_softs)}/{len(required_softs)}) required soft skills: {', '.join(matched_softs[:2])} (+{soft_points:.1f})."
            )

    # Work style alignment rule
    target_styles = career_info["target_workstyles"]
    if profile.work_style in target_styles:
        points += 5.0
        rules_fired.append(f"Rule [R-STYLE-MATCH]: Work style preference '{profile.work_style}' aligns with role requirements (+5).")

    normalized_score = min(max(points / max_possible, 0.0), 1.0)
    return normalized_score, rules_fired


# 6.2 ENGINE B: FUZZY LOGIC SUITABILITY (30% Weight)
def fuzzy_triangular(x: float, a: float, b: float, c: float) -> float:
    """Calculates triangular fuzzy membership degree μ(x) in [0, 1]."""
    if x <= a or x >= c:
        return 0.0
    if a < x <= b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def fuzzy_trapezoidal(x: float, a: float, b: float, c: float, d: float) -> float:
    """Calculates trapezoidal fuzzy membership degree μ(x) in [0, 1]."""
    if x <= a or x >= d:
        return 0.0
    if a < x < b:
        return (x - a) / (b - a)
    if b <= x <= c:
        return 1.0
    return (d - x) / (d - c)


def evaluate_fuzzy_suitability(profile: StudentProfile, career_name: str) -> float:
    """Mamdani-style Fuzzy Inference System incorporating Soft-Skill Archetype Compatibility."""
    gpa = profile.gpa

    # Fuzzify GPA
    mu_gpa_low = fuzzy_trapezoidal(gpa, 0.0, 0.0, 2.4, 2.9)
    mu_gpa_med = fuzzy_triangular(gpa, 2.6, 3.2, 3.6)
    mu_gpa_high = fuzzy_trapezoidal(gpa, 3.3, 3.7, 4.0, 4.0)

    # Compute Technical Mastery Degree
    benchmarks = CAREER_UNIVERSE[career_name]["benchmark_skills"]
    unified = profile.get_unified_skill_dict()

    skill_ratios = []
    for skill_k, req_val in benchmarks.items():
        curr_val = unified.get(skill_k, 0)
        ratio = min(curr_val / req_val, 1.2) if req_val > 0 else 1.0
        skill_ratios.append(ratio)

    avg_skill_ratio = float(np.mean(skill_ratios)) if skill_ratios else 0.7

    mu_tech_weak = fuzzy_trapezoidal(avg_skill_ratio, 0.0, 0.0, 0.5, 0.7)
    mu_tech_mod = fuzzy_triangular(avg_skill_ratio, 0.6, 0.85, 1.05)
    mu_tech_strong = fuzzy_trapezoidal(avg_skill_ratio, 0.9, 1.1, 1.5, 1.5)

    # Fuzzify Soft Skill Archetype Overlap
    req_softs = CAREER_UNIVERSE[career_name].get("required_soft_skills", [])
    matched_softs = [s for s in req_softs if s in profile.soft_skills]
    soft_ratio = len(matched_softs) / len(req_softs) if req_softs else 0.5

    mu_soft_high = fuzzy_trapezoidal(soft_ratio, 0.5, 0.75, 1.0, 1.0)
    mu_soft_mod = fuzzy_triangular(soft_ratio, 0.25, 0.5, 0.75)
    mu_soft_low = fuzzy_trapezoidal(soft_ratio, 0.0, 0.0, 0.25, 0.45)

    # Fuzzify Work-Style Compatibility
    is_preferred_style = profile.work_style in CAREER_UNIVERSE[career_name]["target_workstyles"]
    mu_style_high = 1.0 if is_preferred_style else 0.3

    # Academic Year Readiness
    year_map = {"1st Year": 0.4, "2nd Year": 0.65, "3rd Year": 0.85, "4th Year": 1.0}
    year_factor = year_map.get(profile.year, 0.75)

    # Mamdani Fuzzy Rules:
    w1 = min(mu_tech_strong, mu_gpa_high, mu_soft_high)  # High Fit
    w2 = min(mu_tech_strong, mu_soft_high)                # Strong soft + tech
    w3 = min(mu_tech_mod, max(mu_gpa_med, mu_gpa_high), mu_soft_mod)
    w4 = min(mu_tech_mod, mu_style_high)
    w5 = min(mu_tech_weak, mu_gpa_low, mu_soft_low)
    w6 = 0.15

    consequents = [0.96, 0.88, 0.72, 0.60, 0.25, 0.50]
    weights = [w1, w2, w3, w4, w5, w6]

    total_weight = sum(weights)
    if total_weight > 0:
        crisp_suitability = sum(w * c for w, c in zip(weights, consequents)) / total_weight
    else:
        crisp_suitability = 0.5

    final_fuzzy_score = float(crisp_suitability * (0.85 + 0.15 * year_factor))
    return min(max(final_fuzzy_score, 0.0), 1.0)


# 6.3 ENGINE C: SUPERVISED ML RANDOM FOREST CLASSIFIER (40% Weight)
def build_feature_vector(profile: StudentProfile) -> np.ndarray:
    """Encodes student profile into a fixed-order numeric feature vector."""
    unified = profile.get_unified_skill_dict()
    vec = [
        profile.gpa / 4.0,
        {"1st Year": 0.25, "2nd Year": 0.5, "3rd Year": 0.75, "4th Year": 1.0}.get(profile.year, 0.5),
        unified["DSA"] / 100.0,
        unified["OOP"] / 100.0,
        unified["DBMS"] / 100.0,
        unified["OS_Networks"] / 100.0,
        unified["SE_Principles"] / 100.0,
        unified["Math_Stats"] / 100.0,
        unified["Python"] / 5.0,
        unified["Java_CPP"] / 5.0,
        unified["SQL"] / 5.0,
        unified["WebStack"] / 5.0,
        unified["CloudDocker"] / 5.0,
        unified["ML_AI"] / 5.0,
        unified["MobileDev"] / 5.0,
        unified["Cybersecurity"] / 5.0,
        1.0 if profile.has_internship else 0.0,
        min(profile.projects_count / 5.0, 1.0),
        {"Technical Specialist": 1.0, "Consulting / Management": 0.5, "R&D / Creative": 0.8}.get(profile.work_style, 0.5),
        min(len(profile.soft_skills) / 6.0, 1.0),
    ]
    return np.array(vec, dtype=np.float32)


@st.cache_resource
def get_default_calibrated_ml_classifier():
    """Initializes and trains the baseline Random Forest model on the standard benchmark dataset."""
    benchmark_df = generate_benchmark_dataset(samples_per_class=45)
    train_res = train_custom_classifier(benchmark_df, n_estimators=80, max_depth=10)
    if "model" in train_res:
        return train_res["model"], train_res["careers"]
    return None, list(CAREER_UNIVERSE.keys())


def evaluate_ml_model(profile: StudentProfile) -> Dict[str, float]:
    """Generates class probability distribution across all 9 careers via active ML model."""
    if "active_ml_model" in st.session_state and st.session_state["active_ml_model"] is not None:
        clf = st.session_state["active_ml_model"]
        careers = st.session_state.get("active_ml_careers", list(CAREER_UNIVERSE.keys()))
    else:
        clf, careers = get_default_calibrated_ml_classifier()

    feature_vec = build_feature_vector(profile)

    if clf is not None:
        try:
            probabilities = clf.predict_proba(feature_vec.reshape(1, -1))[0]
            return {career: float(prob) for career, prob in zip(careers, probabilities)}
        except Exception:
            pass

    # Resilient Softmax Fallback
    unified = profile.get_unified_skill_dict()
    logits = []
    for career in careers:
        bench = CAREER_UNIVERSE[career]["benchmark_skills"]
        diffs = []
        for k, v in bench.items():
            curr = unified.get(k, 0)
            diffs.append(1.0 - abs(curr - v) / (100.0 if v > 5 else 5.0))
        logits.append(np.mean(diffs) * 4.0)

    exp_l = np.exp(logits - np.max(logits))
    probs = exp_l / np.sum(exp_l)
    return {career: float(p) for career, p in zip(careers, probs)}


# 6.4 HYBRID SCORE FUSION
def run_hybrid_inference(profile: StudentProfile) -> List[Dict[str, Any]]:
    """Combines Rule-based (30%), Fuzzy logic (30%), and ML probabilities (40%)."""
    ml_probs = evaluate_ml_model(profile)
    results = []

    for career_name, career_info in CAREER_UNIVERSE.items():
        rule_score, rules_fired = evaluate_rule_engine(profile, career_name)
        fuzzy_score = evaluate_fuzzy_suitability(profile, career_name)
        ml_score = ml_probs.get(career_name, 0.1)

        ml_scaled = min(ml_score * 2.2, 1.0)

        # Weighted Score Fusion
        final_score = (0.30 * rule_score) + (0.30 * fuzzy_score) + (0.40 * ml_scaled)
        final_score = min(max(final_score, 0.0), 0.99)

        if final_score >= 0.78:
            confidence = "High (⚡ Strong Fit)"
            conf_color = "#34D399"
        elif final_score >= 0.62:
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
            "ml_score": round(ml_scaled * 100, 1),
            "rules_fired": rules_fired,
            "matched_soft_skills": matched_softs,
            "missing_soft_skills": missing_softs,
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results


# -----------------------------------------------------------------------------
# 7. EXPLAINABILITY (SHAP PROXY) & SKILL GAP ENGINE
# -----------------------------------------------------------------------------
def calculate_shap_proxy_deltas(profile: StudentProfile, top_career: str) -> pd.DataFrame:
    """Computes transparent feature contributions (SHAP TreeExplainer proxy)."""
    career_info = CAREER_UNIVERSE[top_career]
    bench = career_info["benchmark_skills"]
    unified = profile.get_unified_skill_dict()

    feature_deltas = []
    for skill_name, req_val in bench.items():
        curr = unified.get(skill_name, 0)
        scale = 100.0 if req_val > 5 else 5.0
        delta = (curr - req_val) / scale * 16.0

        label_map = {
            "DSA": "Data Structures & Alg",
            "OOP": "OOP Principles",
            "DBMS": "DBMS & Storage",
            "OS_Networks": "OS & Networks",
            "SE_Principles": "Software Eng Principles",
            "Math_Stats": "Math & Statistics",
            "Python": "Python Fluency",
            "Java_CPP": "Java / C++ Fluency",
            "SQL": "SQL / Querying",
            "WebStack": "Web Engineering",
            "CloudDocker": "Cloud & Docker",
            "ML_AI": "ML & AI Frameworks",
            "MobileDev": "Mobile Dev",
            "Cybersecurity": "Cybersecurity",
        }
        feature_deltas.append({
            "Feature": label_map.get(skill_name, skill_name),
            "Delta": round(delta, 1),
            "Color": "#00F0FF" if delta >= 0 else "#FB7185",
        })

    # Add GPA and Internship drivers
    gpa_delta = (profile.gpa - 3.2) * 12.0
    feature_deltas.append({
        "Feature": "Cumulative GPA",
        "Delta": round(gpa_delta, 1),
        "Color": "#00F0FF" if gpa_delta >= 0 else "#FB7185",
    })

    intern_delta = 7.5 if profile.has_internship else -4.0
    feature_deltas.append({
        "Feature": "Internship Experience",
        "Delta": round(intern_delta, 1),
        "Color": "#00F0FF" if intern_delta >= 0 else "#FB7185",
    })

    # Soft Skill Archetype Alignment Driver
    req_softs = career_info.get("required_soft_skills", [])
    matched_softs = [s for s in req_softs if s in profile.soft_skills]
    soft_ratio = len(matched_softs) / len(req_softs) if req_softs else 0.5
    soft_delta = (soft_ratio - 0.45) * 15.0
    feature_deltas.append({
        "Feature": "Archetype Soft Skills",
        "Delta": round(soft_delta, 1),
        "Color": "#00F0FF" if soft_delta >= 0 else "#FB7185",
    })

    df = pd.DataFrame(feature_deltas)
    df = df.sort_values(by="Delta", key=abs, ascending=False).head(7)
    df = df.sort_values(by="Delta", ascending=True)
    return df


def calculate_skill_gaps(profile: StudentProfile, top_career: str) -> List[Dict[str, Any]]:
    """Calculates missing competencies (Target Requisites minus Current Proficiency)."""
    career_info = CAREER_UNIVERSE[top_career]
    bench = career_info["benchmark_skills"]
    unified = profile.get_unified_skill_dict()
    gaps = []

    label_map = {
        "DSA": "Data Structures & Algorithms",
        "OOP": "Object-Oriented Programming",
        "DBMS": "Database Management Systems",
        "OS_Networks": "OS & Computer Networks",
        "SE_Principles": "Software Engineering Principles",
        "Math_Stats": "Mathematics & Statistics",
        "Python": "Python Programming",
        "Java_CPP": "Java / C++ Systems",
        "SQL": "SQL & Relational Querying",
        "WebStack": "Modern Web Stack",
        "CloudDocker": "Cloud Architecture & Docker",
        "ML_AI": "Machine Learning & AI",
        "MobileDev": "Mobile Development",
        "Cybersecurity": "Cybersecurity Defense",
    }

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
                "skill_name": label_map.get(skill_key, skill_key),
                "current": f"{int(curr_val)}/100" if req_val > 5 else f"{int(curr_val)}/5",
                "target": f"{int(req_val)}/100" if req_val > 5 else f"{int(req_val)}/5",
                "deficit": round(deficit, 1),
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
                "urgency": "Medium Priority",
                "badge_class": "pf-badge-med",
            })

    gaps.sort(key=lambda x: x["deficit"], reverse=True)
    return gaps


def recommend_certifications(
    gaps: List[Dict[str, Any]],
    target_career: str = "Software Engineer",
    student_year: str = "3rd Year",
) -> List[Dict[str, Any]]:
    """Computes Gap-Weighted Cosine Similarity with Domain Pruning and Academic Tier Matching."""
    catalog = load_certification_catalog()

    CAREER_DOMAINS = {
        "Software Engineer": {"Software Engineering", "Web Development", "Cloud & DevOps", "Core Systems"},
        "Data Scientist": {"Data & Analytics", "Artificial Intelligence", "Database & Big Data"},
        "AI Engineer": {"Artificial Intelligence", "Data & Analytics", "Software Engineering"},
        "Cloud Architect": {"Cloud & DevOps", "Infrastructure & Networking", "Software Engineering", "Cybersecurity & Defense"},
        "UX Designer": {"UI/UX Design", "Web Development", "Product & Media"},
        "IT Business Analyst": {"Agile & Project Management", "Management & Consulting", "Data & Analytics", "Software Engineering"},
        "Game Developer": {"Game Development & Graphics", "C++ & Systems", "Software Engineering"},
        "CAD-CAM Engineer": {"CAD/CAM & Mechanical", "Engineering Architecture"},
        "Cybersecurity Specialist": {"Cybersecurity & Defense", "Infrastructure & Networking", "Cloud & DevOps"},
    }

    allowed_domains = CAREER_DOMAINS.get(target_career, {"Software Engineering", "Cloud & DevOps"})

    urgency_weights = {
        "High Urgency": 3.0,
        "Medium Priority": 2.0,
        "Low / Minor": 1.0,
    }

    weighted_gaps: Dict[str, float] = {}
    total_gap_weight = 0.0

    for g in gaps:
        sk = g["skill_key"]
        w = urgency_weights.get(g.get("urgency", "Medium Priority"), 2.0)
        def_val = float(g.get("deficit", 1.0))
        weighted_val = def_val * w
        weighted_gaps[sk] = weighted_val
        total_gap_weight += weighted_val

    mag_gap = math.sqrt(sum(v ** 2 for v in weighted_gaps.values())) if weighted_gaps else 0.0

    scored_certs = []

    for cert in catalog:
        cert_domains = set(cert.get("domains", []))
        cert_keys = cert.get("skill_keys", cert.get("skills", []))
        cert_vector = cert.get("vector", {})
        if not cert_vector:
            cert_vector = {k: 1.0 for k in cert_keys}

        domain_overlap = cert_domains.intersection(allowed_domains)
        if not domain_overlap:
            domain_multiplier = 0.10
        else:
            domain_multiplier = 1.30 + (0.15 * len(domain_overlap))

        level_str = cert.get("level", "").lower()
        if student_year in ["1st Year", "2nd Year"]:
            if "foundational" in level_str or "beginner" in level_str:
                tier_multiplier = 1.25
            elif "intermediate" in level_str or "associate" in level_str:
                tier_multiplier = 1.05
            else:
                tier_multiplier = 0.70
        else:
            if "intermediate" in level_str or "associate" in level_str:
                tier_multiplier = 1.25
            elif "advanced" in level_str or "professional" in level_str:
                tier_multiplier = 1.20
            else:
                tier_multiplier = 0.85

        dot_product = 0.0
        covered_gap_weight = 0.0
        matched_skill_names = []

        for k, cert_w in cert_vector.items():
            if k in weighted_gaps:
                gap_w = weighted_gaps[k]
                dot_product += gap_w * cert_w
                covered_gap_weight += gap_w * cert_w

                for g in gaps:
                    if g["skill_key"] == k and g["skill_name"] not in matched_skill_names:
                        matched_skill_names.append(g["skill_name"])

        mag_cert = math.sqrt(sum(v ** 2 for v in cert_vector.values())) if cert_vector else 1.0

        if mag_gap > 0 and mag_cert > 0 and dot_product > 0:
            cosine_sim = dot_product / (mag_gap * mag_cert)
        else:
            cosine_sim = 0.05

        final_ranking_score = cosine_sim * domain_multiplier * tier_multiplier

        if total_gap_weight > 0 and covered_gap_weight > 0:
            raw_coverage = (covered_gap_weight / total_gap_weight) * 100.0
            coverage_pct = int(min(98, max(18, round(raw_coverage * (1.1 if domain_overlap else 0.5)))))
        else:
            coverage_pct = 20 if domain_overlap else 10

        if not matched_skill_names:
            matched_skill_names = cert.get("skills", [])[:2]

        scored_certs.append({
            "id": cert.get("id", ""),
            "title": cert.get("title", cert.get("name", "Certification")),
            "name": cert.get("title", cert.get("name", "Certification")),
            "issuer": cert.get("issuer", "Industry Partner"),
            "level": cert.get("level", "Intermediate"),
            "skills": cert.get("skills", []),
            "description": cert.get("description", cert.get("url_hint", "")),
            "official_url": cert.get("official_url", "https://aws.amazon.com/certification/"),
            "prep_url": cert.get("prep_url", cert.get("official_url", "https://aws.amazon.com/certification/")),
            "exam_guide_url": cert.get("exam_guide_url", cert.get("official_url", "")),
            "similarity": cosine_sim,
            "final_score": final_ranking_score,
            "coverage_pct": coverage_pct,
            "covered_skills": matched_skill_names[:3],
        })

    scored_certs.sort(key=lambda x: x["final_score"], reverse=True)
    return scored_certs[:3]


def generate_executive_narrative(profile: StudentProfile, top_career: Dict[str, Any], gaps: List[Dict[str, Any]]) -> str:
    """Produces a crisp executive guidance narrative."""
    role_title = top_career["title"]
    match_pct = top_career["match_pct"]
    archetype = top_career.get("archetype", "Engineering & Architecture")
    matched_softs = top_career.get("matched_soft_skills", [])

    best_module = max(profile.core_modules.items(), key=lambda x: x[1])

    soft_note = f"and natural strengths in <b style='color: #00F0FF;'>{matched_softs[0]}</b>" if matched_softs else "and analytical work habits"

    if gaps:
        primary_gap_name = gaps[0]["skill_name"]
        gap_advice = f"Targeting competency development in <b style='color: #FB7185;'>{primary_gap_name}</b> while sustaining your <b style='color: #38BDF8;'>{profile.gpa:.2f} GPA</b> will position you as a top tier candidate in the <b style='color: #C084FC;'>{archetype}</b> cluster."
    else:
        gap_advice = f"Your profile exhibits rounded technical depth and soft skill harmony aligned with the <b style='color: #C084FC;'>{archetype}</b> archetype."

    sentence1 = (
        f"Based on multi-engine evaluation, your performance in <b style='color: #38BDF8;'>{best_module[0]}</b> "
        f"{soft_note} demonstrates an exceptional <b style='color: #00F0FF;'>{match_pct}% alignment</b> with the <b style='color: #F8FAFC;'>{role_title}</b> career path."
    )
    return f"{sentence1} {gap_advice}"


# -----------------------------------------------------------------------------
# 8. UI HELPER UTILITIES & PRESET MANAGEMENT
# -----------------------------------------------------------------------------

ALL_PEOPLE_SKILLS = [
    "Communicating & Articulating",
    "Team Leadership & Delegation",
    "Negotiation & Persuasion",
    "Client Facing & Presentation",
    "Mentoring & Teaching",
    "Active Listening & Empathy",
]
ALL_IDEAS_SKILLS = [
    "Problem Solving & Logic",
    "Creative & Visual Design",
    "Research & Investigation",
    "Storytelling & Conceptualizing",
    "Innovation & Prototyping",
    "Technical & Content Writing",
]
ALL_DATA_SKILLS = [
    "Analytical & Critical Thinking",
    "Detail-Oriented & Precision",
    "Planning & Task Organizing",
    "Time & Deadline Management",
    "Quantitative & Mathematical",
    "Monitoring & Evaluation",
]
ALL_EXEC_SKILLS = [
    "Hardware & System Troubleshooting",
    "Hands-on Prototyping",
    "Debugging & Root-Cause Analysis",
    "Working Under Pressure / Incidents",
    "Process Optimization",
    "Process & Protocol Adherence",
]
ALL_24_SOFT_SKILLS = ALL_PEOPLE_SKILLS + ALL_IDEAS_SKILLS + ALL_DATA_SKILLS + ALL_EXEC_SKILLS


def get_score_status(score: float) -> Tuple[str, str, str]:
    """
    Returns (status_label, badge_color_hex, badge_bg_rgba) for a given score (0-100).
    0-49: Needs Improvement (#FB7185)
    50-69: Developing (#FBBF24)
    70-84: Strong (#38BDF8)
    85-100: Excellent (#34D399)
    """
    if score >= 85:
        return "Excellent", "#34D399", "rgba(52, 211, 153, 0.18)"
    elif score >= 70:
        return "Strong", "#38BDF8", "rgba(56, 189, 248, 0.18)"
    elif score >= 50:
        return "Developing", "#FBBF24", "rgba(251, 191, 36, 0.18)"
    else:
        return "Needs Improvement", "#FB7185", "rgba(251, 113, 133, 0.18)"


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
    st.session_state["preset_gpa"] = gpa
    st.session_state["preset_year"] = year
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
    sh = calculate_shap_proxy_deltas(prof, top["career"])
    gp = calculate_skill_gaps(prof, top["career"])
    rc = recommend_certifications(gp, target_career=top["career"], student_year=year)
    nr = generate_executive_narrative(prof, top, gp)
    st.session_state["evaluation_data"] = {
        "profile": prof,
        "results": res,
        "top_rec": top,
        "shap_df": sh,
        "gaps": gp,
        "certs": rc,
        "narrative": nr,
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
    def_sh = calculate_shap_proxy_deltas(def_prof, def_top["career"])
    def_gp = calculate_skill_gaps(def_prof, def_top["career"])
    def_rc = recommend_certifications(def_gp, target_career=def_top["career"], student_year=def_year)
    def_nr = generate_executive_narrative(def_prof, def_top, def_gp)
    st.session_state["evaluation_data"] = {
        "profile": def_prof,
        "results": def_res,
        "top_rec": def_top,
        "shap_df": def_sh,
        "gaps": def_gp,
        "certs": def_rc,
        "narrative": def_nr,
    }


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

# Initialize active navigation state if not set
if "active_nav_tab" not in st.session_state:
    st.session_state["active_nav_tab"] = "① Profile"

# 2. Top Stepper Navigation Header (Right below heading)
nav_steps = [
    ("① Profile", "Profile Intake & Strengths", "👤"),
    ("② Career Roadmap", "AI Career Guidance & Roadmap", "🧭"),
    ("③ Model Studio", "Dataset & Model Studio", "🔬"),
    ("④ History", "Evaluation History & Logs", "📋"),
]

nav_cols = st.columns(4)
for idx, (tab_id, tab_label, tab_icon) in enumerate(nav_steps):
    with nav_cols[idx]:
        is_active = (st.session_state["active_nav_tab"] == tab_id)
        if st.button(
            f"{tab_icon} {tab_id} • {tab_label}",
            key=f"nav_step_btn_{idx}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state["active_nav_tab"] = tab_id
            st.rerun()

current_tab = st.session_state["active_nav_tab"]


# -----------------------------------------------------------------------------
# TAB 1: ① PROFILE (CAREER PROFILE INTAKE & COMPETENCIES)
# -----------------------------------------------------------------------------
if current_tab == "① Profile":
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
            value=float(st.session_state.get("preset_gpa", 3.40)),
            step=0.01,
            key="preset_gpa",
            help="Your overall university cumulative GPA (0.00 to 4.00).",
        )

    with col_year:
        year_list = ["1st Year", "2nd Year", "3rd Year", "4th Year"]
        def_year = st.session_state.get("preset_year", "3rd Year")
        in_year = st.selectbox(
            "📅 Academic Year Level",
            year_list,
            index=year_list.index(def_year) if def_year in year_list else 2,
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
            value=st.session_state.get("slide_dsa", 82),
            key="slide_dsa",
            help="Algorithmic complexity, trees, graphs, sorting, and dynamic programming.",
        )
        m_dbms = st.slider(
            "🗄️ 3. Database Management Systems",
            min_value=0,
            max_value=100,
            value=st.session_state.get("slide_dbms", 75),
            key="slide_dbms",
            help="Relational modeling, SQL optimization, indexing, ACID transactions.",
        )
        m_se = st.slider(
            "⚙️ 5. Software Engineering Principles",
            min_value=0,
            max_value=100,
            value=st.session_state.get("slide_se", 80),
            key="slide_se",
            help="Design patterns, modular architecture, agile lifecycle, CI/CD.",
        )

    with col_m2:
        m_oop = st.slider(
            "💻 2. Object-Oriented Programming",
            min_value=0,
            max_value=100,
            value=st.session_state.get("slide_oop", 78),
            key="slide_oop",
            help="Encapsulation, inheritance, polymorphism, design SOLID principles.",
        )
        m_os = st.slider(
            "🌐 4. Operating Systems & Networks",
            min_value=0,
            max_value=100,
            value=st.session_state.get("slide_os", 72),
            key="slide_os",
            help="Process scheduling, memory paging, socket networking, TCP/IP protocols.",
        )
        m_math = st.slider(
            "📐 6. Mathematics & Statistics",
            min_value=0,
            max_value=100,
            value=st.session_state.get("slide_math", 76),
            key="slide_math",
            help="Discrete math, linear algebra, calculus, probability & inferential statistics.",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Section 2: Specialized University Electives (Previous style)
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
        default=st.session_state.get("selected_electives_input", ["AI & Machine Learning", "Cloud Computing"]),
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
            with e_cols[i % 2]:
                slider_key = f"elec_{elec_name.replace(' ', '_').replace('&', 'and')}"
                elec_score = st.slider(
                    f"📖 {elec_name}",
                    min_value=0,
                    max_value=100,
                    value=st.session_state.get(slider_key, 80),
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

    tc1, tc2, tc3, tc4 = st.columns(4)
    with tc1:
        st.markdown("🐍 **Python**")
        t_python = st.select_slider("Python", options=[1, 2, 3, 4, 5], value=st.session_state.get("sl_py", 4), key="sl_py", label_visibility="collapsed")
        st.caption(f"Level: `{lvl_map[t_python]}`")

        st.markdown("☕ **Java / C++**")
        t_jcpp = st.select_slider("Java / C++", options=[1, 2, 3, 4, 5], value=st.session_state.get("sl_jcpp", 3), key="sl_jcpp", label_visibility="collapsed")
        st.caption(f"Level: `{lvl_map[t_jcpp]}`")

    with tc2:
        st.markdown("🗄️ **SQL / Databases**")
        t_sql = st.select_slider("SQL", options=[1, 2, 3, 4, 5], value=st.session_state.get("sl_sql", 4), key="sl_sql", label_visibility="collapsed")
        st.caption(f"Level: `{lvl_map[t_sql]}`")

        st.markdown("🌐 **Web Stack (React/Node)**")
        t_web = st.select_slider("Web", options=[1, 2, 3, 4, 5], value=st.session_state.get("sl_web", 3), key="sl_web", label_visibility="collapsed")
        st.caption(f"Level: `{lvl_map[t_web]}`")

    with tc3:
        st.markdown("☁️ **Cloud / Docker**")
        t_cloud = st.select_slider("Cloud", options=[1, 2, 3, 4, 5], value=st.session_state.get("sl_cloud", 3), key="sl_cloud", label_visibility="collapsed")
        st.caption(f"Level: `{lvl_map[t_cloud]}`")

        st.markdown("🧠 **ML / AI Frameworks**")
        t_ml = st.select_slider("ML", options=[1, 2, 3, 4, 5], value=st.session_state.get("sl_ml", 4), key="sl_ml", label_visibility="collapsed")
        st.caption(f"Level: `{lvl_map[t_ml]}`")

    with tc4:
        st.markdown("📱 **Mobile Dev**")
        t_mob = st.select_slider("Mobile", options=[1, 2, 3, 4, 5], value=st.session_state.get("sl_mob", 2), key="sl_mob", label_visibility="collapsed")
        st.caption(f"Level: `{lvl_map[t_mob]}`")

        st.markdown("🛡️ **Cybersecurity**")
        t_sec = st.select_slider("Security", options=[1, 2, 3, 4, 5], value=st.session_state.get("sl_sec", 2), key="sl_sec", label_visibility="collapsed")
        st.caption(f"Level: `{lvl_map[t_sec]}`")

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

    col_people, col_ideas, col_data, col_execution = st.columns(4)
    selected_soft_skills = []
    default_checked_softs = {"Problem Solving & Logic", "Analytical & Critical Thinking", "Detail-Oriented & Precision"}

    with col_people:
        st.markdown("<p style='font-size: 0.88rem; font-weight: 700; color: #38BDF8; margin-bottom: 0.5rem;'>👥 People & Leadership</p>", unsafe_allow_html=True)
        for skill in ALL_PEOPLE_SKILLS:
            chk_val = st.session_state.get(f"soft_{skill}", skill in default_checked_softs)
            if st.checkbox(skill, value=chk_val, key=f"soft_{skill}"):
                selected_soft_skills.append(skill)

    with col_ideas:
        st.markdown("<p style='font-size: 0.88rem; font-weight: 700; color: #C084FC; margin-bottom: 0.5rem;'>💡 Ideas & Innovation</p>", unsafe_allow_html=True)
        for skill in ALL_IDEAS_SKILLS:
            chk_val = st.session_state.get(f"soft_{skill}", skill in default_checked_softs)
            if st.checkbox(skill, value=chk_val, key=f"soft_{skill}"):
                selected_soft_skills.append(skill)

    with col_data:
        st.markdown("<p style='font-size: 0.88rem; font-weight: 700; color: #34D399; margin-bottom: 0.5rem;'>📊 Data & Systems</p>", unsafe_allow_html=True)
        for skill in ALL_DATA_SKILLS:
            chk_val = st.session_state.get(f"soft_{skill}", skill in default_checked_softs)
            if st.checkbox(skill, value=chk_val, key=f"soft_{skill}"):
                selected_soft_skills.append(skill)

    with col_execution:
        st.markdown("<p style='font-size: 0.88rem; font-weight: 700; color: #FBBF24; margin-bottom: 0.5rem;'>🛠️ Execution & Practical</p>", unsafe_allow_html=True)
        for skill in ALL_EXEC_SKILLS:
            chk_val = st.session_state.get(f"soft_{skill}", skill in default_checked_softs)
            if st.checkbox(skill, value=chk_val, key=f"soft_{skill}"):
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
        style_list = ["Technical Specialist", "Consulting / Management", "R&D / Creative"]
        def_style = st.session_state.get("work_style_input", "Technical Specialist")
        in_work_style = st.radio(
            "💼 Work Style Preference",
            style_list,
            index=style_list.index(def_style) if def_style in style_list else 0,
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
            default=st.session_state.get("domains_input", ["Artificial Intelligence & ML", "Enterprise Cloud & DevOps"]),
            key="domains_input",
            help="Industries and software engineering subfields you are most interested in.",
        )

    with adv_col2:
        in_internship = st.toggle(
            "💼 Completed University / Industry Internship",
            value=st.session_state.get("intern_toggle", True),
            key="intern_toggle",
            help="Toggle if you have completed an internship or formal work experience.",
        )
        in_projects = st.number_input(
            "🚀 Completed Technical Projects",
            min_value=0,
            max_value=20,
            value=st.session_state.get("projects_input", 3),
            key="projects_input",
            help="Number of major academic or personal projects in your portfolio.",
        )
        in_certs = st.text_input(
            "📜 Existing Certifications (comma separated)",
            value=st.session_state.get("certs_input", "AWS Cloud Practitioner"),
            key="certs_input",
            help="List any credentials you already hold.",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # 6. Profile Snapshot Summary Gauge Cards
    avg_core = (m_dsa + m_oop + m_dbms + m_os + m_se + m_math) / 6.0
    avg_tech = ((t_python + t_jcpp + t_sql + t_web + t_cloud + t_ml + t_mob + t_sec) / 40.0) * 100.0

    st.markdown(
        "<p style='font-size: clamp(0.85rem, 0.95vw, 1.05rem); font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #38BDF8; margin-bottom: 0.75rem;'>📊 YOUR PROFILE SNAPSHOT</p>",
        unsafe_allow_html=True,
    )
    snap1, snap2, snap3, snap4 = st.columns(4)

    with snap1:
        st.markdown(
            f"""
            <div class="pf-snapshot-card">
                <div class="pf-snapshot-title">
                    <span>💻 Technical Skills</span>
                    <span style="color: #00E5FF; font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem;">{int(avg_tech)}%</span>
                </div>
                <div class="pf-progress-track">
                    <div class="pf-progress-fill" style="width: {int(avg_tech)}%; background: linear-gradient(90deg, #0284C7, #00E5FF);"></div>
                </div>
                <div style="font-size: 0.78rem; color: #94A3B8;">8 Hands-on Proficiencies</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with snap2:
        st.markdown(
            f"""
            <div class="pf-snapshot-card">
                <div class="pf-snapshot-title">
                    <span>🎓 Academic Performance</span>
                    <span style="color: #34D399; font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem;">{int(avg_core)}%</span>
                </div>
                <div class="pf-progress-track">
                    <div class="pf-progress-fill" style="width: {int(avg_core)}%; background: linear-gradient(90deg, #059669, #34D399);"></div>
                </div>
                <div style="font-size: 0.78rem; color: #94A3B8;">GPA {in_gpa:.2f} • 6 Core Modules</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with snap3:
        soft_pct = min(100, int((len(selected_soft_skills) / 6.0) * 100))
        st.markdown(
            f"""
            <div class="pf-snapshot-card">
                <div class="pf-snapshot-title">
                    <span>🤝 Soft Skills</span>
                    <span style="color: #C084FC; font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem;">{len(selected_soft_skills)} Active</span>
                </div>
                <div class="pf-progress-track">
                    <div class="pf-progress-fill" style="width: {soft_pct}%; background: linear-gradient(90deg, #7C3AED, #C084FC);"></div>
                </div>
                <div style="font-size: 0.78rem; color: #94A3B8;">4 Pillars Configured</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with snap4:
        st.markdown(
            f"""
            <div class="pf-snapshot-card">
                <div class="pf-snapshot-title">
                    <span>🚀 Projects / Experience</span>
                    <span style="color: #FBBF24; font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem;">{in_projects} Projects</span>
                </div>
                <div class="pf-progress-track">
                    <div class="pf-progress-fill" style="width: {min(100, in_projects * 20)}%; background: linear-gradient(90deg, #D97706, #FBBF24);"></div>
                </div>
                <div style="font-size: 0.78rem; color: #94A3B8;">Verified Portfolio Builds</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 7. Primary Call to Action Section
    completeness = 20
    if in_gpa > 0:
        completeness += 15
    if avg_core > 0:
        completeness += 25
    if avg_tech > 0:
        completeness += 20
    if len(selected_soft_skills) >= 2:
        completeness += 10
    if in_projects > 0:
        completeness += 10
    completeness = min(100, completeness)

    st.markdown(
        f"""
        <div class="pf-cta-card">
            <div class="pf-cta-title">✨ READY TO DISCOVER YOUR PATH?</div>
            <div class="pf-cta-sub">Your profile is <b style="color: #00E5FF;">{completeness}% complete</b>. Our AI engines are ready to evaluate your optimal career trajectory.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    btn_submit = st.button("🚀 Analyze My Career & Generate Roadmap", type="primary", use_container_width=True)

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

        inference_results = run_hybrid_inference(profile)
        top_rec = inference_results[0]

        shap_df = calculate_shap_proxy_deltas(profile, top_rec["career"])
        gaps = calculate_skill_gaps(profile, top_rec["career"])
        rec_certs = recommend_certifications(gaps, target_career=top_rec["career"], student_year=in_year)
        narrative = generate_executive_narrative(profile, top_rec, gaps)

        top_driver = shap_df[shap_df["Delta"] > 0].iloc[-1]["Feature"] if not shap_df[shap_df["Delta"] > 0].empty else "Academic Foundation"
        critical_gap = gaps[0]["skill_name"] if gaps else "None (Target Met)"

        save_student_record(
            gpa=in_gpa,
            academic_year=in_year,
            top_career=top_rec["title"],
            match_score=top_rec["final_score"],
            confidence=top_rec["confidence"],
            work_style=in_work_style,
            top_driver=top_driver,
            critical_gap=critical_gap,
        )

        st.session_state["evaluation_data"] = {
            "profile": profile,
            "results": inference_results,
            "top_rec": top_rec,
            "shap_df": shap_df,
            "gaps": gaps,
            "certs": rec_certs,
            "narrative": narrative,
        }

        st.session_state["active_nav_tab"] = "② Career Roadmap"
        st.toast("⚡ Multi-Engine Evaluation complete! Redirecting to Career Roadmap...", icon="🚀")
        st.rerun()

    # Bottom Navigation for Tab 1
    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(56, 189, 248, 0.2); margin: 2rem 0 1.2rem 0;'>", unsafe_allow_html=True)
    p_nav1, p_nav2, p_nav3 = st.columns([1, 1, 1.3])
    with p_nav1:
        if st.button("🔬 Explore Model Studio", key="p_bot_studio", use_container_width=True):
            st.session_state["active_nav_tab"] = "③ Model Studio"
            st.rerun()
    with p_nav2:
        if st.button("📋 View Evaluation History", key="p_bot_hist", use_container_width=True):
            st.session_state["active_nav_tab"] = "④ History"
            st.rerun()
    with p_nav3:
        if st.button("Next: View Career Roadmap →", key="p_bot_next", type="primary", use_container_width=True):
            st.session_state["active_nav_tab"] = "② Career Roadmap"
            st.rerun()


# -----------------------------------------------------------------------------
# TAB 2: ② CAREER ROADMAP (AI RECOMMENDATIONS & GUIDANCE)
# -----------------------------------------------------------------------------
elif current_tab == "② Career Roadmap":
    if "evaluation_data" not in st.session_state:
        st.info("⚡ No evaluation results available yet. Please complete and submit the **① Profile** form to generate your personalized career roadmap.")
        if st.button("← Go to Profile Intake Form", type="primary", key="roadmap_goto_prof"):
            st.session_state["active_nav_tab"] = "① Profile"
            st.rerun()
    else:
        eval_data = st.session_state["evaluation_data"]
        top_rec = eval_data["top_rec"]
        results = eval_data["results"]
        shap_df = eval_data["shap_df"]
        gaps = eval_data["gaps"]
        certs = eval_data["certs"]
        narrative = eval_data["narrative"]

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
                    <div style="background: rgba(14, 23, 44, 0.65); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 10px; padding: 0.85rem 1.1rem; margin-top: 0.95rem; font-size: clamp(0.82rem, 0.9vw, 0.98rem); color: #FBBF24;">
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
                st.markdown("<div class='pf-card-title'>🔮 Feature Attribution Deltas (SHAP)</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='pf-card-desc'>Factors driving (+) or penalizing (-) the top match (<b style='color: #00E5FF;'>{top_rec['title']}</b>).</div>", unsafe_allow_html=True)

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

        with st.expander("🔍 View Transparent Rule-Based Firing Trace"):
            st.markdown(f"**Rules evaluated and triggered for {top_rec['title']}:**")
            for r in top_rec["rules_fired"]:
                st.markdown(f"- `{r}`")
            st.caption("Score Fusion: 30% Rule Logic + 30% Mamdani Fuzzy System + 40% Random Forest Probability.")

        st.markdown("<hr style='border: 0; border-top: 1px solid rgba(56, 189, 248, 0.2); margin: 1.8rem 0;'>", unsafe_allow_html=True)

        # Two-Column Split: Skill Gap Audit & Recommended Certifications
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

        # Bottom Navigation for Tab 2
        st.markdown("<hr style='border: 0; border-top: 1px solid rgba(56, 189, 248, 0.2); margin: 2rem 0 1.2rem 0;'>", unsafe_allow_html=True)
        r_nav1, r_nav2, r_nav3 = st.columns([1, 1, 1])
        with r_nav1:
            if st.button("← Back to Profile Intake", key="r_bot_back", use_container_width=True):
                st.session_state["active_nav_tab"] = "① Profile"
                st.rerun()
        with r_nav2:
            if st.button("🔬 Dataset & Model Studio →", key="r_bot_studio", use_container_width=True):
                st.session_state["active_nav_tab"] = "③ Model Studio"
                st.rerun()
        with r_nav3:
            if st.button("📋 View Evaluation History →", key="r_bot_hist", type="primary", use_container_width=True):
                st.session_state["active_nav_tab"] = "④ History"
                st.rerun()


# -----------------------------------------------------------------------------
# TAB 3: ③ MODEL STUDIO (DATASET INSPECTION & MODEL TRAINING)
# -----------------------------------------------------------------------------
elif current_tab == "③ Model Studio":
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
            <div class="pf-card-desc">Fit a Random Forest multi-class model on the active dataset to calibrate career predictions.</div>
        """,
        unsafe_allow_html=True,
    )

    tcol1, tcol2, tcol3 = st.columns(3)
    with tcol1:
        n_est = st.slider("Number of Estimators (Trees)", 20, 200, 80, step=10)
    with tcol2:
        m_dep = st.slider("Maximum Tree Depth", 3, 20, 10)
    with tcol3:
        t_split = st.slider("Test Split Ratio", 0.1, 0.4, 0.2, step=0.05)

    if st.button("⚡ Train Model on Active Dataset", type="primary", use_container_width=True):
        with st.spinner("Training Random Forest Classifier on dataset..."):
            train_results = train_custom_classifier(active_df, n_estimators=n_est, max_depth=m_dep, test_size=t_split)

            if "error" in train_results:
                st.error(f"Training failed: {train_results['error']}")
            else:
                st.session_state["active_ml_model"] = train_results["model"]
                st.session_state["active_ml_careers"] = train_results["careers"]
                st.session_state["train_metrics"] = train_results
                st.success(f"⚡ Model successfully trained! Test Accuracy: **{train_results['test_acc']*100:.1f}%** | Train Accuracy: **{train_results['train_acc']*100:.1f}%**")

    st.markdown("</div>", unsafe_allow_html=True)

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

        feat_df = metrics["feat_imp_df"]
        fig_feat = px.bar(
            feat_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Global Feature Importances (Random Forest Gini Impurity)",
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
    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(56, 189, 248, 0.2); margin: 2rem 0 1.2rem 0;'>", unsafe_allow_html=True)
    s_nav1, s_nav2, s_nav3 = st.columns([1, 1, 1])
    with s_nav1:
        if st.button("← Back to Career Roadmap", key="s_bot_back", use_container_width=True):
            st.session_state["active_nav_tab"] = "② Career Roadmap"
            st.rerun()
    with s_nav2:
        if st.button("👤 Edit Student Profile", key="s_bot_prof", use_container_width=True):
            st.session_state["active_nav_tab"] = "① Profile"
            st.rerun()
    with s_nav3:
        if st.button("📋 View Evaluation History →", key="s_bot_next", type="primary", use_container_width=True):
            st.session_state["active_nav_tab"] = "④ History"
            st.rerun()


# -----------------------------------------------------------------------------
# TAB 4: ④ HISTORY (AUDIT TRAIL & LOG STORAGE)
# -----------------------------------------------------------------------------
elif current_tab == "④ History":
    st.markdown(
        """
        <div class="pf-card-title">📋 Academic Advisor & Evaluation History</div>
        <div class="pf-card-desc">Structured audit log of student profile assessments persisted in local SQLite storage.</div>
        """,
        unsafe_allow_html=True,
    )

    records_df = fetch_all_records()

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
        st.dataframe(records_df, use_container_width=True, hide_index=True)

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
    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(56, 189, 248, 0.2); margin: 2rem 0 1.2rem 0;'>", unsafe_allow_html=True)
    h_nav1, h_nav2, h_nav3 = st.columns([1, 1, 1.2])
    with h_nav1:
        if st.button("← Back to Career Roadmap", key="h_bot_back", use_container_width=True):
            st.session_state["active_nav_tab"] = "② Career Roadmap"
            st.rerun()
    with h_nav2:
        if st.button("🔬 Open Model Studio", key="h_bot_studio", use_container_width=True):
            st.session_state["active_nav_tab"] = "③ Model Studio"
            st.rerun()
    with h_nav3:
        if st.button("🚀 Start New Profile Intake →", key="h_bot_new", type="primary", use_container_width=True):
            st.session_state["active_nav_tab"] = "① Profile"
            st.rerun()


