"""Explainable AI: per-student SHAP TreeExplainer attributions, with a benchmark-delta fallback."""

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.knowledge_base import CAREER_UNIVERSE
from src.ml_engine import FEATURE_NAMES, MODEL_LABELS, build_feature_vector, get_active_classifier
from src.profile import StudentProfile


def calculate_shap_proxy_deltas(profile: StudentProfile, top_career: str) -> pd.DataFrame:
    """Benchmark-delta feature contributions; fallback for calculate_shap_contributions when SHAP cannot run."""
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
            "Color": "#00F0FF" if delta >= 0 else "#F43F5E",
        })

    # Add GPA and Internship drivers
    gpa_delta = (profile.gpa - 3.2) * 12.0
    feature_deltas.append({
        "Feature": "Cumulative GPA",
        "Delta": round(gpa_delta, 1),
        "Color": "#00F0FF" if gpa_delta >= 0 else "#F43F5E",
    })

    intern_delta = 7.5 if profile.has_internship else -4.0
    feature_deltas.append({
        "Feature": "Internship Experience",
        "Delta": round(intern_delta, 1),
        "Color": "#00F0FF" if intern_delta >= 0 else "#F43F5E",
    })

    # Soft Skill Archetype Alignment Driver
    req_softs = career_info.get("required_soft_skills", [])
    matched_softs = [s for s in req_softs if s in profile.soft_skills]
    soft_ratio = len(matched_softs) / len(req_softs) if req_softs else 0.5
    soft_delta = (soft_ratio - 0.45) * 15.0
    feature_deltas.append({
        "Feature": "Archetype Soft Skills",
        "Delta": round(soft_delta, 1),
        "Color": "#00F0FF" if soft_delta >= 0 else "#F43F5E",
    })

    df = pd.DataFrame(feature_deltas)
    df = df.sort_values(by="Delta", key=abs, ascending=False).head(7)
    df = df.sort_values(by="Delta", ascending=True)
    df.attrs["method"] = "proxy"
    return df


_SHAP_EXPLAINERS: Dict[int, Any] = {}


def get_tree_explainer(clf):
    """Returns a cached shap.TreeExplainer for the given tree model (one per trained model object)."""
    import shap

    cached = _SHAP_EXPLAINERS.get(id(clf))
    if cached is None or cached[0] is not clf:
        cached = (clf, shap.TreeExplainer(clf))
        _SHAP_EXPLAINERS[id(clf)] = cached
    return cached[1]


def calculate_shap_contributions(profile: StudentProfile, top_career: str, top_n: int = 7) -> pd.DataFrame:
    """SHAP TreeExplainer attributions of the ML engine's probability for top_career, for this student.

    Each Delta is that feature's contribution in percentage points; base value + all contributions equals
    the model's predicted probability. Falls back to the benchmark-delta proxy if SHAP cannot run.
    """
    clf, careers, model_type = get_active_classifier()
    try:
        class_label = careers.index(top_career)
        class_col = list(clf.classes_).index(class_label)
        x = build_feature_vector(profile).reshape(1, -1)

        explainer = get_tree_explainer(clf)
        raw = explainer.shap_values(x)
        # Older shap returns one array per class; newer returns (samples, features, classes)
        if isinstance(raw, list):
            contrib = np.asarray(raw[class_col])[0]
        else:
            arr = np.asarray(raw)
            contrib = arr[0, :, class_col] if arr.ndim == 3 else arr[0]
        base_value = float(np.ravel(explainer.expected_value)[class_col])
    except Exception:
        return calculate_shap_proxy_deltas(profile, top_career)

    df = pd.DataFrame({"Feature": FEATURE_NAMES, "Delta": np.round(contrib * 100.0, 1)})
    df["Color"] = np.where(df["Delta"] >= 0, "#00F0FF", "#F43F5E")
    df = df.reindex(df["Delta"].abs().sort_values(ascending=False).index).head(top_n)
    df = df.sort_values(by="Delta", ascending=True).reset_index(drop=True)
    df.attrs.update({
        "method": "shap",
        "model": MODEL_LABELS.get(model_type, model_type),
        "base_value": base_value,
        "prediction": base_value + float(contrib.sum()),
    })
    return df
