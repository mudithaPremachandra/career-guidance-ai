"""Engine C (40%): benchmark dataset, feature encoding and Random Forest / Decision Tree classifiers."""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from src.knowledge_base import CAREER_UNIVERSE
from src.profile import StudentProfile


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

        for i in range(samples_per_class):
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


# Display names for the 20 columns produced by preprocess_features() / build_feature_vector(), in order
FEATURE_NAMES = [
    "GPA", "Year", "DSA", "OOP", "DBMS", "OS & Networks",
    "SE Principles", "Math & Stats", "Python", "Java/C++",
    "SQL", "Web Stack", "Cloud/Docker", "ML/AI", "Mobile Dev",
    "Cybersecurity", "Internship", "Projects", "Work Style", "Soft Skills Affinity",
]

MODEL_LABELS = {"random_forest": "Random Forest", "decision_tree": "Decision Tree"}

# Configuration of the default model the app trains on startup (also used by src/train.py to save it)
DEFAULT_SAMPLES_PER_CLASS = 45
DEFAULT_N_ESTIMATORS = 80
DEFAULT_MAX_DEPTH = 10


def make_classifier(model_type: str, n_estimators: int, max_depth: int):
    """Creates an untrained Random Forest (ensemble) or Decision Tree (single-tree baseline)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    if model_type == "decision_tree":
        return DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)


def train_custom_classifier(
    df: pd.DataFrame,
    n_estimators: int = 100,
    max_depth: int = 10,
    test_size: float = 0.2,
    model_type: str = "random_forest",
) -> Dict[str, Any]:
    """Trains a Random Forest or Decision Tree classifier on the provided DataFrame and returns metrics."""
    try:
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split

        X, y, careers = preprocess_features(df)

        if len(X) < 10:
            return {"error": "Dataset must contain at least 10 student records for training."}

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) > 1 and min(np.bincount(y)) >= 2 else None
        )

        clf = make_classifier(model_type, n_estimators, max_depth)
        clf.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc = accuracy_score(y_test, clf.predict(X_test))

        importances = clf.feature_importances_
        feat_imp_df = pd.DataFrame({"Feature": FEATURE_NAMES, "Importance": importances}).sort_values(
            by="Importance", ascending=True
        )

        return {
            "model": clf,
            "model_type": model_type,
            "careers": careers,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "feat_imp_df": feat_imp_df,
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "X_test": X_test,
            "y_test": y_test,
        }
    except Exception as e:
        return {"error": str(e)}


def compare_classifiers(
    df: pd.DataFrame, n_estimators: int = 100, max_depth: int = 10, test_size: float = 0.2, cv_folds: int = 5
) -> pd.DataFrame:
    """Benchmarks the Decision Tree baseline against the Random Forest on the same split and CV folds."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    X, y, _ = preprocess_features(df)
    folds = StratifiedKFold(n_splits=min(cv_folds, int(np.bincount(y).min())), shuffle=True, random_state=42)
    rows = []
    for model_type in ["decision_tree", "random_forest"]:
        res = train_custom_classifier(df, n_estimators, max_depth, test_size, model_type=model_type)
        if "error" in res:
            raise ValueError(res["error"])
        cv = cross_val_score(make_classifier(model_type, n_estimators, max_depth), X, y, cv=folds)
        rows.append({
            "Model": MODEL_LABELS[model_type],
            "Train Accuracy": res["train_acc"],
            "Test Accuracy": res["test_acc"],
            "CV Mean": cv.mean(),
            "CV Std": cv.std(),
        })
    return pd.DataFrame(rows)


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
    benchmark_df = generate_benchmark_dataset(samples_per_class=DEFAULT_SAMPLES_PER_CLASS)
    train_res = train_custom_classifier(benchmark_df, n_estimators=DEFAULT_N_ESTIMATORS, max_depth=DEFAULT_MAX_DEPTH)
    if "model" in train_res:
        return train_res["model"], train_res["careers"]
    return None, list(CAREER_UNIVERSE.keys())


def get_active_classifier():
    """Returns (model, careers, model_type) for the user-trained model, else the default Random Forest."""
    if "active_ml_model" in st.session_state and st.session_state["active_ml_model"] is not None:
        return (
            st.session_state["active_ml_model"],
            st.session_state.get("active_ml_careers", list(CAREER_UNIVERSE.keys())),
            st.session_state.get("active_ml_type", "random_forest"),
        )
    clf, careers = get_default_calibrated_ml_classifier()
    return clf, careers, "random_forest"


def evaluate_ml_model(profile: StudentProfile) -> Dict[str, float]:
    """Generates class probability distribution across all 9 careers via active ML model."""
    clf, careers, _ = get_active_classifier()

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
