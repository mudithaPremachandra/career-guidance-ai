"""SQLite persistence for assessments, including per-student progress-tracking columns."""

import datetime
import json
import os
import sqlite3
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.paths import DATA_DIR


# CAREER_DB_FILE lets tests point the app at a throwaway database before import
DB_FILE = os.environ.get("CAREER_DB_FILE", os.path.join(DATA_DIR, "career_records.db"))

# Columns added for per-student progress tracking; older databases are migrated in place by init_database()
PROGRESS_COLUMNS = {
    "student_id": "TEXT NOT NULL DEFAULT ''",
    "career_scores": "TEXT NOT NULL DEFAULT '{}'",
    "profile_json": "TEXT NOT NULL DEFAULT '{}'",
    "gaps_json": "TEXT NOT NULL DEFAULT '[]'",
}


def init_database() -> None:
    """Initializes the SQLite records table and adds any missing progress-tracking columns."""
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
    existing = {row[1] for row in cursor.execute("PRAGMA table_info(records)")}
    for column, definition in PROGRESS_COLUMNS.items():
        if column not in existing:
            cursor.execute(f"ALTER TABLE records ADD COLUMN {column} {definition}")
    conn.commit()
    conn.close()


def normalize_student_id(student_id: str) -> str:
    """Normalizes a pseudonymous student ID so 'd/bit/24/0088 ' and 'D/BIT/24/0088' match."""
    return " ".join((student_id or "").split()).upper()


def save_student_record(
    gpa: float,
    academic_year: str,
    top_career: str,
    match_score: float,
    confidence: str,
    work_style: str,
    top_driver: str,
    critical_gap: str,
    student_id: str = "",
    career_scores: Dict[str, float] = None,
    profile: Dict[str, Any] = None,
    gaps: List[str] = None,
) -> None:
    """Saves a student profile evaluation result into SQLite."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        """
        INSERT INTO records (
            timestamp, gpa, academic_year, top_career, match_score,
            confidence, work_style, top_driver, critical_gap,
            student_id, career_scores, profile_json, gaps_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            normalize_student_id(student_id),
            json.dumps(career_scores or {}),
            json.dumps(profile or {}),
            json.dumps(gaps or []),
        ),
    )
    conn.commit()
    conn.close()


def fetch_all_records(student_id: str = "") -> pd.DataFrame:
    """Retrieves past student submissions from SQLite, optionally for one student ID."""
    conn = sqlite3.connect(DB_FILE)
    try:
        query = (
            "SELECT id AS 'ID', timestamp AS 'Timestamp', student_id AS 'Student ID', gpa AS 'GPA', "
            "academic_year AS 'Year', top_career AS 'Recommended Role', "
            "match_score AS 'Match %', confidence AS 'Confidence', "
            "work_style AS 'Work Style', top_driver AS 'Key Strength', "
            "critical_gap AS 'Primary Gap' FROM records"
        )
        params: Tuple = ()
        if normalize_student_id(student_id):
            query += " WHERE student_id = ?"
            params = (normalize_student_id(student_id),)
        df = pd.read_sql_query(query + " ORDER BY id DESC", conn, params=params)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


def fetch_student_history(student_id: str) -> List[Dict[str, Any]]:
    """Returns one student's past assessments, oldest first, with their stored JSON fields parsed."""
    sid = normalize_student_id(student_id)
    if not sid:
        return []
    conn = sqlite3.connect(DB_FILE)
    try:
        rows = conn.execute(
            "SELECT id, timestamp, top_career, match_score, career_scores, profile_json, gaps_json "
            "FROM records WHERE student_id = ? ORDER BY id ASC",
            (sid,),
        ).fetchall()
    except Exception:
        rows = []
    finally:
        conn.close()
    return [
        {
            "id": r[0],
            "timestamp": r[1],
            "top_career": r[2],
            "match_pct": r[3],
            "career_scores": json.loads(r[4] or "{}"),
            "profile": json.loads(r[5] or "{}"),
            "gaps": json.loads(r[6] or "[]"),
        }
        for r in rows
    ]


def clear_all_records() -> None:
    """Clears history records from SQLite."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM records")
    conn.commit()
    conn.close()
