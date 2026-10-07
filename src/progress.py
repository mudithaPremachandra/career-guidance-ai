"""Adaptive progress tracking: compares a student's new assessment with their previous one."""

import re
from typing import Any, Dict, List

import plotly.graph_objects as go

from src.knowledge_base import CAREER_UNIVERSE, CERTIFICATION_CATALOG, SKILL_LABELS
from src.profile import StudentProfile


CERT_MATCH_STOPWORDS = {"certified", "certification", "certificate", "cert", "the", "of", "and", "for", "in"}


def _cert_tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in CERT_MATCH_STOPWORDS}


def parse_held_certifications(text: str) -> List[str]:
    """Matches the free-text 'existing certifications' field against the catalog.

    An entry is recognised when all its significant words appear in exactly one catalog title, so
    "AWS Cloud Practitioner" matches "AWS Certified Cloud Practitioner (CLF-C02)" while vague
    entries such as "security" (several matches) or "CS50x" (no match) are ignored.
    """
    held: List[str] = []
    for entry in (text or "").split(","):
        tokens = _cert_tokens(entry)
        if not tokens:
            continue
        matches = [c["id"] for c in CERTIFICATION_CATALOG if tokens <= _cert_tokens(c.get("title", ""))]
        if len(matches) == 1 and matches[0] not in held:
            held.append(matches[0])
    return held


def cert_titles(cert_ids: List[str]) -> List[str]:
    """Maps catalog IDs to certification titles."""
    titles = {c["id"]: c.get("title", c["id"]) for c in CERTIFICATION_CATALOG}
    return [titles.get(i, i) for i in cert_ids]


def profile_to_dict(profile: StudentProfile) -> Dict[str, Any]:
    """Serializes the profile fields that progress tracking compares between assessments."""
    return {
        "gpa": round(profile.gpa, 2),
        "year": profile.year,
        "core_modules": dict(profile.core_modules),
        "tech_skills": dict(profile.tech_skills),
        "electives": sorted(profile.electives.keys()),
        "soft_skills": sorted(profile.soft_skills),
        "has_internship": bool(profile.has_internship),
        "projects_count": int(profile.projects_count),
        "held_certs": parse_held_certifications(profile.existing_certs),
    }


def compare_assessments(
    previous: Dict[str, Any], profile: StudentProfile, results: List[Dict[str, Any]], gaps: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Explains how the recommendation adapted since the student's previous assessment."""
    prev_profile = previous.get("profile", {})
    curr_profile = profile_to_dict(profile)
    prev_scores = previous.get("career_scores", {})
    top = results[0]

    # In current ranking order
    career_changes = [
        {"career": r["title"], "before": prev_scores[r["career"]], "after": r["match_pct"], "change": round(r["match_pct"] - prev_scores[r["career"]], 1)}
        for r in results if r["career"] in prev_scores
    ]

    skill_changes = []
    for group, scale in [("core_modules", 100), ("tech_skills", 5)]:
        before = prev_profile.get(group, {})
        for key, now in curr_profile[group].items():
            if key in before and before[key] != now:
                skill_changes.append({"skill": SKILL_LABELS.get(key, key), "before": f"{before[key]:g}/{scale}", "after": f"{now:g}/{scale}", "improved": now > before[key]})

    events = []
    # Only meaningful when the previous run stored a profile (rows from before progress tracking did not)
    new_certs = [c for c in curr_profile["held_certs"] if c not in prev_profile.get("held_certs", [])] if prev_profile else []
    if prev_profile:
        if curr_profile["gpa"] != prev_profile.get("gpa"):
            events.append(f"GPA {prev_profile.get('gpa'):.2f} → {curr_profile['gpa']:.2f}")
        if curr_profile["has_internship"] and not prev_profile.get("has_internship"):
            events.append("Completed an internship")
        if curr_profile["projects_count"] != prev_profile.get("projects_count"):
            events.append(f"Projects {prev_profile.get('projects_count')} → {curr_profile['projects_count']}")
        if new_certs:
            events.append("New certifications: " + ", ".join(cert_titles(new_certs)))
        new_softs = [s for s in curr_profile["soft_skills"] if s not in prev_profile.get("soft_skills", [])]
        if new_softs:
            events.append("New soft skills: " + ", ".join(new_softs))
        new_electives = [e for e in curr_profile["electives"] if e not in prev_profile.get("electives", [])]
        if new_electives:
            events.append("New electives: " + ", ".join(new_electives))

    # Gap lists are measured against the top career, so they only compare when the top career is unchanged
    same_top = previous.get("top_career") == top["title"]
    curr_gap_names = [g["skill_name"] for g in gaps]
    closed_gaps = [g for g in previous.get("gaps", []) if g not in curr_gap_names] if same_top else []
    new_gaps = [g for g in curr_gap_names if g not in previous.get("gaps", [])] if same_top else []

    before_top = prev_scores.get(top["career"])
    insights = []
    if not same_top:
        insights.append(f"Your top recommendation changed from {previous.get('top_career')} to {top['title']}.")
    if before_top is not None:
        diff = round(top["match_pct"] - before_top, 1)
        direction = "rose" if diff > 0 else "fell" if diff < 0 else "stayed"
        insights.append(f"Your {top['title']} match {direction} from {before_top}% to {top['match_pct']}% ({diff:+.1f} pts)." if diff else f"Your {top['title']} match stayed at {top['match_pct']}%.")
    if new_certs:
        insights.append("You now hold " + ", ".join(cert_titles(new_certs)) + (", so it is" if len(new_certs) == 1 else ", so they are") + " no longer recommended and other certifications take their place.")
    if closed_gaps:
        insights.append("Gaps closed since last time: " + ", ".join(closed_gaps) + ".")
    if new_gaps:
        insights.append("New gaps to watch: " + ", ".join(new_gaps) + ".")
    if not skill_changes and not events:
        insights.append("Your profile is unchanged since the last assessment, so the recommendation is the same.")

    return {
        "previous_timestamp": previous.get("timestamp", ""),
        "previous_top": previous.get("top_career", ""),
        "top_before": before_top,
        "top_after": top["match_pct"],
        "same_top": same_top,
        "career_changes": career_changes,
        "skill_changes": skill_changes,
        "events": events,
        "closed_gaps": closed_gaps,
        "new_gaps": new_gaps,
        "insights": insights,
    }


def build_progress_chart(history: List[Dict[str, Any]], careers: List[str]):
    """Line chart of match % across a student's assessments for the given career keys."""
    titles = {k: v["title"] for k, v in CAREER_UNIVERSE.items()}
    fig = go.Figure()
    x = [f"#{i + 1} · {h['timestamp'][:10]}" for i, h in enumerate(history)]
    palette = ["#00F0FF", "#C084FC", "#FBBF24", "#34D399", "#FB7185"]
    for i, career in enumerate(careers):
        y = [h["career_scores"].get(career) for h in history]
        if any(v is not None for v in y):
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=titles.get(career, career),
                                     line=dict(color=palette[i % len(palette)], width=2), connectgaps=True))
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.5)",
        yaxis=dict(title="Match %", tickfont=dict(color="#CBD5E1"), gridcolor="rgba(51, 65, 85, 0.4)"),
        xaxis=dict(tickfont=dict(color="#CBD5E1")),
        legend=dict(font=dict(color="#E2E8F0"), orientation="h", y=-0.25),
    )
    return fig
