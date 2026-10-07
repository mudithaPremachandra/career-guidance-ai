"""Guidance text and learning pathway: template NLG plus optional, verified Gemini narration."""

import html
import json
import os
import re
from typing import Any, Dict, List

import streamlit as st

from src.fusion import HIGH_FIT_THRESHOLD, VIABLE_FIT_THRESHOLD
from src.knowledge_base import ALL_DATA_SKILLS, ALL_EXEC_SKILLS, ALL_IDEAS_SKILLS, ALL_PEOPLE_SKILLS, CERTIFICATION_CATALOG, SKILL_LABELS
from src.profile import StudentProfile


def fit_strength_label(final_score: float) -> str:
    """Maps a fused score to wording that matches its confidence band."""
    if final_score >= HIGH_FIT_THRESHOLD:
        return "strong"
    if final_score >= VIABLE_FIT_THRESHOLD:
        return "promising"
    return "emerging"


def generate_executive_narrative(profile: StudentProfile, top_career: Dict[str, Any], gaps: List[Dict[str, Any]]) -> str:
    """Template-based NLG: a guidance summary whose every claim comes from computed results."""
    role_title = top_career["title"]
    match_pct = top_career["match_pct"]
    archetype = top_career.get("archetype", "Engineering & Architecture")
    matched_softs = top_career.get("matched_soft_skills", [])
    strength = fit_strength_label(top_career["final_score"])

    best_module = max(profile.core_modules.items(), key=lambda x: x[1])
    best_module_name = SKILL_LABELS.get(best_module[0], best_module[0])

    soft_note = f"and natural strengths in <b style='color: #00F0FF;'>{matched_softs[0]}</b>" if matched_softs else "and analytical work habits"

    if gaps:
        primary_gap_name = gaps[0]["skill_name"]
        outcome = {
            "strong": "will position you as a top tier candidate",
            "promising": "will turn this into a strong fit",
            "emerging": "is the first step towards a competitive profile",
        }[strength]
        gap_advice = f"Targeting competency development in <b style='color: #FB7185;'>{primary_gap_name}</b> while sustaining your <b style='color: #38BDF8;'>{profile.gpa:.2f} GPA</b> {outcome} in the <b style='color: #C084FC;'>{archetype}</b> cluster."
    else:
        gap_advice = f"Your profile already meets every benchmark for the <b style='color: #C084FC;'>{archetype}</b> archetype."

    sentence1 = (
        f"Based on multi-engine evaluation, your performance in <b style='color: #38BDF8;'>{best_module_name}</b> "
        f"{soft_note} shows {"an" if strength[0] in "aeiou" else "a"} {strength} <b style='color: #00F0FF;'>{match_pct}% alignment</b> with the <b style='color: #F8FAFC;'>{role_title}</b> career path."
    )
    return f"{sentence1} {gap_advice}"


CAREER_PROJECTS = {
    "Software Engineer": "Build and deploy a full-stack web app with automated tests and a CI pipeline",
    "Data Scientist": "Complete an end-to-end analysis of a public dataset: cleaning, modelling and a written report",
    "AI Engineer": "Train, evaluate and serve a machine learning model behind a small web API",
    "Cloud Architect": "Containerise a multi-service app and deploy it to the cloud with infrastructure as code",
    "UX Designer": "Run a small user study and redesign an existing app screen, written up as a case study",
    "IT Business Analyst": "Write the requirements and process models for a real campus or club system",
    "Game Developer": "Ship a small playable game in a game engine and publish the build",
    "CAD-CAM Engineer": "Model a mechanical part and take it from design through to a manufacturing drawing",
    "Cybersecurity Specialist": "Set up a home security lab and document a capture-the-flag or vulnerability assessment",
}

SOFT_SKILL_ACTIVITIES = {
    "People": "Take a leadership or presentation role in a student society or group project",
    "Ideas": "Join a hackathon, workshop or research project",
    "Data": "Own the planning, tracking or analysis work on a team project",
    "Execution": "Volunteer for hands-on lab, troubleshooting or event-operations work",
}

CERT_MAIN_SKILL_WEIGHT = 0.8

PATHWAY_PHASES = ["Phase 1 · Next 1-3 months", "Phase 2 · 3-6 months", "Phase 3 · 6-12 months"]


def soft_skill_pillar(skill: str) -> str:
    """Returns which of the 4 pillars a soft skill belongs to."""
    for pillar, skills in [("People", ALL_PEOPLE_SKILLS), ("Ideas", ALL_IDEAS_SKILLS),
                           ("Data", ALL_DATA_SKILLS), ("Execution", ALL_EXEC_SKILLS)]:
        if skill in skills:
            return pillar
    return "Ideas"


def build_learning_pathway(
    profile: StudentProfile, top_career: Dict[str, Any], gaps: List[Dict[str, Any]], certs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Sequences gaps, certifications, activities and a portfolio project into a phased learning pathway.

    Urgent technical gaps come first, then medium gaps and soft skills, then portfolio and experience.
    Only certifications chosen by the recommender are used, so the pathway never introduces new ones.
    """
    steps: List[Dict[str, Any]] = []
    unused_certs = list(certs)
    vectors = {c.get("id"): c.get("vector", {}) for c in CERTIFICATION_CATALOG}

    def add(phase: int, kind: str, title: str, detail: str) -> None:
        steps.append({"id": f"S{len(steps) + 1}", "phase": PATHWAY_PHASES[phase], "kind": kind, "title": title, "detail": detail})

    def cert_covering(skill_key: str):
        # Pair only when the gap is one of the cert's main skills, not a minor side topic
        for cert in unused_certs:
            if vectors.get(cert.get("id"), {}).get(skill_key, 0.0) >= CERT_MAIN_SKILL_WEIGHT:
                unused_certs.remove(cert)
                return cert
        return None

    tech_gaps = [g for g in gaps if not g["skill_key"].startswith("SOFT_")]
    soft_gaps = [g for g in gaps if g["skill_key"].startswith("SOFT_")]

    for phase, urgency in [(0, "High Urgency"), (1, "Medium Priority")]:
        for g in [g for g in tech_gaps if g["urgency"] == urgency]:
            cert = cert_covering(g["skill_key"])
            detail = f"Raise {g['skill_name']} from {g['current']} to {g['target']} ({urgency.lower()})."
            if cert:
                add(phase, "Certification", f"Work towards {cert['title']}", f"{detail} This certification covers the gap.")
            else:
                add(phase, "Course", f"Take a structured course in {g['skill_name']}", detail)

    for g in soft_gaps:
        skill = g["skill_name"].replace("Soft Skill: ", "")
        add(1, "Activity", f"Practise {skill}", f"{SOFT_SKILL_ACTIVITIES[soft_skill_pillar(skill)]} to build {skill}, which {top_career['title']} roles expect.")

    add(2, "Project", "Build a portfolio project", f"{CAREER_PROJECTS.get(top_career['career'], 'Build a project in your target field')}.")
    if not profile.has_internship:
        add(2, "Experience", "Secure an internship", f"Apply for a {top_career['title']} internship to gain industry experience.")

    minor = [g["skill_name"] for g in tech_gaps if g["urgency"] == "Low / Minor"]
    if minor:
        add(2, "Course", "Polish minor gaps", f"Close the small remaining gaps in {', '.join(minor)}.")
    for cert in unused_certs:
        add(2, "Certification", f"Optional: {cert['title']}", f"Covers {', '.join(cert.get('covered_skills', [])) or 'related skills'}.")

    # Close up empty phases so the plan always starts now (e.g. no urgent gaps -> medium gaps become Phase 1)
    used = [p for p in PATHWAY_PHASES if any(s["phase"] == p for s in steps)]
    shift = {p: PATHWAY_PHASES[i] for i, p in enumerate(used)}
    for s in steps:
        s["phase"] = shift[s["phase"]]
    return steps


GEMINI_DEFAULT_MODEL = "gemini-3.5-flash-lite"

GEMINI_SYSTEM_PROMPT = (
    "You are the writing assistant for a university career guidance system. Rewrite the verified facts you are given "
    "as warm, plain-English guidance addressed to the student as 'you'.\n"
    "Rules:\n"
    "- Use ONLY the facts provided. Never add a certification, course, company, tool, statistic or claim that is not in them.\n"
    "- Copy every number exactly as given. Do not round, convert or invent numbers.\n"
    "- Return one entry per pathway step, with the same ids, in the same order.\n"
    "- summary: 2-3 sentences, at most 70 words. Each step: 1-2 sentences, at most 45 words.\n"
    "- Plain text only, no markdown."
)

GEMINI_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "text": {"type": "string"}},
                "required": ["id", "text"],
            },
        },
    },
    "required": ["summary", "steps"],
}

# Provider names an invented recommendation would most likely mention
KNOWN_PROVIDERS = {"AWS", "Amazon", "Azure", "Microsoft", "Google", "IBM", "Meta", "Cisco", "CompTIA", "Oracle",
                   "Coursera", "Udemy", "edX", "Unity", "Autodesk", "Scrum", "PMI", "ISC2", "EC-Council", "Salesforce"}


def get_app_setting(name: str, default: str = "") -> str:
    """Reads a setting from Streamlit secrets, then environment variables."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)


def build_narration_facts(
    profile: StudentProfile, top_career: Dict[str, Any], gaps: List[Dict[str, Any]],
    certs: List[Dict[str, Any]], pathway: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Collects the computed facts the narrator may use. Nothing outside this dict may appear in its output."""
    best_module = max(profile.core_modules.items(), key=lambda x: x[1])
    return {
        "career": top_career["title"],
        "match": f"{top_career['match_pct']}%",
        "fit": fit_strength_label(top_career["final_score"]),
        "gpa": round(profile.gpa, 2),
        "best_module": SKILL_LABELS.get(best_module[0], best_module[0]),
        "best_module_score": best_module[1],
        "matched_soft_skills": top_career.get("matched_soft_skills", []),
        "top_gaps": [{"skill": g["skill_name"], "current": g["current"], "target": g["target"], "urgency": g["urgency"]} for g in gaps[:4]],
        "recommended_certifications": [c["title"] for c in certs],
        "pathway": [{k: s[k] for k in ("id", "phase", "kind", "title", "detail")} for s in pathway],
    }


@st.cache_data(show_spinner=False, ttl=86400)
def call_gemini(facts_json: str, model: str, api_key: str) -> Dict[str, Any]:
    """Asks Gemini to narrate the facts as JSON. Raises on any failure (failures are not cached)."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=20000))
    response = client.models.generate_content(
        model=model,
        contents=f"Verified facts (JSON):\n{facts_json}",
        config=types.GenerateContentConfig(
            system_instruction=GEMINI_SYSTEM_PROMPT,
            temperature=0.3,
            max_output_tokens=1024,
            response_mime_type="application/json",
            response_json_schema=GEMINI_RESPONSE_SCHEMA,
        ),
    )
    return json.loads(response.text)


def validate_narration(output: Any, facts: Dict[str, Any]) -> str:
    """Checks LLM narration against the facts. Returns "" if valid, otherwise the reason it was rejected."""
    if not isinstance(output, dict) or not isinstance(output.get("summary"), str) or not isinstance(output.get("steps"), list):
        return "response did not match the expected structure"
    summary = output["summary"].strip()
    if not summary or len(summary) > 600:
        return "summary was empty or too long"

    expected_ids = [s["id"] for s in facts["pathway"]]
    got_ids = [s.get("id") if isinstance(s, dict) else None for s in output["steps"]]
    if got_ids != expected_ids:
        return "pathway steps were missing, added or reordered"
    texts = [summary]
    for s in output["steps"]:
        text = s.get("text")
        if not isinstance(text, str) or not text.strip() or len(text) > 400:
            return f"step {s.get('id')} text was empty or too long"
        texts.append(text)
    combined = " ".join(texts)

    facts_text = json.dumps(facts, ensure_ascii=False)
    allowed_numbers = {float(n) for n in re.findall(r"\d+(?:\.\d+)?", facts_text)}
    for n in re.findall(r"\d+(?:\.\d+)?", combined):
        if float(n) not in allowed_numbers:
            return f"introduced a number not in the facts ({n})"

    for cert in CERTIFICATION_CATALOG:
        title = cert.get("title", "")
        if title and title.lower() in combined.lower() and title not in facts["recommended_certifications"]:
            return f"mentioned a certification the engine did not recommend ({title})"
    for provider in KNOWN_PROVIDERS:
        pattern = rf"\b{re.escape(provider)}\b"
        if re.search(pattern, combined, re.IGNORECASE) and not re.search(pattern, facts_text, re.IGNORECASE):
            return f"mentioned a provider not in the facts ({provider})"
    return ""


def generate_guidance(
    profile: StudentProfile, top_career: Dict[str, Any], gaps: List[Dict[str, Any]], certs: List[Dict[str, Any]],
    narrator=None,
) -> Dict[str, Any]:
    """Builds the summary and learning pathway: Gemini narration when configured and verified, else template NLG.

    `narrator(facts) -> dict` can be passed to replace the Gemini call (used by tests).
    """
    pathway = build_learning_pathway(profile, top_career, gaps, certs)
    guidance = {
        "narrative": generate_executive_narrative(profile, top_career, gaps),
        "pathway": [dict(s, text=s["detail"]) for s in pathway],
        "source": "template",
        "note": "",
    }

    facts = build_narration_facts(profile, top_career, gaps, certs, pathway)
    model = get_app_setting("GEMINI_MODEL", GEMINI_DEFAULT_MODEL)
    if narrator is None:
        api_key = get_app_setting("GEMINI_API_KEY")
        if not api_key:
            guidance["note"] = "Gemini is not configured (no GEMINI_API_KEY), so template narration is shown."
            return guidance
        narrator = lambda f: call_gemini(json.dumps(f, ensure_ascii=False, sort_keys=True), model, api_key)

    try:
        output = narrator(facts)
    except Exception as e:
        guidance["note"] = f"Gemini request failed ({type(e).__name__}), so template narration is shown."
        return guidance

    problem = validate_narration(output, facts)
    if problem:
        guidance["note"] = f"Gemini narration was rejected because it {problem}, so template narration is shown."
        return guidance

    guidance["narrative"] = html.escape(output["summary"].strip())
    for step, narrated in zip(guidance["pathway"], output["steps"]):
        step["text"] = narrated["text"].strip()
    guidance["source"] = "gemini"
    guidance["note"] = f"Narrated by {model} and verified against the engine's facts: no new numbers, certifications or providers."
    return guidance
