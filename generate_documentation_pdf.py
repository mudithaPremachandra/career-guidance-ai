"""
Script to generate a comprehensive, publication-quality PDF technical documentation
for PathFinder AI: Undergraduate AI Career Guidance System.
"""

import os
import sys
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
    HRFlowable,
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print total page count."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Suppress running header on cover/first page
        if self._pageNumber > 1:
            # Header
            self.drawString(54, 750, "PathFinder AI — Comprehensive System Architecture & Technical Manual")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)

            # Footer
            self.line(54, 45, 558, 45)
            self.drawString(54, 32, "Confidential & Proprietary • PathFinder AI Engineering Documentation")
            page_text = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(558, 32, page_text)
        else:
            # First page subtle bottom note
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(54, 45, 558, 45)
            self.drawString(54, 32, "PathFinder AI • Undergraduate AI Career Guidance System")
            page_text = f"Page 1 of {page_count}"
            self.drawRightString(558, 32, page_text)

        self.restoreState()


def create_system_pdf(output_filename="PathFinder_AI_System_Documentation.pdf"):
    pdf_path = os.path.abspath(output_filename)
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom Color Palette
    PRIMARY = colors.HexColor("#0F172A")       # Deep Navy Slate
    ACCENT_CYAN = colors.HexColor("#0284C7")   # Deep Cyan
    ACCENT_PURPLE = colors.HexColor("#6D28D9") # Deep Purple
    TEXT_MAIN = colors.HexColor("#1E293B")     # Dark Slate
    TEXT_MUTED = colors.HexColor("#475569")    # Medium Slate
    BG_LIGHT = colors.HexColor("#F8FAFC")      # Light Slate Background
    BORDER_COLOR = colors.HexColor("#CBD5E1")  # Border Gray
    CARD_BG = colors.HexColor("#F1F5F9")

    # Custom Typography Styles
    styles.add(ParagraphStyle(
        "CoverDocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=PRIMARY,
        spaceAfter=6,
    ))

    styles.add(ParagraphStyle(
        "CoverDocSub",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=ACCENT_CYAN,
        spaceAfter=14,
    ))

    styles.add(ParagraphStyle(
        "CoverMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=TEXT_MUTED,
    ))

    styles.add(ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=PRIMARY,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True,
    ))

    styles.add(ParagraphStyle(
        "SubSectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=ACCENT_PURPLE,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    ))

    styles.add(ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=TEXT_MAIN,
        spaceAfter=6,
    ))

    styles.add(ParagraphStyle(
        "BodyDarkBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13.5,
        textColor=TEXT_MAIN,
        spaceAfter=4,
    ))

    styles.add(ParagraphStyle(
        "BulletItem",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=TEXT_MAIN,
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=3,
    ))

    styles.add(ParagraphStyle(
        "CalloutText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=PRIMARY,
    ))

    styles.add(ParagraphStyle(
        "TableHead",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
    ))

    styles.add(ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=TEXT_MAIN,
    ))

    styles.add(ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=PRIMARY,
    ))

    styles.add(ParagraphStyle(
        "FormulaBox",
        parent=styles["Normal"],
        fontName="Courier-Bold",
        fontSize=9,
        leading=13,
        textColor=ACCENT_PURPLE,
    ))

    story = []

    # =========================================================================
    # COVER / HEADER BLOCK
    # =========================================================================
    story.append(Spacer(1, 10))
    story.append(Paragraph("PATHFINDER AI", styles["CoverDocTitle"]))
    story.append(Paragraph("Undergraduate AI Career Guidance & Multi-Engine Intelligence Platform", styles["CoverDocSub"]))
    story.append(Paragraph(
        "<b>System Architecture, Mathematical Formulation & Full Operational Specification</b><br/>"
        "Document Version: 2.4.0 • Target Domain: Computer Science & Software Engineering Undergraduates<br/>"
        "Platform Core: Hybrid Multi-Engine Score Fusion (Rule + Fuzzy + ML) • XAI • Gap Analysis • SQLite",
        styles["CoverMeta"],
    ))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=ACCENT_CYAN, spaceBefore=4, spaceAfter=12))

    # Executive Summary Box
    exec_summary_html = (
        "<b>Executive Summary:</b> PathFinder AI is an enterprise-grade career intelligence system engineered "
        "to solve the 'Career Ambiguity Dilemma' faced by undergraduate computing students. Traditional career "
        "guidance tools rely either on simplistic rule quizzes or opaque black-box machine learning. PathFinder AI "
        "implements a state-of-the-art <b>Hybrid Multi-Engine Fusion Model (30% Rule-Based Expert System + 30% Mamdani Fuzzy "
        "Logic Suitability Engine + 40% Supervised Machine Learning Classifier)</b> with <b>SHAP-style Explainable AI (XAI)</b>, "
        "<b>4-Pillar Soft Skills Mapping</b>, <b>Gap-Weighted Cosine Similarity Certification Recommendations</b>, and an interactive "
        "<b>Model Studio</b> with local SQLite persistence."
    )
    exec_table = Table([[Paragraph(exec_summary_html, styles["CalloutText"])]], colWidths=[504])
    exec_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BG_LIGHT),
        ("BOX", (0, 0), (-1, -1), 1, ACCENT_CYAN),
        ("PADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(exec_table)
    story.append(Spacer(1, 12))

    # =========================================================================
    # 1. CORE SYSTEM ARCHITECTURE & DATA FLOW
    # =========================================================================
    story.append(Paragraph("1. System Architecture & High-Level Data Flow", styles["SectionHeading"]))
    story.append(Paragraph(
        "PathFinder AI is designed around a modular 4-layer decoupled architecture ensuring high computational "
        "throughput, fault tolerance, and deterministic explainability:",
        styles["BodyDark"]
    ))

    arch_layers = [
        ("Layer 1: Reactive Student Intake Layer", "Captures multi-dimensional student profiles across Academic Foundation (GPA, Core CS modules 0–100), Specialized Electives (10 catalog options), 4-Pillar Soft Skills (People, Ideas, Data, Execution), Technical Tool Proficiencies (1–5 scale), and Work Style / Career Preferences."),
        ("Layer 2: Multi-Engine Inference Core", "Processes intake features through three distinct parallel inference pipelines: (A) Rule-Based Expert System for hard gatekeeper prerequisite validation, (B) Mamdani Fuzzy Logic System for non-linear boundary defuzzification, and (C) Random Forest Multi-Class Classifier for empirical probabilistic pattern matching."),
        ("Layer 3: Explainability (XAI) & Recommendation Layer", "Computes SHAP-proxy feature attribution deltas (isolating driving vs penalizing factors), calculates urgency-weighted competency gap matrices, generates dynamic executive natural language narratives, and queries a curated certification catalog via gap-weighted cosine similarity."),
        ("Layer 4: Persistence & Model Studio Layer", "Maintains local relational history in SQLite (career_records.db), supports synthetic benchmark dataset generation (450+ records), facilitates custom CSV/Excel dataset uploads with schema validation, and enables live on-the-fly model retraining with feature importance diagnostics."),
    ]

    arch_data = [[Paragraph(f"<b>{l[0]}</b>", styles["TableCellBold"]), Paragraph(l[1], styles["TableCell"])] for l in arch_layers]
    arch_table = Table(arch_data, colWidths=[150, 354])
    arch_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(arch_table)
    story.append(Spacer(1, 14))

    # =========================================================================
    # 2. PROFILE INTAKE & 4-PILLAR SOFT SKILLS FRAMEWORK
    # =========================================================================
    story.append(Paragraph("2. Profile Intake & 4-Pillar Soft Skills Framework", styles["SectionHeading"]))
    story.append(Paragraph(
        "Unlike conventional systems that only measure programming languages, PathFinder AI incorporates a comprehensive "
        "holistic assessment model structured into four major intake categories:",
        styles["BodyDark"]
    ))

    story.append(Paragraph("A. Academic Foundation & Core Computer Science Modules (0–100 Scale)", styles["SubSectionHeading"]))
    story.append(Paragraph(
        "• <b>Cumulative GPA</b> (0.00 – 4.00) and <b>Academic Year Level</b> (1st, 2nd, 3rd, 4th Year).<br/>"
        "• <b>6 Compulsory Core CS Disciplines</b>: (1) Data Structures & Algorithms (DSA), (2) Object-Oriented Programming (OOP), "
        "(3) Database Management Systems (DBMS), (4) Operating Systems & Computer Networks (OS/Net), "
        "(5) Software Engineering Principles (SE), and (6) Mathematics & Statistics (Math/Stats).",
        styles["BulletItem"]
    ))

    story.append(Paragraph("B. Specialized University Electives Catalog (10 Modules)", styles["SubSectionHeading"]))
    story.append(Paragraph(
        "Supports dynamic multi-selection and performance tracking for: <i>AI & Machine Learning, Data Mining & Big Data, "
        "Human-Computer Interaction (UI/UX), Computer Graphics & Animation, Game Engine Development, CAD/CAM Principles, "
        "Network Security & Cyber Defense, Cloud Computing, IT Project Management, and Technical Writing</i>.",
        styles["BulletItem"]
    ))

    story.append(Paragraph("C. 4-Pillar Soft Skills & Work Strengths Matrix (24 Granular Competencies)", styles["SubSectionHeading"]))

    soft_pillars = [
        [Paragraph("<b>Pillar</b>", styles["TableHead"]), Paragraph("<b>Competency Focus</b>", styles["TableHead"]), Paragraph("<b>Granular Skills Evaluated</b>", styles["TableHead"])],
        [Paragraph("👥 People & Leadership", styles["TableCellBold"]), Paragraph("Interpersonal & Collaboration", styles["TableCell"]), Paragraph("Communicating & Articulating, Team Leadership & Delegation, Negotiation & Persuasion, Client Facing & Presentation, Mentoring & Teaching, Active Listening & Empathy.", styles["TableCell"])],
        [Paragraph("💡 Ideas & Innovation", styles["TableCellBold"]), Paragraph("Conceptual & Creative Design", styles["TableCell"]), Paragraph("Problem Solving & Logic, Creative & Visual Design, Research & Investigation, Storytelling & Conceptualizing, Innovation & Prototyping, Technical & Content Writing.", styles["TableCell"])],
        [Paragraph("📊 Data & Systems", styles["TableCellBold"]), Paragraph("Methodical & Analytical Logic", styles["TableCell"]), Paragraph("Analytical & Critical Thinking, Detail-Oriented & Precision, Planning & Task Organizing, Time & Deadline Management, Quantitative & Mathematical, Monitoring & Evaluation.", styles["TableCell"])],
        [Paragraph("🛠️ Execution & Practical", styles["TableCellBold"]), Paragraph("Hands-on Engineering & Systems", styles["TableCell"]), Paragraph("Hardware & System Troubleshooting, Hands-on Prototyping, Debugging & Root-Cause Analysis, Working Under Pressure / Incidents, Process Optimization, Process & Protocol Adherence.", styles["TableCell"])],
    ]
    soft_table = Table(soft_pillars, colWidths=[120, 110, 274])
    soft_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(soft_table)
    story.append(Spacer(1, 14))

    # =========================================================================
    # 3. THE 6-ARCHETYPE CAREER REQUIREMENT UNIVERSE
    # =========================================================================
    story.append(Paragraph("3. The 6-Archetype Career Requirement Universe", styles["SectionHeading"]))
    story.append(Paragraph(
        "PathFinder AI maps 9 specialized undergraduate career destinations into 6 overarching industry archetypes. "
        "Each career profile establishes rigorous benchmark requisites across core modules, technical proficiencies, and soft skills:",
        styles["BodyDark"]
    ))

    career_defs = [
        [Paragraph("<b>Role & Icon</b>", styles["TableHead"]), Paragraph("<b>Archetype Cluster</b>", styles["TableHead"]), Paragraph("<b>Core Requisite Benchmarks</b>", styles["TableHead"]), Paragraph("<b>Required Soft Skills (20% Bonus)</b>", styles["TableHead"])],
        [
            Paragraph("<b>💻 Software Engineer</b>", styles["TableCellBold"]),
            Paragraph("Engineering & Architecture", styles["TableCell"]),
            Paragraph("DSA: 85, OOP: 85, SE: 85, Java/C++: 4/5, Web: 4/5", styles["TableCell"]),
            Paragraph("Problem Solving, Debugging, Precision, Prototyping", styles["TableCell"])
        ],
        [
            Paragraph("<b>📊 Data Scientist</b>", styles["TableCellBold"]),
            Paragraph("Data & Analytics", styles["TableCell"]),
            Paragraph("Math/Stats: 90, DBMS: 85, Python: 4.5/5, SQL: 4.5/5", styles["TableCell"]),
            Paragraph("Analytical Thinking, Storytelling, Quantitative, Research", styles["TableCell"])
        ],
        [
            Paragraph("<b>🧠 AI Engineer</b>", styles["TableCellBold"]),
            Paragraph("Engineering & Architecture", styles["TableCell"]),
            Paragraph("Math/Stats: 90, DSA: 85, Python: 5/5, ML/AI: 5/5", styles["TableCell"]),
            Paragraph("Problem Solving, Innovation, Research, Quantitative", styles["TableCell"])
        ],
        [
            Paragraph("<b>☁️ Cloud Architect</b>", styles["TableCellBold"]),
            Paragraph("Infrastructure, Systems & Sec.", styles["TableCell"]),
            Paragraph("OS/Net: 90, DBMS: 80, Cloud/Docker: 5/5, Cyber: 4/5", styles["TableCell"]),
            Paragraph("Troubleshooting, Task Organizing, Pressure/Incidents", styles["TableCell"])
        ],
        [
            Paragraph("<b>🎨 UX Designer</b>", styles["TableCellBold"]),
            Paragraph("Creative, UI/UX & Media", styles["TableCell"]),
            Paragraph("SE: 80, Web: 4/5, Mobile: 3.5/5, HCI Elective", styles["TableCell"]),
            Paragraph("Empathy, Visual Design, Storytelling, Innovation", styles["TableCell"])
        ],
        [
            Paragraph("<b>📈 IT Business Analyst</b>", styles["TableCellBold"]),
            Paragraph("Management & Consulting", styles["TableCell"]),
            Paragraph("SE: 90, DBMS: 75, SQL: 4/5, IT-PM Elective", styles["TableCell"]),
            Paragraph("Communicating, Analytical, Organizing, Negotiation", styles["TableCell"])
        ],
        [
            Paragraph("<b>🎮 Game Developer</b>", styles["TableCellBold"]),
            Paragraph("Creative, UI/UX & Media", styles["TableCell"]),
            Paragraph("OOP: 90, Math/Stats: 85, Java/C++: 5/5, Graphics", styles["TableCell"]),
            Paragraph("Visual Design, Logic, Prototyping, Storytelling", styles["TableCell"])
        ],
        [
            Paragraph("<b>📐 CAD-CAM Engineer</b>", styles["TableCellBold"]),
            Paragraph("Engineering & Architecture", styles["TableCell"]),
            Paragraph("Math/Stats: 85, SE: 75, CAD/CAM Elective", styles["TableCell"]),
            Paragraph("Precision, Logic, Prototyping, Quantitative", styles["TableCell"])
        ],
        [
            Paragraph("<b>🛡️ Cybersecurity Specialist</b>", styles["TableCellBold"]),
            Paragraph("Infrastructure, Systems & Sec.", styles["TableCell"]),
            Paragraph("OS/Net: 95, Cyber: 5/5, Cloud: 3.5/5, Sec Elective", styles["TableCell"]),
            Paragraph("Pressure/Incidents, Debugging, Protocol Adherence", styles["TableCell"])
        ],
    ]

    career_table = Table(career_defs, colWidths=[110, 110, 144, 140])
    career_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 4.5),
    ]))
    story.append(career_table)
    story.append(Spacer(1, 14))

    # =========================================================================
    # 4. HYBRID MULTI-ENGINE SCORE FUSION MATHEMATICAL MODEL
    # =========================================================================
    story.append(Paragraph("4. Hybrid Multi-Engine Score Fusion Mathematical Model", styles["SectionHeading"]))
    story.append(Paragraph(
        "PathFinder AI overcomes single-model biases by unifying three distinct artificial intelligence paradigms "
        "into a calibrated hybrid score fusion formula:",
        styles["BodyDark"]
    ))

    fusion_box_html = (
        "<b>Score Fusion Mathematical Equation:</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>Score_Final = 0.30 × Score_Rule + 0.30 × Score_Fuzzy + 0.40 × Score_ML</b><br/><br/>"
        "• <b>Score_Rule (30% Weight)</b>: Deterministic expert gatekeeper verification + Soft Skill bonus.<br/>"
        "• <b>Score_Fuzzy (30% Weight)</b>: Mamdani Fuzzy Inference with Centroid defuzzification & academic year scaling.<br/>"
        "• <b>Score_ML (40% Weight)</b>: Supervised Random Forest multi-class class probability distribution."
    )
    fusion_table = Table([[Paragraph(fusion_box_html, styles["FormulaBox"])]], colWidths=[504])
    fusion_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BG_LIGHT),
        ("BOX", (0, 0), (-1, -1), 1, ACCENT_PURPLE),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(fusion_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("4.1 Engine A: Rule-Based Expert System (30%)", styles["SubSectionHeading"]))
    story.append(Paragraph(
        "Evaluates explicit hard constraints, prerequisite thresholds, and GPA standing. When a condition evaluates to TRUE, "
        "points are accumulated and a transparent <b>Rule Firing Trace</b> is recorded for auditable decision justification. "
        "An additional +20% weight bonus is awarded based on archetype soft skill matching ratio.",
        styles["BodyDark"]
    ))

    story.append(Paragraph("4.2 Engine B: Mamdani Fuzzy Logic Suitability Engine (30%)", styles["SubSectionHeading"]))
    story.append(Paragraph(
        "Employs Triangular and Trapezoidal Membership Functions (μ(x)) to model academic and technical grades continuously "
        "rather than using rigid Boolean boundaries. Evaluates fuzzy rules (e.g., IF Technical Mastery is Strong AND GPA is High "
        "AND Soft Skills are High THEN Suitability is Excellent [0.96]) and aggregates consequents using Centroid Defuzzification.",
        styles["BodyDark"]
    ))

    story.append(Paragraph("4.3 Engine C: Supervised Machine Learning Classifier (40%)", styles["SubSectionHeading"]))
    story.append(Paragraph(
        "Encodes the 20-dimensional student profile vector into normalized continuous features and runs probabilistic inference "
        "across a 100-tree Random Forest classifier trained on university student benchmark distributions.",
        styles["BodyDark"]
    ))

    story.append(Paragraph("4.4 Confidence Tier Calibration", styles["SubSectionHeading"]))
    story.append(Paragraph(
        "• <b>High (⚡ Strong Fit)</b>: Final Score ≥ 78.0% (Primary recommended career path).<br/>"
        "• <b>Moderate (⚡ Viable Path)</b>: Final Score between 62.0% and 77.9% (Alternative career track).<br/>"
        "• <b>Exploring (⚡ Emerging Fit)</b>: Final Score < 62.0% (Developing alignment requiring targeted upskilling).",
        styles["BulletItem"]
    ))
    story.append(Spacer(1, 14))

    # =========================================================================
    # 5. EXPLAINABLE AI (XAI), SHAP DELTAS & GAP ANALYSIS
    # =========================================================================
    story.append(Paragraph("5. Explainable AI (XAI) & Competency Gap Analysis", styles["SectionHeading"]))
    story.append(Paragraph(
        "To eliminate black-box opacity and foster student trust, PathFinder AI provides two transparent diagnostic engines:",
        styles["BodyDark"]
    ))

    story.append(Paragraph("5.1 SHAP-Proxy Divergent Feature Attribution", styles["SubSectionHeading"]))
    story.append(Paragraph(
        "Calculates exact mathematical delta contributions (Δ%) relative to the benchmark career requirements. "
        "Positive deltas (Cyan bars) highlight competitive advantages that raised the student's ranking, while "
        "negative deltas (Rose bars) isolate specific performance deficits dragging down alignment.",
        styles["BodyDark"]
    ))

    story.append(Paragraph("5.2 Competency & Skill Gap Audit Engine", styles["SubSectionHeading"]))
    story.append(Paragraph(
        "Calculates normalized deficits (Target Requisite - Current Proficiency) across all technical and soft skills, "
        "categorizing them into actionable urgency tiers:<br/>"
        "• <b>High Urgency (Deficit ≥ 25%)</b>: Critical prerequisite gap requiring immediate structured intervention.<br/>"
        "• <b>Medium Priority (12% ≤ Deficit < 25%)</b>: Moderate gap that can be addressed via elective courses or certifications.<br/>"
        "• <b>Low / Minor (Deficit < 12%)</b>: Minor fine-tuning required for senior-level readiness.",
        styles["BodyDark"]
    ))
    story.append(Spacer(1, 14))

    # =========================================================================
    # 6. GAP-WEIGHTED COSINE SIMILARITY CERTIFICATION RECOMMENDER
    # =========================================================================
    story.append(Paragraph("6. Gap-Weighted Cosine Certification Recommender", styles["SectionHeading"]))
    story.append(Paragraph(
        "PathFinder AI connects identified student skill gaps directly to real-world credentials using an intelligent "
        "vector cosine similarity algorithm against a curated JSON catalog (AWS, Google Cloud, Microsoft Azure, Cisco, Meta, PMI):",
        styles["BodyDark"]
    ))

    cert_math_html = (
        "<b>Cosine Similarity Mathematical Formulation:</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>Cosine_Sim(G, C) = ( Σ [ w_i · Gap_i · Cert_i ] ) / ( ||G_w|| · ||C|| )</b><br/><br/>"
        "• <b>Domain Pruning Multiplier (1.30x – 1.60x)</b>: Guarantees industry relevance to the target career archetype.<br/>"
        "• <b>Academic Tier Multiplier (0.70x – 1.25x)</b>: Adapts recommendations based on academic year (Foundational for 1st/2nd Year, Professional for 3rd/4th Year)."
    )
    cert_table = Table([[Paragraph(cert_math_html, styles["FormulaBox"])]], colWidths=[504])
    cert_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BG_LIGHT),
        ("BOX", (0, 0), (-1, -1), 1, ACCENT_CYAN),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(cert_table)
    story.append(Spacer(1, 14))

    # =========================================================================
    # 7. DATASET & ML MODEL STUDIO + SQLITE PERSISTENCE
    # =========================================================================
    story.append(Paragraph("7. Dataset & Model Studio + Local SQLite Persistence", styles["SectionHeading"]))
    story.append(Paragraph(
        "PathFinder AI includes a fully functional Machine Learning Studio and persistent logging subsystem:",
        styles["BodyDark"]
    ))

    story.append(Paragraph("7.1 ML Model Studio Capabilities", styles["SubSectionHeading"]))
    story.append(Paragraph(
        "• <b>Benchmark Dataset Generator</b>: Generates 450+ balanced synthetic undergraduate student records.<br/>"
        "• <b>Custom Dataset Ingestion</b>: Allows professors and career advisors to upload custom student cohorts (.csv / .xlsx).<br/>"
        "• <b>On-the-Fly Retraining</b>: Dynamically retrains Random Forest classifiers with tunable trees and max depth, rendering live Train/Test accuracy metrics, confusion matrices, and Gini feature importance charts.<br/>"
        "• <b>Batch Prediction Studio</b>: Enables bulk evaluation of entire student cohorts with one-click CSV export.",
        styles["BulletItem"]
    ))

    story.append(Paragraph("7.2 SQLite Relational Logging Layer (career_records.db)", styles["SubSectionHeading"]))
    story.append(Paragraph(
        "Every completed career evaluation is securely stored with timestamp, GPA, academic year, top recommendation, match score, "
        "confidence tier, work style, primary SHAP driver, and critical gap. The <b>History Tab</b> provides longitudinal analytics, "
        "cohort score distributions, radar comparisons, and full database export.",
        styles["BulletItem"]
    ))
    story.append(Spacer(1, 14))

    # =========================================================================
    # 8. TECHNICAL SPECIFICATIONS & OPERATIONAL GUIDE
    # =========================================================================
    story.append(Paragraph("8. Technical Stack & Quickstart Operational Guide", styles["SectionHeading"]))

    tech_specs = [
        [Paragraph("<b>Component</b>", styles["TableHead"]), Paragraph("<b>Technology / Library</b>", styles["TableHead"]), Paragraph("<b>Version / Specification</b>", styles["TableHead"])],
        [Paragraph("Frontend & UI Framework", styles["TableCellBold"]), Paragraph("Streamlit Core", styles["TableCell"]), Paragraph(">= 1.30.0 (Reactive Stepper UI)", styles["TableCell"])],
        [Paragraph("Data Manipulation", styles["TableCellBold"]), Paragraph("Pandas & NumPy", styles["TableCell"]), Paragraph("Pandas >= 2.0.0, NumPy >= 1.24.0", styles["TableCell"])],
        [Paragraph("Machine Learning Core", styles["TableCellBold"]), Paragraph("Scikit-Learn", styles["TableCell"]), Paragraph("RandomForestClassifier, Gini Importances", styles["TableCell"])],
        [Paragraph("Interactive Visualizations", styles["TableCellBold"]), Paragraph("Plotly Graph Objects", styles["TableCell"]), Paragraph("Radar charts, SHAP attribution, Bars", styles["TableCell"])],
        [Paragraph("Embedded Database", styles["TableCellBold"]), Paragraph("SQLite3", styles["TableCell"]), Paragraph("Local career_records.db persistence", styles["TableCell"])],
        [Paragraph("Document Engine", styles["TableCellBold"]), Paragraph("ReportLab", styles["TableCell"]), Paragraph("Automated Technical Documentation Generator", styles["TableCell"])],
    ]
    tech_table = Table(tech_specs, colWidths=[140, 160, 204])
    tech_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 4.5),
    ]))
    story.append(tech_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>How to Run PathFinder AI:</b>", styles["BodyDarkBold"]))
    story.append(Paragraph(
        "1. Open PowerShell / Command Prompt and navigate to: <code>cd D:\\PathFinder_AI</code><br/>"
        "2. Launch the application: <code>python -m streamlit run app.py</code><br/>"
        "3. Access the live interface in your web browser at <code>http://localhost:8501</code>.",
        styles["BodyDark"]
    ))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    return pdf_path


if __name__ == "__main__":
    out_file = "PathFinder_AI_System_Documentation.pdf"
    if len(sys.argv) > 1:
        out_file = sys.argv[1]
    res_path = create_system_pdf(out_file)
    print(f"Successfully generated PDF documentation: {res_path}")
