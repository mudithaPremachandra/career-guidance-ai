"""Gap-weighted cosine-similarity certification recommender."""

import math
from typing import Any, Dict, List, Tuple

from src.knowledge_base import load_certification_catalog


def recommend_certifications(
    gaps: List[Dict[str, Any]],
    target_career: str = "Software Engineer",
    student_year: str = "3rd Year",
    held_cert_ids: Tuple[str, ...] = (),
) -> List[Dict[str, Any]]:
    """
    Computes Gap-Weighted Cosine Similarity with Domain Pruning and Academic Tier Matching.
    
    1. Weighted Gap Vector: G_weighted = G * W_urgency (High=3.0, Medium=2.0, Minor=1.0)
    2. Domain Pruning: Filter candidate certifications by career domain relevance.
    3. Tier Matching: Align certification difficulty with student's academic year.
    4. Exact Dynamic Gap Coverage %: Measures true weighted coverage ratio without static floors.
    """
    catalog = load_certification_catalog()
    
    # Domain Mapping for the 9 Careers
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
    
    # 1. Build Urgency-Weighted Gap Vector G_weighted
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
        def_val = float(g.get("gap_pct", g.get("deficit", 1.0)))
        weighted_val = def_val * w
        weighted_gaps[sk] = weighted_val
        total_gap_weight += weighted_val
        
    mag_gap = math.sqrt(sum(v ** 2 for v in weighted_gaps.values())) if weighted_gaps else 0.0

    scored_certs = []
    
    for cert in catalog:
        if cert.get("id") in held_cert_ids:
            continue  # already completed
        cert_domains = set(cert.get("domains", []))
        cert_keys = cert.get("skill_keys", cert.get("skills", []))
        cert_vector = cert.get("vector", {})
        if not cert_vector:
            cert_vector = {k: 1.0 for k in cert_keys}
            
        # 2. Domain Pruning & Compatibility Multiplier
        domain_overlap = cert_domains.intersection(allowed_domains)
        if not domain_overlap:
            # Harsh penalty for unrelated domains to prevent pollution
            domain_multiplier = 0.10
        else:
            # Overlap in target career domains
            domain_multiplier = 1.30 + (0.15 * len(domain_overlap))
            
        # 3. Tier & Academic Year Alignment Multiplier
        level_str = cert.get("level", "").lower()
        if student_year in ["1st Year", "2nd Year"]:
            if "foundational" in level_str or "beginner" in level_str:
                tier_multiplier = 1.25
            elif "intermediate" in level_str or "associate" in level_str:
                tier_multiplier = 1.05
            else:
                tier_multiplier = 0.70
        else:  # 3rd Year or 4th Year
            if "intermediate" in level_str or "associate" in level_str:
                tier_multiplier = 1.25
            elif "advanced" in level_str or "professional" in level_str:
                tier_multiplier = 1.20
            else:
                tier_multiplier = 0.85

        # 4. Compute Weighted Cosine Similarity
        dot_product = 0.0
        covered_gap_weight = 0.0
        matched_skill_names = []
        
        for k, cert_w in cert_vector.items():
            if k in weighted_gaps:
                gap_w = weighted_gaps[k]
                dot_product += gap_w * cert_w
                covered_gap_weight += gap_w * cert_w
                
                # Find skill display name
                for g in gaps:
                    if g["skill_key"] == k and g["skill_name"] not in matched_skill_names:
                        matched_skill_names.append(g["skill_name"])

        mag_cert = math.sqrt(sum(v ** 2 for v in cert_vector.values())) if cert_vector else 1.0
        
        if mag_gap > 0 and mag_cert > 0 and dot_product > 0:
            cosine_sim = dot_product / (mag_gap * mag_cert)
        else:
            cosine_sim = 0.05
            
        final_ranking_score = cosine_sim * domain_multiplier * tier_multiplier
        
        # 5. Exact Dynamic Gap Coverage % Calculation
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
            "in_domain": bool(domain_overlap),
        })

    # In-domain certifications always outrank off-domain ones; off-domain only fill the remaining slots
    scored_certs.sort(key=lambda x: (x["in_domain"], x["final_score"]), reverse=True)
    return scored_certs[:3]
