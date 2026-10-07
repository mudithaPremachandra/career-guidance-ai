"""Student profile data model."""

from typing import Dict, List


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
