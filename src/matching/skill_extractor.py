"""
EthioMatch — Skill Extractor
==============================
PURPOSE:
    Extract skills from any text (CV or job description) by matching
    against a master skill keyword list.

    Also performs skill gap analysis: given a CV skill set and a job
    skill set, returns what's matched, what's missing, and what's partial.

CONCEPT — Why keyword matching (not NER)?
    Named Entity Recognition (NER) needs labeled training data.
    We don't have labeled Ethiopian tech skill data.
    A well-curated keyword list works reliably, is transparent,
    and is easy to expand. We can always upgrade to NER later.

MATCHING STRATEGY:
    - Case-insensitive exact substring matching
    - Alias mapping: 'JS' → 'JavaScript', 'ML' → 'Machine Learning'
    - Partial match: CV has 'Python', job wants 'Python, Django' →
        Python = matched, Django = missing

WHAT THIS MODULE DOES:
    1. Load master skill list from data/skills/master_skills.json
    2. extract_skills(text) → set of skills found in any text
    3. compare_skills(cv_skills, job_skills) → gap analysis result
"""

import json
import re
from pathlib import Path


# ── Alias map: normalize common abbreviations ──────────────────────────────────
# Covers cases like 'JS' meaning 'JavaScript', 'ML' meaning 'Machine Learning'
SKILL_ALIASES: dict[str, str] = {
    # Programming
    "c language":    "C Programming",
    "c programming language": "C Programming",
    "js":            "JavaScript",
    "ts":            "TypeScript",
    "py":            "Python",
    "ml":            "Machine Learning",
    "dl":            "Deep Learning",
    "nlp":           "Natural Language Processing",
    "cv":            "Computer Vision",
    "oop":           "OOP",
    "k8s":           "Kubernetes",
    "gcp":           "Google Cloud",
    "psql":          "PostgreSQL",
    "mssql":         "SQL Server",
    "node":          "Node.js",
    "react.js":      "React",
    "reactjs":       "React",
    "angular.js":    "Angular",
    "angularjs":     "Angular",
    "vue.js":        "Vue",
    # Soft
    "problem-solving": "Problem Solving",
    "teamplayer":    "Teamwork",
}


# ── Skill loader ───────────────────────────────────────────────────────────────
_SKILLS_CACHE: dict | None = None   # loaded once, reused


def load_skill_list() -> dict:
    """
    Load the master skill list from JSON.
    Returns a dict of {category: [skill, ...]} and a flat set of all skills.
    Caches the result after first load.
    """
    global _SKILLS_CACHE
    if _SKILLS_CACHE is not None:
        return _SKILLS_CACHE

    skills_path = Path(__file__).parent.parent.parent / "data" / "skills" / "master_skills.json"

    if not skills_path.exists():
        raise FileNotFoundError(
            f"Skill list not found at {skills_path}. "
            "Make sure data/skills/master_skills.json exists."
        )

    with open(skills_path, encoding="utf-8") as f:
        _SKILLS_CACHE = json.load(f)

    return _SKILLS_CACHE


def get_all_skills() -> list[str]:
    """Return a flat list of all skills across all categories."""
    skill_dict = load_skill_list()
    all_skills = []
    for skills in skill_dict.values():
        all_skills.extend(skills)
    return all_skills


def get_skill_category(skill: str) -> str:
    """Return which category a skill belongs to."""
    skill_dict = load_skill_list()
    for category, skills in skill_dict.items():
        if any(skill.lower() == s.lower() for s in skills):
            return category
    return "other"


# ── Text normalizer ────────────────────────────────────────────────────────────
def normalize_text(text: str) -> str:
    """
    Clean text before skill matching:
    - Lowercase
    - Replace special chars with spaces (but keep . and + for C++, .NET)
    - Normalize whitespace
    """
    text = text.lower()
    text = re.sub(r"[^\w\s.+#/]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ── Core skill extractor ───────────────────────────────────────────────────────
def extract_skills(text: str) -> set[str]:
    """
    Find all skills from the master list that appear in the given text.

    Args:
        text: Any text — CV content, job description body, etc.

    Returns:
        Set of skill names (using canonical names from master_skills.json)

    Example:
        text = "I have 3 years of Python and built React apps with PostgreSQL"
        extract_skills(text) → {'Python', 'React', 'PostgreSQL'}
    """
    if not text or not text.strip():
        return set()

    # Apply alias normalization first
    normalized = normalize_text(text)
    for alias, canonical in SKILL_ALIASES.items():
        # Replace alias with canonical (whole word only)
        normalized = re.sub(
            rf"\b{re.escape(alias)}\b",
            canonical.lower(),
            normalized
        )

    found = set()
    all_skills = get_all_skills()

    for skill in all_skills:
        # Build a pattern that matches the skill as a whole word/phrase
        # e.g., "C++" needs special handling for regex
        pattern = re.escape(skill.lower())
        # Allow word boundary detection (handles "Python" vs "Python3")
        try:
            if re.search(rf"(?<!\w){pattern}(?!\w)", normalized):
                found.add(skill)
        except re.error:
            # If regex fails (rare edge case), try simple substring
            if skill.lower() in normalized:
                found.add(skill)

    return found


# ── Skill gap analysis ─────────────────────────────────────────────────────────
def compare_skills(cv_skills: set[str], job_skills: set[str]) -> dict:
    """
    Compare CV skills against job required skills.

    Args:
        cv_skills:  Skills extracted from the candidate's CV
        job_skills: Skills extracted from the job description

    Returns:
        {
            "matched":  [...],   # skills in both CV and job
            "missing":  [...],   # skills in job but NOT in CV
            "extra":    [...],   # skills in CV but not required by job
            "match_pct": float,  # percentage of job skills covered by CV
        }

    Example:
        cv_skills  = {'Python', 'React', 'PostgreSQL', 'Git'}
        job_skills = {'Python', 'Django', 'PostgreSQL', 'Docker'}

        Result:
            matched  = ['Python', 'PostgreSQL']
            missing  = ['Django', 'Docker']
            extra    = ['React', 'Git']
            match_pct = 50.0  (2 out of 4 job skills covered)

    WHY match_pct is calculated this way:
        We measure how much of the JOB's requirements the CV satisfies.
        Having extra skills is good but doesn't increase the match score.
        This prevents a CV full of irrelevant skills from scoring high.
    """
    # Normalize to lowercase for comparison, then restore canonical names
    cv_lower = {s.lower(): s for s in cv_skills}
    job_lower = {s.lower(): s for s in job_skills}

    matched_lower = set(cv_lower.keys()) & set(job_lower.keys())
    missing_lower = set(job_lower.keys()) - set(cv_lower.keys())
    extra_lower   = set(cv_lower.keys()) - set(job_lower.keys())

    # Restore canonical names
    matched = sorted(job_lower[s] for s in matched_lower)
    missing = sorted(job_lower[s] for s in missing_lower)
    extra   = sorted(cv_lower[s]  for s in extra_lower)

    match_pct = (len(matched) / len(job_skills) * 100) if job_skills else 0.0

    return {
        "matched":   matched,
        "missing":   missing,
        "extra":     extra,
        "match_pct": round(match_pct, 1),
    }


# ── Enricher: add skills to parsed job records ─────────────────────────────────
def enrich_jobs_with_skills(jobs: list[dict]) -> list[dict]:
    """
    Add extracted skills to each parsed job record.
    Modifies jobs in-place and returns the list.
    """
    for job in jobs:
        body = job.get("body", "") or ""
        title = job.get("title", "") or ""
        # Extract from both title and body for better coverage
        job["skills"] = sorted(extract_skills(title + " " + body))
    return jobs


# ── Script entry point (for testing) ──────────────────────────────────────────
if __name__ == "__main__":
    import json
    from pathlib import Path

    # ── Test 1: skill extraction from sample text ──────────────────────────────
    print("── Test 1: Skill Extraction ──────────────────────────────")
    sample_cv = """
    I am a software engineer with 3 years of experience.
    I work with Python, Django, and React for web development.
    I use PostgreSQL and MongoDB for databases.
    Familiar with Docker, Git, and deploying to AWS.
    I have built REST APIs and worked in Agile teams.
    """
    cv_skills = extract_skills(sample_cv)
    print(f"CV skills found ({len(cv_skills)}): {sorted(cv_skills)}")

    sample_jd = """
    We are looking for a Backend Engineer.
    Requirements: Python, FastAPI, PostgreSQL, Docker, Kubernetes, CI/CD.
    Nice to have: Redis, Machine Learning, AWS.
    Must have strong communication skills and work well in teams.
    """
    jd_skills = extract_skills(sample_jd)
    print(f"\nJD skills found ({len(jd_skills)}): {sorted(jd_skills)}")

    # ── Test 2: skill gap analysis ─────────────────────────────────────────────
    print("\n── Test 2: Skill Gap Analysis ────────────────────────────")
    gap = compare_skills(cv_skills, jd_skills)
    print(f"Match score : {gap['match_pct']}%")
    print(f"Matched     : {gap['matched']}")
    print(f"Missing     : {gap['missing']}")
    print(f"Extra (CV)  : {gap['extra']}")

    # ── Test 3: enrich real parsed jobs ───────────────────────────────────────
    print("\n── Test 3: Enrich Parsed Jobs ────────────────────────────")
    parsed_path = Path(__file__).parent.parent.parent / "data" / "processed" / "parsed_jobs.json"

    if parsed_path.exists():
        with open(parsed_path, encoding="utf-8") as f:
            jobs = json.load(f)

        jobs = enrich_jobs_with_skills(jobs)

        # Show 5 sample job skills
        for job in jobs[:5]:
            print(f"\n  {job['title']}")
            print(f"  Skills found: {job['skills'][:8]}")  # first 8 skills

        # Save enriched jobs
        output_path = Path(__file__).parent.parent.parent / "data" / "processed" / "enriched_jobs.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(jobs, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n✅ Saved {len(jobs)} enriched jobs to {output_path}")
    else:
        print("  No parsed_jobs.json found — run job_parser.py first")
