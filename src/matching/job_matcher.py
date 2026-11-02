"""
EthioMatch — SBERT Job Matcher
================================
PURPOSE:
    Compute semantic similarity between a CV and job descriptions
    using Sentence Transformers (SBERT).

CONCEPT — What is SBERT?
    SBERT (Sentence-BERT) converts text into dense vectors (embeddings)
    that capture MEANING, not just keywords.

    Example:
        "Backend developer with Python experience"
        "Server-side engineer proficient in Python"
        → These have DIFFERENT words but SIMILAR meaning.
        → SBERT gives them a high cosine similarity score.

    Traditional TF-IDF would score them low because the exact words differ.
    SBERT understands they mean the same thing.

MODEL CHOICE — all-MiniLM-L6-v2
    - Small (80MB) — runs on CPU in <1 second per match
    - Trained on 1 billion sentence pairs
    - Produces 384-dimensional vectors
    - Best balance of speed and quality for our use case
    - Free, open source, no API key needed

WHAT THIS MODULE DOES:
    1. Load the SBERT model (once, cached in memory)
    2. Encode CV text into a vector
    3. Encode each job description into a vector
    4. Compute cosine similarity between CV vector and each job vector
    5. Combine SBERT score with skill match score for final ranking
    6. Return ranked list of jobs with scores and skill gap analysis
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Import our skill modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from matching.skill_extractor import extract_skills, compare_skills


# ── Model loader (singleton pattern) ──────────────────────────────────────────
_MODEL: Optional[SentenceTransformer] = None


def load_model() -> SentenceTransformer:
    """
    Load the SBERT model. Downloads on first use (~80MB), then cached.

    WHY SINGLETON:
        Loading a model takes 2-3 seconds. If we loaded it for every
        match request, the app would be painfully slow. Loading once
        and reusing is standard practice in ML serving.
    """
    global _MODEL
    if _MODEL is None:
        print("Loading SBERT model (all-MiniLM-L6-v2)... ", end="", flush=True)
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        print("done ✅")
    return _MODEL


# ── Text preparation ──────────────────────────────────────────────────────────
def prepare_text_for_embedding(text: str, max_chars: int = 2000) -> str:
    """
    Prepare text for SBERT encoding.

    WHY TRUNCATE TO 2000 CHARS?
        SBERT models have a max token limit (~256 tokens for MiniLM).
        Text beyond this is silently ignored by the model anyway.
        Truncating ourselves is cleaner and avoids wasting computation.
        The first 2000 chars of a CV/JD contain the most important info.
    """
    if not text:
        return ""
    # Remove excessive whitespace
    text = " ".join(text.split())
    return text[:max_chars]


# ── Core matching function ────────────────────────────────────────────────────
def compute_match_score(cv_text: str, job_text: str) -> float:
    """
    Compute cosine similarity between a CV and a job description.

    Returns:
        float between 0.0 and 1.0
        0.0 = completely unrelated
        1.0 = identical meaning

    WHAT THE SCORE MEANS (from testing on real data):
        0.0 - 0.2  → Very poor match (different field entirely)
        0.2 - 0.4  → Weak match (some overlap but different focus)
        0.4 - 0.6  → Moderate match (relevant field, missing specifics)
        0.6 - 0.8  → Strong match (good fit, minor gaps)
        0.8 - 1.0  → Excellent match (very close to requirements)
    """
    model = load_model()

    cv_prepared = prepare_text_for_embedding(cv_text)
    job_prepared = prepare_text_for_embedding(job_text)

    if not cv_prepared or not job_prepared:
        return 0.0

    # Encode both texts into vectors
    embeddings = model.encode([cv_prepared, job_prepared])

    # Compute cosine similarity
    similarity = cosine_similarity(
        embeddings[0].reshape(1, -1),
        embeddings[1].reshape(1, -1)
    )[0][0]

    # Clamp to [0, 1] — cosine similarity can technically be negative
    return max(0.0, min(1.0, float(similarity)))


# ── Batch matching: CV against all jobs ───────────────────────────────────────
def match_cv_to_jobs(cv_text: str, cv_skills: set, jobs: list[dict],
                     sbert_weight: float = 0.6,
                     skill_weight: float = 0.4,
                     job_embeddings: Optional[np.ndarray] = None,
                     **kwargs) -> list[dict]:
    """
    Match a CV against all jobs and return ranked results.

    Args:
        cv_text:      Cleaned CV text
        cv_skills:    Set of skills extracted from CV
        jobs:         List of parsed job dicts (from job_parser.py)
        sbert_weight: Weight for SBERT semantic score (default 0.6)
        skill_weight: Weight for skill match score (default 0.4)

    Returns:
        List of job results, sorted by combined score (highest first).
        Each result contains the job info + scores + skill gap analysis.

    WHY 60/40 SPLIT?
        SBERT captures overall semantic fit (role type, industry, level).
        Skill matching captures specific technical requirements.
        We weight SBERT slightly higher because:
        1. Job descriptions are often truncated (fewer skills to match)
        2. SBERT handles synonyms that keyword matching misses
        3. A candidate who matches the job's "vibe" is often a good fit
           even if they're missing one specific tool

        These weights are configurable — users or future A/B tests
        could adjust them.
    """
    model = load_model()

    if not cv_text or not jobs:
        return []

    # ── Step 1: Encode CV once ────────────────────────────────────────────────
    cv_prepared = prepare_text_for_embedding(cv_text)
    cv_embedding = model.encode([cv_prepared])[0]

    # ── Step 2: Encode all job descriptions in one batch (or use cached) ──────
    if job_embeddings is None:
        job_texts = []
        for job in jobs:
            text = f"{job.get('title', '')} {job.get('body', '')}"
            job_texts.append(prepare_text_for_embedding(text))
        job_embeddings = model.encode(job_texts, show_progress_bar=False, batch_size=64)

    # ── Step 3: Compute all similarities at once ──────────────────────────────
    similarities = cosine_similarity(
        cv_embedding.reshape(1, -1),
        job_embeddings
    )[0]

    # ── Step 4: Combine with skill matching ───────────────────────────────────
    results = []

    for i, job in enumerate(jobs):
        sbert_score = max(0.0, float(similarities[i]))

        # Skill gap analysis
        job_skills_list = job.get("skills", [])
        job_skills_set = set(job_skills_list) if job_skills_list else set()

        if job_skills_set:
            gap = compare_skills(cv_skills, job_skills_set)
            skill_score = gap["match_pct"] / 100.0
        else:
            # No skills extracted from job → rely entirely on SBERT
            gap = {
                "matched": [],
                "missing": [],
                "extra": sorted(cv_skills),
                "match_pct": 0.0
            }
            skill_score = 0.0
            # Adjust weights when no skill data available
            sbert_weight_adj = 1.0
            skill_weight_adj = 0.0

        # Combined score
        if job_skills_set:
            combined = (sbert_weight * sbert_score) + (skill_weight * skill_score)
        else:
            combined = sbert_score  # pure SBERT when no skills

        results.append({
            # Job info
            "title":        job.get("title", "Unknown"),
            "company":      job.get("company", "Not specified"),
            "deadline_str": job.get("deadline_str", "Not specified"),
            "deadline":     job.get("deadline"),
            "channel":      job.get("channel", ""),
            "url":          job.get("url", ""),
            "language":     job.get("language", "english"),
            # Scores
            "sbert_score":    round(sbert_score * 100, 1),
            "skill_score":    round(skill_score * 100, 1),
            "combined_score": round(combined * 100, 1),
            # Skill gap
            "matched_skills": gap["matched"],
            "missing_skills": gap["missing"],
            "extra_skills":   gap["extra"],
            "job_skills":     sorted(job_skills_set),
        })

    # Sort by combined score, highest first
    results.sort(key=lambda x: x["combined_score"], reverse=True)

    return results


# ── Script entry point (for testing) ──────────────────────────────────────────
if __name__ == "__main__":
    print("── SBERT Matcher Test ─────────────────────────────────────")

    # ── Test 1: Single pair matching ──────────────────────────────────────────
    print("\nTest 1 — Single pair cosine similarity:")

    cv_sample = """
    Software Engineer with 3 years of experience in Python, Django, and React.
    Built REST APIs, worked with PostgreSQL and MongoDB databases.
    Familiar with Docker, Git, AWS deployment, and Agile practices.
    BSc in Computer Science.
    """
    jd_sample = """
    Backend Engineer needed. Requirements: Python, FastAPI, PostgreSQL,
    Docker, Kubernetes, CI/CD. Experience with cloud deployment on AWS.
    Strong communication skills and Agile experience required.
    """
    score = compute_match_score(cv_sample, jd_sample)
    print(f"  CV vs Backend Engineer JD → Similarity: {score:.1%}")

    jd_unrelated = """
    We are hiring an Accountant. Requirements: QuickBooks, Excel,
    financial reporting, tax compliance, and CPA certification.
    Must have 5 years of accounting experience.
    """
    score2 = compute_match_score(cv_sample, jd_unrelated)
    print(f"  CV vs Accountant JD      → Similarity: {score2:.1%}")
    print(f"  (Backend should score higher than Accountant — {'✅ PASS' if score > score2 else '❌ FAIL'})")

    # ── Test 2: Batch matching with real parsed jobs ──────────────────────────
    print("\nTest 2 — Batch matching against real parsed jobs:")
    jobs_path = Path(__file__).parent.parent.parent / "data" / "processed" / "enriched_jobs.json"

    if jobs_path.exists():
        with open(jobs_path, encoding="utf-8") as f:
            jobs = json.load(f)

        cv_skills = extract_skills(cv_sample)
        print(f"  CV skills: {sorted(cv_skills)}")
        print(f"  Matching against {len(jobs)} jobs...")

        results = match_cv_to_jobs(cv_sample, cv_skills, jobs)

        print(f"\n  Top 10 matches:")
        print(f"  {'Rank':<5} {'Score':>6} {'SBERT':>6} {'Skill':>6}  {'Title':<40} {'Company'}")
        print(f"  {'─'*5} {'─'*6} {'─'*6} {'─'*6}  {'─'*40} {'─'*20}")

        for rank, r in enumerate(results[:10], 1):
            print(
                f"  {rank:<5} {r['combined_score']:>5.1f}% "
                f"{r['sbert_score']:>5.1f}% "
                f"{r['skill_score']:>5.1f}%  "
                f"{r['title'][:40]:<40} {r['company'][:20]}"
            )

        print(f"\n  Bottom 3 matches:")
        for r in results[-3:]:
            print(f"  {r['combined_score']:>5.1f}%  {r['title'][:50]}")
    else:
        print("  No enriched_jobs.json found — run skill_extractor.py first")

    print("\n✅ SBERT Matcher test complete")
