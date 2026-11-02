"""
EthioMatch — World-Class AI Job Matching & Skill Gap Analysis Platform
========================================================================
Design System: Modern Glassmorphic Dark/Slate SaaS with Neon Accents
Powered by: SBERT (all-MiniLM-L6-v2) + Semantic Skill Extraction Engine
"""

import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import src.parser.cv_parser as cv_parser_mod
import src.matching.skill_extractor as skill_extractor_mod
import src.matching.job_matcher as job_matcher_mod
import src.recommender.learning_recommender as learning_recommender_mod

# Force reload local modules so running Streamlit processes pick up changes
importlib.reload(cv_parser_mod)
importlib.reload(skill_extractor_mod)
importlib.reload(job_matcher_mod)
importlib.reload(learning_recommender_mod)

from src.parser.cv_parser import parse_cv
from src.matching.skill_extractor import extract_skills, compare_skills
from src.matching.job_matcher import match_cv_to_jobs, load_model, prepare_text_for_embedding
from src.recommender.learning_recommender import get_recommendations

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EthioMatch AI | Smart Job Intelligence & Skill Gap Radar",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── World-Class Custom CSS & Typography ───────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">

<style>
    /* Global Overrides */
    * {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 15% 15%, rgba(20, 24, 40, 0.95), rgba(10, 12, 20, 1));
        color: #f1f5f9;
    }

    /* Top Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 36px 40px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: -60px;
        right: -60px;
        width: 240px;
        height: 240px;
        background: radial-gradient(circle, rgba(16, 185, 129, 0.25) 0%, rgba(59, 130, 246, 0.15) 50%, transparent 70%);
        filter: blur(40px);
        pointer-events: none;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 30px;
        padding: 6px 14px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 14px;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        line-height: 1.15;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 12px 0;
    }

    .hero-subtitle {
        font-size: 1.08rem;
        color: #94a3b8;
        max-width: 680px;
        line-height: 1.6;
        margin: 0 0 22px 0;
    }

    .pulse-stats-row {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
    }

    .stat-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 8px 16px;
        font-size: 0.86rem;
        color: #cbd5e1;
    }

    .stat-pill strong {
        color: #f8fafc;
        font-weight: 700;
    }

    /* KPI Summary Cards */
    .kpi-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }
    .kpi-val {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 4px;
        background: linear-gradient(135deg, #ffffff, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .kpi-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }

    /* Job Card Design */
    .job-card {
        background: rgba(30, 41, 59, 0.55);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 20px;
        transition: all 0.25s ease;
        position: relative;
    }
    .job-card:hover {
        border-color: rgba(99, 102, 241, 0.45);
        box-shadow: 0 16px 32px -10px rgba(0, 0, 0, 0.6);
        transform: translateY(-2px);
    }

    .job-header-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 16px;
        margin-bottom: 16px;
    }

    .company-avatar {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 1.2rem;
        color: white;
        flex-shrink: 0;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35);
    }

    .job-meta-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 4px 0;
    }

    .job-meta-sub {
        font-size: 0.9rem;
        color: #94a3b8;
        display: flex;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
    }

    .score-badge {
        padding: 8px 16px;
        border-radius: 30px;
        font-weight: 800;
        font-size: 1.05rem;
        font-family: 'Space Grotesk', sans-serif;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        white-space: nowrap;
    }
    .score-badge-high {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
    }
    .score-badge-med {
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.4);
    }
    .score-badge-low {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }

    /* Skills Pill Tags */
    .chip {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 8px;
        margin: 3px 4px 3px 0;
    }
    .chip-matched {
        background: rgba(16, 185, 129, 0.12);
        color: #6ee7b7;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    .chip-missing {
        background: rgba(239, 68, 68, 0.12);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.25);
    }
    .chip-neutral {
        background: rgba(148, 163, 184, 0.1);
        color: #cbd5e1;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }

    /* Learning Card */
    .resource-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 14px 18px;
        margin-top: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .resource-link {
        color: #818cf8;
        font-weight: 600;
        text-decoration: none;
    }
    .resource-link:hover {
        text-decoration: underline;
        color: #a5b4fc;
    }

    /* Telegram CTA Button */
    .btn-tg {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(135deg, #229ED9 0%, #0088cc 100%);
        color: #ffffff !important;
        font-size: 0.86rem;
        font-weight: 700;
        padding: 8px 18px;
        border-radius: 10px;
        text-decoration: none !important;
        box-shadow: 0 4px 12px rgba(34, 158, 217, 0.3);
        transition: all 0.2s ease;
    }
    .btn-tg:hover {
        box-shadow: 0 6px 16px rgba(34, 158, 217, 0.5);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ───────────────────────────────────────────────────────────
def get_initials(name: str) -> str:
    """Extract clean 1-2 letter uppercase initials for company avatar."""
    if not name or name == "Not specified":
        return "ET"
    words = [w for w in name.replace("-", " ").split() if w and w[0].isalnum()]
    if not words:
        return "ET"
    if len(words) == 1:
        return words[0][:2].upper()
    return (words[0][0] + words[1][0]).upper()


@st.cache_data(show_spinner=False)
def load_cached_jobs() -> list[dict]:
    """Load pre-parsed jobs from enriched_jobs.json with caching."""
    jobs_path = PROJECT_ROOT / "data" / "processed" / "enriched_jobs.json"
    if jobs_path.exists():
        with open(jobs_path, encoding="utf-8") as f:
            return json.load(f)
    return []


@st.cache_resource(show_spinner=False)
def get_cached_job_embeddings(_jobs: list[dict]):
    """Pre-compute and cache embeddings for all parsed jobs to make matching instantaneous."""
    embed_path = PROJECT_ROOT / "data" / "processed" / "job_embeddings.npy"
    if embed_path.exists():
        try:
            return np.load(embed_path)
        except Exception:
            pass

    model = load_model()
    job_texts = [
        prepare_text_for_embedding(f"{job.get('title', '')} {job.get('body', '')}")
        for job in _jobs
    ]
    embeddings = model.encode(job_texts, batch_size=64, show_progress_bar=False)
    try:
        np.save(embed_path, embeddings)
    except Exception:
        pass
    return embeddings


# ── Sidebar Configuration ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:16px;">
            <div style="font-size:1.8rem;">🇪🇹</div>
            <div>
                <div style="font-weight:800; font-size:1.15rem; color:#f8fafc; letter-spacing:-0.02em;">EthioMatch AI</div>
                <div style="font-size:0.75rem; color:#94a3b8;">Ethiopian Talent & Job Engine</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📄 Upload Resume")
    uploaded_file = st.file_uploader(
        "Upload your CV (PDF or DOCX)",
        type=["pdf", "docx"],
        help="Processed entirely on your device with local neural embeddings.",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Matching Model Tuning")

    sbert_weight = st.slider(
        "Semantic Relevance (SBERT)",
        min_value=0.0, max_value=1.0, value=0.6, step=0.05,
        help="Weights conceptual and contextual fit (Sentence-BERT MiniLM)."
    )
    skill_weight = round(1.0 - sbert_weight, 2)
    st.caption(f"Exact Skill Coverage: **{int(skill_weight * 100)}%**")

    min_score = st.slider(
        "Minimum Match Cutoff",
        min_value=10, max_value=85, value=25, step=5,
        format="%d%%",
        help="Only display job listings exceeding this match threshold."
    )

    channel_filter = st.selectbox(
        "Filter by Channel",
        options=["All Channels", "josad_it", "josad_software", "josad_digital", "hahujobs", "afriworkamharic", "gadaanetwork", "jobs_in_ethio", "effoyjobs", "ethio_job_vacancy1", "vacancyforallethio"]
    )

    st.markdown("---")
    st.markdown("""
        <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:12px; padding:14px;">
            <div style="font-weight:700; font-size:0.82rem; color:#cbd5e1; margin-bottom:6px;">⚡ LIVE INTELLIGENCE</div>
            <div style="font-size:0.78rem; color:#94a3b8; line-height:1.5;">
                • 10 Telegram Job Hubs<br>
                • 705 Validated Active Vacancies<br>
                • Real-Time SBERT Similarity Engine
            </div>
        </div>
    """, unsafe_allow_html=True)


# ── Hero Section ───────────────────────────────────────────────────────────────
jobs = load_cached_jobs()
total_jobs = len(jobs)

st.markdown(f"""
<div class="hero-container">
    <div class="hero-badge">⚡ Real-Time Ethiopian Tech Talent Radar</div>
    <h1 class="hero-title">AI-Powered Job Matching &<br>Skill Gap Intelligence</h1>
    <p class="hero-subtitle">
        Bridging Ethiopian developers with verified vacancies across Telegram & national boards. 
        Deep neural semantic matching paired with immediate upskilling roadmaps.
    </p>
    <div class="pulse-stats-row">
        <div class="stat-pill">📍 <strong>Addis Ababa & Remote</strong></div>
        <div class="stat-pill">📊 <strong>{total_jobs}</strong> Active Vacancies Tracked</div>
        <div class="stat-pill">🤖 <strong>Sentence-BERT</strong> Neural Engine</div>
        <div class="stat-pill">🛡️ <strong>Zero Data Retention</strong> (100% Private)</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Guard: No File Uploaded ────────────────────────────────────────────────────
if uploaded_file is None:
    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.3); border: 2px dashed rgba(255, 255, 255, 0.12); border-radius: 20px; padding: 60px 30px; text-align: center; margin-top: 10px;">
        <div style="font-size: 3rem; margin-bottom: 12px;">📂</div>
        <h3 style="color:#ffffff; margin:0 0 8px 0; font-weight:700;">Upload Your CV to Unlock Your Matches</h3>
        <p style="color:#94a3b8; max-width:540px; margin:0 auto 24px auto; font-size:0.95rem;">
            Upload your resume in the left sidebar (PDF or DOCX). Our AI will parse your skills, 
            run cross-semantic alignment against 705+ Ethiopian tech jobs, and compute your personalized skill gap.
        </p>
        <div style="display:inline-flex; gap:12px; color:#cbd5e1; font-size:0.85rem;">
            <span>✨ Instant ATS Compatibility</span> • 
            <span>🎯 Missing Skills Radar</span> • 
            <span>📚 Free Verified Curricula</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Process Uploaded CV ────────────────────────────────────────────────────────
with st.spinner("⚡ Extracting technical competencies from your resume..."):
    cv_result = parse_cv(uploaded_file, uploaded_file.name)

if cv_result["error"]:
    st.error(f"❌ {cv_result['error']}")
    st.stop()

cv_text = cv_result["raw_text"]
cv_skills = set(cv_result["skills"])

# ── Matching Calculation ───────────────────────────────────────────────────────
with st.spinner(f"🧠 Computing neural match across {total_jobs} live positions..."):
    cached_job_embeds = get_cached_job_embeddings(jobs) if jobs else None
    try:
        raw_results = match_cv_to_jobs(
            cv_text, cv_skills, jobs,
            sbert_weight=sbert_weight,
            skill_weight=skill_weight,
            job_embeddings=cached_job_embeds,
        )
    except TypeError:
        raw_results = match_cv_to_jobs(
            cv_text, cv_skills, jobs,
            sbert_weight=sbert_weight,
            skill_weight=skill_weight,
        )

# Apply Channel Filter & Score Cutoff
filtered = [
    r for r in raw_results
    if r["combined_score"] >= min_score and (channel_filter == "All Channels" or r["channel"] == channel_filter)
]

# ── Summary KPI Dashboard ──────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-val">{cv_result['skill_count']}</div>
        <div class="kpi-label">Skills Found in CV</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-val">{len(filtered)}</div>
        <div class="kpi-label">Qualified Matches</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    top_score = f"{filtered[0]['combined_score']:.0f}%" if filtered else "0%"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-val">{top_score}</div>
        <div class="kpi-label">Highest Match Fit</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    total_missing = sum(len(r["missing_skills"]) for r in filtered[:10])
    avg_gap = f"{total_missing // max(1, min(len(filtered), 10))}"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-val">{avg_gap}</div>
        <div class="kpi-label">Avg. Skill Gaps / Job</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# ── Detected Skills Pill Cloud ─────────────────────────────────────────────────
if cv_skills:
    chips_html = "".join([f'<span class="chip chip-neutral">✓ {s}</span>' for s in sorted(cv_skills)])
    st.markdown(f"""
    <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255,255,255,0.06); border-radius: 14px; padding: 16px 20px; margin-bottom: 24px;">
        <div style="font-weight: 700; font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; margin-bottom: 8px;">
            Recognized Profile Competencies:
        </div>
        <div>{chips_html}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Main Tabs Navigation ───────────────────────────────────────────────────────
tab_jobs, tab_gap, tab_roadmap, tab_trends = st.tabs([
    "🎯 Top Matched Jobs",
    "🔍 Skill Gap Radar",
    "🎓 Upskilling Roadmap",
    "📊 Ethiopia Tech Market Insights"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Matched Jobs
# ═══════════════════════════════════════════════════════════════════════════════
with tab_jobs:
    if not filtered:
        st.warning("⚠️ No jobs match your current cutoff filter. Lower the threshold in the sidebar to view more roles.")
    else:
        st.markdown(f"<div style='font-size:0.95rem; color:#94a3b8; margin-bottom:18px;'>Showing <strong>{min(len(filtered), 25)}</strong> highest-ranking roles (out of {len(filtered)} candidates):</div>", unsafe_allow_html=True)

        # Shared CSS injected once for all cards (components.html is sandboxed per call)
        CARD_CSS = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');
        body { margin:0; padding:0; background:transparent; font-family:'Plus Jakarta Sans',sans-serif; color:#f1f5f9; }
        .job-card {
            background: rgba(30,41,59,0.85);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 18px;
            padding: 22px 24px;
            transition: all 0.25s ease;
        }
        .job-header-row { display:flex; justify-content:space-between; align-items:flex-start; gap:16px; margin-bottom:16px; }
        .company-avatar {
            width:48px; height:48px; border-radius:12px;
            background: linear-gradient(135deg,#4f46e5 0%,#06b6d4 100%);
            display:flex; align-items:center; justify-content:center;
            font-weight:800; font-size:1.2rem; color:white; flex-shrink:0;
            box-shadow: 0 4px 12px rgba(79,70,229,0.35);
        }
        .job-meta-title { font-size:1.18rem; font-weight:700; color:#ffffff; margin:0 0 5px 0; }
        .job-meta-sub  { font-size:0.88rem; color:#94a3b8; display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
        .score-badge {
            padding:8px 16px; border-radius:30px; font-weight:800; font-size:1rem;
            font-family:'Space Grotesk',sans-serif; display:inline-flex; align-items:center;
            gap:6px; white-space:nowrap;
        }
        .score-badge-high { background:rgba(16,185,129,0.15); color:#34d399; border:1px solid rgba(16,185,129,0.4); box-shadow:0 0 15px rgba(16,185,129,0.2); }
        .score-badge-med  { background:rgba(59,130,246,0.15);  color:#60a5fa; border:1px solid rgba(59,130,246,0.4); }
        .score-badge-low  { background:rgba(245,158,11,0.15);  color:#fbbf24; border:1px solid rgba(245,158,11,0.4); }
        .skills-grid {
            display:grid; grid-template-columns:1fr 1fr; gap:16px;
            margin:14px 0; padding:12px 16px;
            background:rgba(15,23,42,0.55); border-radius:12px;
            border:1px solid rgba(255,255,255,0.05);
        }
        .skill-col-label { font-size:0.72rem; font-weight:700; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:6px; }
        .chip { display:inline-flex; align-items:center; gap:5px; font-size:0.78rem; font-weight:600; padding:4px 10px; border-radius:8px; margin:3px 4px 3px 0; }
        .chip-matched { background:rgba(16,185,129,0.12); color:#6ee7b7; border:1px solid rgba(16,185,129,0.25); }
        .chip-missing  { background:rgba(239,68,68,0.12);  color:#fca5a5; border:1px solid rgba(239,68,68,0.25); }
        .card-footer { display:flex; justify-content:space-between; align-items:center; margin-top:14px; }
        .score-detail { font-size:0.8rem; color:#64748b; }
        .score-detail strong { color:#cbd5e1; }
        .btn-tg {
            display:inline-flex; align-items:center; gap:8px;
            background:linear-gradient(135deg,#229ED9 0%,#0088cc 100%);
            color:#ffffff !important; font-size:0.86rem; font-weight:700;
            padding:8px 18px; border-radius:10px; text-decoration:none !important;
            box-shadow:0 4px 12px rgba(34,158,217,0.3);
        }
        </style>
        """

        for idx, r in enumerate(filtered[:25]):
            score = r["combined_score"]
            if score >= 65:
                badge_class = "score-badge-high"
                badge_icon = "🟢"
            elif score >= 45:
                badge_class = "score-badge-med"
                badge_icon = "🔵"
            else:
                badge_class = "score-badge-low"
                badge_icon = "🟡"

            initials = get_initials(r["company"])

            if r["matched_skills"]:
                matched_chips = "".join(
                    f'<span class="chip chip-matched">✅ {s}</span>'
                    for s in r["matched_skills"]
                )
            else:
                matched_chips = '<span style="font-size:0.8rem;color:#64748b;">Semantic match only</span>'

            if r["missing_skills"]:
                missing_chips = "".join(
                    f'<span class="chip chip-missing">❌ {s}</span>'
                    for s in r["missing_skills"]
                )
            else:
                missing_chips = '<span style="font-size:0.8rem;color:#34d399;">✓ No skill gaps detected!</span>'

            card_html = (
                CARD_CSS
                + f"""
                <div class="job-card">
                    <div class="job-header-row">
                        <div style="display:flex;gap:14px;align-items:center;">
                            <div class="company-avatar">{initials}</div>
                            <div>
                                <div class="job-meta-title">{r['title']}</div>
                                <div class="job-meta-sub">
                                    <span>🏢 {r['company']}</span>
                                    <span style="color:#334155;">•</span>
                                    <span>📢 @{r['channel']}</span>
                                    <span style="color:#334155;">•</span>
                                    <span>⏰ Deadline: <strong style="color:#e2e8f0;">{r['deadline_str']}</strong></span>
                                </div>
                            </div>
                        </div>
                        <div class="score-badge {badge_class}">{badge_icon} {score:.0f}% Match</div>
                    </div>

                    <div class="skills-grid">
                        <div>
                            <div class="skill-col-label">✅ Covered Skills</div>
                            <div>{matched_chips}</div>
                        </div>
                        <div>
                            <div class="skill-col-label">🔴 Missing Requirements</div>
                            <div>{missing_chips}</div>
                        </div>
                    </div>

                    <div class="card-footer">
                        <div class="score-detail">
                            Semantic: <strong>{r['sbert_score']:.0f}%</strong>
                            &nbsp;|&nbsp;
                            Skill: <strong>{r['skill_score']:.0f}%</strong>
                        </div>
                        <a href="{r['url']}" target="_blank" class="btn-tg">
                            ✈ View on Telegram
                        </a>
                    </div>
                </div>
                """
            )
            components.html(card_html, height=270, scrolling=False)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Skill Gap Radar
# ═══════════════════════════════════════════════════════════════════════════════
with tab_gap:
    st.markdown("### 🔍 Collective Skill Gap Analysis")
    st.markdown("Which skills are standing between you and the highest number of hiring companies?")

    gap_freq = {}
    for r in filtered[:40]:
        for sk in r["missing_skills"]:
            gap_freq[sk] = gap_freq.get(sk, 0) + 1

    if gap_freq:
        sorted_gaps = sorted(gap_freq.items(), key=lambda x: x[1], reverse=True)[:15]
        df_gaps = pd.DataFrame(sorted_gaps, columns=["Skill", "Roles Requiring It"])

        col_chart, col_detail = st.columns([3, 2])
        
        with col_chart:
            st.markdown("#### Most Requested Missing Skills")
            st.bar_chart(df_gaps.set_index("Skill"), color="#6366f1")

        with col_detail:
            st.markdown("#### High-ROI Action Items")
            st.markdown("Mastering these top 3 competencies unlocks the most immediate roles:")
            for rank, (sk, cnt) in enumerate(sorted_gaps[:5], 1):
                st.markdown(f"""
                <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 10px 14px; margin-bottom: 8px;">
                    <strong style="color:#818cf8;">#{rank} {sk}</strong> — Needed by <strong>{cnt} matching jobs</strong>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.success("🎉 You possess all required technical skills for the matched roles!")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Upskilling Roadmap
# ═══════════════════════════════════════════════════════════════════════════════
with tab_roadmap:
    st.markdown("### 🎓 Tailored Learning Roadmap")
    st.markdown("Direct, free, high-reputation resources to bridge your current vacancies:")

    if gap_freq:
        priority_skills = [sk for sk, _ in sorted(gap_freq.items(), key=lambda x: x[1], reverse=True)[:8]]
        recs = get_recommendations(priority_skills)

        for rec in recs:
            sk_name = rec["skill"]
            cnt = gap_freq.get(sk_name, 0)
            
            st.markdown(f"""
            <div style="margin-top:18px; margin-bottom:6px; font-weight:700; font-size:1.05rem; color:#f1f5f9; display:flex; align-items:center; gap:8px;">
                <span style="color:#ef4444;">●</span> {sk_name}
                <span style="font-size:0.75rem; background:rgba(239,68,68,0.15); color:#fca5a5; padding:2px 8px; border-radius:6px;">{cnt} matching jobs</span>
            </div>
            """, unsafe_allow_html=True)

            for item in rec["resources"]:
                badge_type = item.get("type", "course").upper()
                st.markdown(f"""
                <div class="resource-card">
                    <div>
                        <div style="font-weight:600; color:#ffffff; font-size:0.95rem;">{item['name']}</div>
                        <div style="font-size:0.75rem; color:#94a3b8;">Format: {badge_type} • Cost: {item['cost']}</div>
                    </div>
                    <a href="{item['url']}" target="_blank" class="resource-link">
                        Start Learning ↗
                    </a>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No gaps to address! You're ready to apply immediately.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: Ethiopian Tech Market Insights
# ═══════════════════════════════════════════════════════════════════════════════
with tab_trends:
    st.markdown("### 📊 Live Ethiopian Tech Market Intelligence")
    st.markdown("Empirical insights derived from 705 scraped vacancies across active Telegram channels:")

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("#### Vacancies by Telegram Channel")
        channel_counts = pd.Series([j["channel"] for j in jobs]).value_counts().reset_index()
        channel_counts.columns = ["Channel", "Active Posts"]
        st.dataframe(channel_counts, use_container_width=True)

    with col_t2:
        st.markdown("#### Top 10 In-Demand Technologies in Ethiopia")
        all_skills_in_jobs = []
        for j in jobs:
            all_skills_in_jobs.extend(j.get("skills", []))
        
        if all_skills_in_jobs:
            skill_counts = pd.Series(all_skills_in_jobs).value_counts().head(10).reset_index()
            skill_counts.columns = ["Skill", "Demand Count"]
            st.bar_chart(skill_counts.set_index("Skill"), color="#10b981")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center; color:#64748b; font-size:0.8rem; padding: 12px 0;">
    <div>EthioMatch v2.0 Enterprise • Designed for Ethiopian Engineers & Tech Hiring Hubs</div>
    <div>Powered by SBERT Neural Embeddings & Telethon</div>
</div>
""", unsafe_allow_html=True)
