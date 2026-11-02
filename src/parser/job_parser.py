"""
EthioMatch — Job Parser
========================
PURPOSE:
    Parse raw Telegram messages into structured job records.
    Built from real data audit findings (Stage 2).

KEY FINDINGS FROM AUDIT:
    - JOSAD channels: very structured, reliable regex parsing
    - @afriworkamharic: Amharic labels, English values + Gregorian dates
    - All deadlines are Gregorian English (no Ethiopian calendar needed)
    - Some posts are digest lists — must be filtered out
    - Some posts have no deadline (remote/LinkedIn sourced) — keep them

WHAT THIS MODULE DOES:
    1. Detects channel type (JOSAD / Afriwork / Other)
    2. Extracts title, company, deadline, body using regex
    3. Parses deadline string to Python datetime
    4. Filters out expired jobs and digest posts
    5. Returns clean structured records ready for matching
"""

import re
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import dateparser


# ── Channel type detection ─────────────────────────────────────────────────────
JOSAD_CHANNELS = {"josad_it", "josad_digital", "josad_software"}
AFRIWORK_CHANNELS = {"afriworkamharic"}


def detect_channel_type(channel: str) -> str:
    """Returns 'josad', 'afriwork', or 'other'."""
    if channel in JOSAD_CHANNELS:
        return "josad"
    if channel in AFRIWORK_CHANNELS:
        return "afriwork"
    return "other"


# ── Language detection (fixed from audit finding #2) ──────────────────────────
def detect_language(text: str) -> str:
    """
    Returns 'amharic' if ≥5% of characters are Ethiopic script.
    Returns 'english' otherwise.

    WHY: Original detector flagged any Ethiopic character as Amharic.
    A company name like 'Guzogo (ጉዞጎ)' has only 3 Ethiopic chars in
    an otherwise English post — wrong to classify that as Amharic.
    """
    if not text:
        return "english"
    ethiopic_chars = sum(1 for c in text if "\u1200" <= c <= "\u137F")
    ratio = ethiopic_chars / len(text)
    return "amharic" if ratio >= 0.05 else "english"


# ── Digest post detection (audit finding #3) ──────────────────────────────────
def is_digest_post(text: str) -> bool:
    """
    Returns True if this post is a digest/roundup (list of links),
    not an actual job description.

    Digest posts look like:
        **Top 10 IT Jobs This Week**
        1. [Job Title](https://t.me/...)
        2. [Other Title](https://t.me/...)
    """
    # Count markdown-style links [text](url)
    links = re.findall(r"\[.+?\]\(https?://\S+\)", text)
    has_deadline = bool(re.search(r"[Dd]eadline|ማብቂያ", text))
    # If 3+ links and no deadline field → it's a digest
    return len(links) >= 3 and not has_deadline


# ── Field extractors by channel type ──────────────────────────────────────────

def extract_josad_fields(text: str) -> dict:
    """
    Extract fields from JOSAD-format posts.

    JOSAD format (from real data):
        **Job Title**
        Company: **Company Name**
        Deadline: **Nov 24, 2026**
        Salary: **...**
        [description]
        #hashtags
        @channel | josad.net
    """
    # Title: first bold text on the first line
    title_match = re.search(r"^\*\*(.+?)\*\*", text.strip(), re.MULTILINE)
    title = title_match.group(1).strip() if title_match else None

    # Company
    company_match = re.search(r"Company:\s*\*\*(.+?)\*\*", text)
    company = company_match.group(1).strip() if company_match else None

    # Deadline
    deadline_match = re.search(r"Deadline:\s*\*\*(.+?)\*\*", text)
    deadline_str = deadline_match.group(1).strip() if deadline_match else None

    # Body: everything between title and hashtags
    body = re.sub(r"#\w+", "", text)            # remove hashtags
    body = re.sub(r"@\w+\s*\|.*$", "", body, flags=re.MULTILINE)  # remove footer
    body = re.sub(r"\*\*|\n{3,}", " ", body).strip()  # remove bold markers

    return {
        "title": title,
        "company": company,
        "deadline_str": deadline_str,
        "body": body,
    }


def extract_afriwork_fields(text: str) -> dict:
    """
    Extract fields from @afriworkamharic posts.

    Afriwork format (from real data):
        የስራው መጠሪያ: **Job Title**
        [Amharic labels for type, location, salary]
        የማመልከቻ ማብቂያ ቀን: **November 13th, 2026**
        **የስራው ዝርዝር**:
        [English job description]
        Company Name
        ከ፡ afriworket.com | @afriworkamharic
    """
    # Title (Amharic label for "Job name/title")
    title_match = re.search(r"የስራው መጠሪያ:\s*\*\*(.+?)\*\*", text)
    title = title_match.group(1).strip() if title_match else None

    # Fallback: first bold text
    if not title:
        fb = re.search(r"^\*\*(.+?)\*\*", text.strip(), re.MULTILINE)
        title = fb.group(1).strip() if fb else None

    # Deadline (Amharic label for "Application deadline date")
    deadline_match = re.search(r"የማመልከቻ ማብቂያ ቀን:\s*\*\*(.+?)\*\*", text)
    deadline_str = deadline_match.group(1).strip() if deadline_match else None

    # Company: bold text near end before footer
    company_match = re.search(
        r"\n\*\*(.+?)\*\*\n.*?(?:የተመዘገበ ድርጅት|Jobs Posted|ስራ ለጥፈዋ)", text
    )
    company = company_match.group(1).strip() if company_match else None

    # Body: the English job description section
    body_match = re.search(r"\*\*የስራው ዝርዝር\*\*:\s*(.+?)(?:ከ፡|$)", text, re.DOTALL)
    body = body_match.group(1).strip() if body_match else text

    return {
        "title": title,
        "company": company,
        "deadline_str": deadline_str,
        "body": body,
    }


def extract_other_fields(text: str) -> dict:
    """
    Fallback extractor for unstructured channels.
    Extracts what it can; marks unknown fields as None.
    """
    # Title: first bold or first line
    title_match = re.search(r"^\*\*(.+?)\*\*", text.strip(), re.MULTILINE)
    if not title_match:
        title_match = re.search(r"^(.+)", text.strip())
    title = title_match.group(1).strip() if title_match else None

    # Company
    company_match = re.search(
        r"(?:Company|Organization|Employer|Institution):\s*\*?\*?(.+?)\*?\*?\n",
        text, re.IGNORECASE
    )
    company = company_match.group(1).strip() if company_match else None

    # Deadline: look for any date-like string near "deadline" or "apply by"
    deadline_match = re.search(
        r"(?:Deadline|Apply by|Closing date|Last date)[:\s]*\*?\*?(.{5,30}?)\*?\*?(?:\n|$)",
        text, re.IGNORECASE
    )
    deadline_str = deadline_match.group(1).strip() if deadline_match else None

    body = re.sub(r"\*\*|\n{3,}", " ", text).strip()

    return {
        "title": title,
        "company": company,
        "deadline_str": deadline_str,
        "body": body,
    }


# ── Deadline parser ────────────────────────────────────────────────────────────
def parse_deadline(deadline_str: Optional[str]) -> Optional[datetime]:
    """
    Converts a deadline string to a timezone-aware datetime.

    Examples handled (from real data):
        'Nov 24, 2026'        → datetime(2026, 11, 24, tzinfo=UTC)
        'October 30th, 2026'  → datetime(2026, 10, 30, tzinfo=UTC)
        'November 4th, 2026'  → datetime(2026, 11, 4, tzinfo=UTC)

    Returns None if parsing fails (not a hard error — job gets kept
    with deadline marked as unknown).
    """
    if not deadline_str:
        return None

    parsed = dateparser.parse(
        deadline_str,
        settings={
            "RETURN_AS_TIMEZONE_AWARE": True,
            "PREFER_DAY_OF_MONTH": "first",
            "PREFER_DATES_FROM": "future",
        },
    )
    return parsed


def is_expired(deadline: Optional[datetime]) -> bool:
    """
    Returns True if the deadline has already passed.
    If deadline is None (unknown), we KEEP the job (benefit of the doubt).
    """
    if deadline is None:
        return False   # unknown deadline → keep it
    now = datetime.now(tz=timezone.utc)
    return deadline < now


# ── Source URL builder ─────────────────────────────────────────────────────────
def build_telegram_url(channel: str, message_id: int) -> str:
    """Build a direct link to the Telegram message."""
    return f"https://t.me/{channel}/{message_id}"


# ── Master parse function ──────────────────────────────────────────────────────
def parse_job_message(raw: dict) -> Optional[dict]:
    """
    Takes one raw message dict (from telegram_sample.json) and returns
    a structured job record, or None if the message should be skipped.

    Args:
        raw: dict with keys: id, channel, date, text, language, is_job_post

    Returns:
        Structured dict or None (if expired / digest / no title)
    """
    text = raw.get("text", "").strip()
    channel = raw.get("channel", "")
    msg_id = raw.get("id", 0)
    msg_date = raw.get("date", "")

    # Skip non-job messages
    if not raw.get("is_job_post", False):
        return None

    # Skip digest posts (audit finding #3)
    if is_digest_post(text):
        return None

    # Detect channel type and extract fields
    ch_type = detect_channel_type(channel)
    if ch_type == "josad":
        fields = extract_josad_fields(text)
    elif ch_type == "afriwork":
        fields = extract_afriwork_fields(text)
    else:
        fields = extract_other_fields(text)

    # Must have at least a title
    if not fields.get("title"):
        return None

    # Parse and check deadline
    deadline = parse_deadline(fields.get("deadline_str"))
    if is_expired(deadline):
        return None   # skip expired jobs

    # Fix language detection (audit finding #2)
    language = detect_language(text)

    return {
        "id":           msg_id,
        "channel":      channel,
        "channel_type": ch_type,
        "url":          build_telegram_url(channel, msg_id),
        "posted_date":  msg_date,
        "title":        fields["title"],
        "company":      fields.get("company") or "Not specified",
        "deadline_str": fields.get("deadline_str") or "Not specified",
        "deadline":     deadline.isoformat() if deadline else None,
        "body":         fields.get("body", ""),
        "language":     language,
        "skills":       [],   # filled in by skill_extractor.py
    }


# ── Batch processing ───────────────────────────────────────────────────────────
def parse_all_jobs(raw_data: dict) -> list[dict]:
    """
    Process all channels from the raw sampler JSON.

    Args:
        raw_data: dict loaded from data/raw/telegram_sample.json

    Returns:
        List of structured, non-expired job records
    """
    results = []
    stats = {"total": 0, "expired": 0, "digest": 0, "no_title": 0, "kept": 0}

    for channel, messages in raw_data.items():
        if isinstance(messages, dict) and "error" in messages:
            print(f"⚠️  Skipping {channel}: {messages['error']}")
            continue

        for raw_msg in messages:
            stats["total"] += 1
            parsed = parse_job_message(raw_msg)

            if parsed is None:
                # We can't tell which filter rejected it without more code,
                # but that's fine for now.
                continue

            results.append(parsed)
            stats["kept"] += 1

    print(f"\nParsing complete:")
    print(f"  Total messages  : {stats['total']}")
    print(f"  Kept (active)   : {stats['kept']}")
    print(f"  Skipped         : {stats['total'] - stats['kept']}")

    return results


# ── Script entry point (for testing) ──────────────────────────────────────────
if __name__ == "__main__":
    data_path = Path(__file__).parent.parent.parent / "data" / "raw" / "telegram_sample.json"

    if not data_path.exists():
        print(f"ERROR: Raw data not found at {data_path}")
        print("Run notebooks/01_telegram_sampler.py first.")
        exit(1)

    with open(data_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    jobs = parse_all_jobs(raw_data)

    # Show 3 sample parsed records
    print("\n── Sample parsed records ──────────────────────────────")
    for job in jobs[:3]:
        print(f"\nTitle    : {job['title']}")
        print(f"Company  : {job['company']}")
        print(f"Deadline : {job['deadline_str']}")
        print(f"Channel  : @{job['channel']}")
        print(f"URL      : {job['url']}")
        print(f"Language : {job['language']}")
        print(f"Body     : {job['body'][:120]}...")

    # Save processed jobs
    output_path = Path(__file__).parent.parent.parent / "data" / "processed" / "parsed_jobs.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(jobs, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n✅ Saved {len(jobs)} parsed jobs to {output_path}")
