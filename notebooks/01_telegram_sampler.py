"""
EthioMatch — Telegram Data Sampler (Stage 2: Data Audit)
=========================================================
PURPOSE:
    Connects to Telegram, reads recent messages from each job channel,
    saves raw data to data/raw/telegram_sample.json for inspection.

HOW TO RUN:
    python notebooks/01_telegram_sampler.py

WHAT IT DOES:
    1. Reads your API credentials from .env
    2. Connects to Telegram (you will be asked for your phone + OTP once)
    3. Fetches last N messages from each channel
    4. Saves raw JSON for data audit
    5. Prints a summary so you can see what real job posts look like

NOTE:
    - Your session is saved in ethiomatch.session (DO NOT share this file)
    - This script only READS messages. It never posts anything.
    - Only works on PUBLIC channels.
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.errors import (
    ChannelPrivateError,
    UsernameInvalidError,
    UsernameNotOccupiedError,
)
from telethon.tl.types import Message

# ── Load credentials from .env ─────────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / ".env")

API_ID   = os.getenv("TELEGRAM_API_ID")
API_HASH = os.getenv("TELEGRAM_API_HASH")
PHONE    = os.getenv("TELEGRAM_PHONE")

if not all([API_ID, API_HASH, PHONE]):
    print("ERROR: Missing credentials in .env file.")
    print("Please copy .env.example → .env and fill in your values.")
    print("Get API_ID and API_HASH from: https://my.telegram.org")
    sys.exit(1)

# ── Channel list ───────────────────────────────────────────────────────────────
CHANNELS = [
    "josad_it",
    "josad_digital",
    "josad_software",
    "gadaanetwork",
    "vacancyforallethio",
    "hahujobs",
    "jobs_in_ethio",
    "effoyjobs",
    "ethio_job_vacancy1",
    "afriworkamharic",
]

# How many messages to fetch per channel for the data audit
MESSAGES_PER_CHANNEL = 100

# Output path
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "raw" / "telegram_sample.json"


# ── Helper: detect Amharic text ────────────────────────────────────────────────
def is_amharic(text: str) -> bool:
    """
    Returns True if the text contains Ethiopic Unicode characters.
    Ethiopic Unicode block: U+1200 to U+137F
    """
    return any("\u1200" <= char <= "\u137F" for char in text)


# ── Helper: check if message looks like a job post ────────────────────────────
JOB_KEYWORDS_EN = [
    "vacancy", "job", "position", "hiring", "apply", "deadline",
    "required", "engineer", "developer", "analyst", "officer",
    "manager", "intern", "experience", "qualification", "salary",
]
JOB_KEYWORDS_AM = [
    "ስራ", "ቦታ", "ይፈለጋል", "ማመልከቻ", "የስራ", "ተፈላጊ", "ብቃት",
]


def looks_like_job_post(text: str) -> bool:
    """
    Simple heuristic: does this message look like a job vacancy post?
    We check for common job-related keywords.
    """
    text_lower = text.lower()
    en_hits = sum(1 for kw in JOB_KEYWORDS_EN if kw in text_lower)
    am_hits = sum(1 for kw in JOB_KEYWORDS_AM if kw in text)
    return (en_hits >= 2) or (am_hits >= 1)


# ── Main async function ────────────────────────────────────────────────────────
async def sample_channels():
    """
    Connects to Telegram and samples messages from each channel.
    """
    print("\n" + "=" * 60)
    print("EthioMatch — Telegram Data Sampler")
    print("=" * 60)

    # Session file is saved in the project root
    session_path = Path(__file__).parent.parent / "ethiomatch"

    client = TelegramClient(str(session_path), int(API_ID), API_HASH)

    await client.start(phone=PHONE)
    print(f"\n✅ Connected to Telegram as: {PHONE}\n")

    all_results = {}
    total_posts = 0
    total_job_posts = 0

    for channel_username in CHANNELS:
        print(f"📡 Fetching: @{channel_username} ...", end=" ")

        try:
            messages = []
            async for msg in client.iter_messages(
                channel_username,
                limit=MESSAGES_PER_CHANNEL,
            ):
                # Only process text messages
                if not isinstance(msg, Message) or not msg.text:
                    continue

                text = msg.text.strip()
                if len(text) < 30:   # skip very short messages (links, emojis)
                    continue

                language = "amharic" if is_amharic(text) else "english"
                job_post  = looks_like_job_post(text)

                messages.append({
                    "id":          msg.id,
                    "channel":     channel_username,
                    "date":        msg.date.isoformat(),
                    "text":        text,
                    "language":    language,
                    "is_job_post": job_post,
                    "char_count":  len(text),
                })

                if job_post:
                    total_job_posts += 1

            all_results[channel_username] = messages
            total_posts += len(messages)
            print(f"✅ {len(messages)} messages ({sum(1 for m in messages if m['is_job_post'])} job posts)")

        except ChannelPrivateError:
            print(f"❌ PRIVATE — cannot access @{channel_username}")
            all_results[channel_username] = {"error": "channel_private"}

        except (UsernameInvalidError, UsernameNotOccupiedError):
            print(f"❌ NOT FOUND — @{channel_username} does not exist")
            all_results[channel_username] = {"error": "not_found"}

        except Exception as e:
            print(f"❌ ERROR — {type(e).__name__}: {e}")
            all_results[channel_username] = {"error": str(e)}

    await client.disconnect()

    # ── Save results ────────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    # ── Print summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total messages fetched : {total_posts}")
    print(f"Identified as job posts: {total_job_posts}")
    print(f"Saved to               : {OUTPUT_PATH}")
    print()
    print("NEXT STEP:")
    print("  Open notebooks/02_data_audit.ipynb to inspect the data.")
    print("=" * 60 + "\n")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(sample_channels())
