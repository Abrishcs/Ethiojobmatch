"""
EthioMatch — CV Parser
=======================
PURPOSE:
    Extract text and skills from a user's uploaded CV (PDF or DOCX).

WHY pdfplumber OVER PyPDF2?
    pdfplumber handles complex PDF layouts better — multi-column CVs,
    tables, and formatted text. PyPDF2 often garbles text from
    professionally formatted resumes.

WHAT THIS MODULE DOES:
    1. Accept a file path or file-like object (from Streamlit uploader)
    2. Detect file type (PDF or DOCX)
    3. Extract all text
    4. Clean and normalize the text
    5. Extract skills using skill_extractor.py
    6. Return a structured CV dict

LIMITATION (document honestly):
    - Scanned PDFs (images of text) cannot be read without OCR.
      We detect this and show a clear error message.
    - Some creative CV designs with heavy graphics may extract poorly.
    - We extract skills from text only — we don't parse CV sections
      (education, experience) in this version.
"""

import io
import re
from pathlib import Path

import pdfplumber
from docx import Document

# Import skill extractor from our matching module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from matching.skill_extractor import extract_skills


# ── Text cleaning ──────────────────────────────────────────────────────────────
def clean_cv_text(text: str) -> str:
    """
    Clean raw extracted CV text.

    Removes:
    - Excessive whitespace and blank lines
    - Common CV noise (page numbers, headers repeated on each page)
    - Non-printable characters

    Keeps:
    - All meaningful words and phrases
    - Punctuation that helps sentence boundary detection
    """
    if not text:
        return ""

    # Remove non-printable characters (but keep Ethiopic Unicode)
    text = re.sub(r"[^\x20-\x7E\u1200-\u137F\n]", " ", text)

    # Normalize whitespace: collapse multiple spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse more than 2 consecutive newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove lines that are just numbers (page numbers)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    return text.strip()


# ── PDF extractor ──────────────────────────────────────────────────────────────
def extract_text_from_pdf(file) -> str:
    """
    Extract text from a PDF file using pdfplumber.

    Args:
        file: file path (str/Path) or file-like object (BytesIO from Streamlit)

    Returns:
        Extracted text as a single string.

    Raises:
        ValueError: if the PDF appears to be scanned (no extractable text)
    """
    all_text = []

    with pdfplumber.open(file) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                all_text.append(page_text)

    full_text = "\n".join(all_text)

    # Detect scanned PDF: very little text extracted despite having pages
    if len(full_text.strip()) < 100:
        raise ValueError(
            "This PDF appears to be a scanned image. "
            "EthioMatch cannot read scanned PDFs. "
            "Please save your CV as a text-based PDF or DOCX file."
        )

    return full_text


# ── DOCX extractor ─────────────────────────────────────────────────────────────
def extract_text_from_docx(file) -> str:
    """
    Extract text from a DOCX file using python-docx.

    Args:
        file: file path (str/Path) or file-like object (BytesIO from Streamlit)

    Returns:
        Extracted text as a single string.
    """
    # python-docx needs a file path or BytesIO
    if isinstance(file, (str, Path)):
        doc = Document(file)
    else:
        doc = Document(io.BytesIO(file.read()) if hasattr(file, "read") else file)

    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]

    # Also extract text from tables (skills are often in table format in CVs)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    paragraphs.append(cell.text.strip())

    return "\n".join(paragraphs)


# ── Main CV parser ─────────────────────────────────────────────────────────────
def parse_cv(file, filename: str) -> dict:
    """
    Parse a CV file and return a structured dict.

    Args:
        file:     File path or file-like object (Streamlit UploadedFile)
        filename: Original filename (used to detect PDF vs DOCX)

    Returns:
        {
            "raw_text":     str,           Full extracted text
            "word_count":   int,           Approximate word count
            "skills":       list[str],     Extracted skills (sorted)
            "skill_count":  int,           Number of skills found
            "file_type":    str,           'pdf' or 'docx'
            "error":        str | None,    Error message if parsing failed
        }
    """
    filename_lower = filename.lower()

    try:
        # Detect file type and extract text
        if filename_lower.endswith(".pdf"):
            raw_text = extract_text_from_pdf(file)
            file_type = "pdf"

        elif filename_lower.endswith((".docx", ".doc")):
            raw_text = extract_text_from_docx(file)
            file_type = "docx"

        else:
            return {
                "raw_text":    "",
                "word_count":  0,
                "skills":      [],
                "skill_count": 0,
                "file_type":   "unknown",
                "error":       f"Unsupported file type: {filename}. Please upload a PDF or DOCX file.",
            }

        # Clean the extracted text
        cleaned_text = clean_cv_text(raw_text)

        # Extract skills
        skills_found = sorted(extract_skills(cleaned_text))

        return {
            "raw_text":    cleaned_text,
            "word_count":  len(cleaned_text.split()),
            "skills":      skills_found,
            "skill_count": len(skills_found),
            "file_type":   file_type,
            "error":       None,
        }

    except ValueError as e:
        # Known error (e.g. scanned PDF) — show to user
        return {
            "raw_text":    "",
            "word_count":  0,
            "skills":      [],
            "skill_count": 0,
            "file_type":   filename_lower.split(".")[-1],
            "error":       str(e),
        }

    except Exception as e:
        # Unexpected error — show generic message, log the real error
        print(f"[cv_parser] Unexpected error parsing {filename}: {e}")
        return {
            "raw_text":    "",
            "word_count":  0,
            "skills":      [],
            "skill_count": 0,
            "file_type":   filename_lower.split(".")[-1],
            "error":       "Could not read this file. Please try a different PDF or DOCX.",
        }


# ── Script entry point (for testing) ──────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("── CV Parser Test ─────────────────────────────────────────")

    # If a file path is passed as argument, test with it
    if len(sys.argv) > 1:
        cv_path = Path(sys.argv[1])
        if not cv_path.exists():
            print(f"ERROR: File not found: {cv_path}")
            sys.exit(1)

        print(f"Testing with: {cv_path.name}")
        result = parse_cv(cv_path, cv_path.name)

    else:
        # Create a minimal test DOCX in memory
        print("No CV file provided — creating a test document in memory...")

        from docx import Document as DocxDocument
        doc = DocxDocument()
        doc.add_paragraph("John Doe — Software Engineer")
        doc.add_paragraph(
            "I have 3 years of experience with Python, Django, and React. "
            "I build REST APIs and work with PostgreSQL and MongoDB. "
            "Familiar with Docker, Git, AWS, and Agile methodology. "
            "Some experience with Machine Learning using scikit-learn and Pandas."
        )
        doc.add_paragraph("Education: BSc Computer Science")
        doc.add_paragraph("Skills: Python, JavaScript, TypeScript, Node.js, Linux")

        # Save to a temp buffer
        buf = io.BytesIO()
        doc.save(buf)
        buf.seek(0)

        result = parse_cv(buf, "test_cv.docx")

    # Display results
    if result["error"]:
        print(f"\n❌ Error: {result['error']}")
    else:
        print(f"\n✅ Parsed successfully")
        print(f"   File type  : {result['file_type'].upper()}")
        print(f"   Word count : {result['word_count']}")
        print(f"   Skills found ({result['skill_count']}):")
        for skill in result["skills"]:
            print(f"     • {skill}")
        print(f"\n   Raw text preview (first 300 chars):")
        print(f"   {result['raw_text'][:300]}...")
