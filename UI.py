import html
import re
import time
import unicodedata
import json
from datetime import datetime
from io import BytesIO
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import pandas as pd
import requests
import streamlit as st
import urllib3
import bing
from bs4 import BeautifulSoup

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        HRFlowable,
        KeepTogether,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Online Adverse News Search",
    page_icon="🔎",
    layout="wide"
)


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

REQUEST_TIMEOUT_SECONDS = 30

DEFAULT_KEYWORD_OPTIONS = [
    "fraud",
    "money laundering",
    "investigation",
    "corruption",
    "bribery",
    "sanctions",
    "lawsuit",
    "litigation",
    "prosecution",
    "criminal",
]

# Common question words do not help identify relevant article results.
CHAT_STOP_WORDS = {
    "a", "about", "all", "an", "and", "any", "are", "article",
    "articles", "been", "can", "count", "did", "do", "find", "for",
    "found", "from", "give", "had", "has", "have", "how", "i", "in",
    "is", "list", "many", "me", "mention", "mentions", "number", "of",
    "on", "please", "related", "results", "show", "tell", "that", "the",
    "there", "these", "this", "to", "total", "was", "were", "what",
    "when", "where", "which", "who", "will", "with", "would", "you",
}
MAXIMUM_CHAT_ARTICLES = 5

# These rules are applied only to the collected article titles and snippets.
# They indicate screening priority, not proof of wrongdoing or a final vendor
# onboarding decision.
HIGH_RISK_AML_KEYWORDS = [
    "money laundering",
    "anti-money laundering",
    "terrorist financing",
    "financing of terrorism",
    "sanction",
    "sanctions evasion",
    "proceeds of crime",
    "criminal prosecution",
    "prosecution",
    "conviction",
    "indictment",
    "arrest",
    "criminal",
    "fraud",
    "corruption",
    "bribery",
    "embezzlement",
    "shell company",
    "regulatory action"
]

MEDIUM_RISK_AML_KEYWORDS = [

    "investigation",
    "lawsuit",
    "litigation",
    "charge",
    "fine",
    "penalty"
]

RISK_LEVEL_ORDER = {
    "High": 3,
    "Medium": 2,
    "Low": 1,
}

RISK_RECOMMENDATIONS = {
    "High": (
        "Do not approve automatically. Escalate for enhanced due diligence "
        "and compliance review before any onboarding decision."
    ),
    "Medium": (
        "Hold for enhanced review. Obtain supporting information and "
        "complete due diligence before onboarding."
    ),
    "Low": (
        "No major AML keyword was found in this result's title or snippet. "
        "Continue with standard due diligence; this is not an approval."
    ),
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_whitespace(value: str) -> str:
    """
    Replace repeated whitespace with a single space.
    """
    if not value:
        return ""

    return re.sub(r"\s+", " ", value).strip()


def clean_text(value: str) -> str:
    """
    Decode HTML entities and normalize whitespace.
    """
    if not value:
        return ""

    return normalize_whitespace(html.unescape(value))


def strip_trailing_ellipsis(value: str) -> str:
    """Remove trailing ellipsis or repeated dot characters commonly added by RSS/snippet feeds.

    Examples: "...", "…", "...." at the end of a snippet.
    """
    if not value:
        return ""

    # Normalize and remove trailing runs of dots or ellipsis characters
    value = value.strip()
    # Remove Unicode ellipsis and multiple dot sequences at end
    value = re.sub(r"[\u2026]+$", "", value)
    value = re.sub(r"[.]{2,}$", "", value)
    return value.strip()


def sanitize_text(value: str) -> str:
    """Normalize and clean text to avoid common mojibake and control characters.

    This attempts to fix typical encoding artifacts such as smart quotes
    rendered as sequences like "â€™" and removes non-printable control
    characters. It's intentionally conservative to avoid altering meaning.
    """
    if not value:
        return ""

    # Decode HTML entities and normalize Unicode form
    text = html.unescape(value)
    text = unicodedata.normalize("NFKC", text)

    # Replace common mojibake sequences returned by some feeds/APIs
    replacements = {
        "â€™": "'",
        "â€˜": "'",
        "â€œ": '"',
        "â€�": '"',
        "â€“": "-",
        "â€”": "-",
        "Ã©": "é",
        "Ã±": "ñ",
        "â€¦": "...",
    }

    for k, v in replacements.items():
        if k in text:
            text = text.replace(k, v)

    # Map common Unicode curly quotes to ASCII equivalents
    unicode_quote_map = {
        "\u2019": "'",
        "\u2018": "'",
        "\u201B": "'",
        "\u201A": "'",
        "\u201C": '"',
        "\u201D": '"',
        "\u201E": '"',
        "\u201F": '"',
    }
    for k, v in unicode_quote_map.items():
        if k in text:
            text = text.replace(k, v)

    # Remove control characters except whitespace-like ones
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]+", "", text)

    # Fix lone replacement-question-marks that often appear where an
    # apostrophe/typographic quote belonged. Replace only when the
    # question mark appears inside a word or before a capitalised word
    # (common in names), to avoid changing real sentence-ending ? marks.
    text = re.sub(r"(?<=\w)\?(?=\w)", "'", text)
    text = re.sub(r"(?<=\w)\?(?=\s[A-Z])", "'", text)
    # Also handle Unicode replacement char (U+FFFD) in the same positions.
    text = re.sub(r"(?<=\w)\uFFFD(?=\w)", "'", text)
    text = re.sub(r"(?<=\w)\uFFFD(?=\s[A-Z])", "'", text)

    # Also convert question marks that directly follow a word and are
    # followed by whitespace (common in truncated/encoded snippets)
    text = re.sub(r"(?<=\w)\?(?=\s)", "'", text)
    # And question marks that appear after whitespace before a word
    # (e.g. 'Dato ?Najib') -> "Dato 'Najib".
    text = re.sub(r"(?<=\s)\?(?=\w)", "'", text)

    # Fallback: if any single non-word punctuation sits between letters
    # (e.g. unusual encoding artifact displayed as a punctuation glyph),
    # conservatively convert it to an apostrophe.
    text = re.sub(r"(?<=\w)[^\w\s](?=\w)", "'", text)
    text = re.sub(r"(?<=\w)[^\w\s](?=\s[A-Z])", "'", text)

    # Normalize repeated whitespace
    text = normalize_whitespace(text)

    return text


def prepare_snippet_for_display(snippet: str) -> str:
    """Normalize a snippet for UI/PDF display and append an ellipsis if appropriate.

    - Decodes HTML entities and normalizes whitespace.
    - Removes existing trailing ellipsis characters or repeated dots.
    - Appends a single space + three dots (` ...`) if the snippet doesn't already
      end with sentence punctuation or an ellipsis.
    """
    s = sanitize_text(snippet or "")
    s = strip_trailing_ellipsis(s)
    if not s:
        return ""

    # If it already ends with punctuation or an ellipsis, leave as-is.
    if re.search(r"(\.{3}|\u2026|[.!?])$", s):
        return s

    return s + " ..."


def format_keyword_for_search(keyword: str) -> str:
    """
    Format a keyword for use in the Boolean search expression.

    Multi-word terms are wrapped in quotation marks so they are searched
    as an exact phrase.
    """
    keyword = normalize_whitespace(keyword)

    if " " in keyword:
        return f'"{keyword}"'

    return keyword


def parse_additional_keywords(value: str) -> list[str]:
    """
    Read optional user-added keywords separated by commas or new lines.

    Users may include quotation marks around a phrase, but these are
    removed because format_keyword_for_search adds them where needed.
    """
    keywords = []
    seen_keywords = set()

    for value_part in re.split(r"[,;\n]+", value):
        keyword = normalize_whitespace(value_part).strip('"')

        if not keyword:
            continue

        keyword_key = keyword.casefold()

        if keyword_key not in seen_keywords:
            keywords.append(keyword)
            seen_keywords.add(keyword_key)

    return keywords


def build_keyword_expression(keywords: list[str]) -> str:
    """
    Turn selected keyword terms into a parenthesized Boolean expression.
    """
    return "(" + " OR ".join(
        format_keyword_for_search(keyword)
        for keyword in keywords
    ) + ")" if keywords else ""


def build_search_query(subject: str, keyword_expression: str) -> str:
    """
    Construct the final search query.

    Example:
        "DHL" (fraud OR "money laundering" OR corruption)
    """
    subject = normalize_whitespace(subject)
    keyword_expression = normalize_whitespace(keyword_expression)

    if not subject:
        raise ValueError("Please enter a company, person, or search subject.")

    quoted_subject = subject

    # Avoid adding another set of quotes if the user already entered them.
    if not (
        subject.startswith('"')
        and subject.endswith('"')
    ):
        quoted_subject = f'"{subject}"'

    if keyword_expression:
        return f"{quoted_subject} {keyword_expression}"

    return quoted_subject




def calculate_keyword_matches(
    title: str,
    snippet: str,
    keywords: list[str]
) -> tuple[int, list[str]]:
    """
    Calculate simple keyword matches against the title and snippet.

    This affects only local result ordering. It does not verify whether
    the content of the linked article actually supports the keyword.
    """
    combined_text = f"{title} {snippet}".casefold()

    matched_keywords = []

    for keyword in keywords:
        if keyword.casefold() in combined_text:
            matched_keywords.append(keyword)

    # Remove duplicates while preserving order.
    matched_keywords = list(dict.fromkeys(matched_keywords))

    return len(matched_keywords), matched_keywords


def find_keyword_hits(text: str, keywords: list[str]) -> list[str]:
    """
    Return the major AML keywords that appear in a text value.
    """
    normalized_text = text.casefold()

    return [
        keyword
        for keyword in keywords
        if keyword.casefold() in normalized_text
    ]


def is_msn_result(result: dict) -> bool:
    """Return True if the search result appears to be from MSN.

    We check several result fields (link, snippet, title, source/publisher)
    for MSN indicators so aggregated or redirected MSN items are caught.
    """
    # Check link first (covers direct MSN URLs and common redirects)
    link = (result.get("link") or "").strip()
    try:
        if link:
            if "msn.com" in link.lower() or "msn." in urlparse(link).netloc.lower():
                return True
    except Exception:
        pass

    # Check snippet, title, or source fields for explicit 'MSN' publisher text.
    snippet = (result.get("snippet") or "").strip()
    title = (result.get("title") or "").strip()
    source = (result.get("source") or result.get("publisher") or "").strip()

    for text in (snippet, title, source):
        if re.search(r"\bmsn\b", text, flags=re.IGNORECASE):
            return True

    return False


def assess_article_aml_risk(result: dict) -> dict:
    """
    Assign a screening risk flag using the article title and snippet only.
    """
    article_text = " ".join(
        [
            result.get("title", ""),
            result.get("snippet", ""),
        ]
    )

    high_risk_hits = find_keyword_hits(
        article_text,
        HIGH_RISK_AML_KEYWORDS,
    )

    if high_risk_hits:
        risk_level = "High"
        keyword_hits = high_risk_hits
    else:
        medium_risk_hits = find_keyword_hits(
            article_text,
            MEDIUM_RISK_AML_KEYWORDS,
        )

        if medium_risk_hits:
            risk_level = "Medium"
            keyword_hits = medium_risk_hits
        else:
            risk_level = "Low"
            keyword_hits = []

    return {
        "risk_level": risk_level,
        "aml_keyword_flags": ", ".join(keyword_hits) or "None found",
        "onboarding_recommendation": RISK_RECOMMENDATIONS[risk_level],
    }


def create_aml_risk_assessments(results: list[dict]) -> list[dict]:
    """
    Create one AML risk assessment record for every collected article.
    """
    return [
        assess_article_aml_risk(result)
        for result in results
    ]


def overall_vendor_screening_result(
    assessments: list[dict]
) -> tuple[str, str]:
    """
    Return the highest screening level across the collected articles.
    """
    if not assessments:
        return "Low", RISK_RECOMMENDATIONS["Low"]

    highest_risk_level = max(
        (assessment["risk_level"] for assessment in assessments),
        key=lambda risk_level: RISK_LEVEL_ORDER[risk_level],
    )

    return (
        highest_risk_level,
        RISK_RECOMMENDATIONS[highest_risk_level],
    )


def pdf_safe_text(value: str) -> str:
    """
    Prepare text for ReportLab's built-in Helvetica font and Paragraph XML.
    """
    text = clean_text(str(value))
    text = text.encode("latin-1", "replace").decode("latin-1")

    return html.escape(text, quote=True)


def pdf_footer(canvas, document) -> None:
    """
    Add a footer to each vendor-screening PDF page.
    """
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D1D5DB"))
    canvas.line(
        document.leftMargin,
        12 * mm,
        document.pagesize[0] - document.rightMargin,
        12 * mm,
    )
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#4B5563"))
    canvas.drawString(
        document.leftMargin,
        7 * mm,
        "Vendor AML article screening - unverified search-result metadata",
    )
    canvas.drawRightString(
        document.pagesize[0] - document.rightMargin,
        7 * mm,
        f"Page {document.page}",
    )
    canvas.restoreState()


def generate_vendor_screening_pdf(
    subject: str,
    results: list[dict],
    assessments: list[dict],
    include_ai_summary: bool = False,
) -> bytes:
    """
    Generate a vendor AML screening report listing every collected article.
    """
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError(
            "PDF generation requires the ReportLab package to be installed."
        )

    overall_risk_level, overall_recommendation = (
        overall_vendor_screening_result(assessments)
    )
    risk_counts = {
        risk_level: sum(
            assessment["risk_level"] == risk_level
            for assessment in assessments
        )
        for risk_level in RISK_LEVEL_ORDER
    }

    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=20 * mm,
        title="Vendor AML Article Screening Report",
        author="Online Article Search",
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="VendorReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#0F172A"),
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="VendorReportSubtitle",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#475569"),
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="VendorSectionHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#0F766E"),
            spaceBefore=10,
            spaceAfter=6,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="VendorArticleTitle",
            parent=styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#111827"),
            spaceBefore=8,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="VendorBody",
            parent=styles["BodyText"],
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#1F2937"),
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="VendorTableHeader",
            parent=styles["BodyText"],
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.white,
        )
    )
    styles.add(
        ParagraphStyle(
            name="VendorTableBody",
            parent=styles["BodyText"],
            alignment=TA_LEFT,
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#1F2937"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="VendorLink",
            parent=styles["BodyText"],
            fontSize=7.5,
            leading=9,
            textColor=colors.HexColor("#0F766E"),
            wordWrap="CJK",
        )
    )

    story = [
        Paragraph(
            "Vendor AML Article Screening Report",
            styles["VendorReportTitle"],
        ),
        Paragraph(
            f"Screened subject: <b>{pdf_safe_text(subject)}</b><br/>"
            f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}<br/>"
            "Scope: article titles and snippets collected by this search only.",
            styles["VendorReportSubtitle"],
        ),
        Paragraph(
            "Important: this report flags search-result metadata for review. "
            "It does not verify allegations, establish misconduct, or make a "
            "final vendor-acceptance decision.",
            styles["VendorBody"],
        ),
        Spacer(1, 4 * mm),
        Paragraph("Screening overview", styles["VendorSectionHeading"]),
    ]

    overview_data = [
        [
            Paragraph("<b>Overall screening flag</b>", styles["VendorTableBody"]),
            Paragraph(
                f"<b>{overall_risk_level.upper()}</b>",
                styles["VendorTableBody"],
            ),
            Paragraph("<b>Recommended next step</b>", styles["VendorTableBody"]),
            Paragraph(
                pdf_safe_text(overall_recommendation),
                styles["VendorTableBody"],
            ),
        ],
        [
            Paragraph("<b>Articles collected</b>", styles["VendorTableBody"]),
            Paragraph(str(len(results)), styles["VendorTableBody"]),
            Paragraph("<b>High / Medium / Low</b>", styles["VendorTableBody"]),
            Paragraph(
                (
                    f"{risk_counts['High']} / {risk_counts['Medium']} / "
                    f"{risk_counts['Low']}"
                ),
                styles["VendorTableBody"],
            ),
        ],
    ]
    overview_table = Table(
        overview_data,
        colWidths=[39 * mm, 27 * mm, 43 * mm, 138 * mm],
    )
    overview_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#ECFDF5")),
                ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#ECFDF5")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#A7F3D0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([overview_table, Spacer(1, 4 * mm)])

    story.append(Paragraph("Risk-flag guide", styles["VendorSectionHeading"]))
    guide_data = [
        [
            Paragraph("Risk", styles["VendorTableHeader"]),
            Paragraph("Major AML keyword trigger", styles["VendorTableHeader"]),
            Paragraph("Vendor-screening direction", styles["VendorTableHeader"]),
        ],
        [
            Paragraph("HIGH", styles["VendorTableBody"]),
            Paragraph(
                pdf_safe_text(
                    ", ".join(HIGH_RISK_AML_KEYWORDS)
                ),
                styles["VendorTableBody"],
            ),
            Paragraph(
                pdf_safe_text(RISK_RECOMMENDATIONS["High"]),
                styles["VendorTableBody"],
            ),
        ],
        [
            Paragraph("MEDIUM", styles["VendorTableBody"]),
            Paragraph(
                pdf_safe_text(
                    ", ".join(MEDIUM_RISK_AML_KEYWORDS)
                ),
                styles["VendorTableBody"],
            ),
            Paragraph(
                pdf_safe_text(RISK_RECOMMENDATIONS["Medium"]),
                styles["VendorTableBody"],
            ),
        ],
        [
            Paragraph("LOW", styles["VendorTableBody"]),
            Paragraph(
                "No listed major AML keyword found in the title or snippet.",
                styles["VendorTableBody"],
            ),
            Paragraph(
                pdf_safe_text(RISK_RECOMMENDATIONS["Low"]),
                styles["VendorTableBody"],
            ),
        ],
    ]
    guide_table = Table(
        guide_data,
        colWidths=[24 * mm, 95 * mm, 128 * mm],
        repeatRows=1,
    )
    guide_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F766E")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("BACKGROUND", (0, 1), (0, 1), colors.HexColor("#FEE2E2")),
                ("BACKGROUND", (0, 2), (0, 2), colors.HexColor("#FEF3C7")),
                ("BACKGROUND", (0, 3), (0, 3), colors.HexColor("#DCFCE7")),
            ]
        )
    )
    story.extend([guide_table, Spacer(1, 4 * mm)])
    article_entries = []

    for index, (result, assessment) in enumerate(
        zip(results, assessments),
        start=1,
    ):
        article_flowables = [
            Paragraph(
                f"Article {index}: {pdf_safe_text(result['title'])}",
                styles["VendorArticleTitle"],
            ),
            Paragraph(
                f"<b>Risk flag:</b> {assessment['risk_level'].upper()}<br/>"
                f"<b>Major AML keyword flags:</b> "
                f"{pdf_safe_text(assessment['aml_keyword_flags'])}<br/>"
                f"<b>Vendor-screening direction:</b> "
                f"{pdf_safe_text(assessment['onboarding_recommendation'])}",
                styles["VendorBody"],
            ),
        ]

        # Include either the snippet or the AI summary based on caller preference
        if include_ai_summary:
            article_flowables.append(
                Paragraph(
                    f"<b>AI summary:</b> {pdf_safe_text(result.get('summary', '') or 'No summary generated.')}",
                    styles["VendorBody"],
                )
            )
        else:
            snippet_display = prepare_snippet_for_display(result.get('snippet', '') or '') or 'No snippet available.'
            article_flowables.append(
                Paragraph(
                    f"<b>Search-result snippet:</b> "
                    f"{pdf_safe_text(snippet_display)}",
                    styles["VendorBody"],
                )
            )
        safe_link = pdf_safe_text(result["link"])
        article_flowables.append(
            Paragraph(
                f'<link href="{safe_link}" color="#0F766E">{safe_link}</link>',
                styles["VendorLink"],
            )
        )
        article_flowables.extend(
            [
                Spacer(1, 2 * mm),
                HRFlowable(
                    width="100%",
                    thickness=0.4,
                    color=colors.HexColor("#CBD5E1"),
                    spaceAfter=2 * mm,
                ),
            ]
        )
        article_entries.append(KeepTogether(article_flowables))

    if article_entries:
        article_section_heading = Paragraph(
            "Article-by-article screening",
            styles["VendorSectionHeading"],
        )
        story.append(
            KeepTogether([article_section_heading, article_entries[0]])
        )
        story.extend(article_entries[1:])

    document.build(
        story,
        onFirstPage=pdf_footer,
        onLaterPages=pdf_footer,
    )

    return buffer.getvalue()


def simple_local_summary(text: str, max_sentences: int = 3) -> str:
    """Create a short heuristic summary by taking the first few sentences."""
    if not text:
        return ""
    # Normalize whitespace and split on sentence-like punctuation.
    text = clean_text(text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chosen = sentences[:max_sentences]
    summary = " ".join(chosen).strip()
    # Fallback to truncation if no sentences found.
    if not summary:
        return text[:300].strip()
    return summary


def summarize_with_gemini(text: str, api_key: str, model: str = "models/gemini-3.5-flash-lite", timeout: int = 15) -> str:
    """Call a Vertex/Gemini-style REST endpoint to generate a concise summary.

    This implementation uses the Google Generative Language REST pattern:
    POST https://generativelanguage.googleapis.com/v1beta2/{model}:generateText

    If the request fails or the response shape is unexpected, an exception is raised.
    """
    if not text:
        return ""

    endpoint = f"https://generativelanguage.googleapis.com/v1beta2/{model}:generateText"
    # prompt = (
    #     "Provide a concise summary of the following article in 2-3 sentences. "

    # )
    
    prompt = (
        "You are a concise compliance assistant. Read the full article text below and produce a clear, "
        "human-friendly 2–3 sentence summary focused on any misconduct, investigations, legal action, "
        "or regulatory matters. Do not copy full sentences from the article. Paraphrase and synthesize "
        "the information in your own words. Follow these rules:\n"
        "- Produce exactly 2–3 short sentences.\n"
        "- Sentence 1: State the main point (what happened).\n"
        "- Sentence 2: Provide key specifics (who, what, when, where) if available.\n"
        "- Optional Sentence 3: One short implication or current status (e.g., 'under investigation', 'charged', 'settled').\n"
        "- Prioritize facts about investigations, charges, regulatory actions, fines, arrests, or legal outcomes; omit generic background context.\n"
        "- If the article lacks concrete details, write 'no details provided'.\n"
        "- Do not include quotes, verbatim sentences, or extra commentary. Output the summary only.\n\n"
        "Important: Do NOT simply repeat the article's first paragraph — synthesize across the full text and avoid verbatim copying.\n\n"
        "Article:\n\n"
        + text
        + "\n\nProvide the summary only (no extra commentary)."
    )
    

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    body = {
        "prompt": {"text": prompt},
        "temperature": 0.0,
        "maxOutputTokens": 180,
    }

    resp = requests.post(endpoint, headers=headers, json=body, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    # Try common response shapes.
    # Older PaLM/Generative API used 'candidates'[0]['output'] or 'candidates'[0]['content']
    if isinstance(data, dict):
        # 'candidates' style
        if "candidates" in data and data["candidates"]:
            first = data["candidates"][0]
            for key in ("output", "content", "text"):
                if key in first:
                    return first[key].strip()
        # 'results' or 'outputs' style
        for arr_key in ("results", "outputs"):
            if arr_key in data and data[arr_key]:
                first = data[arr_key][0]
                for key in ("content", "text", "output"):
                    if key in first:
                        # some shapes use first['content'][0]['text']
                        if isinstance(first[key], list) and first[key]:
                            candidate = first[key][0]
                            if isinstance(candidate, dict) and "text" in candidate:
                                return candidate["text"].strip()
                            return str(candidate).strip()
                        return str(first[key]).strip()

    # If not returned yet, raise to let caller fallback.
    raise RuntimeError("Unexpected Gemini response shape: %s" % json.dumps(data)[:1000])


def fetch_article_text(url: str, verify_ssl: bool = True, timeout: int = 10) -> str:
    """Fetch an article URL and extract the main textual content.

    Strategy:
    - Prefer content inside an <article> tag.
    - Otherwise, choose the parent element that contains the largest
      combined length of <p> text nodes.
    """
    if not url:
        return ""

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=timeout, verify=verify_ssl)
        resp.raise_for_status()
        html_text = resp.text
    except Exception:
        return ""

    try:
        soup = BeautifulSoup(html_text, "html.parser")

        # Remove script/style and irrelevant tags
        for tag in soup(["script", "style", "noscript", "iframe", "footer", "nav", "header", "aside"]):
            tag.decompose()

        # Prefer <article>
        article_tag = soup.find("article")
        if article_tag:
            paragraphs = [p.get_text(" ", strip=True) for p in article_tag.find_all("p")]
            text = "\n\n".join([p for p in paragraphs if p])
            if len(text) > 200:
                return normalize_whitespace(text)

        # Otherwise pick the parent with largest combined <p> text
        p_tags = soup.find_all("p")
        if not p_tags:
            # Fallback: whole page text
            return normalize_whitespace(soup.get_text(" ", strip=True))[:5000]

        parent_scores = {}
        for p in p_tags:
            parent = p.find_parent()
            if parent is None:
                continue
            parent_key = id(parent)
            parent_scores.setdefault(parent_key, {"parent": parent, "length": 0, "texts": []})
            text = p.get_text(" ", strip=True)
            parent_scores[parent_key]["length"] += len(text)
            parent_scores[parent_key]["texts"].append(text)

        best = max(parent_scores.values(), key=lambda v: v["length"])
        combined = "\n\n".join(best["texts"]) if best and best.get("texts") else ""
        return normalize_whitespace(combined)[:5000]
    except Exception:
        return ""


# Article Assistant removed per user request.


def create_session() -> requests.Session:
    """
    Create and configure the HTTP session.
    """
    session = requests.Session()

    # Prevent requests from automatically inheriting local proxy settings.
    # Change this to True if your organization requires an HTTP proxy.
    session.trust_env = False

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/125.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;"
                "q=0.9,image/avif,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
        }
    )

    return session





def convert_results_to_csv(results: list[dict]) -> bytes:
    """
    Convert search results into a UTF-8 CSV file with a BOM.

    UTF-8 with BOM helps Microsoft Excel display non-English
    characters correctly.
    """
    export_rows = [
        {
            "title": result["title"],
            "link": result["link"],
            "snippet": result["snippet"],
            "summary": result.get("summary", ""),
        }
        for result in results
    ]

    dataframe = pd.DataFrame(
        export_rows,
        columns=["title", "link", "snippet", "summary"]
    )

    return dataframe.to_csv(
        index=False
    ).encode("utf-8-sig")


def safe_filename(value: str) -> str:
    """
    Create a filesystem-friendly file name component.
    """
    cleaned = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        value.strip()
    )

    cleaned = cleaned.strip("_")

    return cleaned or "search"


# ============================================================
# STREAMLIT USER INTERFACE
# ============================================================

st.markdown(
    """
    <style>
    .app-header {
        align-items: center;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, .06);
        color: #0f172a;
        display: flex;
        gap: 14px;
        margin-bottom: 18px;
        padding: 18px 22px;
    }
    .app-brand-lockup {
        align-items: center;
        display: flex;
        gap: 6px;
        min-width: 112px;
    }
    .app-brand-icon {
        height: 30px;
        width: 30px;
    }
    .app-brand-wordmark {
        color: #d71920;
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: -.04em;
    }
    .app-header-divider {
        background: #cbd5e1;
        height: 30px;
        width: 1px;
    }
    .app-header h1 { font-size: 1.7rem; margin: 0; }
    .app-header p { color: #64748b; margin: 3px 0 0; }
    .filter-heading {
        color: #9f1239;
        font-size: 1.1rem;
        font-weight: 750;
        margin: 4px 0 2px;
    }
    .filter-caption { color: #64748b; font-size: .88rem; margin-bottom: 14px; }
    .query-preview {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        color: #334155;
        font-family: "Consolas", "Courier New", monospace;
        font-size: .78rem;
        line-height: 1.5;
        overflow-wrap: anywhere;
        padding: 10px 12px;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .welcome-card {
        background: #fff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, .07);
        margin-top: 12px;
        padding: 42px 46px;
    }
    .welcome-icon {
        align-items: center;
        background: #fff1f2;
        border-radius: 14px;
        color: #c8102e;
        display: flex;
        font-size: 2rem;
        height: 58px;
        justify-content: center;
        margin-bottom: 18px;
        width: 58px;
    }
    .welcome-card h2 { color: #0f172a; margin: 0 0 8px; }
    .welcome-card p, .welcome-card li { color: #475569; line-height: 1.55; }
    .welcome-card ol { padding-left: 22px; }
    .export-card {
        align-items: center;
        background: #fff7f7;
        border: 1px solid #fecdd3;
        border-radius: 12px;
        display: flex;
        gap: 14px;
        margin: 14px 0 16px;
        padding: 14px 16px;
    }
    .export-card-icon {
        align-items: center;
        background: #c8102e;
        border-radius: 9px;
        color: white;
        display: flex;
        font-size: 1.2rem;
        height: 38px;
        justify-content: center;
        width: 38px;
    }
    .export-card strong { color: #881337; }
    .export-card span { color: #64748b; display: block; font-size: .83rem; margin-top: 2px; }
    @media (max-width: 900px) {
        .welcome-card { padding: 28px 24px; }
        .app-header h1 { font-size: 1.35rem; }
        .app-brand-lockup { min-width: auto; }
        .app-header-divider { display: none; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <div class="app-brand-lockup">
            <svg class="app-brand-icon" viewBox="0 0 32 32" role="img" aria-label="OCBC logo">
                <circle cx="16" cy="16" r="15" fill="#d71920"/>
                <path fill="#ffffff" d="M7 7h7l4 4-4 4H7l4-4-4-4zm0 10h7l4 4-4 4H7l4-4-4-4zm12-7 6 6-6 6-4-4 2-2-2-2 4-4z"/>
            </svg>
            <span class="app-brand-wordmark">OCBC</span>
        </div>
        <div class="app-header-divider"></div>
        <div>
            <h1>OCBC Vendor Screening</h1>
            <p>Adverse news and AML triage prototype</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Search-result flags are review leads only. Always open and verify the "
    "full article before making a vendor decision."
)

filter_column, results_column = st.columns([0.30, 0.70], gap="large")

with filter_column:
    st.markdown('<div class="filter-heading">Search filters</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="filter-caption">Complete the filters, then run a screening.</div>',
        unsafe_allow_html=True,
    )

    with st.form("article_search_form"):
        subject = st.text_input(
            label="Company, person, or subject",
            value="DHL",
            placeholder="Example: DHL",
            help=(
                "The application automatically places the subject "
                "inside quotation marks for a more exact search."
            ),
            key="subject",
        )

        selected_keywords = st.multiselect(
            label="Risk keywords",
            options=DEFAULT_KEYWORD_OPTIONS,
            default=DEFAULT_KEYWORD_OPTIONS,
            help=(
                "Choose the keywords to include in the search. "
                "You can remove any default option or add it back later."
            ),
            key="selected_keywords",
        )

        additional_keywords = st.text_area(
            label="Additional keywords (optional)",
            height=100,
            placeholder="Example: embezzlement, tax evasion\nshell company",
            help=(
                "Add one or more extra keywords or phrases, separated by "
                "commas, semicolons, or new lines. Multi-word phrases are "
                "automatically searched in quotation marks."
            ),
            key="additional_keywords",
        )

        additional_keyword_terms = parse_additional_keywords(additional_keywords)
        keyword_terms = list(dict.fromkeys(selected_keywords + additional_keyword_terms))
        keyword_expression = build_keyword_expression(keyword_terms)

        final_query_preview = ""
        try:
            final_query_preview = build_search_query(
                subject=subject,
                keyword_expression=keyword_expression,
            )
        except ValueError:
            pass

        st.markdown("**Final query preview**")
        if final_query_preview:
            st.markdown(
                f'<div class="query-preview">{html.escape(final_query_preview)}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Enter a company, person, or subject to see the final query.")

        results_mode = st.radio(
            label="Results mode",
            options=["Get all results", "Limit results"],
            index=0,
            help=(
                "Choose 'Get all results' to collect everything from the "
                "search provider, or choose 'Limit results' to enter a maximum."
            ),
            key="results_mode",
        )

        requested_results = st.session_state.get("requested_results", 10)
        if results_mode == "Limit results":
            requested_results = st.number_input(
                label="Maximum number of articles",
                min_value=1,
                max_value=10000,
                value=requested_results,
                step=10,
                help=(
                    "The application stops when it reaches this number "
                    "or when the search provider has no more results."
                ),
                key="requested_results",
            )

        with st.expander("Advanced settings"):
            delay_seconds = st.number_input(
                label="Delay between search pages, in seconds",
                min_value=0.5,
                max_value=10.0,
                value=1.5,
                step=0.5,
                help=(
                    "A delay reduces request frequency and lowers the "
                    "chance of temporary blocking."
                ),
                key="delay_seconds",
            )

        generate_summaries = st.checkbox(
            label="Generate AI summaries for each article",
            value=False,
            help="Enable to generate a short 2-3 sentence summary for each collected article.",
            key="generate_summaries",
        )

        search_button = st.form_submit_button(
            label="🔍  Search articles",
            type="primary",
            use_container_width=True,
            key="search_button",
        )


# ============================================================
# SEARCH EXECUTION
# ============================================================

if search_button:
    if not subject.strip():
        st.error("Please enter a company, person, or subject.")

    elif not keyword_terms:
        st.error("Select at least one risk keyword or add an additional keyword.")

    else:
        final_query = build_search_query(
            subject=subject,
            keyword_expression=keyword_expression,
        )
        # Do not leave stale results visible while a new filter is running.
        st.session_state.pop("search_results", None)
        st.session_state.pop("vendor_screening_pdf", None)
        st.session_state.pop("vendor_screening_pdf_subject", None)

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        try:
            with st.spinner("Searching for articles..."):
                keyword_input = ", ".join(keyword_terms)
                max_results = (
                    None
                    if st.session_state.get("results_mode", "Get all results") == "Get all results"
                    else int(requested_results)
                )

                results, query_used, fallback_used = bing.search_articles(
                    subject=subject,
                    keyword_input=keyword_input,
                    maximum_results=max_results,
                    verify_ssl=False,
                    use_system_proxy=False,
                    show_debug=False,
                )

                try:
                    results = [result for result in results if not is_msn_result(result)]
                except Exception:
                    pass

                st.session_state["search_query"] = query_used
                st.session_state["fallback_used"] = fallback_used

                try:
                    if st.session_state.get("generate_summaries", False) and results:
                        api_key = ""
                        if hasattr(st, "secrets"):
                            api_key = st.secrets.get("GEMINI_API_KEY", "")
                            model = st.secrets.get("GEMINI_MODEL", "models/text-bison-001")
                        else:
                            import os as _os
                            api_key = _os.environ.get("GEMINI_API_KEY", "")
                            model = _os.environ.get("GEMINI_MODEL", "models/text-bison-001")

                        for result in results:
                            article_text = ""
                            try:
                                article_text = fetch_article_text(result.get("link", ""), verify_ssl=False)
                            except Exception:
                                pass

                            snippet = clean_text(strip_trailing_ellipsis(result.get("snippet", "") or ""))
                            text_to_summarize = article_text or "\n".join([result.get("title", ""), snippet])
                            try:
                                if api_key and article_text:
                                    result["summary"] = summarize_with_gemini(text_to_summarize, api_key=api_key, model=model)
                                else:
                                    result["summary"] = simple_local_summary(text_to_summarize)
                            except Exception:
                                result["summary"] = simple_local_summary(text_to_summarize)

                            try:
                                summary = strip_trailing_ellipsis(sanitize_text(result.get("summary", "")))
                                if summary and summary[-1] not in ".!?":
                                    summary = summary.rstrip(".") + "."
                                result["summary"] = summary
                            except Exception:
                                pass
                except Exception:
                    pass

            st.session_state["search_results"] = results
            st.session_state["search_subject"] = subject

        except Exception as exc:
            st.error(str(exc))


# ============================================================
# RESULTS WORKSPACE
# ============================================================

with results_column:
    if "search_results" not in st.session_state:
        st.markdown(
            """
            <div class="welcome-card">
                <div class="welcome-icon">⌕</div>
                <h2>Ready to screen a vendor?</h2>
                <p>Use the filters in the left panel to begin an adverse-news and AML screening.</p>
                <ol>
                    <li>Enter the company, person, or subject.</li>
                    <li>Keep, remove, or add risk keywords.</li>
                    <li>Choose the result count and click <b>Search articles</b>.</li>
                </ol>
                <p>Your results, risk flags, and vendor PDF export will appear here after the search completes.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        results = st.session_state["search_results"]
        try:
            results = [result for result in results if not is_msn_result(result)]
            st.session_state["search_results"] = results
        except Exception:
            pass

        search_subject = st.session_state.get("search_subject", "search")

        if not results:
            st.warning(
                "No results were collected. Try selecting fewer risk keywords, "
                "removing extra keywords, or trying again later."
            )
        else:
            st.markdown("### Screening results")
            submitted_query = st.session_state.get("search_query", "")
            if submitted_query:
                st.caption(f"Query submitted: {submitted_query}")

            aml_risk_assessments = create_aml_risk_assessments(results)
            display_results = []
            for result, assessment in zip(results, aml_risk_assessments):
                display_result = dict(result)
                display_result.setdefault("summary", "")
                display_result.update(assessment)
                display_results.append(display_result)

            display_dataframe = pd.DataFrame(display_results)
            overall_risk_level, overall_recommendation = overall_vendor_screening_result(aml_risk_assessments)
            risk_counts = display_dataframe["risk_level"].value_counts()

            st.success(f"Collected {len(results):,} unique search results.")
            high_column, medium_column, low_column, total_column = st.columns(4)
            with high_column:
                st.metric("High risk", int(risk_counts.get("High", 0)))
            with medium_column:
                st.metric("Medium risk", int(risk_counts.get("Medium", 0)))
            with low_column:
                st.metric("Low risk", int(risk_counts.get("Low", 0)))
            with total_column:
                st.metric("Total results", len(results))

            st.markdown("#### Evidence found")
            st.dataframe(
                display_dataframe[
                    [
                        "title", "link", "snippet", "summary", "keyword_score",
                        "matched_keywords", "risk_level", "aml_keyword_flags",
                        "onboarding_recommendation",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "title": st.column_config.TextColumn("Title", width="medium"),
                    "link": st.column_config.LinkColumn("Link", display_text="Open article", width="small"),
                    "snippet": st.column_config.TextColumn("Snippet", width="large"),
                    "summary": st.column_config.TextColumn("AI summary", width="large"),
                    "keyword_score": st.column_config.NumberColumn("Keyword score", format="%d"),
                    "matched_keywords": st.column_config.TextColumn("Matched keywords", width="medium"),
                    "risk_level": st.column_config.TextColumn("AML risk flag", width="small"),
                    "aml_keyword_flags": st.column_config.TextColumn("Major AML keyword flags", width="medium"),
                    "onboarding_recommendation": st.column_config.TextColumn("Vendor-screening direction", width="large"),
                },
            )

            if overall_risk_level == "High":
                st.error(f"Overall screening flag: HIGH. {overall_recommendation}")
            elif overall_risk_level == "Medium":
                st.warning(f"Overall screening flag: MEDIUM. {overall_recommendation}")
            else:
                st.success(f"Overall screening flag: LOW. {overall_recommendation}")

            st.caption(
                "Risk flags use only article titles and snippets. They are a "
                "triage aid, not proof of misconduct or final vendor acceptance."
            )

            if REPORTLAB_AVAILABLE:
                st.markdown(
                    """
                    <div class="export-card">
                        <div class="export-card-icon">▣</div>
                        <div><strong>Vendor AML screening report</strong>
                        <span>Create a consolidated, review-ready PDF with every article and risk flag.</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if "vendor_screening_pdf" not in st.session_state:
                    if st.button(
                        "Prepare vendor AML screening PDF",
                        key="prepare_vendor_screening_pdf",
                        type="primary",
                        use_container_width=True,
                    ):
                        try:
                            with st.spinner("Generating vendor AML screening PDF..."):
                                st.session_state["vendor_screening_pdf"] = generate_vendor_screening_pdf(
                                    subject=search_subject,
                                    results=results,
                                    assessments=aml_risk_assessments,
                                    include_ai_summary=st.session_state.get("generate_summaries", False),
                                )
                            st.session_state["vendor_screening_pdf_subject"] = search_subject
                        except Exception as exc:
                            st.error(f"Unable to generate the PDF report: {exc}")

                if "vendor_screening_pdf" in st.session_state:
                    pdf_filename = f"{safe_filename(search_subject)}_vendor_aml_screening.pdf"
                    st.download_button(
                        label="⬇️ Download vendor AML screening PDF",
                        data=st.session_state["vendor_screening_pdf"],
                        file_name=pdf_filename,
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True,
                    )
            else:
                st.warning(
                    "PDF generation is unavailable because ReportLab is not installed in the application environment."
                )

            with st.expander("Preview individual links"):
                for index, result in enumerate(results, start=1):
                    st.markdown(f"### {index}. {result['title']}")
                    st.markdown(f"[Open article]({result['link']})")

                    if st.session_state.get("generate_summaries", False):
                        if result.get("summary"):
                            st.info(result["summary"])
                    else:
                        snippet_text = prepare_snippet_for_display(result.get("snippet", "") or "")
                        if snippet_text:
                            st.write(snippet_text)

                    if result["matched_keywords"]:
                        st.caption(f"Matched keywords: {result['matched_keywords']}")
                    st.divider()
