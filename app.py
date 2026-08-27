import html
import re
import time
from datetime import datetime
from io import BytesIO
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import pandas as pd
import requests
import streamlit as st
import urllib3
import bing

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
]

MEDIUM_RISK_AML_KEYWORDS = [
    "fraud",
    "corruption",
    "bribery",
    "investigation",
    "lawsuit",
    "litigation",
    "charge",
    "fine",
    "penalty",
    "regulatory action",
    "embezzlement",
    "shell company",
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
            Paragraph(
                f"<b>Search-result snippet:</b> "
                f"{pdf_safe_text(result['snippet'] or 'No snippet available.')}",
                styles["VendorBody"],
            ),
        ]
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
        }
        for result in results
    ]

    dataframe = pd.DataFrame(
        export_rows,
        columns=["title", "link", "snippet"]
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

st.title("🔎 Online Adverse News Search")

st.write(
    "Search for online articles using a company or subject together "
    "with selected risk keywords. Add your own optional keywords or "
    "phrases, then review the results on screen or download them as a "
    "CSV file."
)

st.warning(
    "Search results are leads for further review. A title or snippet "
    "containing a risk-related keyword does not prove misconduct. "
    "Always open the source and verify the full article."
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

    additional_keyword_terms = parse_additional_keywords(
        additional_keywords
    )
    keyword_terms = list(dict.fromkeys(
        selected_keywords + additional_keyword_terms
    ))
    keyword_expression = build_keyword_expression(keyword_terms)

    final_query_preview = ""

    try:
        final_query_preview = build_search_query(
            subject=subject,
            keyword_expression=keyword_expression
        )
    except ValueError:
        pass

    st.text_input(
        label="Final search query preview",
        value=final_query_preview,
        disabled=True,
        key="final_query_preview",
    )

    # Let the user choose to collect all results or enter a maximum.
    collect_all = st.checkbox(
        label="Collect all available results",
        value=False,
        help="When checked, the app will collect all available results from the search provider.",
        key="collect_all",
    )

    requested_results = st.number_input(
        label="Maximum number of articles",
        min_value=1,
        max_value=10000,
        value=10,
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

        # Advanced options removed for simplicity; sensible defaults used.
        # SSL verification is disabled by default for easier use in
        # corporate or restricted networks; proxy and diagnostics are
        # disabled.

    search_button = st.form_submit_button(
        label="Search articles",
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
        st.error(
            "Select at least one risk keyword or add an additional keyword."
        )

    else:
        final_query = build_search_query(
            subject=subject,
            keyword_expression=keyword_expression
        )

        st.subheader("Search query")
        st.code(final_query, language=None)

        # SSL verification disabled by default for simplified UX.
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        try:
            with st.spinner("Searching for articles..."):

                # Use the Bing searcher from bing.py. It returns
                # (results, query_used, fallback_used).
                keyword_input = ", ".join(keyword_terms)

                max_results = None if collect_all else int(requested_results)

                results, query_used, fallback_used = bing.search_articles(
                    subject=subject,
                    keyword_input=keyword_input,
                    maximum_results=max_results,
                    # Keep SSL verification disabled for non-technical users.
                    verify_ssl=False,
                    use_system_proxy=False,
                    show_debug=False,
                )

                # Preserve the original name for downstream UI logic.
                st.session_state["search_query"] = query_used
                st.session_state["fallback_used"] = fallback_used

            st.session_state["search_results"] = results
            st.session_state["search_subject"] = subject
            st.session_state.pop("vendor_screening_pdf", None)
            st.session_state.pop("vendor_screening_pdf_subject", None)

        except Exception as exc:
            st.error(str(exc))


# ============================================================
# DISPLAY AND DOWNLOAD RESULTS
# ============================================================

if "search_results" in st.session_state:

    results = st.session_state["search_results"]
    search_subject = st.session_state.get(
        "search_subject",
        "search"
    )

    if not results:
        st.warning(
            "No results were collected. Try selecting fewer risk keywords, "
            "removing extra keywords, or trying again later."
        )

    else:
        st.success(
            f"Collected {len(results):,} unique search results."
        )

        aml_risk_assessments = create_aml_risk_assessments(results)
        display_results = []

        for result, assessment in zip(results, aml_risk_assessments):
            display_result = dict(result)
            display_result.update(assessment)
            display_results.append(display_result)

        display_dataframe = pd.DataFrame(display_results)
        overall_risk_level, overall_recommendation = (
            overall_vendor_screening_result(aml_risk_assessments)
        )

        left_column, middle_column, right_column = st.columns(3)

        with left_column:
            st.metric(
                label="Unique results",
                value=f"{len(display_dataframe):,}"
            )

        with middle_column:
            st.metric(
                label="High AML risk results",
                value=(
                    display_dataframe["risk_level"] == "High"
                ).sum(),
            )

        with right_column:
            results_with_keyword_matches = (
                display_dataframe["keyword_score"] > 0
            ).sum()

            st.metric(
                label="Results with local keyword matches",
                value=f"{results_with_keyword_matches:,}"
            )

        st.subheader("Search results")

        st.dataframe(
            display_dataframe[
                [
                    "title",
                    "link",
                    "snippet",
                    "keyword_score",
                    "matched_keywords",
                    "risk_level",
                    "aml_keyword_flags",
                    "onboarding_recommendation",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "title": st.column_config.TextColumn(
                    "Title",
                    width="medium"
                ),
                "link": st.column_config.LinkColumn(
                    "Link",
                    display_text="Open article",
                    width="small"
                ),
                "snippet": st.column_config.TextColumn(
                    "Snippet",
                    width="large"
                ),
                "keyword_score": st.column_config.NumberColumn(
                    "Keyword score",
                    format="%d"
                ),
                "matched_keywords": st.column_config.TextColumn(
                    "Matched keywords",
                    width="medium"
                ),
                "risk_level": st.column_config.TextColumn(
                    "AML risk flag",
                    width="small"
                ),
                "aml_keyword_flags": st.column_config.TextColumn(
                    "Major AML keyword flags",
                    width="medium"
                ),
                "onboarding_recommendation": st.column_config.TextColumn(
                    "Vendor-screening direction",
                    width="large"
                ),
            },
        )

        st.subheader("Vendor AML screening")
        risk_message = (
            f"Overall screening flag: {overall_risk_level.upper()}. "
            f"{overall_recommendation}"
        )

        if overall_risk_level == "High":
            st.error(risk_message)
        elif overall_risk_level == "Medium":
            st.warning(risk_message)
        else:
            st.success(risk_message)

        st.caption(
            "The risk flag is based only on the title and snippet of each "
            "search result. It is a triage aid, not proof of misconduct or a "
            "final vendor-acceptance decision."
        )

        if REPORTLAB_AVAILABLE:
            if st.button(
                "Prepare vendor AML screening PDF",
                key="prepare_vendor_screening_pdf",
                use_container_width=True,
            ):
                try:
                    with st.spinner("Generating vendor AML screening PDF..."):
                        st.session_state["vendor_screening_pdf"] = (
                            generate_vendor_screening_pdf(
                                subject=search_subject,
                                results=results,
                                assessments=aml_risk_assessments,
                            )
                        )
                    st.session_state["vendor_screening_pdf_subject"] = (
                        search_subject
                    )
                except Exception as exc:
                    st.error(f"Unable to generate the PDF report: {exc}")

            if "vendor_screening_pdf" in st.session_state:
                pdf_filename = (
                    f"{safe_filename(search_subject)}_"
                    "vendor_aml_screening.pdf"
                )
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
                "PDF generation is unavailable because ReportLab is not "
                "installed in the application environment."
            )

        csv_data = convert_results_to_csv(results)

        csv_filename = (
            f"{safe_filename(search_subject)}_article_results.csv"
        )

        st.download_button(
            label="⬇️ Download results as CSV",
            data=csv_data,
            file_name=csv_filename,
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )

        st.markdown(
            """
            <style>
            div[data-testid="stPopover"] {
                position: fixed;
                right: 1.5rem;
                bottom: 1.5rem;
                z-index: 999;
            }
            div[data-testid="stPopover"] > button {
                align-items: center;
                background: #0f766e;
                border: 0;
                border-radius: 50%;
                box-shadow: 0 5px 18px rgba(0, 0, 0, 0.25);
                color: white;
                display: flex;
                font-size: 1.4rem;
                height: 3.5rem;
                justify-content: center;
                min-height: 3.5rem;
                min-width: 3.5rem;
                padding: 0;
                width: 3.5rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Article Assistant removed.

        with st.expander("Preview individual links"):
            for index, result in enumerate(results, start=1):

                st.markdown(
                    f"### {index}. {result['title']}"
                )

                st.markdown(
                    f"{result['link']}"
                )

                if result["snippet"]:
                    st.write(result["snippet"])

                if result["matched_keywords"]:
                    st.caption(
                        "Matched keywords: "
                        f"{result['matched_keywords']}"
                    )

                st.divider()
