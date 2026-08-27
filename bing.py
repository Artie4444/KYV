import html
import re
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import pandas as pd
import requests
import streamlit as st
import urllib3
try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except Exception:
    LANGDETECT_AVAILABLE = False


# ============================================================
# STREAMLIT PAGE CONFIGURATION (only when run directly)
# ============================================================

def _configure_streamlit_page():
    st.set_page_config(
        page_title="Online Adverse News Search",
        page_icon="🔎",
        layout="wide",
    )


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

# Free Bing News RSS endpoint. This avoids parsing Bing result-page HTML.
BING_NEWS_RSS_URL = "https://www.bing.com/news/search"
REQUEST_TIMEOUT_SECONDS = 30

DEFAULT_KEYWORD_EXPRESSION = (
    'fraud, money laundering, investigation, corruption, bribery, '
    'sanctions, lawsuit, litigation, prosecution, criminal'
)

DEFAULT_MATCH_KEYWORDS = [
    "fraud",
    "fraudulent",
    "money laundering",
    "anti-money laundering",
    "investigation",
    "investigated",
    "corruption",
    "bribery",
    "bribe",
    "sanction",
    "sanctions",
    "lawsuit",
    "litigation",
    "prosecution",
    "prosecuted",
    "criminal",
    "crime",
    "arrest",
    "arrested",
    "charge",
    "charged",
    "conviction",
    "convicted",
    "fine",
    "fined",
    "penalty",
    "regulatory action",
]


# ============================================================
# TEXT HELPERS
# ============================================================


def normalize_whitespace(value: str) -> str:
    """Replace repeated whitespace with one space."""
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def strip_html(value: str) -> str:
    """Remove HTML tags, decode entities, and normalize whitespace."""
    if not value:
        return ""
    without_tags = re.sub(r"<[^>]+>", " ", value)
    return normalize_whitespace(html.unescape(without_tags))


def parse_keywords(value: str) -> List[str]:
    """
    Convert comma-separated or OR-separated input into a clean list.

    Supported examples:
      fraud, money laundering, sanctions
      fraud OR "money laundering" OR sanctions
      (fraud OR "money laundering" OR sanctions)
    """
    if not value:
        return DEFAULT_MATCH_KEYWORDS.copy()

    cleaned = value.strip()
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = cleaned[1:-1]

    parts = re.split(r"\s+OR\s+|,|\n", cleaned, flags=re.IGNORECASE)
    keywords = []

    for part in parts:
        keyword = part.strip().strip('"').strip("'")
        keyword = normalize_whitespace(keyword)
        if keyword:
            keywords.append(keyword)

    # Remove duplicates case-insensitively while preserving order.
    unique_keywords = []
    seen = set()
    for keyword in keywords:
        key = keyword.casefold()
        if key not in seen:
            seen.add(key)
            unique_keywords.append(keyword)

    return unique_keywords or DEFAULT_MATCH_KEYWORDS.copy()


def build_search_query(subject: str, keywords: List[str]) -> str:
    """Build a Bing News query from the subject and keywords."""
    subject = normalize_whitespace(subject)
    if not subject:
        raise ValueError("Please enter a company, person, or subject.")

    quoted_subject = subject
    if not (subject.startswith('"') and subject.endswith('"')):
        quoted_subject = f'"{subject}"'

    if not keywords:
        return quoted_subject

    keyword_query = " OR ".join(
        f'"{keyword}"' if " " in keyword else keyword
        for keyword in keywords
    )
    return f"{quoted_subject} ({keyword_query})"


def calculate_keyword_matches(
    title: str,
    snippet: str,
    keywords: List[str],
) -> Tuple[int, List[str]]:
    """
    Match complete keywords or phrases in title and snippet.

    Boundary checks prevent a short keyword from matching inside a
    longer unrelated word. For example, 'fine' will not match 'refined'.
    """
    combined_text = normalize_whitespace(f"{title} {snippet}")
    matched = []

    for keyword in keywords:
        pattern = rf"(?<!\w){re.escape(keyword)}(?!\w)"
        if re.search(pattern, combined_text, flags=re.IGNORECASE):
            matched.append(keyword)

    # Remove duplicates case-insensitively while preserving order.
    unique_matched = []
    seen = set()
    for keyword in matched:
        key = keyword.casefold()
        if key not in seen:
            seen.add(key)
            unique_matched.append(keyword)

    return len(unique_matched), unique_matched


def contains_non_latin(text: str) -> bool:
    """Return True if the text contains non-Latin (CJK) characters.

    This filters Chinese, Japanese, and Korean characters commonly
    present in East Asian languages.
    """
    if not text:
        return False
    # CJK Unified Ideographs, Hangul Syllables, Hiragana, Katakana
    return bool(re.search(r"[\u4e00-\u9fff\u3400-\u4dbf\u1100-\u11ff\u3130-\u318f\u3040-\u309f\u30a0-\u30ff]", text))


def is_probably_english(text: str) -> bool:
    """Fallback heuristic to guess if `text` is English when langdetect is unavailable.

    Checks for a high ratio of ASCII letters and presence of common English words.
    """
    if not text:
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    ascii_letters = sum(1 for c in letters if ord(c) < 128)
    if ascii_letters / len(letters) < 0.8:
        return False

    words = re.findall(r"[A-Za-z']+", text.lower())
    common = {"the", "and", "is", "in", "of", "for", "to", "with", "on", "as", "are", "at"}
    matches = sum(1 for w in words if w in common)
    return matches >= 1


def is_english(text: str) -> bool:
    """Return True if text is detected as English.

    Prefer `langdetect` when available, otherwise use a heuristic.
    """
    if not text:
        return False
    if LANGDETECT_AVAILABLE:
        try:
            return detect(text) == "en"
        except Exception:
            return is_probably_english(text)
    return is_probably_english(text)


# ============================================================
# HTTP AND RSS HELPERS
# ============================================================


def create_session(use_system_proxy: bool) -> requests.Session:
    """Create a request session with optional environment proxy use."""
    session = requests.Session()

    # Keep this False by default because a malformed HTTP_PROXY or
    # HTTPS_PROXY environment variable caused proxy error.
    session.trust_env = use_system_proxy

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
        }
    )
    return session


def safe_xml_text(element: ET.Element, child_name: str) -> str:
    """Return cleaned text from a direct RSS child element."""
    child = element.find(child_name)
    if child is None or child.text is None:
        return ""
    return strip_html(child.text)


def format_publication_date(raw_date: str) -> Tuple[str, Optional[datetime]]:
    """Return a readable publication date and a sortable datetime."""
    if not raw_date:
        return "", None

    try:
        parsed = parsedate_to_datetime(raw_date)
        return parsed.strftime("%Y-%m-%d %H:%M %Z").strip(), parsed
    except (TypeError, ValueError, OverflowError):
        return raw_date, None


def validate_external_link(link: str) -> bool:
    """Accept only normal HTTP or HTTPS links with a hostname."""
    try:
        parsed = urlparse(link)
        return parsed.scheme in {"http", "https"} and bool(parsed.hostname)
    except ValueError:
        return False


def fetch_bing_news_rss(
    query: str,
    verify_ssl: bool,
    use_system_proxy: bool,
    show_debug: bool,
) -> List[Dict]:
    """Fetch and parse one Bing News RSS feed."""
    session = create_session(use_system_proxy=use_system_proxy)

    params = {
        "q": query,
        "format": "rss",
        # Force English-language results when possible
        "setlang": "en-US",
        "cc": "US",
    }

    try:
        response = session.get(
            BING_NEWS_RSS_URL,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
            verify=verify_ssl,
            allow_redirects=True,
        )
        response.raise_for_status()
    except requests.exceptions.ProxyError as exc:
        raise RuntimeError(
            "The proxy configuration is invalid or unavailable. "
            "Keep 'Use system proxy settings' turned off, then try again."
        ) from exc
    except requests.exceptions.SSLError as exc:
        raise RuntimeError(
            "SSL certificate verification failed. If this is caused by the "
            "organization network, use the SSL option carefully."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"The Bing News request exceeded {REQUEST_TIMEOUT_SECONDS} seconds."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"The Bing News request failed: {exc}") from exc
    finally:
        session.close()

    content_type = response.headers.get("Content-Type", "")

    if show_debug:
        st.write("Search provider:", "Bing News RSS")
        st.write("HTTP status:", response.status_code)
        st.write("Returned URL:", response.url)
        st.write("Content type:", content_type or "Not provided")
        st.write("Response length:", len(response.content))

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as exc:
        preview = response.text[:1000]
        if show_debug:
            st.code(preview, language="xml")
        raise RuntimeError(
            "Bing did not return a valid RSS feed. Enable diagnostics to "
            "inspect the response."
        ) from exc

    items = root.findall(".//item")

    if show_debug:
        st.write("RSS items found:", len(items))

    results = []
    for item in items:
        title = safe_xml_text(item, "title")
        link = safe_xml_text(item, "link")
        snippet = safe_xml_text(item, "description")
        source = safe_xml_text(item, "source")
        raw_date = safe_xml_text(item, "pubDate")
        published, published_dt = format_publication_date(raw_date)

        if not title or not validate_external_link(link):
            continue

        results.append(
            {
                "title": title,
                "link": link,
                "snippet": snippet,
                "source": source,
                "published": published,
                "_published_dt": published_dt,
            }
        )

    return results


def fetch_bing_web_results(
    query: str,
    maximum_results: int = 0,
):
    """
    Web scraping disabled. This project uses only Bing News RSS.
    """
    return []


# ============================================================
# SEARCH ORCHESTRATION
# ============================================================

def search_articles(
    subject: str,
    keyword_input: str,
    maximum_results: Optional[int],
    verify_ssl: bool,
    use_system_proxy: bool,
    show_debug: bool,
) -> Tuple[List[Dict], str, bool]:
    """
    Search risk terms first. If the risk query returns no RSS items,
    automatically retry with the subject only and score matches locally.
    """
    keywords = parse_keywords(keyword_input)
    primary_query = build_search_query(subject, keywords)

    rss_results = fetch_bing_news_rss(
        query=primary_query,
        verify_ssl=verify_ssl,
        use_system_proxy=use_system_proxy,
        show_debug=show_debug,
    )

    web_max = maximum_results if maximum_results is not None else 10000
    web_results = fetch_bing_web_results(
        query=primary_query,
        maximum_results=web_max,
    )

    all_results = rss_results + web_results

    unique_results = []

    seen = set()

    for result in all_results:

        link = (
            result["link"]
            .strip()
            .lower()
            .rstrip("/")
        )

        if link in seen:
            continue

        seen.add(link)

        unique_results.append(
            result
        )

    fallback_used = False
    query_used = primary_query

    results = unique_results

    if not results:

        fallback_used = True
        query_used = build_search_query(subject, [])
        results = fetch_bing_news_rss(
            query=query_used,
            verify_ssl=verify_ssl,
            use_system_proxy=use_system_proxy,
            show_debug=show_debug,
        )

    final_results = []
    seen_links = set()

    for result in results:
        link_key = result["link"].strip().casefold().rstrip("/")
        if link_key in seen_links:
            continue
        seen_links.add(link_key)
        # Filter out non-Latin (CJK) language results early.
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        if contains_non_latin(title) or contains_non_latin(snippet):
            continue

        # Ensure the combined text is English.
        combined_text = f"{title} {snippet}".strip()
        if not is_english(combined_text):
            continue

        score, matched_keywords = calculate_keyword_matches(
            title=result["title"],
            snippet=result["snippet"],
            keywords=keywords,
        )

        final_results.append(
            {
                "title": result["title"],
                "link": result["link"],
                "snippet": result["snippet"],
                "source": result["source"],
                "published": result["published"],
                "keyword_score": score,
                "matched_keywords": ", ".join(matched_keywords),
                "_published_dt": result["_published_dt"],
            }
        )

    # Rank risk matches first, then dated items, then longer snippets.
    final_results.sort(
        key=lambda item: (
            item["keyword_score"],
            item["_published_dt"] is not None,
            item["_published_dt"] or datetime.min,
            len(item["snippet"]),
        ),
        reverse=True,
    )

    for result in final_results:
        result.pop("_published_dt", None)

    if maximum_results is None:
        return final_results, query_used, fallback_used

    return final_results[:maximum_results], query_used, fallback_used


# ============================================================
# CSV AND FILENAME HELPERS
# ============================================================

def convert_results_to_csv(results: List[Dict]) -> bytes:
    """Convert results into a UTF-8 CSV with BOM for Excel."""
    columns = [
        "title",
        "link",
        "snippet",
        "source",
        "published",
        "keyword_score",
        "matched_keywords",
    ]
    dataframe = pd.DataFrame(results, columns=columns)
    return dataframe.to_csv(index=False).encode("utf-8-sig")


def safe_filename(value: str) -> str:
    """Create a filesystem-friendly filename component."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    return cleaned or "search"


if __name__ == "__main__":
    _configure_streamlit_page()

    # ============================================================
    # SESSION STATE
    # ============================================================

    if "search_results" not in st.session_state:
        st.session_state["search_results"] = None
    if "search_subject" not in st.session_state:
        st.session_state["search_subject"] = ""
    if "search_query" not in st.session_state:
        st.session_state["search_query"] = ""
    if "fallback_used" not in st.session_state:
        st.session_state["fallback_used"] = False


    # ============================================================
    # STREAMLIT USER INTERFACE
    # ============================================================

    st.title("🔎 Online Adverse News Search")

    st.write(
        "Search Bing News RSS for articles related to a company, person, or "
        "subject. Results are scored locally using matched risk keywords and "
        "can be downloaded as CSV."
    )

    st.warning(
        "Search results are leads for review. A keyword match does not prove "
        "misconduct. Open the source and verify the full article."
    )

    with st.form("article_search_form"):
        subject = st.text_input(
            label="Company, person, or subject",
            value="HSBC",
            placeholder="Example: HSBC",
        )

        keyword_input = st.text_area(
            label="Risk keywords",
            value=DEFAULT_KEYWORD_EXPRESSION,
            height=120,
            help=(
                "Enter terms separated by commas or uppercase OR. "
                "Exact phrases such as money laundering are supported."
            ),
        )

        parsed_keyword_preview = parse_keywords(keyword_input)
        final_query_preview = ""
        if subject.strip():
            final_query_preview = build_search_query(
                subject=subject,
                keywords=parsed_keyword_preview,
            )

        st.text_input(
            label="Final search query preview",
            value=final_query_preview,
            disabled=True,
        )

        maximum_results = st.number_input(
            label="Maximum number of articles",
            min_value=1,
            max_value=100,
            value=30,
            step=10,
            help="The RSS feed may contain fewer items than this maximum.",
        )

        with st.expander("Advanced settings"):
            use_system_proxy = st.checkbox(
                label="Use system proxy settings",
                value=False,
                help=(
                    "Keep this off unless the organization requires a configured "
                    "HTTP or HTTPS proxy."
                ),
            )

            disable_ssl_verification = st.checkbox(
                label="Disable SSL certificate verification",
                value=False,
                help=(
                    "Only use this if the organization network causes certificate "
                    "verification errors. Keeping verification enabled is safer."
                ),
            )

            show_debug = st.checkbox(
                label="Show search diagnostics",
                value=False,
            )

        search_button = st.form_submit_button(
            label="Search articles",
            type="primary",
            use_container_width=True,
        )


    # ============================================================
    # SEARCH EXECUTION
    # ============================================================

    if search_button:
        if not subject.strip():
            st.error("Please enter a company, person, or subject.")
        else:
            if disable_ssl_verification:
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

            try:
                with st.spinner("Searching Bing News..."):
                    results, query_used, fallback_used = search_articles(
                        subject=build_search_query(subject, parse_keywords(keyword_input)),
                        verify_ssl=not disable_ssl_verification,
                        use_system_proxy=use_system_proxy,
                        show_debug=show_debug,
                    )

                st.session_state["search_results"] = results
                st.session_state["search_subject"] = subject
                st.session_state["search_query"] = query_used
                st.session_state["fallback_used"] = fallback_used

            except Exception as exc:
                st.session_state["search_results"] = None
                st.error(str(exc))


    # ============================================================
    # DISPLAY AND DOWNLOAD RESULTS
    # ============================================================

    results = st.session_state.get("search_results")

    if results is not None:
        search_subject = st.session_state.get("search_subject", "search")
        query_used = st.session_state.get("search_query", "")
        fallback_used = st.session_state.get("fallback_used", False)

        st.subheader("Search query used")
        st.code(query_used, language=None)

        if fallback_used:
            st.info(
                "The risk-keyword query returned no RSS items, so the app retried "
                "with the subject only. Keyword scores below were still calculated "
                "locally from each title and snippet."
            )

        if not results:
            st.warning(
                "No RSS results were returned. Try a broader company name or fewer "
                "keywords. This can also happen when Bing News has no current feed "
                "items for the query."
            )
        else:
            dataframe = pd.DataFrame(results)
            matched_count = int((dataframe["keyword_score"] > 0).sum())

            st.success(f"Collected {len(results):,} unique results.")

            left_column, right_column = st.columns(2)
            with left_column:
                st.metric("Unique results", f"{len(results):,}")
            with right_column:
                st.metric("Results with keyword matches", f"{matched_count:,}")

            st.subheader("Search results")
            st.dataframe(
                dataframe[
                    [
                        "title",
                        "link",
                        "snippet",
                        "source",
                        "published",
                        "keyword_score",
                        "matched_keywords",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "title": st.column_config.TextColumn("Title", width="medium"),
                    "link": st.column_config.LinkColumn(
                        "Link",
                        display_text="Open article",
                        width="small",
                    ),
                    "snippet": st.column_config.TextColumn("Snippet", width="large"),
                    "source": st.column_config.TextColumn("Source", width="small"),
                    "published": st.column_config.TextColumn(
                        "Published",
                        width="small",
                    ),
                    "keyword_score": st.column_config.NumberColumn(
                        "Keyword score",
                        format="%d",
                    ),
                    "matched_keywords": st.column_config.TextColumn(
                        "Matched keywords",
                        width="medium",
                    ),
                },
            )

            csv_data = convert_results_to_csv(results)
            csv_filename = (
                f"{safe_filename(search_subject)}_bing_news_results.csv"
            )

            st.download_button(
                label="⬇️ Download results as CSV",
                data=csv_data,
                file_name=csv_filename,
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )

            with st.expander("Preview individual results"):
                for index, result in enumerate(results, start=1):
                    st.markdown(f"### {index}. {result['title']}")
                    st.markdown(f"[Open article]({result['link']})")

                    details = []
                    if result["source"]:
                        details.append(result["source"])
                    if result["published"]:
                        details.append(result["published"])
                    if details:
                        st.caption(" | ".join(details))

                    if result["snippet"]:
                        st.write(result["snippet"])

                    if result["matched_keywords"]:
                        st.caption(
                            "Matched keywords: " + result["matched_keywords"]
                        )
                    else:
                        st.caption("Matched keywords: None")

                    st.divider()
