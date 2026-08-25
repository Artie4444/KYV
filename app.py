import html
import re
import time
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import pandas as pd
import requests
import streamlit as st
import urllib3
from bs4 import BeautifulSoup


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Online Article Search",
    page_icon="🔎",
    layout="wide"
)


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

# DuckDuckGo's HTML search interface is generally easier to parse
# than its JavaScript-rendered search page.
SEARCH_URL = "https://html.duckduckgo.com/html/"

REQUEST_TIMEOUT_SECONDS = 30
DEFAULT_DELAY_SECONDS = 1.5

# Safety limit to prevent an accidental endless search.
# This only applies when "Collect all available results" is selected.
MAXIMUM_SAFETY_PAGES = 50

DEFAULT_KEYWORD_EXPRESSION = (
    '(fraud OR "money laundering" OR investigation OR '
    'corruption OR bribery OR sanctions OR lawsuit OR '
    'litigation OR prosecution OR criminal)'
)

DEFAULT_MATCH_KEYWORDS = [
    "fraud",
    "money laundering",
    "investigation",
    "corruption",
    "bribery",
    "sanction",
    "sanctions",
    "lawsuit",
    "litigation",
    "prosecution",
    "criminal",
    "crime",
    "arrest",
    "arrested",
    "charge",
    "charged",
    "conviction",
    "convicted",
    "fine",
    "penalty",
    "regulatory action",
]


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


def validate_keyword_expression(keyword_expression: str) -> tuple[bool, str]:
    """
    Perform basic validation of the Boolean keyword expression.

    This is not intended to be a complete Boolean-query parser.
    It catches common input mistakes such as unbalanced brackets
    or quotation marks.
    """
    expression = keyword_expression.strip()

    if not expression:
        return True, ""

    if expression.count("(") != expression.count(")"):
        return False, "The keyword expression has unbalanced parentheses."

    if expression.count('"') % 2 != 0:
        return False, "The keyword expression has an unbalanced quotation mark."

    return True, ""


def decode_duckduckgo_link(link: str) -> str:
    """
    Convert a DuckDuckGo redirect link into the underlying destination URL.

    DuckDuckGo may return links similar to:
        //duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com

    This function extracts and decodes the 'uddg' destination.
    """
    if not link:
        return ""

    link = html.unescape(link.strip())

    if link.startswith("//"):
        link = f"https:{link}"
    elif link.startswith("/"):
        link = urljoin("https://duckduckgo.com", link)

    parsed = urlparse(link)
    query_parameters = parse_qs(parsed.query)

    if "uddg" in query_parameters and query_parameters["uddg"]:
        return unquote(query_parameters["uddg"][0]).strip()

    return link


def extract_result(item: BeautifulSoup) -> dict | None:
    """
    Extract title, link, and snippet from one DuckDuckGo result container.
    """
    title_tag = item.select_one("a.result__a")

    if title_tag is None:
        return None

    title = clean_text(title_tag.get_text(" ", strip=True))
    link = decode_duckduckgo_link(title_tag.get("href", ""))

    snippet_tag = item.select_one(".result__snippet")
    snippet = (
        clean_text(snippet_tag.get_text(" ", strip=True))
        if snippet_tag
        else ""
    )

    if not title or not link:
        return None

    if not link.startswith(("http://", "https://")):
        return None

    return {
        "title": title,
        "link": link,
        "snippet": snippet,
    }


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


def search_duckduckgo(
    query: str,
    requested_results: int | None,
    delay_seconds: float,
    verify_ssl: bool = True
) -> list[dict]:
    """
    Search DuckDuckGo HTML results.

    Parameters
    ----------
    query:
        Final search query.

    requested_results:
        Maximum number of unique results to return.
        When None, continue until no more new results are found.

    delay_seconds:
        Delay between page requests.

    verify_ssl:
        Whether to verify SSL certificates.

    Returns
    -------
    list[dict]
        Search results containing title, link and snippet.
    """
    session = create_session()

    results = []
    seen_links = set()
    previous_page_signature = None

    progress_bar = st.progress(0)
    status_placeholder = st.empty()

    for page_number in range(1, MAXIMUM_SAFETY_PAGES + 1):

        if requested_results is not None:
            if len(results) >= requested_results:
                break

        # DuckDuckGo HTML commonly uses the "s" parameter as an offset.
        offset = (page_number - 1) * 30

        params = {
            "q": query,
            "s": offset,
            "dc": offset + 1,
            "kl": "wt-wt",
        }

        status_placeholder.info(
            f"Searching page {page_number}. "
            f"Collected {len(results)} unique results so far."
        )

        try:
            response = session.get(
                SEARCH_URL,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
                verify=verify_ssl,
            )
            response.raise_for_status()

        except requests.exceptions.SSLError as exc:
            raise RuntimeError(
                "SSL certificate verification failed. "
                "If this is caused by your organization's network, "
                "use the SSL verification option carefully."
            ) from exc

        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"The search request exceeded "
                f"{REQUEST_TIMEOUT_SECONDS} seconds."
            ) from exc

        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                f"The search request failed: {exc}"
            ) from exc

        soup = BeautifulSoup(response.text, "html.parser")

        search_items = soup.select(".result")

        if not search_items:
            search_items = soup.select(".web-result")

        if not search_items:
            status_placeholder.warning(
                "No result containers were found. "
                "The search provider may have changed its page structure, "
                "returned no matches, or blocked automated requests."
            )
            break

        page_results = []

        for item in search_items:
            extracted = extract_result(item)

            if extracted is None:
                continue

            link_key = extracted["link"].strip().casefold()

            if link_key in seen_links:
                continue

            keyword_score, matched_keywords = calculate_keyword_matches(
                title=extracted["title"],
                snippet=extracted["snippet"],
                keywords=DEFAULT_MATCH_KEYWORDS,
            )

            page_results.append(
                {
                    "title": extracted["title"],
                    "link": extracted["link"],
                    "snippet": extracted["snippet"],
                    "keyword_score": keyword_score,
                    "matched_keywords": ", ".join(matched_keywords),
                    "search_page": page_number,
                }
            )

            seen_links.add(link_key)

            if requested_results is not None:
                if len(results) + len(page_results) >= requested_results:
                    break

        # Stop if a page produces no additional unique results.
        if not page_results:
            status_placeholder.info(
                "The search returned no additional unique results."
            )
            break

        page_signature = tuple(
            item["link"]
            for item in page_results
        )

        # Stop if DuckDuckGo begins repeating the same page.
        if page_signature == previous_page_signature:
            status_placeholder.info(
                "The search provider repeated the previous result page."
            )
            break

        previous_page_signature = page_signature
        results.extend(page_results)

        if requested_results is not None:
            completion = min(
                len(results) / requested_results,
                1.0
            )
        else:
            completion = min(
                page_number / MAXIMUM_SAFETY_PAGES,
                1.0
            )

        progress_bar.progress(completion)

        if requested_results is not None:
            if len(results) >= requested_results:
                break

        time.sleep(delay_seconds)

    progress_bar.progress(1.0)
    status_placeholder.empty()

    if requested_results is not None:
        results = results[:requested_results]

    results.sort(
        key=lambda item: (
            item["keyword_score"],
            len(item["snippet"])
        ),
        reverse=True
    )

    return results


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

st.title("🔎 Online Article Search")

st.write(
    "Search for online articles using a company or subject together "
    "with an optional Boolean keyword expression. Results can be "
    "reviewed on screen and downloaded as a CSV file."
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
    )

    keyword_expression = st.text_area(
        label="Keyword expression",
        value=DEFAULT_KEYWORD_EXPRESSION,
        height=130,
        placeholder=(
            '(fraud OR "money laundering" OR investigation '
            'OR corruption)'
        ),
        help=(
            "Use uppercase OR between alternatives. "
            "Use quotation marks for exact phrases such as "
            '"money laundering".'
        ),
    )

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
    )

    result_mode = st.radio(
        label="How many results should be collected?",
        options=[
            "Collect all available results",
            "Set a maximum number of results",
        ],
        horizontal=True,
    )

    requested_results = None

    if result_mode == "Set a maximum number of results":
        requested_results = st.number_input(
            label="Maximum number of articles",
            min_value=1,
            max_value=1000,
            value=100,
            step=10,
            help=(
                "The application stops when it reaches this number "
                "or when the search provider has no more results."
            ),
        )

    with st.expander("Advanced settings"):

        delay_seconds = st.number_input(
            label="Delay between search pages, in seconds",
            min_value=0.5,
            max_value=10.0,
            value=DEFAULT_DELAY_SECONDS,
            step=0.5,
            help=(
                "A delay reduces request frequency and lowers the "
                "chance of temporary blocking."
            ),
        )

        disable_ssl_verification = st.checkbox(
            label="Disable SSL certificate verification",
            value=False,
            help=(
                "Only use this if your organization's network causes "
                "certificate verification errors. Keeping SSL "
                "verification enabled is safer."
            ),
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

    is_valid, validation_message = validate_keyword_expression(
        keyword_expression
    )

    if not subject.strip():
        st.error("Please enter a company, person, or subject.")

    elif not is_valid:
        st.error(validation_message)

    else:
        final_query = build_search_query(
            subject=subject,
            keyword_expression=keyword_expression
        )

        st.subheader("Search query")
        st.code(final_query, language=None)

        if disable_ssl_verification:
            urllib3.disable_warnings(
                urllib3.exceptions.InsecureRequestWarning
            )

        try:
            with st.spinner("Searching for articles..."):

                results = search_duckduckgo(
                    query=final_query,
                    requested_results=(
                        int(requested_results)
                        if requested_results is not None
                        else None
                    ),
                    delay_seconds=float(delay_seconds),
                    verify_ssl=not disable_ssl_verification,
                )

            st.session_state["search_results"] = results
            st.session_state["search_subject"] = subject

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
            "No results were collected. Try simplifying the keyword "
            "expression, reducing the number of terms, or trying again "
            "later."
        )

    else:
        st.success(
            f"Collected {len(results):,} unique search results."
        )

        display_dataframe = pd.DataFrame(results)

        left_column, right_column = st.columns(2)

        with left_column:
            st.metric(
                label="Unique results",
                value=f"{len(display_dataframe):,}"
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
            },
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