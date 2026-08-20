import requests
from bs4 import BeautifulSoup
from ddgs import DDGS


# ============================================================
# WEBPAGE CONTENT FETCHER
# ============================================================

def fetch_page_content(
    url: str,
    max_chars: int = 6000
) -> str:
    """
    Fetch readable text from a webpage.

    Returns cleaned webpage text.
    """

    if not url:
        return ""

    try:

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/151.0 Safari/537.36"
            )
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # ----------------------------------------------------
        # Remove unnecessary HTML elements
        # ----------------------------------------------------

        for element in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "form"
        ]):

            element.decompose()

        # ----------------------------------------------------
        # Extract visible text
        # ----------------------------------------------------

        text = soup.get_text(
            separator=" ",
            strip=True
        )

        # ----------------------------------------------------
        # Clean excessive whitespace
        # ----------------------------------------------------

        text = " ".join(
            text.split()
        )

        # ----------------------------------------------------
        # Limit context size
        # ----------------------------------------------------

        return text[:max_chars]

    except Exception as e:

        print(
            f"Page fetch failed for {url}: {e}"
        )

        return ""


# ============================================================
# WEB SEARCH
# ============================================================

def web_search(
    query: str,
    max_results: int = 5
) -> list[dict]:
    """
    Search the web using DuckDuckGo.

    Returns:

    [
        {
            "title": "...",
            "url": "...",
            "snippet": "...",
            "content": "..."
        }
    ]
    """

    results = []

    try:

        with DDGS() as ddgs:

            search_results = ddgs.text(
                query,
                max_results=max_results
            )

            for result in search_results:

                title = result.get(
                    "title",
                    ""
                )

                url = result.get(
                    "href",
                    ""
                )

                snippet = result.get(
                    "body",
                    ""
                )

                # ------------------------------------------------
                # Fetch actual webpage content
                # ------------------------------------------------

                content = fetch_page_content(
                    url
                )

                results.append({

                    "title": title,

                    "url": url,

                    "snippet": snippet,

                    "content": content
                })

    except Exception as e:

        print(
            f"Web search error: {e}"
        )

    return results


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    query = (
        "What is the latest development in AI?"
    )

    print("\n" + "=" * 60)
    print("WEB SEARCH TEST")
    print("=" * 60)

    print("\nQuery:")
    print(query)

    results = web_search(
        query,
        max_results=5
    )

    print("\n" + "=" * 60)
    print("SEARCH RESULTS")
    print("=" * 60)

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nRESULT {i}"
        )

        print(
            f"Title: {result['title']}"
        )

        print(
            f"URL: {result['url']}"
        )

        print(
            f"\nSnippet:\n{result['snippet']}"
        )

        print(
            "\nPage Content Preview:"
        )

        content = result.get(
            "content",
            ""
        )

        if content:

            print(
                content[:1000]
            )

        else:

            print(
                "No webpage content available."
            )

        print(
            "-" * 60
        )