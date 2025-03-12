import argparse
import json
import requests
from typing import Dict, List, Any
import html2text


def search_wikipedia(query: str, limit: int = 2) -> List[Dict[str, Any]]:
    """
    Search Wikipedia and return detailed information for each result.

    Args:
        query: Search term to look for on Wikipedia
        limit: Maximum number of results to return

    Returns:
        List of dictionaries containing detailed information for each result
    """
    # First, search for pages matching the query
    search_url = "https://en.wikipedia.org/w/api.php"
    search_params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "utf8": 1,
    }

    search_response = requests.get(search_url, params=search_params)
    search_data = search_response.json()

    if "query" not in search_data or "search" not in search_data["query"]:
        return []

    search_results = search_data["query"]["search"]
    if not search_results:
        return []

    # For each search result, get the detailed content
    detailed_results = []

    for result in search_results:
        page_id = result["pageid"]
        title = result["title"]

        # Get the full page content and sections (TOC)
        content_params = {
            "action": "parse",
            "format": "json",
            "pageid": page_id,
            "prop": "text|sections",
            "utf8": 1,
        }

        content_response = requests.get(search_url, params=content_params)
        content_data = content_response.json()

        if "parse" in content_data:
            # Extract page text and truncate to reduce size
            html_content = content_data["parse"]["text"]["*"]

            # Convert HTML to markdown
            try:
                # Create html2text instance with improved settings for Wikipedia content
                h = html2text.HTML2Text()
                h.ignore_links = False
                h.ignore_images = False
                h.ignore_tables = False
                h.body_width = 0  # Don't wrap text
                h.unicode_snob = True  # Use Unicode instead of ASCII
                h.mark_code = True  # Properly format code sections
                h.skip_internal_links = False
                h.inline_links = True
                h.protect_links = True  # Don't replace links with numbers
                h.wrap_links = False  # Don't wrap links

                markdown_content = h.handle(html_content)
            except Exception as e:
                markdown_content = f"Error converting HTML to markdown: {str(e)}"

            # Extract table of contents (sections) - limited to 5 for brevity
            toc = []
            if "sections" in content_data["parse"]:
                for i, section in enumerate(content_data["parse"]["sections"]):
                    if i >= 5:  # Limit to first 5 sections
                        break
                    toc.append(
                        {
                            "index": section.get("index", ""),
                            "level": section.get("level", ""),
                            "title": section.get("line", ""),
                            "anchor": section.get("anchor", ""),
                        }
                    )

            # Truncate content for both HTML and markdown
            # if len(html_content) > 1000:
            #     html_content = html_content[:1000] + "... [content truncated]"
            #
            # if len(markdown_content) > 1000:
            #     markdown_content = markdown_content[:1000] + "... [content truncated]"

            # Create result object
            result_object = {
                "page_id": page_id,
                "title": title,
                "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                "html_content": html_content,
                "text_content": markdown_content,
                "table_of_contents": toc,
            }

            detailed_results.append(result_object)

    return detailed_results


def main():
    parser = argparse.ArgumentParser(
        description="Search Wikipedia and return detailed JSON results"
    )
    parser.add_argument("query", help="Search term to look for on Wikipedia")
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=2,
        help="Maximum number of results (default: 2)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path for JSON results (default: print to stdout)",
    )

    args = parser.parse_args()

    # Perform the search
    results = search_wikipedia(args.query, args.limit)

    # Format the output as JSON
    output_json = json.dumps(results, indent=2, ensure_ascii=False)

    # Output the results
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"Results saved to {args.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
