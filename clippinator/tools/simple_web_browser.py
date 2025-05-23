import requests
from bs4 import BeautifulSoup
from yarl import URL
import re
from collections import deque
from typing import Tuple, List, Set, Dict, Any

class SimpleWebBrowserTool:
    """
    A simple web browser tool that fetches web page content using HTTP requests
    and parses it using BeautifulSoup to extract text and links.
    It can optionally crawl to a specified depth.
    """

    def __init__(self, user_agent: str = "ClippinatorBrowser/1.0"):
        """
        Initializes the web browser tool.
        Args:
            user_agent: The User-Agent string to use for HTTP requests.
        """
        self.user_agent = user_agent
        print(f"[INFO] SimpleWebBrowserTool initialized with User-Agent: {self.user_agent}")

    def _fetch_and_parse(self, current_url: str) -> Tuple[str, List[str]]:
        """
        Fetches a single URL, parses its HTML content, and extracts text and links.

        Args:
            current_url: The URL to fetch and parse.

        Returns:
            A tuple containing:
                - The extracted visible text from the page.
                - A list of unique, absolute HTTP/HTTPS URLs found on the page.
        """
        print(f"[INFO] Fetching URL: {current_url}")
        try:
            response = requests.get(current_url, headers={'User-Agent': self.user_agent}, timeout=10)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Failed to fetch {current_url}: {e}")
            return "", []

        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract visible text
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()  # Remove script and style elements
        text = soup.get_text(separator='\n', strip=True)
        
        # Extract links
        links = set()
        base_url_yarl = URL(current_url)
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if href:
                try:
                    absolute_link = base_url_yarl.join(URL(href))
                    if absolute_link.scheme in ['http', 'https']:
                        links.add(str(absolute_link).split('#')[0]) # Remove fragment
                except Exception as e:
                    print(f"[DEBUG] Could not process link '{href}' on page {current_url}: {e}")
        
        print(f"[INFO] Fetched {current_url}. Text length: {len(text)}, Links found: {len(links)}")
        return text, sorted(list(links))

    def run(self, url: str, depth: int = 0, max_pages: int = 5, max_total_text_len: int = 10000) -> Dict[str, Any]:
        """
        Fetches web page(s) starting from the given URL, up to a specified depth.

        Args:
            url: The starting URL to browse.
            depth: The depth of crawling. 0 means only the initial URL.
                   1 means the initial URL and pages linked directly from it, etc.
            max_pages: The maximum number of pages to visit during crawling.
            max_total_text_len: The maximum total length of accumulated text.

        Returns:
            A dictionary containing:
                - "text": The combined text from all visited pages (truncated if over limit).
                - "links": A list of all unique absolute HTTP/HTTPS links found across all visited pages.
                - "visited_urls": A list of URLs that were successfully fetched and parsed.
        """
        if not url.startswith(('http://', 'https://')):
            print(f"[ERROR] Invalid URL scheme: {url}. Must be HTTP or HTTPS.")
            return {"text": "Error: Invalid URL scheme. Must be HTTP or HTTPS.", "links": [], "visited_urls": []}

        queue = deque([(url, 0)])  # (url, current_depth)
        visited_urls: Set[str] = set()
        all_texts: List[str] = []
        all_unique_links: Set[str] = set()
        pages_visited_count = 0
        current_total_text_len = 0

        results: Dict[str, Any] = {"text": "", "links": [], "visited_urls": []}

        while queue and pages_visited_count < max_pages:
            current_url, current_d = queue.popleft()

            if current_url in visited_urls:
                continue
            
            if current_d > depth:
                continue

            visited_urls.add(current_url)
            pages_visited_count += 1
            
            page_text, page_links = self._fetch_and_parse(current_url)

            if page_text: # Add text only if page was successfully fetched
                if current_total_text_len + len(page_text) > max_total_text_len:
                    remaining_len = max_total_text_len - current_total_text_len
                    all_texts.append(page_text[:remaining_len])
                    current_total_text_len += remaining_len
                    print(f"[INFO] Reached max total text length. Truncating text from {current_url}.")
                    # No more text processing after this
                else:
                    all_texts.append(page_text)
                    current_total_text_len += len(page_text)
            
            results["visited_urls"].append(current_url)
            all_unique_links.update(page_links)

            if current_d < depth and pages_visited_count < max_pages and current_total_text_len < max_total_text_len :
                for link in page_links:
                    if link not in visited_urls:
                        queue.append((link, current_d + 1))
            
            if current_total_text_len >= max_total_text_len:
                break # Stop if max text length reached

        results["text"] = "\n\n--- Page Break ---\n\n".join(all_texts)
        results["links"] = sorted(list(all_unique_links))
        
        print(f"[INFO] Browsing complete. Visited {pages_visited_count} page(s). Total text length: {current_total_text_len}. Total unique links: {len(all_unique_links)}.")
        return results

if __name__ == '__main__':
    # Example Usage
    browser = SimpleWebBrowserTool()

    print("\\n--- Example 1: Single page ---")
    result1 = browser.run("http://example.com", depth=0)
    print(f"Text from example.com:\\n{result1['text'][:500]}...")
    print(f"Links from example.com: {result1['links']}")
    print(f"Visited URLs: {result1['visited_urls']}")

    # Note: For depth > 0, be mindful of the sites you are crawling.
    # This example might try to fetch external sites like iana.org.
    # print("\\n--- Example 2: Depth 1 (use a controlled site for testing) ---")
    # Create a dummy html file for testing depth=1 locally if needed.
    # For instance, create test.html:
    # <html><body><p>Test page 1</p><a href="test2.html">Link to Test Page 2</a></body></html>
    # And test2.html:
    # <html><body><p>Test page 2</p><a href="http://example.com">Example</a></body></html>
    # Then run a local HTTP server: python -m http.server
    # result2 = browser.run("http://localhost:8000/test.html", depth=1)
    # print(f"Combined Text (depth 1):\\n{result2['text']}")
    # print(f"All Links (depth 1): {result2['links']}")
    # print(f"Visited URLs: {result2['visited_urls']}")

    print("\\n--- Example 3: Fetching a non-existent page ---")
    result3 = browser.run("http://domainthatdoesnotexistforsure123.com")
    print(f"Result for non-existent page: {result3}")
    
    print("\\n--- Example 4: Fetching an invalid URL scheme ---")
    result4 = browser.run("ftp://example.com")
    print(f"Result for invalid scheme: {result4}")

```
