import asyncio
import httpx
from urllib.parse import quote
from xml.etree import ElementTree
from app.agents.base import BaseAgent

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
# Wikimedia asks API clients for a descriptive User-Agent with contact info.
USER_AGENT = "Meridian/1.0 (https://github.com/Nityaa08/meridian)"


class ResearcherAgent(BaseAgent):
    def __init__(self):
        super().__init__("ResearcherAgent")

    async def run(self, sub_query: str, queue: asyncio.Queue) -> list[dict]:
        await self.emit(queue, "SEARCHING", f"Searching: '{sub_query}'")
        passages = []

        wiki_results = await self._search_wikipedia(sub_query)
        passages.extend(wiki_results)

        arxiv_results = await self._search_arxiv(sub_query)
        passages.extend(arxiv_results)

        if len(passages) < 3:
            ddg_results = await self._search_duckduckgo(sub_query)
            passages.extend(ddg_results)

        await self.emit(
            queue, "SEARCHING",
            f"Found {len(passages)} passages for: '{sub_query}'",
            {"query": sub_query, "count": len(passages)},
        )
        return passages

    async def _search_wikipedia(self, query: str) -> list[dict]:
        """
        Full-text search plus intro extracts in one MediaWiki Action API call.
        (The old REST path /api/rest_v1/page/search/title returned HTTP 404.)
        """
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    WIKIPEDIA_API,
                    params={
                        "action": "query",
                        "format": "json",
                        "formatversion": "2",
                        "generator": "search",
                        "gsrsearch": query,
                        "gsrlimit": 2,
                        "prop": "extracts|info",
                        "exintro": 1,
                        "explaintext": 1,
                        "inprop": "url",
                    },
                    headers={"User-Agent": USER_AGENT},
                )
                resp.raise_for_status()
                return parse_wikipedia_response(resp.json())
        except Exception:
            return []

    async def _search_arxiv(self, query: str) -> list[dict]:
        try:
            # Scope to CS + stats categories to avoid unrelated science papers
            scoped = f"(ti:{query} OR abs:{query}) AND (cat:cs.* OR cat:stat.ML OR cat:eess.*)"
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    "https://export.arxiv.org/api/query",
                    params={"search_query": scoped, "max_results": 3, "sortBy": "relevance"},
                )
                resp.raise_for_status()
                root = ElementTree.fromstring(resp.text)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                entries = root.findall("atom:entry", ns)

                passages = []
                for entry in entries[:2]:
                    title_el = entry.find("atom:title", ns)
                    summary_el = entry.find("atom:summary", ns)
                    id_el = entry.find("atom:id", ns)
                    if title_el is None or summary_el is None:
                        continue
                    title = title_el.text.strip().replace("\n", " ")
                    summary = summary_el.text.strip().replace("\n", " ")
                    url = id_el.text.strip() if id_el is not None else ""
                    if summary:
                        passages.append({
                            "title": title,
                            "url": url,
                            "text": summary,
                            "source": "arxiv",
                        })
                return passages
        except Exception:
            return []

    async def _search_duckduckgo(self, query: str) -> list[dict]:
        try:
            return await asyncio.to_thread(self._ddg_sync, query)
        except Exception:
            return []

    def _ddg_sync(self, query: str) -> list[dict]:
        # "ddgs" is the maintained successor of the duckduckgo-search package,
        # whose 6.x release failed with "Invalid impersonate".
        try:
            from ddgs import DDGS
            results = DDGS().text(query, max_results=3) or []
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "text": r.get("body", ""),
                    "source": "duckduckgo",
                }
                for r in results
                if r.get("body")
            ]
        except Exception:
            return []


def parse_wikipedia_response(data: dict) -> list[dict]:
    """Turn an Action API generator=search response into passages, in search-rank order."""
    pages = data.get("query", {}).get("pages", [])
    pages = sorted(pages, key=lambda p: p.get("index", 0))
    passages = []
    for page in pages:
        extract = (page.get("extract") or "").strip()
        if not extract:
            continue
        title = page.get("title", "")
        url = page.get("fullurl") or "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"))
        passages.append({"title": title, "url": url, "text": extract, "source": "wikipedia"})
    return passages
