"""Extractor for DMPs hosted on dmponline.org (or saved HTML).

This extractor accepts a URL (http/https) or a local HTML file path passed
as the *pdf_path* argument. It fetches/parses the page, pulls headings and
paragraphs, and returns the same flat block schema used by other extractors.
"""
from pathlib import Path
from typing import List
import os

from .base import BaseExtractor

import requests
from bs4 import BeautifulSoup


class DmponlineExtractor(BaseExtractor):
    """Extract text blocks from a DMP shown on dmponline.org.

    The method is intentionally conservative: it extracts headings (h1..h6),
    paragraphs, and list items. Visual layout features (bbox, font size)
    are not available for HTML; those fields are set to None/empty.
    """

    def extract(self, pdf_path: Path) -> List[dict]:
        url_or_path = str(pdf_path)

        # Fetch remote HTML or read local file
        if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
            # Allow authenticated access via DMPONLINE_COOKIE env var (.env OK).
            cookie = os.getenv("DMPONLINE_COOKIE")
            headers = {"Cookie": cookie} if cookie else None
            resp = requests.get(url_or_path, headers=headers, timeout=30)
            resp.raise_for_status()
            html = resp.text
        else:
            html = Path(url_or_path).read_text(encoding="utf-8")

        soup = BeautifulSoup(html, "html.parser")

        # Prefer semantic containers when present
        container = soup.find("main") or soup.find("article") or soup.body
        if container is None:
            container = soup

        blocks: List[dict] = []
        order = 0
        page = 1

        for elem in container.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "blockquote"]):
            text = elem.get_text(separator=" ", strip=True)
            if not text:
                continue
            is_bold = elem.name.startswith("h")
            is_italic = bool(elem.find(["em", "i"]))

            blocks.append({
                "text": text,
                "page": page,
                "is_bold": is_bold,
                "is_italic": is_italic,
                "line_order": order,
                # HTML extraction does not provide layout/visual metrics
                "x0": None,
                "top": None,
                "x1": None,
                "bottom": None,
                "avg_font_size": None,
                "font_names": [],
            })
            order += 1

        return blocks
