from typing import Optional

import requests
from bs4 import BeautifulSoup


def extract_text_from_url(url: str, timeout: float = 8.0) -> Optional[str]:
    try:
        resp = requests.get(
            url, timeout=timeout, headers={"User-Agent": "FakeScope/1.0"}
        )
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove scripts/styles
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        # Prefer article tags
        article = soup.find("article")
        if article:
            text = "\n".join(
                p.get_text(separator=" ", strip=True) for p in article.find_all("p")
            )
            if len(text.split()) > 50:
                return text
        # Fallback: concatenate paragraph texts
        paragraphs = soup.find_all("p")
        text = "\n".join(p.get_text(separator=" ", strip=True) for p in paragraphs)
        return text if len(text.split()) > 30 else None
    except Exception:
        return None
