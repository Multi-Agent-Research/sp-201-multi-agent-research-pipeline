import os
import sqlite3
import uuid
from pathlib import Path
from pydantic import BaseModel, Field


# Database Configuration & Absolute Path

DB_PATH = Path(__file__).resolve().parent / "evidence_store.db"

def init_db():
    """Creates the SQLite evidence table at a single, fixed path."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evidence (
            evidence_id TEXT PRIMARY KEY,
            query_topic TEXT NOT NULL,
            source_type TEXT NOT NULL,
            title TEXT NOT NULL,
            url TEXT,
            raw_content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()


# Pydantic Schemas

class EvidenceMetadata(BaseModel):
    evidence_id: str = Field(description="Unique identifier stored in SQLite")
    title: str = Field(description="Title of the source article or document")
    url: str = Field(description="Source URL or reference link")
    source_type: str = Field(description="Origin: wikipedia, arxiv, or web")

class SearchResponse(BaseModel):
    sub_question: str
    retrieved_evidence: list[EvidenceMetadata]

# Database Persistence Helper

def store_evidence(query_topic: str, source_type: str, title: str, url: str, content: str) -> str:
    """Inserts raw search text into SQLite using DB_PATH and returns a reference ID."""
    evidence_id = f"EVID-{uuid.uuid4().hex[:8].upper()}"
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO evidence (evidence_id, query_topic, source_type, title, url, raw_content)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (evidence_id, query_topic, source_type, title, url, content))
    
    conn.commit()
    conn.close()
    return evidence_id

# Search Tool Functions
# ---------------------------------------------------------------------------
# Wikipedia answers the wikipedia package's default User-Agent with a 429
# ("too many requests"), so every fetch failed. A descriptive agent fixes it.
WIKIPEDIA_USER_AGENT = (
    "SP201-multi-agent-research/0.1 "
    "(KSU CS 4850 senior project; https://github.com/Multi-Agent-Research/sp-201-multi-agent-research)"
)
ARXIV_TIMEOUT_SECONDS = 30

def fetch_wikipedia_source(sub_question: str) -> EvidenceMetadata:
    """Performs a 2-step Wikipedia search to locate the relevant page and extract text."""
    try:
        import wikipedia
        wikipedia.set_user_agent(WIKIPEDIA_USER_AGENT)
        search_results = wikipedia.search(sub_question)
        if search_results:
            page_title = search_results[0]
            page = wikipedia.page(page_title, auto_suggest=False)
            title = page.title
            url = page.url
            raw_text = page.content[:3000]
        else:
            title = f"No Wikipedia page found for {sub_question}"
            url = ""
            raw_text = "No content retrieved."
    except Exception as e:
        title = f"Wikipedia search: {sub_question}"
        url = f"https://en.wikipedia.org/wiki/Special:Search?search={sub_question.replace(' ', '+')}"
        raw_text = f"[Wikipedia API Error/Fallback: {str(e)}]"

    eid = store_evidence(sub_question, "wikipedia", title, url, raw_text)
    return EvidenceMetadata(evidence_id=eid, title=title, url=url, source_type="wikipedia")

def fetch_arxiv_source(sub_question: str) -> EvidenceMetadata:
    """Queries arXiv API for academic papers related to the sub-question."""
    try:
        import urllib.request
        import xml.etree.ElementTree as ET
        
        query_url = f"http://export.arxiv.org/api/query?search_query=all:{sub_question.replace(' ', '+')}&start=0&max_results=1"
        response = urllib.request.urlopen(query_url, timeout=ARXIV_TIMEOUT_SECONDS)
        xml_data = response.read()
        
        root = ET.fromstring(xml_data)
        ns = {'arxiv': 'http://www.w3.org/2005/Atom'}
        entry = root.find('arxiv:entry', ns)
        
        if entry is not None:
            title = entry.find('arxiv:title', ns).text.strip().replace('\n', ' ')
            url = entry.find('arxiv:id', ns).text.strip()
            raw_text = entry.find('arxiv:summary', ns).text.strip().replace('\n', ' ')
        else:
            title = f"No arXiv paper found for {sub_question}"
            url = ""
            raw_text = "No paper content found."
    except Exception as e:
        title = f"arXiv query: {sub_question}"
        url = "https://arxiv.org"
        raw_text = f"[arXiv API Error/Fallback: {str(e)}]"

    eid = store_evidence(sub_question, "arxiv", title, url, raw_text)
    return EvidenceMetadata(evidence_id=eid, title=title, url=url, source_type="arxiv")

def fetch_web_source(sub_question: str) -> EvidenceMetadata:
    """Performs a web search via DuckDuckGo search wrapper."""
    try:
        from duckduckgo_search import DDGS
        results = list(DDGS().text(sub_question, max_results=1))
        if results:
            title = results[0].get('title', f"Web search for {sub_question}")
            url = results[0].get('href', '')
            raw_text = results[0].get('body', '')
        else:
            title = f"Web result for {sub_question}"
            url = ""
            raw_text = "No web search results."
    except Exception as e:
        title = f"Web Search for {sub_question}"
        url = ""
        raw_text = f"[Web Search Error/Fallback: {str(e)}]"

    eid = store_evidence(sub_question, "web", title, url, raw_text)
    return EvidenceMetadata(evidence_id=eid, title=title, url=url, source_type="web")


# Execution / Test Block

if __name__ == "__main__":
    init_db()
    print(f"Database initialized at: {DB_PATH}")
    
    meta = fetch_wikipedia_source("Agentic AI Architectures")
    print("Retrieved & Persisted Evidence:", meta.model_dump())
