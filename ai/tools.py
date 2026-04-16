from duckduckgo_search import DDGS
from database.sqlite_db import get_db_session, NewsCache

def search_internet(query: str) -> str:
    """Searches the internet for general information or memes."""
    try:
        results = DDGS().text(query, max_results=3)
        if not results:
            return "No results found."
        
        snippets = []
        for r in results:
            snippets.append(f"Title: {r['title']}\nSnippet: {r['body']}\nLink: {r['href']}")
            
        return "\n\n".join(snippets)
    except Exception as e:
        return f"Error searching internet: {e}"

def search_youtube(query: str) -> str:
    """Searches YouTube for videos."""
    try:
        results = DDGS().videos(query, max_results=3)
        if not results:
            return "No videos found."
        
        snippets = []
        for r in results:
            snippets.append(f"Video Title: {r.get('title')}\nDescription: {r.get('description')}\nURL: {r.get('content')}")
            
        return "\n\n".join(snippets)
    except Exception as e:
        return f"Error searching youtube: {e}"

def search_saved_news(query: str) -> str:
    """Searches the local database of recently saved posts from Telegram channels."""
    db = get_db_session()
    try:
        # Simple LIKE search. For better search, we could use FTS or Qdrant for news too.
        # But for now, basic LIKE is sufficient or just return the latest N news if query is empty.
        if not query or query.lower() in ["latest", "news", "новости"]:
            news = db.query(NewsCache).order_by(NewsCache.date_added.desc()).limit(5).all()
        else:
            search_pattern = f"%{query}%"
            news = db.query(NewsCache).filter(NewsCache.text.ilike(search_pattern)).order_by(NewsCache.date_added.desc()).limit(5).all()
        
        if not news:
            return "No saved news found."
            
        snippets = []
        for n in news:
            snippets.append(f"[{n.date_added.strftime('%Y-%m-%d %H:%M')}] News from channel {n.channel_id}:\n{n.text}")
            
        return "\n\n".join(snippets)
    except Exception as e:
        return f"Error searching news: {e}"
    finally:
        db.close()
