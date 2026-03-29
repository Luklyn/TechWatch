import os
import re
import time
import feedparser
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from bs4 import BeautifulSoup
from pathlib import Path

app = FastAPI()

# ─── HUGGING FACE CONFIG ─────────────────────────────────────────────────────
# Le token est lu depuis le fichier .env (jamais dans le code)
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN manquant — créez un fichier .env avec HF_TOKEN=hf_xxx")
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HF_URL   = "https://router.huggingface.co/v1/chat/completions"

# ─── CATÉGORIES ───────────────────────────────────────────────────────────────
# Personnalisez librement cette liste — les noms sont utilisés tels quels dans l'UI
CATEGORIES = ["Tech", "Gaming", "Smartphone"]
CATEGORY_OTHER = "Autre"  # fallback si aucune catégorie ne correspond

# ─── SOURCES ─────────────────────────────────────────────────────────────────

RSS_FEEDS = {
    "TechPowerUp":    "https://www.techpowerup.com/rss/news",
    "Hardware & Co":  "https://hardwareand.co/actualites?format=feed&type=rss",
    "The Verge":      "https://www.theverge.com/tech/rss/index.xml",
    "Ars Technica":   "https://feeds.arstechnica.com/arstechnica/index",
    "9to5Mac":        "https://9to5mac.com/feed/",
    "9to5Google":     "https://9to5google.com/feed/",
    "FrAndroid":      "https://www.frandroid.com/feed",
    "GamesIndustry":  "https://www.gamesindustry.biz/feed",
    "Digital Foundry": "https://www.eurogamer.net/feed/digitalfoundry",
    "Les Numériques": "https://www.lesnumeriques.com/rss.xml",
}


YOUTUBE_CHANNELS = {
    "Gamers Nexus":     "https://www.youtube.com/feeds/videos.xml?channel_id=UChIs72whgZI9w6d6FhwGGHA",
    "VCG":              "https://www.youtube.com/feeds/videos.xml?channel_id=UCjrj3gdo-KL2S_JN_gdNyPw",
    "Hardware Canucks": "https://www.youtube.com/feeds/videos.xml?channel_id=UCTzLRZUgelatKZ4nyIKcAbg",
    "Hardware Unboxed": "https://www.youtube.com/feeds/videos.xml?channel_id=UCI8iQa1hv7oV_Z8D35vVuSg",
    "Matt Lee":         "https://www.youtube.com/feeds/videos.xml?channel_id=UCGHzpEcSwfBQJAitgw2pgVQ",
    "Digital Foundry":  "https://www/youtube.com/feeds/videos.xml?channel_id=UC9PBzalIcEQCsiIkq36PyUA",
    "Marques Brownlee": "https://www.youtube.com/feeds/videos.xml?channel_id=UCBJycsmduvYEL83R_U4JriQ",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# ─── CACHE ───────────────────────────────────────────────────────────────────

_cache: dict = {}
CACHE_TTL = 600  # 10 minutes


def get_cached(key):
    entry = _cache.get(key)
    if entry and (time.time() - entry["ts"]) < CACHE_TTL:
        return entry["data"], entry["ts"]
    return None, None


def set_cached(key, data):
    _cache[key] = {"data": data, "ts": time.time()}


# ─── FETCH ───────────────────────────────────────────────────────────────────

def fetch_one(name, url, is_youtube):
    try:
        r = requests.get(url, headers=HEADERS, timeout=6)
        r.raise_for_status()
        feed = feedparser.parse(r.content)
        items = []
        for entry in feed.entries:
            try:
                dt = datetime(*entry.published_parsed[:6])
                if is_youtube:
                    v_id = (
                        entry.link.split("v=")[1].split("&")[0]
                        if "v=" in entry.link
                        else entry.id.split(":")[-1]
                    )
                    img = f"https://img.youtube.com/vi/{v_id}/hqdefault.jpg"
                    summary = ""
                else:
                    img = ""
                    if hasattr(entry, "media_content") and entry.media_content:
                        img = entry.media_content[0].get("url", "")
                    elif hasattr(entry, "enclosures") and entry.enclosures:
                        img = entry.enclosures[0].get("href", "")
                    raw = re.sub(r"<[^<]+?>", "", entry.get("summary", ""))
                    summary = (raw[:130] + "…") if len(raw) > 130 else raw

                items.append({
                    "source": name,
                    "title": entry.title,
                    "link": entry.link,
                    "date": dt.isoformat(),
                    "date_display": dt.strftime("%d %b %Y · %H:%M"),
                    "image": img,
                    "summary": summary,
                    "is_video": is_youtube,
                    "category": CATEGORY_OTHER,  # sera mis à jour par categorize_batch
                })
            except Exception:
                continue
        return name, items, None
    except requests.Timeout:
        return name, [], f"{name} : timeout (6s)"
    except Exception as e:
        return name, [], f"{name} : {str(e)[:80]}"


# ─── CATÉGORISATION ───────────────────────────────────────────────────────────

# 4. Pré-filtre par mots-clés (évite des appels Llama inutiles sur les cas évidents)
KEYWORD_RULES = {
    "Tech":       ["GPU", "CPU", "RTX", "RX", "RAM", "DDR", "benchmark", "carte mère",
                   "processeur", "overclocking", "watercooling", "SSD", "NVMe", "PCIe",
                   "socket", "chipset", "BIOS", "driver", "Windows", "Linux", "MacOS"],
    "Gaming":     ["jeu", "game", "console", "PS5", "Xbox", "Nintendo", "Switch",
                   "esport", "DLC", "patch", "update", "FPS", "RPG", "trailer", "Steam"],
    "Smartphone": ["iPhone", "Android", "Samsung", "Pixel", "smartphone", "tablette",
                   "iOS", "application", "app", "mobile", "5G", "OnePlus", "Xiaomi"],
}

def quick_categorize(title: str, summary: str) -> str | None:
    """Retourne une catégorie si les mots-clés sont évidents, sinon None (→ Llama)."""
    text = (title + " " + summary).lower()
    for cat, keywords in KEYWORD_RULES.items():
        if any(kw.lower() in text for kw in keywords):
            return cat
    return None


def categorize_batch(items: list) -> dict:
    """Categorise une liste d'articles. Pré-filtre par mots-clés, puis Llama pour le reste."""
    if not items:
        return {}

    result = {}
    to_llama = []

    # 4. Pré-filtre mots-clés
    for item in items:
        cat = quick_categorize(item["title"], item.get("summary", ""))
        if cat:
            result[item["title"]] = cat
        else:
            to_llama.append(item)

    if not to_llama:
        return result

    categories_str = ", ".join(CATEGORIES)

    # 2. Inclure les 60 premiers caractères du résumé pour plus de contexte
    items_text = "\n".join(
        f"{i+1}. {item['title']} — {item.get('summary', '')[:60]}"
        for i, item in enumerate(to_llama)
    )

    # 1. Prompt amélioré avec critères stricts + autorisation "je ne sais pas"
    # 3. Demande d'un score de confiance pour basculer en Autre si faible
    prompt = (
        f"Classify each article title into exactly one category.\n"
        f"Categories: {categories_str}, {CATEGORY_OTHER}\n"
        f"Tech=PC hardware(GPU/CPU/RAM/SSD/motherboard/cooling/PSU/case/monitor/driver/OS)\n"
        f"Gaming=video games/consoles/esport/game releases/gaming peripherals\n"
        f"Smartphone=phones/tablets/iOS/Android/mobile apps\n"
        f"{CATEGORY_OTHER}=business/finance/politics/general news/unclear\n"
        f"Reply ONLY: number.Category, one per line, no explanations. Use {CATEGORY_OTHER} if unsure.\n\n"
        f"{items_text}"
    )

    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "model": HF_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": len(to_llama) * 12,
        "temperature": 0.0,  # 5. Température 0.0 pour catégorisation pure
    }
    try:
        r = requests.post(HF_URL, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        valid = CATEGORIES + [CATEGORY_OTHER]
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(".", 1)
            if len(parts) == 2:
                try:
                    idx = int(parts[0].strip()) - 1
                    cat_raw = parts[1].strip().split("(")[0].strip()
                    matched = next(
                        (c for c in valid if c.lower() == cat_raw.lower()),
                        next((c for c in valid if c.lower() in cat_raw.lower()), CATEGORY_OTHER)
                    )
                    if 0 <= idx < len(to_llama):
                        result[to_llama[idx]["title"]] = matched
                except (ValueError, IndexError):
                    continue
    except Exception:
        pass

    return result


def fetch_all(source_dict, is_youtube):
    all_items, errors = [], []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(fetch_one, n, u, is_youtube): n for n, u in source_dict.items()}
        for fut in as_completed(futures):
            name, items, err = fut.result()
            all_items.extend(items)
            if err:
                errors.append(err)
    all_items.sort(key=lambda x: x["date"], reverse=True)

    # Catégoriser par batch de 20 pour limiter les tokens
    if not is_youtube:
        BATCH = 20
        for i in range(0, len(all_items), BATCH):
            batch = all_items[i:i+BATCH]
            cats = categorize_batch(batch)
            for item in batch:
                if item["title"] in cats:
                    item["category"] = cats[item["title"]]

    return all_items, errors


# ─── ROUTES ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html = Path("templates/index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.get("/api/feed")
async def api_feed(
    type: str = Query("articles"),
    source: str = Query(""),
    search: str = Query(""),
    period: str = Query(""),
    request_category: str = Query("", alias="category"),
    refresh: bool = Query(False),
):
    is_youtube = type == "videos"
    cache_key = "youtube" if is_youtube else "articles"

    if refresh:
        _cache.pop(cache_key, None)

    items, fetch_ts = get_cached(cache_key)
    errors = []

    if items is None:
        source_dict = YOUTUBE_CHANNELS if is_youtube else RSS_FEEDS
        items, errors = fetch_all(source_dict, is_youtube)
        set_cached(cache_key, items)
        fetch_ts = time.time()

    # Filters
    filtered = items
    if search:
        q = search.lower()
        filtered = [i for i in filtered if q in i["title"].lower()]
    if source:
        filtered = [i for i in filtered if i["source"] == source]
    category_param = request_category
    if category_param:
        cats = [c.strip() for c in category_param.split(",") if c.strip()]
        if cats:
            filtered = [i for i in filtered if i.get("category", CATEGORY_OTHER) in cats]
    if period:
        now = datetime.utcnow()
        def in_period(item):
            d = datetime.fromisoformat(item["date"])
            if period == "today":
                return d.date() == now.date()
            elif period == "week":
                return (now - d).days <= 7
            elif period == "month":
                return d.month == now.month and d.year == now.year
            return True
        filtered = [i for i in filtered if in_period(i)]

    sources = list(YOUTUBE_CHANNELS.keys() if is_youtube else RSS_FEEDS.keys())

    return JSONResponse({
        "items": filtered,
        "total": len(filtered),
        "errors": errors,
        "fetch_ts": fetch_ts,
        "sources": sources,
        "categories": CATEGORIES,
    })

# ─── SUMMARIZE ────────────────────────────────────────────────────────────────

def scrape_article(url: str) -> str:
    """Scrape le texte principal d'un article."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=8)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "aside",
                          "header", "form", "iframe", "noscript"]):
            tag.decompose()
        main = (
            soup.find("article")
            or soup.find("main")
            or soup.find(class_=re.compile(r"(content|article|post|entry)", re.I))
            or soup.body
        )
        text = main.get_text(separator="\n", strip=True) if main else ""
        return text[:4000]
    except Exception:
        return ""


def call_hf(text: str, title: str, is_video: bool = False, is_digest: bool = False) -> str:
    """Appelle Llama via HuggingFace Inference API et retourne un résumé en français."""
    if is_digest:
        messages = [
            {
                "role": "system",
                "content": (
                    "Tu es un assistant spécialisé en technologie. "
                    "À partir d'une liste de titres d'actualités tech, rédige un digest journalier "
                    "en français en 8 à 12 phrases fluides. Regroupe les sujets similaires, "
                    "identifie les tendances du jour, et conclus par les points à retenir. "
                    "Commence directement par les faits, sans introduction générique."
                ),
            },
            {
                "role": "user",
                "content": text[:3500],
            },
        ]
    else:
        content_type = "vidéo YouTube" if is_video else "article"
        messages = [
            {
                "role": "system",
                "content": (
                    f"Tu es un assistant spécialisé en technologie. "
                    f"Tu résumes des {content_type}s tech en français, en prose fluide de 5 à 8 phrases. "
                    f"Tu vas droit au but sans commencer par 'Cet article' ou 'Cette vidéo'."
                ),
            },
            {
                "role": "user",
                "content": f"Résume cette {content_type} intitulée « {title} » :\n\n{text[:3500]}",
            },
        ]
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": HF_MODEL,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.3,
    }
    try:
        r = requests.post(HF_URL, headers=headers, json=payload, timeout=60)
        if r.status_code == 503:
            # Modèle en cours de chargement (cold start), attendre et réessayer
            time.sleep(20)
            r = requests.post(HF_URL, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Impossible de générer le résumé : {str(e)[:150]}"


_summary_cache: dict = {}

@app.post("/api/summarize")
async def api_summarize(body: dict):
    url   = body.get("url", "")
    title = body.get("title", "")
    if not url:
        return JSONResponse({"error": "URL manquante"}, status_code=400)

    # Retourner le cache si deja genere
    if url in _summary_cache:
        return JSONResponse({"summary": _summary_cache[url], "cached": True})

    is_video = body.get("is_video", False)

    # Cas spécial : Digest du jour (liste de titres fournie directement)
    digest_titles = body.get("digest_titles", "")
    if digest_titles:
        today = datetime.utcnow().strftime("%d/%m/%Y")
        digest_text = f"Voici les titres des actualités tech du {today} :\n{digest_titles}"
        summary = call_hf(
            digest_text,
            "Digest tech du jour",
            is_video=False,
            is_digest=True
        )
    else:
        text    = scrape_article(url)
        summary = call_hf(text or f"Contenu non lisible pour : {title}", title, is_video=is_video)

    # Mettre en cache uniquement si succes
    if not summary.startswith("Impossible"):
        _summary_cache[url] = summary

    return JSONResponse({"summary": summary})


# ─── IMAGE PROXY ─────────────────────────────────────────────────────────────

from fastapi import Response
from urllib.parse import urlparse

@app.get("/api/img")
async def img_proxy(url: str = Query(...)):
    """
    Proxy d'images : récupère l'image côté serveur avec le bon Referer
    pour contourner les restrictions hotlinking des sites.
    """
    try:
        parsed   = urlparse(url)
        referrer = f"{parsed.scheme}://{parsed.netloc}/"
        headers  = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Referer":    referrer,
            "Accept":     "image/webp,image/apng,image/*,*/*;q=0.8",
        }
        r = requests.get(url, headers=headers, timeout=8, stream=True)
        r.raise_for_status()
        content_type = r.headers.get("Content-Type", "image/jpeg")
        return Response(content=r.content, media_type=content_type, headers={
            "Cache-Control": "public, max-age=86400",  # cache 24h côté navigateur
        })
    except Exception:
        # Retourner une image placeholder SVG si l'image est introuvable
        svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="800" height="160" viewBox="0 0 800 160">
  <rect width="800" height="160" fill="#1c1c1c"/>
  <text x="400" y="88" font-family="sans-serif" font-size="32" fill="#333" text-anchor="middle">📰</text>
</svg>'''
        return Response(content=svg, media_type="image/svg+xml", headers={
            "Cache-Control": "public, max-age=3600",
        })
