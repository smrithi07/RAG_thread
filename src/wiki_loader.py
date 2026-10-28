import os
import re
import sys
import json
import time
from io import StringIO

import requests
import pandas as pd
from bs4 import BeautifulSoup

import config

API_URL = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "RAG_thread/1.0 (https://github.com/smrithi07/RAG_thread)"}
SKIP_SECTIONS = ["See also", "References", "Notes", "External links", "Further reading",
                 "Bibliography", "Citations", "Sources", "Footnotes"]
REMOVE = "sup.reference, span.mw-editsection, style, script, .navbox, .reflist, .mw-empty-elt, .noprint, .hatnote"


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def file_name(title):
    return re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_") + ".json"


def fetch_html(title):
    params = {"action": "parse", "page": title, "prop": "text", "format": "json",
              "formatversion": "2", "redirects": "1"}
    for attempt in range(5):
        r = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
        if r.status_code == 429:
            wait = r.headers.get("Retry-After", "")
            if wait.isdigit():
                wait = int(wait)
            else:
                wait = 10 * (attempt + 1)
            print("  rate limited, waiting", wait, "s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            print("  error:", data["error"].get("info"))
            return None, None
        return data["parse"]["title"], data["parse"]["text"]
    return None, None


def get_heading(el):
    if el.name in ["h2", "h3", "h4"]:
        return clean(el.get_text()), el.name
    if el.name == "div" and "mw-heading" in el.get("class", []):
        h = el.find(["h2", "h3", "h4"])
        if h:
            return clean(h.get_text()), h.name
    return None, None


def parse_infobox(table):
    rows = []
    group = ""
    for tr in table.find_all("tr"):
        th = tr.find("th")
        td = tr.find("td")
        if th and not td:
            group = clean(th.get_text(" "))
            continue
        if th and td:
            label = clean(th.get_text(" "))
            if label.startswith("•"):
                label = group + " " + label.lstrip("• ").strip()
            rows.append([label, clean(td.get_text(" ")).replace("km 2", "km2")])
    return rows


def parse_table(table):
    classes = table.get("class", [])
    caption = ""
    cap = table.find("caption")
    if cap:
        caption = clean(cap.get_text(" "))

    if "infobox" in classes:
        return {"type": "infobox", "caption": caption, "columns": ["Field", "Value"],
                "rows": parse_infobox(table)}

    if "wikitable" not in classes:
        return None
    try:
        df = pd.read_html(StringIO(str(table)))[0]
    except ValueError:
        return None

    columns = []
    for col in df.columns:
        if isinstance(col, tuple):
            parts = []
            for p in col:
                p = str(p)
                if not p.startswith("Unnamed") and p not in parts:
                    parts.append(p)
            columns.append(" ".join(parts))
        else:
            columns.append(str(col))

    rows = []
    for row in df.astype(str).values.tolist():
        new_row = []
        for v in row:
            v = clean(str(v))
            if v == "nan":
                v = ""
            new_row.append(v)
        rows.append(new_row)
    return {"type": "wikitable", "caption": caption, "columns": columns, "rows": rows}


def new_section(name):
    return {"heading": name, "text": [], "tables": [], "captions": []}


def has_data(sec):
    return len(sec["text"]) > 0 or len(sec["tables"]) > 0 or len(sec["captions"]) > 0


def parse_page(html):
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.select(REMOVE):
        tag.decompose()
    root = soup.find("div", class_="mw-parser-output")
    if root is None:
        root = soup.body

    sections = []
    current = new_section("Introduction")
    top = "Introduction"
    skip = False

    for el in root.find_all(recursive=False):
        heading, level = get_heading(el)
        if heading:
            if has_data(current):
                sections.append(current)
            if level == "h2":
                top = heading
                skip = heading in SKIP_SECTIONS
                current = new_section(heading)
            else:
                current = new_section(top + " > " + heading)
            continue
        if skip:
            continue

        if el.name == "p":
            text = clean(el.get_text())
            if text:
                current["text"].append(text)
        elif el.name in ["ul", "ol"]:
            for li in el.find_all("li"):
                text = clean(li.get_text(" "))
                if text:
                    current["text"].append(text)
        elif el.name == "table":
            t = parse_table(el)
            if t and len(t["rows"]) > 0:
                current["tables"].append(t)
        elif el.name in ["div", "figure"]:
            for table in el.find_all("table"):
                t = parse_table(table)
                if t and len(t["rows"]) > 0:
                    current["tables"].append(t)
            for cap in el.select("figcaption, .thumbcaption, .gallerytext"):
                text = clean(cap.get_text(" "))
                if text:
                    current["captions"].append(text)

    if has_data(current):
        sections.append(current)

    for sec in sections:
        sec["text"] = "\n".join(sec["text"])
    return sections


def load_titles():
    titles = []
    with open(config.PAGES_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                titles.append(line)

    district_titles = []
    for t in titles:
        if t.startswith("States and union territories"):
            continue
        name = re.sub(r"\s*\(.*?\)", "", t)
        district_titles.append("List of districts of " + name)
    return titles + district_titles


def main():
    refresh = "--refresh" in sys.argv
    os.makedirs(config.RAW_DIR, exist_ok=True)
    titles = load_titles()

    total_sec = 0
    total_tab = 0
    total_cap = 0
    failed = []

    for title in titles:
        path = os.path.join(config.RAW_DIR, file_name(title))
        if os.path.exists(path) and not refresh:
            with open(path, encoding="utf-8") as f:
                page = json.load(f)
            status = "cached"
        else:
            try:
                real_title, html = fetch_html(title)
            except requests.RequestException as e:
                print("  failed:", title, e)
                failed.append(title)
                continue
            if html is None:
                failed.append(title)
                continue
            page = {
                "title": real_title,
                "url": "https://en.wikipedia.org/wiki/" + real_title.replace(" ", "_"),
                "sections": parse_page(html),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(page, f, ensure_ascii=False, indent=1)
            status = "fetched"
            time.sleep(1.5)

        n_sec = len(page["sections"])
        n_tab = 0
        n_cap = 0
        for sec in page["sections"]:
            n_tab += len(sec["tables"])
            n_cap += len(sec["captions"])
        total_sec += n_sec
        total_tab += n_tab
        total_cap += n_cap
        print(f"{status:8s} {page['title']:50s} sections={n_sec:3d} tables={n_tab:3d} captions={n_cap:3d}")

    print(f"\nPages: {len(titles) - len(failed)}/{len(titles)}  sections={total_sec}  tables={total_tab}  captions={total_cap}")
    if failed:
        print("Failed:", failed)


if __name__ == "__main__":
    main()