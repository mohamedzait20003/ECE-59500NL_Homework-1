import os
import re
import csv
import time
import requests
from tqdm import tqdm
from io import BytesIO
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup

# Constants for file paths

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', "data", "raw")

TRUMP_DIR = os.path.join(RAW_DIR, "trump")
BIDEN_DIR = os.path.join(RAW_DIR, "biden")

os.makedirs(BIDEN_DIR, exist_ok=True)
os.makedirs(TRUMP_DIR, exist_ok=True)


# Headers for HTTP requests to mimic a browser

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Trump PDFs to collect

TRUMP_PDFS = {
    "art_of_the_deal.txt": (
        "https://ia601405.us.archive.org/19/items/TrumpTheArtOfTheDeal/"
        "Trump_%20The%20Art%20of%20the%20Deal.pdf"
    ),
    "think_like_a_champion.txt": (
        "https://www.reboxu.com/uploads/8/6/0/3/86031326/"
        "think_like_a_champion.pdf"
    ),
}

# Rev.com search URLs for individual speeches (secondary source)

TRUMP_REV_SEARCH = "https://www.rev.com/blog/transcripts?s=trump+speech"
BIDEN_REV_SEARCH = "https://www.rev.com/blog/transcripts?s=biden+speech"

# UCSB American Presidency Project reliable debate transcripts

UCSB_DEBATE_URLS = {
    "debate_2020_1.txt": (
        "https://www.presidency.ucsb.edu/documents/"
        "presidential-debate-case-western-reserve-university-cleveland-ohio"
    ),
    "debate_2020_2.txt": (
        "https://www.presidency.ucsb.edu/documents/"
        "presidential-debate-belmont-university-nashville-tennessee-0"
    ),
}

# Rev.com debate URLs (fallback)

REV_DEBATE_URLS = {
    "debate_2020_1.txt": (
        "https://www.rev.com/blog/transcripts/"
        "donald-trump-joe-biden-1st-presidential-debate-transcript-2020"
    ),
    "debate_2020_2.txt": (
        "https://www.rev.com/blog/transcripts/"
        "donald-trump-joe-biden-final-presidential-debate-transcript-2020"
    ),
}

# Biden speeches from UCSB (reliable, clean text)
BIDEN_SPEECH_URLS = {
    "biden_inaugural_2021.txt": (
        "https://www.presidency.ucsb.edu/documents/"
        "inaugural-address-53"
    ),
    "biden_sotu_2022.txt": (
        "https://www.presidency.ucsb.edu/documents/"
        "address-before-joint-session-the-congress-the-state-the-union-28"
    ),
    "biden_sotu_2023.txt": (
        "https://www.presidency.ucsb.edu/documents/"
        "address-before-joint-session-the-congress-the-state-the-union-29"
    ),
    "biden_sotu_2024.txt": (
        "https://www.presidency.ucsb.edu/documents/"
        "address-before-joint-session-the-congress-the-state-the-union-30"
    ),
}

# White House speeches (additional source for Biden speeches)

WH_SPEECHES_URL = (
    "https://www.whitehouse.gov/briefing-room/speeches-remarks/"
)

# Handle directory setup and cleanup

def ensure_dirs():
    for dir_path in [BIDEN_DIR, TRUMP_DIR]:
        os.makedirs(dir_path, exist_ok=True)


# Handle Trump PDFs

def download_pdf_text(url, output_path):
    print(f"\n=== Downloading PDF from {url} ===")

    try:
        response = requests.get(url, headers=HEADERS, timeout=60)
        response.raise_for_status()

        pages = []
        reader = PdfReader(BytesIO(response.content))

        for page in tqdm(reader.pages, desc="Extracting PDF text", leave=False):
            text = page.extract_text()
            if text:
                pages.append(text)
            
        return "\n\n".join(pages)
    except Exception as e:
        print(f"Error downloading PDF from {url}: {e}")

def collect_trump_pdfs():
    print("\n=== Collecting Trump PDFs ===")

    for filename, url in TRUMP_PDFS.items():
        output_path = os.path.join(TRUMP_DIR, filename)

        if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
            print(f"  Skipping (already exists): {filename}")
            continue

        text = download_pdf_text(url, output_path)

        if text:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Saved PDF text to {output_path}")

# Handle UCSB American Presidency Project scraping

def scrape_ucsb_page(url: str) -> str:
    """Scrape text content from presidency.ucsb.edu pages."""
    print(f"  Fetching UCSB: {url[:80]}...")

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        content_div = soup.find("div", class_="field-docs-content")
        if content_div:
            text = content_div.get_text(separator="\n", strip=True)
            if text and len(text) > 200:
                return text

        article = soup.find("article")
        if article:
            body = article.find("div", class_="field--type-text-with-summary")
            if body:
                return body.get_text(separator="\n", strip=True)
            paragraphs = article.find_all("p")
            return "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

        return ""
    except Exception as e:
        print(f"  [WARN] Failed to scrape UCSB: {e}")
        return ""


def split_debate_by_speaker(text: str):
    biden_lines = []
    trump_lines = []

    raw_lines = text.split("\n")
    merged = []
    current = ""

    speaker_re = re.compile(
        r'^('
        r'TRUMP|BIDEN|WALLACE|WELKER|MODERATOR'
        r'|CHRIS WALLACE|KRISTEN WELKER'
        r'|(President\s+(Donald\s+J\.\s+)?)?Trump'
        r'|(Vice\s+President\s+)?(Joe\s+)?Biden'
        r')\s*:',
        re.IGNORECASE,
    )

    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            if current:
                merged.append(current)
                current = ""
            continue
        if speaker_re.match(stripped):
            if current:
                merged.append(current)
            current = stripped
        else:
            current = (current + " " + stripped) if current else stripped

    if current:
        merged.append(current)

    for turn in merged:
        if re.match(r'^TRUMP\s*:', turn, re.IGNORECASE) or \
           re.match(r'^(President\s+)?(Donald\s+(J\.\s+)?)?Trump\s*:', turn, re.IGNORECASE):
            content = re.sub(r'^[^:]+:\s*', '', turn)
            content = re.sub(r'\(\[?\d{1,2}:\d{2}(:\d{2})?\]?\)\s*', '', content)
            content = re.sub(r'\[crosstalk[^\]]*\]', '', content).strip()
            if len(content) > 5:
                trump_lines.append(content)

        elif re.match(r'^BIDEN\s*:', turn, re.IGNORECASE) or \
             re.match(r'^(Vice\s+President\s+)?(Joe\s+)?Biden\s*:', turn, re.IGNORECASE):
            content = re.sub(r'^[^:]+:\s*', '', turn)
            content = re.sub(r'\(\[?\d{1,2}:\d{2}(:\d{2})?\]?\)\s*', '', content)
            content = re.sub(r'\[crosstalk[^\]]*\]', '', content).strip()
            if len(content) > 5:
                biden_lines.append(content)

    return biden_lines, trump_lines


# Handle Rev.com transcripts (secondary / fallback source)

def scrape_rev_transcript(url: str) -> str:
    print(f"  Fetching Rev.com: {url[:80]}...")

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Try multiple selectors for Rev.com's various layouts
        transcript_div = soup.find("div", class_="fl-callout-text")
        if transcript_div:
            return transcript_div.get_text(separator="\n", strip=True)

        content = soup.find("div", class_="fl-post-content")
        if content:
            paragraphs = content.find_all("p")
            text = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
            if len(text) > 500:
                return text

        article = soup.find("article")
        if article:
            paragraphs = article.find_all("p")
            text = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
            if len(text) > 500:
                return text

        main = soup.find("main") or soup.find("div", {"role": "main"})
        if main:
            paragraphs = main.find_all("p")
            text = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
            if len(text) > 500:
                return text

        paragraphs = soup.find_all("p")
        return "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
    except Exception as e:
        print(f"  [WARN] Failed to scrape transcript: {e}")
        return ""


def collect_debate_transcripts():
    """Collect debate transcripts from UCSB (primary) with Rev.com fallback."""
    print("\n=== Collecting Debate Transcripts (UCSB primary) ===")

    for filename, url in UCSB_DEBATE_URLS.items():
        biden_path = os.path.join(BIDEN_DIR, f"biden_{filename}")
        trump_path = os.path.join(TRUMP_DIR, f"trump_{filename}")

        # Skip only if both files exist AND have content
        if os.path.exists(biden_path) and os.path.exists(trump_path) and \
           os.path.getsize(biden_path) > 100 and os.path.getsize(trump_path) > 100:
            print(f"  Skipping (already exists with content): {filename}")
            continue

        # Try UCSB first
        text = scrape_ucsb_page(url)

        # Fallback to Rev.com
        if not text or len(text) < 500:
            rev_url = REV_DEBATE_URLS.get(filename)
            if rev_url:
                print(f"  UCSB failed, trying Rev.com fallback...")
                text = scrape_rev_transcript(rev_url)

        if not text:
            print(f"  [WARN] Could not get transcript for {filename}")
            continue

        biden_lines, trump_lines = split_debate_by_speaker(text)

        with open(biden_path, "w", encoding="utf-8") as f:
            f.write("\n".join(biden_lines))

        with open(trump_path, "w", encoding="utf-8") as f:
            f.write("\n".join(trump_lines))

        print(f"  Saved {filename}: Biden={len(biden_lines)} lines, Trump={len(trump_lines)} lines")
        time.sleep(2)


# ===== Biden speeches from UCSB =====

def collect_biden_speeches_ucsb():
    """Collect Biden speeches from the American Presidency Project."""
    print("\n=== Collecting Biden Speeches (UCSB Presidency Project) ===")

    for filename, url in BIDEN_SPEECH_URLS.items():
        out_path = os.path.join(BIDEN_DIR, filename)

        if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
            print(f"  Skipping (already exists): {filename}")
            continue

        text = scrape_ucsb_page(url)

        if text and len(text) > 200:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"  Saved {filename} ({len(text):,} chars)")
        else:
            print(f"  [WARN] No content for {filename}")

        time.sleep(2)

# Handle Rev.com individual speeches

def scrape_rev_search_links(search_url: str, max_pages: int = 3) -> list:
    links = []

    for page in range(1, max_pages + 1):
        url = f"{search_url}&page={page}" if page > 1 else search_url
        print(f"  Fetching search page {page}: {url[:80]}...")

        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")

            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                if "/blog/transcripts/" in href and href not in links and "?s=" not in href:
                    if href.startswith("/"):
                        href = "https://www.rev.com" + href
                    links.append(href)

            # Also try finding links in article/post cards
            for a_tag in soup.select("a[href*='/transcripts/']"):
                href = a_tag.get("href", "")
                if href and href not in links and "?s=" not in href:
                    if href.startswith("/"):
                        href = "https://www.rev.com" + href
                    links.append(href)

            time.sleep(2)
        except Exception as e:
            print(f"  [WARN] Failed to fetch search page: {e}")
            break

    return links

def collect_trump_speeches():
    print("\n=== Collecting Trump Speech Transcripts (Rev.com) ===")
    links = scrape_rev_search_links(TRUMP_REV_SEARCH, max_pages=3)

    print(f"  Found {len(links)} transcript links")

    for i, url in enumerate(links[:15]):
        filename = f"trump_rev_speech_{i:02d}.txt"
        out_path = os.path.join(TRUMP_DIR, filename)

        if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
            print(f"  Skipping (already exists): {filename}")
            continue

        text = scrape_rev_transcript(url)
        if text and len(text) > 500:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"  Saved {filename} ({len(text):,} chars)")

        time.sleep(2)

def collect_biden_speeches_rev():
    print("\n=== Collecting Biden Speech Transcripts (Rev.com) ===")
    links = scrape_rev_search_links(BIDEN_REV_SEARCH, max_pages=3)

    print(f"  Found {len(links)} transcript links")

    for i, url in enumerate(links[:15]):
        filename = f"biden_rev_speech_{i:02d}.txt"
        out_path = os.path.join(BIDEN_DIR, filename)

        if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
            print(f"  Skipping (already exists): {filename}")
            continue

        text = scrape_rev_transcript(url)

        if text and len(text) > 500:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"  Saved {filename} ({len(text):,} chars)")

        time.sleep(2)

# Handle White House speeches

def collect_whitehouse_speeches():
    print("\n=== Collecting White House Speeches ===")

    try:
        resp = requests.get(WH_SPEECHES_URL, headers=HEADERS, timeout=30)

        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        links = []

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if "/briefing-room/speeches-remarks/" in href and href != WH_SPEECHES_URL:
                if href.startswith("/"):
                    href = "https://www.whitehouse.gov" + href

                if href not in links:
                    links.append(href)

        print(f"  Found {len(links)} speech links")

        for i, url in enumerate(links[:10]):
            filename = f"biden_wh_speech_{i:02d}.txt"
            out_path = os.path.join(BIDEN_DIR, filename)

            if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
                print(f"  Skipping (already exists): {filename}")
                continue

            try:
                r = requests.get(url, headers=HEADERS, timeout=30)

                r.raise_for_status()
                s = BeautifulSoup(r.text, "html.parser")

                body = s.find("div", class_="body-content") or s.find("article")
                if body:
                    text = body.get_text(separator="\n", strip=True)
                else:
                    paragraphs = s.find_all("p")
                    text = "\n".join(p.get_text(strip=True) for p in paragraphs)
                if text and len(text) > 500:
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(text)
                    print(f"  Saved {filename} ({len(text):,} chars)")
                
                time.sleep(2)
            except Exception as e:
                print(f"  [WARN] Failed to scrape {url}: {e}")
    except Exception as e:
        print(f"  [WARN] Failed to fetch WH speeches index: {e}")

# Handle Trump Tweets

def collect_trump_tweets():
    print("\n=== Collecting Trump Tweets ===")
    csv_path = os.path.join(TRUMP_DIR, "trump_tweets.csv")
    out_path = os.path.join(TRUMP_DIR, "trump_tweets.txt")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
        print("  Skipping (already exists): trump_tweets.txt")
        return
    
    if not os.path.exists(csv_path):
        print(
            "  [INFO] trump_tweets.csv not found. Download from:\n"
            "    https://www.kaggle.com/datasets/headsortails/trump-twitter-archive\n"
            "    Place CSV in data/raw/trump/trump_tweets.csv"
        )
        return
    
    try:
        tweets = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                text = row.get("text") or row.get("content") or row.get("tweet", "")
                text = text.strip()
                
                if text and not text.startswith("RT "):
                    tweets.append(text)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(tweets))

        print(f"  Saved trump_tweets.txt ({len(tweets):,} tweets)")
    except Exception as e:
        print(f"  [WARN] Failed to parse trump_tweets.csv: {e}")


# Main function to run all collection steps

def main():
    print("=" * 60)
    print("  Data Collection for Biden vs. Trump Debate Chatbot")
    print("=" * 60)

    ensure_dirs()

    # Trump data
    collect_trump_pdfs()
    collect_trump_speeches()
    collect_trump_tweets()

    # Biden data (UCSB speeches are the most reliable source)
    collect_biden_speeches_ucsb()
    collect_biden_speeches_rev()
    collect_whitehouse_speeches()

    # Shared debate transcripts (split by speaker) - UCSB primary source
    collect_debate_transcripts()

    # Summary
    trump_files = [f for f in os.listdir(TRUMP_DIR) if os.path.getsize(os.path.join(TRUMP_DIR, f)) > 0]
    biden_files = [f for f in os.listdir(BIDEN_DIR) if os.path.getsize(os.path.join(BIDEN_DIR, f)) > 0]

    print("\n" + "=" * 60)
    print("  Data collection complete!")
    print(f"  Trump files with content: {trump_files}")
    print(f"  Biden files with content: {biden_files}")
    print("=" * 60)

# Run the main function

if __name__ == "__main__":
    main()