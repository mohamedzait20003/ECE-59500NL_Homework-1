import os
import re
import nltk
import glob
import random
from nltk.tokenize import sent_tokenize

# Ensure necessary NLTK resources are available

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


# Define directories

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Constants for chunking

MIN_EXAMPLE_CHARS = 50
MAX_EXAMPLE_CHARS = 3000

# Cleaning function

def clean_text(text: str) -> str:
    text = text.encode("ascii", errors="ignore").decode("ascii")

    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([!?.]){3,}", r"\1", text)

    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)
    return text.strip()

# chunking function

def chunk_text(text: str, max_chars: int = MAX_EXAMPLE_CHARS) -> list:
    sentences = sent_tokenize(text)

    chunks = []
    current_chunk = []

    current_len = 0

    for sent in sentences:
        sent = sent.strip()

        if not sent:
            continue

        if current_len + len(sent) + 1 > max_chars and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_len = 0

        current_chunk.append(sent)
        current_len += len(sent) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return [c for c in chunks if len(c) >= MIN_EXAMPLE_CHARS]

# Loading and example-building functions

def load_raw_files(persona_dir: str) -> str:
    all_text = []
    pattern = os.path.join(persona_dir, "*.txt")
    files = sorted(glob.glob(pattern))

    for fpath in files:
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            all_text.append(f.read())

    return "\n\n".join(all_text)


def deduplicate_chunks(chunks: list, threshold: float = 0.9) -> list:
    """Remove near-duplicate chunks using simple Jaccard similarity."""
    unique = []

    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        is_dup = False

        for existing in unique:
            existing_words = set(existing.lower().split())
            if not chunk_words or not existing_words:
                continue
            intersection = chunk_words & existing_words
            union = chunk_words | existing_words
            similarity = len(intersection) / len(union)

            if similarity >= threshold:
                is_dup = True
                break

        if not is_dup:
            unique.append(chunk)

    return unique


def build_monologue_examples(chunks: list, speaker_tag: str) -> list:
    examples = []

    for chunk in chunks:
        example = f"<|startoftext|>\n[{speaker_tag}]: {chunk}\n<|endoftext|>"
        examples.append(example)
    return examples


# Build debate examples by pairing chunks from both personas

def build_debate_examples(biden_chunks: list, trump_chunks: list) -> list:
    examples = []
    n = min(len(biden_chunks), len(trump_chunks))
    indices = list(range(n))
    random.shuffle(indices)

    for idx in indices:
        example_t = (
            f"<|startoftext|>\n"
            f"[BIDEN]: {biden_chunks[idx]}\n"
            f"[TRUMP]: {trump_chunks[idx]}\n"
            f"<|endoftext|>"
        )
        examples.append(example_t)

        example_b = (
            f"<|startoftext|>\n"
            f"[TRUMP]: {trump_chunks[idx]}\n"
            f"[BIDEN]: {biden_chunks[idx]}\n"
            f"<|endoftext|>"
        )
        examples.append(example_b)

    return examples


def process_persona(persona: str) -> list:
    persona_dir = os.path.join(RAW_DIR, persona)
    if not os.path.isdir(persona_dir):
        print(f"  [WARN] Directory not found: {persona_dir}")
        return []

    raw_text = load_raw_files(persona_dir)
    print(f"  Raw text length for {persona}: {len(raw_text):,} chars")

    cleaned = clean_text(raw_text)
    print(f"  Cleaned text length: {len(cleaned):,} chars")

    chunks = chunk_text(cleaned)
    print(f"  Number of chunks (before dedup): {len(chunks)}")

    chunks = deduplicate_chunks(chunks)
    print(f"  Number of chunks (after dedup):  {len(chunks)}")

    tag = persona.upper()
    examples = build_monologue_examples(chunks, tag)
    return examples, chunks

# Main function

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    random.seed(42)

    print("=" * 60)
    print("  Preprocessing Training Data")
    print("=" * 60)

    biden_examples, biden_chunks = process_persona("biden") if os.path.isdir(
        os.path.join(RAW_DIR, "biden")
    ) else ([], [])
    trump_examples, trump_chunks = process_persona("trump") if os.path.isdir(
        os.path.join(RAW_DIR, "trump")
    ) else ([], [])

    print("\n  Building debate-style examples...")
    debate_examples = build_debate_examples(biden_chunks, trump_chunks)
    print(f"  Debate examples: {len(debate_examples)}")

    biden_train = biden_examples + debate_examples
    trump_train = trump_examples + debate_examples

    random.shuffle(biden_train)
    random.shuffle(trump_train)

    # Save
    biden_path = os.path.join(PROCESSED_DIR, "biden_train.txt")
    trump_path = os.path.join(PROCESSED_DIR, "trump_train.txt")

    with open(biden_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(biden_train))
    print(f"\n  Saved {biden_path} ({len(biden_train)} examples)")

    with open(trump_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(trump_train))
    print(f"  Saved {trump_path} ({len(trump_train)} examples)")

    print("\n" + "=" * 60)
    print("  Preprocessing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()