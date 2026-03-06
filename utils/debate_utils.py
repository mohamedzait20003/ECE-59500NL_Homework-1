import os
import re
import textwrap

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR          = os.path.join(SCRIPT_DIR, "..")
DEFAULT_TRUMP_DIR = os.path.join(ROOT_DIR, "models", "trump")
DEFAULT_BIDEN_DIR = os.path.join(ROOT_DIR, "models", "biden")

# ── Constants ──────────────────────────────────────────────────────────────────

TTS_CHAR_LIMIT = 500
MAX_RETRIES    = 3
WIDTH          = 70

DEFAULT_TOPIC = "war between Iran and US"
DEFAULT_TURNS = 3

_BAD_STARTERS = {
    "vernacular", "ernest", "earnest", "interviewer", "moderator",
    "host", "anchor", "reporter", "narrator", "speaker",
}

_BPE_SPLIT_RE = re.compile(
    r'([a-zA-Z]{3,})'
    r'(about|they|them|their|which|also|because|between|against|from|into|during)\b',
    re.IGNORECASE,
)

# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(model_dir: str, persona: str):
    if os.path.isdir(model_dir) and os.listdir(model_dir):
        print(f"  [{persona.upper()}] Loading fine-tuned model from {model_dir} …")
        tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
        model     = GPT2LMHeadModel.from_pretrained(model_dir)
    else:
        print(f"  [{persona.upper()}] Model not found → falling back to base gpt2-medium.")
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-medium")
        model     = GPT2LMHeadModel.from_pretrained("gpt2-medium")

    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            "<|startoftext|>", "<|endoftext|>",
            "[BIDEN]:", "[TRUMP]:",
        ]
    })
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return model, tokenizer, device


def build_bad_word_ids(tokenizer) -> list:
    bad = [
        "vernacular", " vernacular", "Vernacular", " Vernacular",
        "earnest",    " earnest",    "Earnest",    " Earnest",
        "ernest",     " ernest",     "Ernest",     " Ernest",
        "(@", " (@", "twitter", " twitter", "Twitter",
        "retweet", " retweet", "RT @",
        "rev.com", "Rev.com", "transcript", " transcript",
    ]
    ids = []
    for w in bad:
        enc = tokenizer.encode(w, add_special_tokens=False)
        if enc:
            ids.append(enc)
    return ids


# ══════════════════════════════════════════════════════════════════════════════
# Text generation
# ══════════════════════════════════════════════════════════════════════════════

def _build_prompt(speaker: str, topic: str,
                  opponent_text: str, own_prev_text: str) -> str:
    topic_clean = topic.rstrip(".,!?").strip()

    if speaker == "trump":
        if own_prev_text and opponent_text:
            return (
                f"<|startoftext|>\n"
                f"[TRUMP]: {own_prev_text.strip()}\n"
                f"[BIDEN]: {opponent_text.strip()}\n"
                f"<|endoftext|>\n\n"
                f"<|startoftext|>\n"
                f"[TRUMP]: Let me tell you exactly what happened with {topic_clean}. "
            )
        return (
            f"<|startoftext|>\n"
            f"[TRUMP]: I'll tell you something about {topic_clean}. "
        )
    else:  # biden
        if opponent_text:
            return (
                f"<|startoftext|>\n"
                f"[TRUMP]: {opponent_text.strip()}\n"
                f"[BIDEN]: Here's what I know about {topic_clean}. "
            )
        return (
            f"<|startoftext|>\n"
            f"[BIDEN]: Here's what I know about {topic_clean}. "
        )


def _is_valid(text: str) -> bool:
    if len(text) < 40:
        return False
    words = text.split()
    if len(words) < 6:
        return False
    if not text[0].isupper():
        return False
    first_word = words[0].lower().rstrip(".,!?")
    if first_word in _BAD_STARTERS:
        return False
    if "@" in text:
        return False
    if re.match(r"^[A-Z][a-z]+ [A-Z][a-z]+[:\s(]", text):
        return False
    if re.match(
        r"^(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s", text
    ):
        return False
    return True


def _clean_generated(text: str) -> str:
    text = re.sub(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*\(@[^)]+\)[^:]*:\s*", "", text)
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    text = re.sub(r"[\(\[]\s*\d{1,2}:\d{2}(?::\d{2})?\s*[\)\]]", "", text)
    text = re.sub(r"\[[^\]]{0,60}\]", "", text)
    text = re.sub(
        r"^(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\s*[:\-]?\s*",
        "", text, flags=re.IGNORECASE,
    )
    text = re.sub(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s*(?:\([^)]*\))?:\s*", "", text)
    lines = [ln for ln in text.splitlines()
             if not re.match(r"^\s*\([^)]{0,40}\)\s*$", ln)]
    text = " ".join(" ".join(lines).split())
    for tag in ("[BIDEN]:", "[TRUMP]:", "[MODERATOR]:"):
        if text.upper().startswith(tag.upper()):
            text = text[len(tag):].strip()
    for tag in ("[BIDEN]:", "[TRUMP]:"):
        idx = text.find(tag)
        if idx > 0:
            text = text[:idx].strip()
    # Fix BPE word-joining artifacts (apply repeatedly for nested joins)
    prev = None
    while prev != text:
        prev = text
        text = _BPE_SPLIT_RE.sub(r'\1 \2', text)
    return text.strip()


def _trim_to_last_sentence(text: str, char_limit: int = TTS_CHAR_LIMIT) -> str:
    text = text.strip()
    if len(text) > char_limit:
        text = text[:char_limit]

    # 1) sentence-ending punctuation
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if not sentences:
        return text
    if sentences[-1][-1:] in ".!?":
        return text
    if len(sentences) > 1:
        return " ".join(sentences[:-1]).strip() or text
    
    # 2) no sentence boundary — fall back to semicolon, then comma
    for sep in (';', ','):
        idx = text.rfind(sep)
        if idx > len(text) // 3:
            return text[:idx + 1].strip()
        
    # 3) at least trim to the last complete word
    idx = text.rfind(' ')
    if idx > 0:
        return text[:idx].strip()
    return text


def generate_response(
    model, tokenizer, device,
    speaker: str,
    topic: str = "",
    opponent_text: str = "",
    own_prev_text: str = "",
    max_new: int = 120,
    temperature: float = 0.82,
    top_p: float = 0.90,
    bad_word_ids=None,
) -> str:
    prompt = _build_prompt(speaker, topic, opponent_text, own_prev_text)
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512,
    ).to(device)

    for attempt in range(MAX_RETRIES):
        temp = min(temperature + attempt * 0.05, 1.0)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens     = max_new,
                do_sample          = True,
                temperature        = temp,
                top_p              = top_p,
                repetition_penalty = 1.4,
                bad_words_ids      = bad_word_ids or [],
                pad_token_id       = tokenizer.eos_token_id,
                eos_token_id       = tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        generated = _clean_generated(generated)
        generated = _trim_to_last_sentence(generated)
        if _is_valid(generated):
            return generated

    return "I have nothing to add on that at this time."


# ══════════════════════════════════════════════════════════════════════════════
# Display
# ══════════════════════════════════════════════════════════════════════════════

def banner(label: str, text: str) -> None:
    print("\n" + "─" * WIDTH)
    print(f"  {label}")
    print("─" * WIDTH)
    for line in textwrap.wrap(text, WIDTH - 4):
        print(f"    {line}")
    print("─" * WIDTH)


# ══════════════════════════════════════════════════════════════════════════════
# Q&A routing helper
# ══════════════════════════════════════════════════════════════════════════════

def determine_qa_target(question: str) -> str:
    """Return 'trump', 'biden', or 'both' based on who the question targets."""
    q = question.lower()
    has_trump = "trump" in q
    has_biden = "biden" in q
    if has_trump and not has_biden:
        return "trump"
    if has_biden and not has_trump:
        return "biden"
    return "both"
