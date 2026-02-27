import os
import re
import sys
import time
import argparse
import textwrap

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR          = os.path.join(SCRIPT_DIR, "..")
DEFAULT_TRUMP_DIR = os.path.join(ROOT_DIR, "models", "trump")
DEFAULT_BIDEN_DIR = os.path.join(ROOT_DIR, "models", "biden")

sys.path.insert(0, ROOT_DIR)

from utils.voice_synth_utils import speak as _tts_speak  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────

TTS_CHAR_LIMIT = 240   # XTTS-v2 hard-warns at 250 chars per sentence
MAX_RETRIES    = 3     # retry generation this many times if output looks bad
WIDTH          = 70

# Transcript-artifact words that should never start a response
_BAD_STARTERS = {
    "vernacular", "ernest", "earnest", "interviewer", "moderator",
    "host", "anchor", "reporter", "narrator", "speaker",
}

# ── CLI & interactive setup ────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Biden vs. Trump AI Debate",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--topic",       type=str,   default=None,
                   help="Debate topic (prompted if omitted).")
    p.add_argument("--turns",       type=int,   default=None,
                   help="Number of exchanges (prompted if omitted).")
    p.add_argument("--persona",     type=str,   default=None,
                   choices=["both", "trump", "biden"],
                   help=(
                       "both  — single machine, both personas generated here.\n"
                       "trump — two-laptop mode: this machine plays Trump;\n"
                       "        Biden's replies are entered from the other laptop.\n"
                       "biden — two-laptop mode: this machine plays Biden;\n"
                       "        Trump's replies are entered from the other laptop."
                   ))
    p.add_argument("--max_new",     type=int,   default=120,
                   help="Max new tokens per response (default: 120).")
    p.add_argument("--temperature", type=float, default=0.82,
                   help="Sampling temperature (default: 0.82).")
    p.add_argument("--top_p",       type=float, default=0.90,
                   help="Nucleus sampling p (default: 0.90).")
    p.add_argument("--trump_model", type=str,   default=DEFAULT_TRUMP_DIR,
                   help="Path to fine-tuned Trump model directory.")
    p.add_argument("--biden_model", type=str,   default=DEFAULT_BIDEN_DIR,
                   help="Path to fine-tuned Biden model directory.")
    p.add_argument("--tts",         type=str,   default=None,
                   choices=["auto", "none"],
                   help="auto = voice output via Coqui/edge-tts; none = silent.")
    return p.parse_args()


def _ask(prompt: str, default: str, validator=None):
    """Ask the user a question with a default, re-prompting until valid."""
    while True:
        raw = input(f"  {prompt} [{default}]: ").strip()
        val = raw or default
        if validator is None or validator(val):
            return val
        print(f"    Invalid input. Expected one of: {validator.__doc__}")


def interactive_setup(args):
    """Fill in any args the user did not supply on the command line."""
    if not any(v is None for v in [args.topic, args.turns, args.persona, args.tts]):
        return args

    print("\n" + "─" * WIDTH)
    print("  Debate Setup — press Enter to accept defaults")
    print("─" * WIDTH)

    if args.topic is None:
        args.topic = _ask("Debate topic", "the economy")

    if args.turns is None:
        raw = _ask("Number of exchanges", "3")
        args.turns = int(raw) if raw.isdigit() else 3

    if args.persona is None:
        def _valid_persona(v):
            """both / trump / biden"""
            return v in ("both", "trump", "biden")
        args.persona = _ask("Persona — both / trump / biden", "both", _valid_persona)

    if args.tts is None:
        def _valid_tts(v):
            """auto / none"""
            return v in ("auto", "none")
        args.tts = _ask("TTS — auto / none", "auto", _valid_tts)

    print("─" * WIDTH + "\n")
    return args

# ── Model loading ──────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_dir, persona):
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


def _build_bad_word_ids(tokenizer) -> list:
    """Encode known artifact words so the model never generates them."""
    bad = [
        "vernacular", " vernacular", "Vernacular", " Vernacular",
        "earnest",    " earnest",    "Earnest",    " Earnest",
        "ernest",     " ernest",     "Ernest",     " Ernest",
        # Twitter / social-media noise
        "(@", " (@", "twitter", " twitter", "Twitter",
        "retweet", " retweet", "RT @",
        # Rev.com artefacts
        "rev.com", "Rev.com", "transcript", " transcript",
    ]
    ids = []
    for w in bad:
        enc = tokenizer.encode(w, add_special_tokens=False)
        if enc:
            ids.append(enc)
    return ids


# ── Text generation ────────────────────────────────────────────────────────────

def _build_prompt(speaker: str, topic: str, opponent_text: str, own_prev_text: str) -> str:
    """
    Build a GPT-2 prompt matching the training data structure.

    Training format:  <|startoftext|>\\n[TRUMP]: …\\n[BIDEN]: …\\n<|endoftext|>
    Trump always appears first; Biden always responds to Trump.
    Each speaker is seeded with an in-character opener so the model
    stays on topic and away from interviewer / transcript territory.

    The seed phrase is a *complete* clause so the first generated token
    is always a new word — prevents mid-word continuation artefacts.
    The topic is stripped of trailing punctuation so it integrates cleanly.
    """
    # Strip trailing punctuation from topic to avoid mid-sentence breaks
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
        return (
            f"<|startoftext|>\n"
            f"[TRUMP]: {opponent_text.strip()}\n"
            f"[BIDEN]: Here's what I know about {topic_clean}. "
        )


def _is_valid(text: str) -> bool:
    """Return True if the generated text looks like a genuine debate response."""
    if len(text) < 40:
        return False
    words = text.split()
    if len(words) < 6:
        return False
    # Must start with a capital letter (not a mid-word continuation)
    if not text[0].isupper():
        return False
    first_word = words[0].lower().rstrip(".,!?")
    if first_word in _BAD_STARTERS:
        return False
    # Reject Twitter handles
    if "@" in text:
        return False
    # Reject Name Surname: or Name (@handle) patterns
    if re.match(r"^[A-Z][a-z]+ [A-Z][a-z]+[:\s(]", text):
        return False
    # Reject lines that start with a date: "January 2nd", "February 23, 2017"
    if re.match(r"^(January|February|March|April|May|June|July|August|September|October|November|December)\s", text):
        return False
    return True


def generate_response(
    model, tokenizer, device,
    speaker,
    topic="",
    opponent_text="",
    own_prev_text="",
    max_new=120,
    temperature=0.82,
    top_p=0.90,
    bad_word_ids=None,
):
    """Generate a debate response with retry logic to filter bad outputs."""
    prompt = _build_prompt(speaker, topic, opponent_text, own_prev_text)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
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


def _clean_generated(text: str) -> str:
    """Strip Rev.com / transcript / social-media artifacts from model output."""
    text = re.sub(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*\(@[^)]+\)[^:]*:\s*", "", text)
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)

    text = re.sub(r"[\(\[]\s*\d{1,2}:\d{2}(?::\d{2})?\s*[\)\]]", "", text)
    text = re.sub(r"\[[^\]]{0,60}\]", "", text)
    # Remove leading date stamps: "October 11, 2017", "February 2nd :"
    text = re.sub(
        r"^(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\s*[:\-]?\s*",
        "", text, flags=re.IGNORECASE
    )
    # Remove interviewer/speaker intro lines: "Firstname Lastname:"
    text = re.sub(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s*(?:\([^)]*\))?:\s*", "", text)
    # Remove lines that look like stage directions or metadata
    lines = [ln for ln in text.splitlines() if not re.match(r"^\s*\([^)]{0,40}\)\s*$", ln)]
    text = " ".join(" ".join(lines).split())
    # Strip leaked speaker tags at the start
    for tag in ("[BIDEN]:", "[TRUMP]:", "[MODERATOR]:"):
        if text.upper().startswith(tag.upper()):
            text = text[len(tag):].strip()
    # Cut off at the first occurrence of another speaker's tag
    for tag in ("[BIDEN]:", "[TRUMP]:"):
        idx = text.find(tag)
        if idx > 0:
            text = text[:idx].strip()
    return text.strip()


def _trim_to_last_sentence(text: str, char_limit: int = TTS_CHAR_LIMIT) -> str:
    """Trim to the last complete sentence, staying within char_limit."""
    text = text.strip()
    if len(text) > char_limit:
        text = text[:char_limit]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if not sentences:
        return text
    if sentences[-1][-1:] in ".!?":
        return text
    return " ".join(sentences[:-1]).strip() or text

# ── TTS helper ────────────────────────────────────────────────────────────────

def build_speak_fn(tts_mode):
    if tts_mode == "none":
        return lambda text, persona="biden": None
    print("[debate] TTS backend loaded.", file=sys.stderr)
    return lambda text, persona="biden": _tts_speak(text, persona)

# ── Display ────────────────────────────────────────────────────────────────────

def banner(label, text):
    print("\n" + "─" * WIDTH)
    print(f"  {label}")
    print("─" * WIDTH)
    for line in textwrap.wrap(text, WIDTH - 4):
        print(f"    {line}")
    print("─" * WIDTH)

# ── Debate modes ──────────────────────────────────────────────────────────────

def run_debate_both(args, speak_fn):
    """Single-machine mode: both personas generated on this computer."""
    print("\n" + "=" * WIDTH)
    print("  Biden vs. Trump AI Debate  [single-machine mode]")
    print(f"  Topic: {args.topic}  |  Exchanges: {args.turns}")
    print("=" * WIDTH)

    print("\n  Loading models …")
    trump_model, trump_tok, trump_dev = load_model_and_tokenizer(args.trump_model, "trump")
    biden_model, biden_tok, biden_dev = load_model_and_tokenizer(args.biden_model, "biden")

    trump_bad = _build_bad_word_ids(trump_tok)
    biden_bad = _build_bad_word_ids(biden_tok)

    intro = (
        f"Welcome to the Biden versus Trump AI Debate. "
        f"Tonight's topic is {args.topic}. "
        f"We will have {args.turns} exchange"
        + ("s" if args.turns != 1 else "") +
        ". Mr. Trump, you have the opening statement."
    )
    banner("MODERATOR", intro)
    speak_fn(intro, "moderator")
    time.sleep(0.5)

    trump_prev = ""
    biden_prev = ""

    for turn in range(1, args.turns + 1):
        print(f"\n{'═' * WIDTH}")
        print(f"  Exchange {turn} of {args.turns}")
        print(f"{'═' * WIDTH}")

        trump_response = generate_response(
            trump_model, trump_tok, trump_dev,
            speaker       = "trump",
            topic         = args.topic,
            opponent_text = biden_prev,
            own_prev_text = trump_prev,
            max_new       = args.max_new,
            temperature   = args.temperature,
            top_p         = args.top_p,
            bad_word_ids  = trump_bad,
        )
        banner("TRUMP", trump_response)
        speak_fn(trump_response, "trump")
        trump_prev = trump_response
        time.sleep(0.3)

        biden_response = generate_response(
            biden_model, biden_tok, biden_dev,
            speaker       = "biden",
            topic         = args.topic,
            opponent_text = trump_response,
            max_new       = args.max_new,
            temperature   = args.temperature,
            top_p         = args.top_p,
            bad_word_ids  = biden_bad,
        )
        banner("BIDEN", biden_response)
        speak_fn(biden_response, "biden")
        biden_prev = biden_response
        time.sleep(0.3)

    print("\n" + "=" * WIDTH)
    print("  Debate complete.")
    print("=" * WIDTH)


def run_debate_solo(args, speak_fn):
    """Two-laptop mode: this machine generates ONE persona.
    The opponent's text is entered manually via stdin from the other laptop."""
    persona  = args.persona
    opponent = "biden" if persona == "trump" else "trump"

    print("\n" + "=" * WIDTH)
    print(f"  Biden vs. Trump AI Debate  [{persona.upper()} laptop]")
    print(f"  Topic: {args.topic}  |  Exchanges: {args.turns}")
    print(f"  Opponent ({opponent.upper()}) text will be pasted from the other laptop.")
    print("=" * WIDTH)

    model_dir = args.trump_model if persona == "trump" else args.biden_model
    model, tok, dev = load_model_and_tokenizer(model_dir, persona)
    bad_ids = _build_bad_word_ids(tok)

    own_prev = ""
    opp_prev = ""

    for turn in range(1, args.turns + 1):
        print(f"\n{'═' * WIDTH}")
        print(f"  Exchange {turn} of {args.turns}")
        print(f"{'═' * WIDTH}")

        if persona == "trump":
            response = generate_response(
                model, tok, dev,
                speaker       = "trump",
                topic         = args.topic,
                opponent_text = opp_prev,
                own_prev_text = own_prev,
                max_new       = args.max_new,
                temperature   = args.temperature,
                top_p         = args.top_p,
                bad_word_ids  = bad_ids,
            )
            banner("TRUMP  [this machine]", response)
            speak_fn(response, "trump")
            own_prev = response

            if turn < args.turns:
                print("\n  Copy the above to the Biden laptop, then paste Biden's reply here:")
                opp_prev = input("  BIDEN: ").strip() or "I disagree."

        else:  # biden
            label = "Trump's opening statement" if turn == 1 else "Trump's latest statement"
            print(f"\n  Paste {label} from the Trump laptop:")
            opp_prev = input("  TRUMP: ").strip() or "No comment."

            response = generate_response(
                model, tok, dev,
                speaker       = "biden",
                topic         = args.topic,
                opponent_text = opp_prev,
                max_new       = args.max_new,
                temperature   = args.temperature,
                top_p         = args.top_p,
                bad_word_ids  = bad_ids,
            )
            banner("BIDEN  [this machine]", response)
            speak_fn(response, "biden")
            own_prev = response

    print("\n" + "=" * WIDTH)
    print("  Debate complete.")
    print("=" * WIDTH)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    args = interactive_setup(args)

    if args.topic   is None: args.topic   = "the economy"
    if args.turns   is None: args.turns   = 3
    if args.persona is None: args.persona = "both"
    if args.tts     is None: args.tts     = "auto"

    speak_fn = build_speak_fn(args.tts)

    if args.persona == "both":
        run_debate_both(args, speak_fn)
    else:
        run_debate_solo(args, speak_fn)


if __name__ == "__main__":
    try:
        main()
    except (SystemExit, KeyboardInterrupt):
        pass
    sys.exit(0)
