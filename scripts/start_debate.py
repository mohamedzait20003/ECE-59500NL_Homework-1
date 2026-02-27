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
WIDTH          = 70

# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Biden vs. Trump AI Debate")
    p.add_argument("--topic",       type=str,   default="the economy")
    p.add_argument("--turns",       type=int,   default=5)
    p.add_argument("--max_new",     type=int,   default=120)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top_p",       type=float, default=0.92)
    p.add_argument("--trump_model", type=str,   default=DEFAULT_TRUMP_DIR)
    p.add_argument("--biden_model", type=str,   default=DEFAULT_BIDEN_DIR)
    p.add_argument(
        "--tts", choices=["auto", "none"], default="auto",
        help="auto = voice output (Coqui → edge-tts → pyttsx3); none = silent.",
    )
    return p.parse_args()

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

# ── Text generation ────────────────────────────────────────────────────────────

def generate_response(
    model, tokenizer, device,
    speaker,
    topic="",
    opponent_text="",
    own_prev_text="",
    max_new=120,
    temperature=0.85,
    top_p=0.92,
):
    """
    Build a prompt matching training-data structure and generate a response.

    Training format:  <|startoftext|>\\n[TRUMP]: …\\n[BIDEN]: …\\n<|endoftext|>
    Trump always appears first; Biden always responds to Trump.
    Each speaker is seeded with a strong in-character opener to keep
    the model on topic and out of interviewer / moderator territory.
    """
    if speaker == "trump":
        seed = "Let me tell you —"
        if own_prev_text and opponent_text:
            prompt = (
                f"<|startoftext|>\n"
                f"[TRUMP]: {own_prev_text.strip()}\n"
                f"[BIDEN]: {opponent_text.strip()}\n"
                f"<|endoftext|>\n\n"
                f"<|startoftext|>\n"
                f"[TRUMP]: {seed} on {topic}, "
            )
        else:
            prompt = (
                f"<|startoftext|>\n"
                f"[TRUMP]: {seed} when it comes to {topic}, "
            )
    else:  # biden
        seed = "Here\u2019s the deal —"
        prompt = (
            f"<|startoftext|>\n"
            f"[TRUMP]: {opponent_text.strip()}\n"
            f"[BIDEN]: {seed} on {topic}, "
        )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens     = max_new,
            do_sample          = True,
            temperature        = temperature,
            top_p              = top_p,
            repetition_penalty = 1.3,
            pad_token_id       = tokenizer.eos_token_id,
            eos_token_id       = tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    generated = _clean_generated(generated)
    generated = _trim_to_last_sentence(generated)
    return generated if generated else "I have nothing to add at this time."


def _clean_generated(text: str) -> str:
    """Strip Rev.com / transcript artifacts from model output."""
    # Remove timestamps: (44:39), (1:23:45), [00:44]
    text = re.sub(r"[\(\[]\s*\d{1,2}:\d{2}(?::\d{2})?\s*[\)\]]", "", text)
    # Remove [inaudible ...], [crosstalk ...], [applause], etc.
    text = re.sub(r"\[[^\]]{0,60}\]", "", text)
    # Remove interviewer/speaker intro lines: "Firstname Lastname (HH:MM):"
    text = re.sub(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s*(?:\([^)]*\))?:\s*", "", text)
    # Remove lines that look like stage directions or metadata
    lines = [l for l in text.splitlines() if not re.match(r"^\s*\([^)]{0,40}\)\s*$", l)]
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

# ── Debate loop ────────────────────────────────────────────────────────────────

def run_debate(args, speak_fn):
    print("\n" + "=" * WIDTH)
    print("  Biden vs. Trump AI Debate")
    print(f"  Topic: {args.topic}  |  Exchanges: {args.turns}")
    print("=" * WIDTH)

    print("\n  Loading models …")
    trump_model, trump_tok, trump_dev = load_model_and_tokenizer(args.trump_model, "trump")
    biden_model, biden_tok, biden_dev = load_model_and_tokenizer(args.biden_model, "biden")

    # Moderator opens the debate
    intro = (
        f"Welcome to the Biden versus Trump AI Debate. "
        f"Tonight's topic is {args.topic}. "
        f"We will have {args.turns} exchange"
        + ("s" if args.turns != 1 else "") +
        f". Mr. Trump, you have the opening statement."
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

        # Trump's turn
        trump_response = generate_response(
            trump_model, trump_tok, trump_dev,
            speaker       = "trump",
            topic         = args.topic,
            opponent_text = biden_prev,
            own_prev_text = trump_prev,
            max_new       = args.max_new,
            temperature   = args.temperature,
            top_p         = args.top_p,
        )
        banner("TRUMP", trump_response)
        speak_fn(trump_response, "trump")
        trump_prev = trump_response
        time.sleep(0.3)

        # Biden's turn — responds directly to what Trump just said
        biden_response = generate_response(
            biden_model, biden_tok, biden_dev,
            speaker       = "biden",
            topic         = args.topic,
            opponent_text = trump_response,
            max_new       = args.max_new,
            temperature   = args.temperature,
            top_p         = args.top_p,
        )
        banner("BIDEN", biden_response)
        speak_fn(biden_response, "biden")
        biden_prev = biden_response
        time.sleep(0.3)

    print("\n" + "=" * WIDTH)
    print("  Debate complete.")
    print("=" * WIDTH)

# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    speak_fn = build_speak_fn(args.tts)
    run_debate(args, speak_fn)

if __name__ == "__main__":
    try:
        main()
    except (SystemExit, KeyboardInterrupt):
        pass
    sys.exit(0)
