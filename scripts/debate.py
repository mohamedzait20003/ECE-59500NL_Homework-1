import os
import sys
import argparse
import textwrap
import time

SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR          = os.path.join(SCRIPT_DIR, "..")
DEFAULT_TRUMP_DIR = os.path.join(ROOT_DIR, "models", "trump")
DEFAULT_BIDEN_DIR = os.path.join(ROOT_DIR, "models", "biden")

sys.path.insert(0, ROOT_DIR)

DEBATE_RULES = (
    "This is a presidential debate. "
    "Candidates speak one at a time. "
    "Each candidate has up to 90 seconds per response. "
    "No interruptions. Candidates must stay on topic."
)

MODERATOR_INTRO = (
    "Good evening. I'm your moderator. "
    "Tonight we have Former President Joe Biden and President Donald Trump "
    "debating the following topic: {topic}. "
    "President Trump, please give your opening statement."
)

MODERATOR_BIDEN_PROMPT = "Thank you. Vice President Biden, your response."
MODERATOR_TRUMP_PROMPT = "Thank you. President Trump, your response."
MODERATOR_CLOSING = (
    "That concludes tonight's debate. "
    "Thank you to both candidates and to our audience."
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Biden vs. Trump spoken debate using fine-tuned GPT-2 models."
    )
    parser.add_argument(
        "--mode", choices=["trump", "biden", "both"], default="both",
        help="trump = this machine plays Trump, biden = this machine plays Biden, both = single-machine demo.",
    )
    parser.add_argument("--topic",       type=str,   default="the economy")
    parser.add_argument("--turns",       type=int,   default=5)
    parser.add_argument("--max_new",     type=int,   default=150)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p",       type=float, default=0.92)
    parser.add_argument("--trump_model", type=str,   default=DEFAULT_TRUMP_DIR)
    parser.add_argument("--biden_model", type=str,   default=DEFAULT_BIDEN_DIR)
    parser.add_argument("--tts", choices=["azure", "pyttsx3", "none", "auto"], default="auto")
    parser.add_argument("--stt", choices=["azure", "keyboard", "auto"],        default="auto")
    return parser.parse_args()

def load_model_and_tokenizer(model_dir: str, persona: str):
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    if os.path.isdir(model_dir) and os.listdir(model_dir):
        print(f"  [{persona.upper()}] Loading fine-tuned model from  {model_dir}  …")
        tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
        model     = GPT2LMHeadModel.from_pretrained(model_dir)
    else:
        print(f"  [{persona.upper()}] Fine-tuned model not found at  {model_dir}.")
        print(f"  [{persona.upper()}] Falling back to base gpt2-medium (untrained).")
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-medium")
        model     = GPT2LMHeadModel.from_pretrained("gpt2-medium")

    special_tokens = {
        "additional_special_tokens": [
            "<|startoftext|>", "<|endoftext|>",
            "[BIDEN]:", "[TRUMP]:",
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer, device

def generate_response(
    model,
    tokenizer,
    device: str,
    speaker_tag: str,
    opponent_tag: str,
    context: str,
    max_new: int = 150,
    temperature: float = 0.85,
    top_p: float = 0.92,
) -> str:
    import torch

    prompt = (
        f"<|startoftext|>\n"
        f"{opponent_tag} {context.strip()}\n"
        f"{speaker_tag} "
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=400,
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

    for tag in ["[BIDEN]:", "[TRUMP]:", "[MODERATOR]:"]:
        if generated.upper().startswith(tag.upper()):
            generated = generated[len(tag):].strip()

    generated = _trim_to_last_sentence(generated)
    return generated if generated else "I have nothing to add at this time."

def _trim_to_last_sentence(text: str) -> str:
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if not sentences:
        return text
    if sentences[-1][-1:] in ".!?":
        return text.strip()
    return " ".join(sentences[:-1]).strip() or text.strip()

def build_speak_fn(tts_mode: str):
    if tts_mode == "none":
        def speak(text, persona="moderator"):
            pass
        return speak

    if tts_mode in ("azure", "pyttsx3", "auto"):
        try:
            from utils.tts_utils import speak as _speak
            if tts_mode == "pyttsx3":
                import utils.tts_utils as tts_mod
                tts_mod._azure_available = False
            return _speak
        except ImportError:
            print("[debate] Could not import tts_utils; TTS disabled.", file=sys.stderr)

    def speak(text, persona="moderator"):
        pass
    return speak

def build_listen_fn(stt_mode: str):
    if stt_mode in ("azure", "auto"):
        try:
            from utils.tts_utils import listen as _listen
            return _listen
        except ImportError:
            pass

    def listen(prompt="Listening…", timeout_seconds=10):
        try:
            return input(f"  {prompt} (type the opponent's statement): ").strip()
        except (EOFError, KeyboardInterrupt):
            return ""
    return listen

BANNER_WIDTH = 70

def print_banner(label: str, text: str, width: int = BANNER_WIDTH):
    print("\n" + "─" * width)
    print(f"  {label}")
    print("─" * width)
    for line in textwrap.wrap(text, width - 4):
        print(f"    {line}")
    print("─" * width)

def run_both_mode(args, speak_fn):
    print("\n" + "=" * BANNER_WIDTH)
    print("  Biden vs. Trump Debate  [single-machine mode]")
    print(f"  Topic: {args.topic}")
    print("=" * BANNER_WIDTH)

    trump_model, trump_tok, trump_dev = load_model_and_tokenizer(args.trump_model, "trump")
    biden_model, biden_tok, biden_dev = load_model_and_tokenizer(args.biden_model, "biden")

    intro = MODERATOR_INTRO.format(topic=args.topic)
    print_banner("MODERATOR", intro)
    speak_fn(intro, "moderator")
    time.sleep(1)

    last_trump_statement = f"Let me talk about {args.topic}."
    last_biden_statement = ""

    for turn_num in range(1, args.turns + 1):
        print(f"\n{'═' * BANNER_WIDTH}")
        print(f"  EXCHANGE {turn_num} of {args.turns}")
        print(f"{'═' * BANNER_WIDTH}")

        if turn_num == 1:
            trump_context      = f"The topic tonight is {args.topic}."
            trump_opponent_tag = "[MODERATOR]:"
        else:
            trump_context      = last_biden_statement
            trump_opponent_tag = "[BIDEN]:"

        trump_response = generate_response(
            trump_model, trump_tok, trump_dev,
            speaker_tag  = "[TRUMP]:",
            opponent_tag = trump_opponent_tag,
            context      = trump_context,
            max_new      = args.max_new,
            temperature  = args.temperature,
            top_p        = args.top_p,
        )
        print_banner("TRUMP", trump_response)
        speak_fn(trump_response, "trump")
        last_trump_statement = trump_response
        time.sleep(1)

        print_banner("MODERATOR", MODERATOR_BIDEN_PROMPT)
        speak_fn(MODERATOR_BIDEN_PROMPT, "moderator")
        time.sleep(0.5)

        biden_response = generate_response(
            biden_model, biden_tok, biden_dev,
            speaker_tag  = "[BIDEN]:",
            opponent_tag = "[TRUMP]:",
            context      = last_trump_statement,
            max_new      = args.max_new,
            temperature  = args.temperature,
            top_p        = args.top_p,
        )
        print_banner("BIDEN", biden_response)
        speak_fn(biden_response, "biden")
        last_biden_statement = biden_response
        time.sleep(1)

        if turn_num < args.turns:
            print_banner("MODERATOR", MODERATOR_TRUMP_PROMPT)
            speak_fn(MODERATOR_TRUMP_PROMPT, "moderator")
            time.sleep(0.5)

    print_banner("MODERATOR", MODERATOR_CLOSING)
    speak_fn(MODERATOR_CLOSING, "moderator")
    print("\n" + "=" * BANNER_WIDTH)
    print("  Debate complete.")
    print("=" * BANNER_WIDTH)

def run_single_mode(args, speak_fn, listen_fn):
    my_persona  = args.mode
    opp_persona = "biden" if my_persona == "trump" else "trump"
    my_tag      = f"[{my_persona.upper()}]:"
    opp_tag     = f"[{opp_persona.upper()}]:"
    model_dir   = args.trump_model if my_persona == "trump" else args.biden_model

    print("\n" + "=" * BANNER_WIDTH)
    print(f"  Running as:  {my_persona.upper()}")
    print(f"  Topic:       {args.topic}")
    print(f"  Turns:       {args.turns}")
    print("=" * BANNER_WIDTH)

    model, tokenizer, device = load_model_and_tokenizer(model_dir, my_persona)
    is_trump = (my_persona == "trump")

    for turn_num in range(1, args.turns + 1):
        print(f"\n{'═' * BANNER_WIDTH}")
        print(f"  EXCHANGE {turn_num} of {args.turns}")
        print(f"{'═' * BANNER_WIDTH}")

        if is_trump:
            if turn_num == 1:
                opponent_text  = f"The topic tonight is {args.topic}."
                opponent_label = "MODERATOR"
            else:
                print(f"\n  [Waiting for {opp_persona.upper()} to speak…]")
                opponent_text  = listen_fn(prompt=f"Listening for {opp_persona.upper()}…")
                opponent_label = opp_persona.upper()

            print_banner(f"{opponent_label} (heard)", opponent_text)

            my_response = generate_response(
                model, tokenizer, device,
                speaker_tag  = my_tag,
                opponent_tag = opp_tag if turn_num > 1 else "[MODERATOR]:",
                context      = opponent_text,
                max_new      = args.max_new,
                temperature  = args.temperature,
                top_p        = args.top_p,
            )
            print_banner(my_persona.upper(), my_response)
            speak_fn(my_response, my_persona)

        else:
            print(f"\n  [Waiting for {opp_persona.upper()} to speak…]")
            opponent_text = listen_fn(prompt=f"Listening for {opp_persona.upper()}…")
            print_banner(f"{opp_persona.upper()} (heard)", opponent_text)

            my_response = generate_response(
                model, tokenizer, device,
                speaker_tag  = my_tag,
                opponent_tag = opp_tag,
                context      = opponent_text,
                max_new      = args.max_new,
                temperature  = args.temperature,
                top_p        = args.top_p,
            )
            print_banner(my_persona.upper(), my_response)
            speak_fn(my_response, my_persona)

        time.sleep(1)

    print("\n" + "=" * BANNER_WIDTH)
    print("  Debate complete.")
    print("=" * BANNER_WIDTH)

def main():
    args      = parse_args()
    speak_fn  = build_speak_fn(args.tts)
    listen_fn = build_listen_fn(args.stt)

    if args.mode == "both":
        run_both_mode(args, speak_fn)
    else:
        run_single_mode(args, speak_fn, listen_fn)

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
    sys.exit(0)
