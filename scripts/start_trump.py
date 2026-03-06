import os
import sys
import time
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, ROOT_DIR)

from utils import (                                                    # noqa: E402
    speak, transcribe_from_microphone,
    generate_response, banner,
    load_model_and_tokenizer, build_bad_word_ids, determine_qa_target,
    WIDTH, DEFAULT_TOPIC, DEFAULT_TURNS, DEFAULT_TRUMP_DIR, DEFAULT_BIDEN_DIR,
    DebateSync,
)

PERSONA  = "trump"
OPPONENT = "biden"

# ══════════════════════════════════════════════════════════════════════════════
# Debate exchange loops
# ══════════════════════════════════════════════════════════════════════════════

def _run_speak_mode(model, tok, dev, bad_ids,
                    topic, turns, max_new, temperature, top_p):
    """
    Speak mode — this terminal acts as the moderator host.
    Flow: moderator intro → Trump speaks → mic-listen for Biden → repeat.
    No filesystem sync — communication is purely through audio.
    """
    intro = (
        f"Welcome to the Biden versus Trump AI Debate. "
        f"Tonight's topic is {topic}. "
        f"We will have {turns} exchange{'s' if turns != 1 else ''}. "
        f"Mr. {PERSONA.title()}, you have the opening statement."
    )
    banner("MODERATOR", intro)
    speak(intro, "moderator")
    time.sleep(1.0)

    own_prev = ""
    opp_prev = ""

    for turn in range(1, turns + 1):
        print(f"\n{'═' * WIDTH}")
        print(f"  Exchange {turn} of {turns}")
        print(f"{'═' * WIDTH}")

        response = generate_response(
            model, tok, dev, PERSONA, topic,
            opponent_text=opp_prev, own_prev_text=own_prev,
            max_new=max_new, temperature=temperature, top_p=top_p,
            bad_word_ids=bad_ids,
        )
        banner(PERSONA.upper(), response)
        speak(response, PERSONA)
        own_prev = response

        print(f"\n  [mic] Listening for {OPPONENT.upper()}'s response …")
        opp_text = transcribe_from_microphone(timeout_seconds=120)
        if not opp_text:
            opp_text = "(No response captured via microphone.)"
        banner(f"{OPPONENT.upper()} (heard)", opp_text)
        opp_prev = opp_text

    _run_qa(model, tok, dev, bad_ids, topic, max_new, temperature, top_p,
            is_host=True)


def _run_listen_mode(model, tok, dev, bad_ids,
                     topic, turns, max_new, temperature, top_p):
    """
    Listen mode — mic-listen for the opponent's speak-mode terminal.
    Flow: hear moderator intro → hear opponent → Trump responds → repeat.
    No filesystem sync — communication is purely through audio.
    """
    print(f"\n  [mic] Listening for moderator intro from "
          f"{OPPONENT.upper()}'s terminal (up to 5 min) …")
    mod_text = transcribe_from_microphone(timeout_seconds=300)
    if mod_text:
        banner("MODERATOR (heard)", mod_text)
    else:
        print("  (No moderator audio captured — continuing anyway.)")

    own_prev = ""
    opp_prev = ""

    for turn in range(1, turns + 1):
        print(f"\n{'═' * WIDTH}")
        print(f"  Exchange {turn} of {turns}")
        print(f"{'═' * WIDTH}")

        print(f"\n  [mic] Listening for {OPPONENT.upper()}'s response …")
        opp_text = transcribe_from_microphone(timeout_seconds=120)
        if not opp_text:
            opp_text = "(No response captured via microphone.)"
        banner(f"{OPPONENT.upper()} (heard)", opp_text)
        opp_prev = opp_text

        response = generate_response(
            model, tok, dev, PERSONA, topic,
            opponent_text=opp_prev, own_prev_text=own_prev,
            max_new=max_new, temperature=temperature, top_p=top_p,
            bad_word_ids=bad_ids,
        )
        banner(PERSONA.upper(), response)
        speak(response, PERSONA)
        own_prev = response

    _run_qa(model, tok, dev, bad_ids, topic, max_new, temperature, top_p,
            is_host=False)


# ══════════════════════════════════════════════════════════════════════════════
# Redis-synchronised debate exchange loops
# ══════════════════════════════════════════════════════════════════════════════

def _run_redis_speak_mode(model, tok, dev, bad_ids,
                          topic, turns, max_new, temperature, top_p,
                          sync: "DebateSync"):
    """
    Redis speak mode — this terminal is the moderator host.
    Text is exchanged over Redis; TTS still plays locally for the audience.
    """
    intro = (
        f"Welcome to the Biden versus Trump AI Debate. "
        f"Tonight's topic is {topic}. "
        f"We will have {turns} exchange{'s' if turns != 1 else ''}. "
        f"Mr. {PERSONA.title()}, you have the opening statement."
    )
    banner("MODERATOR", intro)
    speak(intro, "moderator")
    sync.send(intro, msg_type="moderator")
    time.sleep(1.0)

    own_prev = ""
    opp_prev = ""

    for turn in range(1, turns + 1):
        print(f"\n{'═' * WIDTH}")
        print(f"  Exchange {turn} of {turns}")
        print(f"{'═' * WIDTH}")

        response = generate_response(
            model, tok, dev, PERSONA, topic,
            opponent_text=opp_prev, own_prev_text=own_prev,
            max_new=max_new, temperature=temperature, top_p=top_p,
            bad_word_ids=bad_ids,
        )
        banner(PERSONA.upper(), response)
        speak(response, PERSONA)
        sync.send(response, msg_type="response")
        own_prev = response

        print(f"\n  [redis] Waiting for {OPPONENT.upper()}'s response …")
        msg_type, opp_text = sync.receive(timeout=180)
        if msg_type == "timeout" or not opp_text:
            opp_text = "(No response received from opponent.)"
        banner(f"{OPPONENT.upper()} (via Redis)", opp_text)
        opp_prev = opp_text

    _run_qa_redis(model, tok, dev, bad_ids, topic, max_new, temperature,
                  top_p, sync, is_host=True)


def _run_redis_listen_mode(model, tok, dev, bad_ids,
                           topic, turns, max_new, temperature, top_p,
                           sync: "DebateSync"):
    """
    Redis listen mode — wait for opponent via Redis, then respond.
    """
    print(f"\n  [redis] Waiting for moderator intro from "
          f"{OPPONENT.upper()}'s terminal …")
    msg_type, mod_text = sync.receive(timeout=300)
    if mod_text:
        banner("MODERATOR (via Redis)", mod_text)

    own_prev = ""
    opp_prev = ""

    for turn in range(1, turns + 1):
        print(f"\n{'═' * WIDTH}")
        print(f"  Exchange {turn} of {turns}")
        print(f"{'═' * WIDTH}")

        print(f"\n  [redis] Waiting for {OPPONENT.upper()}'s response …")
        msg_type, opp_text = sync.receive(timeout=180)
        if msg_type == "timeout" or not opp_text:
            opp_text = "(No response received from opponent.)"
        banner(f"{OPPONENT.upper()} (via Redis)", opp_text)
        opp_prev = opp_text

        response = generate_response(
            model, tok, dev, PERSONA, topic,
            opponent_text=opp_prev, own_prev_text=own_prev,
            max_new=max_new, temperature=temperature, top_p=top_p,
            bad_word_ids=bad_ids,
        )
        banner(PERSONA.upper(), response)
        speak(response, PERSONA)
        sync.send(response, msg_type="response")
        own_prev = response

    _run_qa_redis(model, tok, dev, bad_ids, topic, max_new, temperature,
                  top_p, sync, is_host=False)


def _run_qa_redis(model, tok, dev, bad_ids,
                  topic, max_new, temperature, top_p,
                  sync: "DebateSync", is_host: bool):
    """
    Q&A phase over Redis — audience questions still come from the local mic;
    answers are exchanged via Redis so both sides stay in sync.
    """
    if is_host:
        qa_intro = (
            "We have now concluded the main exchanges. "
            "The floor is open to audience questions. "
            "Please ask your question."
        )
        banner("MODERATOR", qa_intro)
        speak(qa_intro, "moderator")
        sync.send(qa_intro, msg_type="moderator")
    else:
        msg_type, qa_text = sync.receive(timeout=60)
        if qa_text:
            banner("MODERATOR (via Redis)", qa_text)
        print(f"\n  [Q&A] {PERSONA.upper()} is ready for audience questions.")

    while True:
        print(f"\n  [Q&A] [mic] Listening for audience question "
              f"(15 s silence = end) …")
        question = transcribe_from_microphone(timeout_seconds=15)

        if not question:
            if is_host:
                end_msg = (
                    "As there are no further questions, this concludes "
                    "tonight's debate. Thank you all for watching."
                )
                banner("MODERATOR", end_msg)
                speak(end_msg, "moderator")
                sync.send("__END_QA__", msg_type="control")
            print(f"\n  [Q&A] No further questions — "
                  f"{PERSONA.upper()} Q&A done.")
            break

        banner("AUDIENCE QUESTION", question)
        target = determine_qa_target(question)
        print(f"  [Q&A] Question directed at: {target.upper()}")

        if target in (PERSONA, "both"):
            answer = generate_response(
                model, tok, dev, PERSONA, question,
                opponent_text="", own_prev_text="",
                max_new=max_new, temperature=temperature, top_p=top_p,
                bad_word_ids=bad_ids,
            )
            banner(f"{PERSONA.upper()} [answer]", answer)
            speak(answer, PERSONA)
        else:
            print(f"  [Q&A] Question is for {OPPONENT.upper()} — skipping.")

        if is_host:
            more_msg = ("Thank you. Are there any more questions "
                        "from the audience?")
            banner("MODERATOR", more_msg)
            speak(more_msg, "moderator")


# ══════════════════════════════════════════════════════════════════════════════
# Q&A phase — each terminal handles its own persona independently
# ══════════════════════════════════════════════════════════════════════════════

def _run_qa(model, tok, dev, bad_ids,
            topic, max_new, temperature, top_p, is_host: bool):
    """
    Q&A phase — listen for audience questions via microphone.
    The host terminal announces Q&A; both terminals answer questions
    directed at their persona.  15 seconds of silence ends Q&A.
    """
    if is_host:
        qa_intro = (
            "We have now concluded the main exchanges. "
            "The floor is open to audience questions. "
            "Please ask your question."
        )
        banner("MODERATOR", qa_intro)
        speak(qa_intro, "moderator")
    else:
        print(f"\n  [Q&A] {PERSONA.upper()} is ready for audience questions.")

    while True:
        print(f"\n  [Q&A] [mic] Listening for audience question "
              f"(15 s silence = end) …")
        question = transcribe_from_microphone(timeout_seconds=15)

        if not question:
            if is_host:
                end_msg = (
                    "As there are no further questions, this concludes "
                    "tonight's debate. Thank you all for watching."
                )
                banner("MODERATOR", end_msg)
                speak(end_msg, "moderator")
            print(f"\n  [Q&A] No further questions — "
                  f"{PERSONA.upper()} Q&A done.")
            break

        banner("AUDIENCE QUESTION", question)
        target = determine_qa_target(question)
        print(f"  [Q&A] Question directed at: {target.upper()}")

        if target in (PERSONA, "both"):
            answer = generate_response(
                model, tok, dev, PERSONA, question,
                opponent_text="", own_prev_text="",
                max_new=max_new, temperature=temperature, top_p=top_p,
                bad_word_ids=bad_ids,
            )
            banner(f"{PERSONA.upper()} [answer]", answer)
            speak(answer, PERSONA)
        else:
            print(f"  [Q&A] Question is for {OPPONENT.upper()} — skipping.")

        if is_host:
            more_msg = ("Thank you. Are there any more questions "
                        "from the audience?")
            banner("MODERATOR", more_msg)
            speak(more_msg, "moderator")

# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Trump AI Debate Terminal",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--mode", type=str, default="speak",
        choices=["speak", "listen"],
        help=(
            "speak  — moderator intro + Trump speaks first each exchange\n"
            "listen — waits for Biden, then responds  (default: speak)"
        ),
    )
    p.add_argument(
        "--sync", type=str, default="redis",
        choices=["audio", "redis"],
        help=(
            "redis — Redis-based text sync (default, config via .env)\n"
            "audio — mic-to-speaker sync between laptops (fallback)"
        ),
    )
    p.add_argument(
        "--topic", type=str, default=DEFAULT_TOPIC,
        help=f"Debate topic (default: '{DEFAULT_TOPIC}')",
    )
    p.add_argument(
        "--turns", type=int, default=DEFAULT_TURNS,
        help=f"Number of exchanges (default: {DEFAULT_TURNS})",
    )
    p.add_argument("--max_new",     type=int,   default=120,
                   help="Max new tokens per response (default: 120)")
    p.add_argument("--temperature", type=float, default=0.82,
                   help="Sampling temperature (default: 0.82)")
    p.add_argument("--top_p",       type=float, default=0.90,
                   help="Nucleus sampling p (default: 0.90)")
    p.add_argument("--trump_model", type=str, default=DEFAULT_TRUMP_DIR,
                   help="Path to fine-tuned Trump model directory")
    p.add_argument("--biden_model", type=str, default=DEFAULT_BIDEN_DIR,
                   help="Path to fine-tuned Biden model directory")
    return p.parse_args()

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run_debate(args) -> None:
    """Entry point for the Trump debate terminal."""
    model_dir = args.trump_model

    print(f"\n{'=' * WIDTH}")
    print(f"  {PERSONA.upper()} Debate Terminal  [{args.mode} mode]  [sync: {args.sync}]")
    print(f"  Topic : {args.topic}")
    print(f"  Turns : {args.turns}")
    print(f"{'=' * WIDTH}")

    print("\n  Loading model …")
    model, tok, dev = load_model_and_tokenizer(model_dir, PERSONA)
    bad_ids = build_bad_word_ids(tok)

    # ── Redis sync path ───────────────────────────────────────────────
    if args.sync == "redis":
        if DebateSync is None:
            print("  ERROR: 'redis' package not installed. "
                  "Run:  pip install redis")
            return
        sync = DebateSync(persona=PERSONA)
        print(f"\n  Waiting for {OPPONENT.upper()} to connect …")
        if not sync.wait_for_opponent(timeout=300):
            print("  Opponent did not connect in time. Exiting.")
            sync.close()
            return

        if args.mode == "speak":
            _run_redis_speak_mode(
                model, tok, dev, bad_ids,
                args.topic, args.turns,
                args.max_new, args.temperature, args.top_p,
                sync,
            )
        else:
            _run_redis_listen_mode(
                model, tok, dev, bad_ids,
                args.topic, args.turns,
                args.max_new, args.temperature, args.top_p,
                sync,
            )
        sync.flush_session()
        sync.close()

    # ── Original audio sync path ──────────────────────────────────────
    else:
        if args.mode == "speak":
            _run_speak_mode(
                model, tok, dev, bad_ids,
                args.topic, args.turns,
                args.max_new, args.temperature, args.top_p,
            )
        else:
            _run_listen_mode(
                model, tok, dev, bad_ids,
                args.topic, args.turns,
                args.max_new, args.temperature, args.top_p,
            )

    print(f"\n{'=' * WIDTH}")
    print(f"  {PERSONA.upper()} terminal — debate complete.")
    print(f"{'=' * WIDTH}")


def main():
    args = parse_args()
    run_debate(args)


if __name__ == "__main__":
    try:
        main()
    except (SystemExit, KeyboardInterrupt):
        print("\n  Trump terminal interrupted.")
    sys.exit(0)
