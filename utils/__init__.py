"""utils package — Public re-exports for all helpers used by the debate scripts."""

# TTS / STT
from .voice_utils import speak
from .stt_utils import transcribe_from_microphone, transcribe_from_file

# Debate helpers
from .debate_utils import (
    generate_response,
    banner,
    load_model_and_tokenizer,
    build_bad_word_ids,
    determine_qa_target,
    WIDTH,
    DEFAULT_TOPIC,
    DEFAULT_TURNS,
    DEFAULT_TRUMP_DIR,
    DEFAULT_BIDEN_DIR,
)

# Redis sync (optional — only used with --sync redis)
try:
    from .redis_sync import DebateSync
except ImportError:
    DebateSync = None

__all__ = [
    # TTS / STT
    "speak",
    "transcribe_from_microphone",
    "transcribe_from_file",

    # Debate helpers
    "generate_response",
    "banner",
    "load_model_and_tokenizer",
    "build_bad_word_ids",
    "determine_qa_target",
    "WIDTH",
    "DEFAULT_TOPIC",
    "DEFAULT_TURNS",
    "DEFAULT_TRUMP_DIR",
    "DEFAULT_BIDEN_DIR",
    
    # Redis sync
    "DebateSync",
]
