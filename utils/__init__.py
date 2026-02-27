"""
utils package
=============
Public re-exports for TTS and STT helpers used by the debate scripts.
"""

from .voice_synth_utils import speak
from .stt_utils import transcribe_from_microphone, transcribe_from_file

__all__ = [
    "speak",
    "transcribe_from_microphone",
    "transcribe_from_file",
]
