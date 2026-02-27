import os
import sys
import threading

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Azure Speech-to-Text configuration (if available)

AZURE_KEY    = os.getenv("AZURE_SPEECH_KEY",    "")
AZURE_REGION = os.getenv("AZURE_SPEECH_REGION", "eastus")

_azure_available = False
_sdk = None

if AZURE_KEY:
    try:
        import azure.cognitiveservices.speech as speechsdk
        _sdk = speechsdk
        _azure_available = True
        print("[stt_utils] Azure Speech SDK ready.", file=sys.stderr)
    except ImportError:
        print(
            "[stt_utils] azure-cognitiveservices-speech not installed; "
            "falling back to speech_recognition.",
            file=sys.stderr,
        )
else:
    print(
        "[stt_utils] AZURE_SPEECH_KEY not set; "
        "falling back to speech_recognition.",
        file=sys.stderr,
    )


# Public API

def transcribe_from_microphone(timeout_seconds: int = 10) -> str:
    if _azure_available:
        return _azure_mic(timeout_seconds)
    if sr is not None:
        return _sr_mic(timeout_seconds)
    print(
        "[stt_utils] No STT backend available (Azure SDK not configured, "
        "SpeechRecognition not installed).",
        file=sys.stderr,
    )
    return ""


def transcribe_from_file(audio_path: str) -> str:
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"[stt_utils] Audio file not found: {audio_path}")

    if _azure_available:
        return _azure_file(audio_path)
    if sr is not None:
        return _sr_file(audio_path)
    print(
        "[stt_utils] No STT backend available.",
        file=sys.stderr,
    )
    return ""


# Azure STT implementation

def _make_speech_config() -> "speechsdk.SpeechConfig":
    """Return a configured SpeechConfig with common timeout settings."""
    cfg = _sdk.SpeechConfig(subscription=AZURE_KEY, region=AZURE_REGION)
    cfg.speech_recognition_language = "en-US"
    cfg.set_property(
        _sdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,
        "5000",
    )
    cfg.set_property(
        _sdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs,
        "1500",
    )
    return cfg


def _handle_result(result, source: str) -> str:
    """Extract text from an Azure recognition result or log the error."""
    if result.reason == _sdk.ResultReason.RecognizedSpeech:
        return result.text.strip()
    if result.reason == _sdk.ResultReason.NoMatch:
        print(f"[stt_utils][Azure] No speech recognised from {source}.", file=sys.stderr)
    elif result.reason == _sdk.ResultReason.Canceled:
        details = _sdk.CancellationDetails.from_result(result)
        print(
            f"[stt_utils][Azure] Cancelled ({source}): "
            f"{details.reason} – {details.error_details}",
            file=sys.stderr,
        )
    return ""


def _azure_mic(timeout_seconds: int) -> str:
    """Azure STT from the default microphone."""
    cfg = _make_speech_config()
    cfg.set_property(
        _sdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,
        str(timeout_seconds * 1000),
    )
    audio_cfg = _sdk.audio.AudioConfig(use_default_microphone=True)
    recognizer = _sdk.SpeechRecognizer(speech_config=cfg, audio_config=audio_cfg)
    print("[stt_utils] Listening via Azure (microphone)…", file=sys.stderr)
    result = recognizer.recognize_once_async().get()
    return _handle_result(result, "microphone")


def _azure_file(audio_path: str) -> str:
    """Azure STT from a local audio file."""
    cfg = _make_speech_config()
    audio_cfg = _sdk.audio.AudioConfig(filename=audio_path)
    recognizer = _sdk.SpeechRecognizer(speech_config=cfg, audio_config=audio_cfg)
    print(f"[stt_utils] Transcribing via Azure: {audio_path}", file=sys.stderr)

    done   = threading.Event()
    chunks: list[str] = []

    def on_recognised(evt):
        if evt.result.text.strip():
            chunks.append(evt.result.text.strip())

    def on_stopped(evt):
        done.set()

    recognizer.recognized   += on_recognised
    recognizer.session_stopped += on_stopped
    recognizer.canceled       += on_stopped

    recognizer.start_continuous_recognition()
    done.wait(timeout=300)          # max 5 min for long files
    recognizer.stop_continuous_recognition()

    return " ".join(chunks)


# ── speech_recognition fallback ───────────────────────────────────────────────

def _sr_mic(timeout_seconds: int) -> str:
    """speech_recognition STT from the default microphone (Google Web Speech)."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        print("[stt_utils] Listening via speech_recognition (microphone)…", file=sys.stderr)
        try:
            audio = recognizer.listen(source, timeout=timeout_seconds, phrase_time_limit=30)
        except sr.WaitTimeoutError:
            print("[stt_utils] Microphone timeout.", file=sys.stderr)
            return ""
    try:
        return recognizer.recognize_google(audio).strip()
    except sr.UnknownValueError:
        print("[stt_utils] Could not understand audio.", file=sys.stderr)
        return ""
    except sr.RequestError as exc:
        print(f"[stt_utils] Google Speech API error: {exc}", file=sys.stderr)
        return ""


def _sr_file(audio_path: str) -> str:
    """speech_recognition STT from an audio file."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio).strip()
    except sr.UnknownValueError:
        print("[stt_utils] Could not understand audio.", file=sys.stderr)
        return ""
    except sr.RequestError as exc:
        print(f"[stt_utils] Google Speech API error: {exc}", file=sys.stderr)
        return ""


# ── CLI smoke-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="STT smoke-test")
    parser.add_argument("--file",    help="Path to an audio file to transcribe")
    parser.add_argument("--timeout", type=int, default=10,
                        help="Mic listen timeout in seconds (default 10)")
    args = parser.parse_args()

    if args.file:
        text = transcribe_from_file(args.file)
    else:
        text = transcribe_from_microphone(args.timeout)

    print(f"\nTranscription: {text!r}")
