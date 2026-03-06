import os
import sys
import time
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
    """Return a configured SpeechConfig with common settings."""
    cfg = _sdk.SpeechConfig(subscription=AZURE_KEY, region=AZURE_REGION)
    cfg.speech_recognition_language = "en-US"
    # Disable profanity filtering so synthesised speech is never silenced.
    try:
        cfg.set_profanity(_sdk.ProfanityOption.Raw)
    except Exception:
        pass
    return cfg


def _handle_result(result, source: str) -> str:
    """Extract text from an Azure recognition result or log the error."""
    if result.reason == _sdk.ResultReason.RecognizedSpeech:
        return result.text.strip()
    if result.reason == _sdk.ResultReason.NoMatch:
        print(f"[stt_utils][Azure] No speech recognised from {source}.", file=sys.stderr)
    elif result.reason == _sdk.ResultReason.Canceled:
        details = _sdk.CancellationDetails(result)
        print(
            f"[stt_utils][Azure] Cancelled ({source}): "
            f"{details.reason} – {details.error_details}",
            file=sys.stderr,
        )
    return ""


def _azure_mic(timeout_seconds: int) -> str:
    """Azure STT from the default microphone — continuous recognition."""
    cfg = _make_speech_config()
    # For short timeouts (≤15 s, e.g. Q&A silence detection) use single-shot
    if timeout_seconds <= 15:
        cfg.set_property(
            _sdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,
            str(timeout_seconds * 1000),
        )
        audio_cfg = _sdk.audio.AudioConfig(use_default_microphone=True)
        recognizer = _sdk.SpeechRecognizer(speech_config=cfg, audio_config=audio_cfg)
        print("[stt_utils] Listening via Azure (microphone, single-shot)…", file=sys.stderr)
        result = recognizer.recognize_once_async().get()
        return _handle_result(result, "microphone")

    # ── Continuous recognition with auto-retry ────────────────────────
    # The Azure session sometimes terminates early (EndOfStream on
    # Windows, InitialSilenceTimeout, mic-exclusive-mode, etc.).
    # We transparently re-open the session until speech is captured
    # or the overall deadline expires.
    deadline    = time.monotonic() + timeout_seconds
    all_chunks: list[str] = []
    attempt     = 0

    print("[stt_utils] Listening via Azure (microphone, continuous)…", file=sys.stderr)

    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining < 3:
            break
        attempt += 1

        sess_cfg = _make_speech_config()
        # Do NOT set InitialSilenceTimeoutMs — it prematurely cancels
        # continuous sessions in some Azure SDK builds.
        # EndSilenceTimeoutMs = 15 s tolerates pauses between TTS
        # chunks arriving from a remote speaker.
        sess_cfg.set_property(
            _sdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs,
            "15000",
        )

        audio_cfg  = _sdk.audio.AudioConfig(use_default_microphone=True)
        recognizer = _sdk.SpeechRecognizer(
            speech_config=sess_cfg, audio_config=audio_cfg,
        )

        done   = threading.Event()
        chunks: list[str] = []

        # Closures use default-arg binding so each iteration captures
        # its own *chunks* list and *done* event.
        def _on_recognised(evt, _c=chunks):
            txt = evt.result.text.strip()
            if txt:
                _c.append(txt)
                print(f"[stt_utils] \u2713 heard: {txt[:80]}", file=sys.stderr)

        def _on_recognizing(evt):
            partial = evt.result.text.strip()
            if partial:
                print(f"[stt_utils] \u2026 hearing: {partial[:60]}",
                      file=sys.stderr)

        def _on_stopped(evt, _d=done):
            _d.set()

        def _on_canceled(evt, _d=done):
            details = _sdk.CancellationDetails(evt.result)
            if details.reason == _sdk.CancellationReason.Error:
                print(
                    f"[stt_utils][Azure] Error: {details.error_details}",
                    file=sys.stderr,
                )
            _d.set()

        recognizer.recognized.connect(_on_recognised)
        recognizer.recognizing.connect(_on_recognizing)
        recognizer.session_stopped.connect(_on_stopped)
        recognizer.canceled.connect(_on_canceled)

        recognizer.start_continuous_recognition()
        done.wait(timeout=remaining + 2)
        try:
            recognizer.stop_continuous_recognition()
        except Exception:
            pass

        if chunks:
            all_chunks.extend(chunks)
            break                  # got speech — stop retrying

        # No results — brief pause, then re-open the mic session
        if time.monotonic() < deadline:
            print("[stt_utils] (no speech yet — restarting listener …)",
                  file=sys.stderr)
            time.sleep(0.5)

    text = " ".join(all_chunks).strip()
    if not text:
        print("[stt_utils][Azure] No speech recognised from microphone.",
              file=sys.stderr)
    return text


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

    recognizer.recognized.connect(on_recognised)
    recognizer.session_stopped.connect(on_stopped)
    recognizer.canceled.connect(on_stopped)

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
