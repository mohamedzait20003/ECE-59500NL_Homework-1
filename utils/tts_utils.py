import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

AZURE_KEY    = os.getenv("AZURE_SPEECH_KEY",    "")
AZURE_REGION = os.getenv("AZURE_SPEECH_REGION", "eastus")

AZURE_VOICE_MAP = {
    "biden":     os.getenv("BIDEN_VOICE",     "en-US-GuyNeural"),
    "trump":     os.getenv("TRUMP_VOICE",     "en-US-DavisNeural"),
    "moderator": os.getenv("MODERATOR_VOICE", "en-US-AriaNeural"),
}

_azure_available = False
_sdk = None

if AZURE_KEY:
    try:
        import azure.cognitiveservices.speech as speechsdk
        _sdk = speechsdk
        _azure_available = True
    except ImportError:
        print("[tts_utils] azure-cognitiveservices-speech not installed; "
              "falling back to pyttsx3.", file=sys.stderr)

_pyttsx3_engine = None

def _get_pyttsx3_engine():
    global _pyttsx3_engine
    if _pyttsx3_engine is None:
        import pyttsx3
        _pyttsx3_engine = pyttsx3.init()
        _pyttsx3_engine.setProperty("rate", 165)
        _pyttsx3_engine.setProperty("volume", 1.0)
    return _pyttsx3_engine

def speak(text: str, persona: str = "biden") -> None:
    text = text.strip()
    if not text:
        return
    persona = persona.lower()
    if _azure_available:
        _speak_azure(text, persona)
    else:
        _speak_pyttsx3(text, persona)

def _speak_azure(text: str, persona: str) -> None:
    speech_config = _sdk.SpeechConfig(subscription=AZURE_KEY, region=AZURE_REGION)
    speech_config.speech_synthesis_voice_name = AZURE_VOICE_MAP.get(persona, "en-US-GuyNeural")
    synthesizer = _sdk.SpeechSynthesizer(speech_config=speech_config)
    result = synthesizer.speak_text_async(text).get()
    if result.reason == _sdk.ResultReason.Canceled:
        details = _sdk.SpeechSynthesisCancellationDetails.from_result(result)
        print(f"[tts_utils][Azure TTS] Cancelled: {details.reason} – {details.error_details}",
              file=sys.stderr)

def _speak_pyttsx3(text: str, persona: str) -> None:
    engine = _get_pyttsx3_engine()
    rates = {"biden": 155, "trump": 175, "moderator": 165}
    engine.setProperty("rate", rates.get(persona, 165))
    engine.say(text)
    engine.runAndWait()

def listen(prompt: str = "Listening…", timeout_seconds: int = 10) -> str:
    if _azure_available:
        return _listen_azure(timeout_seconds)
    return _listen_keyboard(prompt)

def _listen_azure(timeout_seconds: int) -> str:
    speech_config = _sdk.SpeechConfig(subscription=AZURE_KEY, region=AZURE_REGION)
    speech_config.set_property(
        _sdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,
        str(timeout_seconds * 1000),
    )
    recognizer = _sdk.SpeechRecognizer(speech_config=speech_config)
    result = recognizer.recognize_once_async().get()
    if result.reason == _sdk.ResultReason.RecognizedSpeech:
        return result.text.strip()
    elif result.reason == _sdk.ResultReason.Canceled:
        details = _sdk.CancellationDetails.from_result(result)
        print(f"[tts_utils][Azure STT] Cancelled: {details.reason} – {details.error_details}",
              file=sys.stderr)
    return ""

def _listen_keyboard(prompt: str) -> str:
    try:
        return input(f"{prompt} (type response): ").strip()
    except (EOFError, KeyboardInterrupt):
        return ""
