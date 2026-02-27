import os
import sys
import tempfile
import asyncio

# ── Fix coqui-tts <-> transformers incompatibility ────────────────────────────
# transformers >=4.40 removed `isin_mps_friendly`; XTTS-v2 still imports it.
# Monkey-patch it back BEFORE any TTS module is imported.
try:
    import torch as _pt
    import transformers.pytorch_utils as _tpu
    if not hasattr(_tpu, "isin_mps_friendly"):
        def _isin_mps_friendly(elements, test_elements, **kwargs):
            if elements.device.type == "mps":
                return _pt.isin(elements.cpu(), test_elements.cpu(), **kwargs).to(elements.device)
            return _pt.isin(elements, test_elements, **kwargs)
        _tpu.isin_mps_friendly = _isin_mps_friendly
except Exception:
    pass

# ── Optional imports ───────────────────────────────────────────────────────────

try:
    import torch; _torch_ok = True
except ImportError:
    torch = None; _torch_ok = False

try:
    import soundfile as sf
    import sounddevice as sd
    _sd_ok = True
except ImportError:
    sf = sd = None; _sd_ok = False

# ── Patch torchaudio to use soundfile instead of torchcodec ───────────────────
# torchcodec requires FFmpeg DLLs on Windows.  We already have soundfile which
# handles WAV natively — patch load/save so coqui-tts never triggers torchcodec.
try:
    import torchaudio as _ta
    import soundfile as _sf2
    import torch as _tc2
    import numpy as _np

    def _sf_load(filepath, *args, **kwargs):
        data, sr = _sf2.read(str(filepath), always_2d=True)
        return _tc2.from_numpy(data.T.copy()).float(), sr

    def _sf_save(filepath, src, sample_rate, *args, **kwargs):
        arr = src.squeeze().detach().cpu().numpy()
        if arr.ndim == 1:
            _sf2.write(str(filepath), arr, sample_rate)
        else:
            _sf2.write(str(filepath), arr.T, sample_rate)

    _ta.load = _sf_load
    _ta.save = _sf_save
except Exception:
    pass

try:
    from TTS.api import TTS
    _coqui_ok = True
except Exception:
    TTS = None; _coqui_ok = False

try:
    import edge_tts; _edge_ok = True
except ImportError:
    edge_tts = None; _edge_ok = False

# ── Paths & reference voices ──────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
VOICES_DIR = os.getenv(
    "VOICES_DIR",
    os.path.join(_HERE, "..", "data", "voices"),
)

REFERENCE_WAVS = {
    "biden": os.path.join(VOICES_DIR, "biden_reference.wav"),
    "trump": os.path.join(VOICES_DIR, "trump_reference.wav"),
}

_EDGE_VOICES = {
    "biden":     "en-US-GuyNeural",
    "trump":     "en-US-RogerNeural",
    "moderator": "en-US-JennyNeural",
}

# ── Coqui XTTS-v2 (lazy-loaded, real voice cloning) ──────────────────────────

_xtts_model = None


def _load_xtts() -> bool:
    global _xtts_model
    if _xtts_model is not None:
        return True
    if not (_coqui_ok and _torch_ok):
        return False
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(
            f"[voice_synth] Loading XTTS-v2 on {device} "
            "(first run downloads ~2 GB) …",
            file=sys.stderr,
        )
        _xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        print("[voice_synth] XTTS-v2 ready — voice cloning active.", file=sys.stderr)
        return True
    except Exception as exc:
        print(f"[voice_synth] XTTS-v2 load failed: {exc}", file=sys.stderr)
        return False


def _xtts_speak(text: str, ref_wav: str) -> None:
    """Clone voice from ref_wav and play immediately."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out = tmp.name; tmp.close()
    try:
        _xtts_model.tts_to_file(
            text=text, speaker_wav=ref_wav,
            language="en", file_path=out, split_sentences=True,
        )
        _play_wav(out)
    finally:
        try: os.remove(out)
        except OSError: pass


# ── edge-tts fallback ────────────────────────────────────────────────────────

def _edge_speak(text: str, persona: str) -> None:
    voice = _EDGE_VOICES.get(persona, "en-US-GuyNeural")
    tmp   = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, prefix=f"{persona}_")
    out   = tmp.name; tmp.close()

    async def _run():
        await edge_tts.Communicate(text, voice).save(out)

    try:
        asyncio.run(_run())
    except RuntimeError:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(asyncio.run, _run()).result()
    try:
        _play_any(out)
    finally:
        try: os.remove(out)
        except OSError: pass


# ── Public API ────────────────────────────────────────────────────────────────

def speak(text: str, persona: str = "biden") -> None:
    """
    Speak *text* in *persona*'s voice.  Blocks until audio finishes.

    1. Coqui XTTS-v2  – real voice cloning from data/voices/*_reference.wav
    2. edge-tts        – Microsoft Neural TTS fallback (no cloning)
    """
    text    = text.strip()
    persona = persona.lower()
    if not text:
        return

    ref_wav   = REFERENCE_WAVS.get(persona)
    can_clone = bool(ref_wav and os.path.isfile(ref_wav))

    if can_clone and _load_xtts():
        try:
            _xtts_speak(text, ref_wav)
            return
        except Exception as exc:
            print(f"[voice_synth] XTTS-v2 error: {exc}", file=sys.stderr)

    if _edge_ok:
        _edge_speak(text, persona)
        return

    print(f"[voice_synth] No TTS backend available for '{persona}'.", file=sys.stderr)


# ── Playback helpers ──────────────────────────────────────────────────────────

def _play_wav(path: str) -> None:
    """Play a WAV file via sounddevice (pure Python, no OS deps)."""
    if _sd_ok:
        try:
            data, rate = sf.read(path, dtype="float32")
            sd.play(data, rate); sd.wait()
            return
        except Exception as exc:
            print(f"[voice_synth] sounddevice error: {exc}", file=sys.stderr)
    _play_any(path)


def _play_any(path: str) -> None:
    """Play wav or mp3 via Windows MCI (built-in ctypes, no ffmpeg needed)."""
    if sys.platform != "win32":
        print(f"[voice_synth] Saved to: {path}", file=sys.stderr)
        return
    try:
        import ctypes
        mm  = ctypes.windll.winmm
        p   = os.path.abspath(path).replace("\\", "\\\\")
        ali = "debate_audio"
        mm.mciSendStringW(f"close {ali}", None, 0, 0)
        if mm.mciSendStringW(f'open "{p}" type mpegvideo alias {ali}', None, 0, 0) == 0:
            mm.mciSendStringW(f"play {ali} wait", None, 0, 0)
            mm.mciSendStringW(f"close {ali}",     None, 0, 0)
            return
    except Exception as exc:
        print(f"[voice_synth] MCI error: {exc}", file=sys.stderr)
    if not path.lower().endswith(".mp3"):
        try:
            import winsound; winsound.PlaySound(path, winsound.SND_FILENAME)
        except Exception:
            pass
