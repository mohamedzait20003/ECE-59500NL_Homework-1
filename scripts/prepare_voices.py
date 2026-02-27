import os
import av
import sys
import yt_dlp
import argparse
import tempfile
import numpy as np
from math import gcd
import soundfile as sf
from scipy.signal import resample_poly, butter, sosfilt


# Global constants

TARGET_SR  = 22050
VOICES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "voices")

# Sources 

SOURCES = {
    "biden": {
        "urls": [
            "https://www.youtube.com/watch?v=GSb2FSBGQLY",
            "https://www.youtube.com/watch?v=N2YnF3BsKYI",
            "https://www.youtube.com/watch?v=4xbBQfFkuOM",

            "ytsearch1:Biden farewell address January 2025 full speech",
            "ytsearch1:Joe Biden farewell speech 2025 solo",
        ],
        "start": 60,
        "duration": 30,
    },
    "trump": {
        "urls": [
            "https://www.youtube.com/watch?v=BXsHGpFGb7g",
            "https://www.youtube.com/watch?v=XsNGWS-0Dvo",
            "https://www.youtube.com/watch?v=7dUP2p9_XZo",

            "ytsearch1:Trump inauguration speech January 20 2025 full",
            "ytsearch1:Donald Trump inaugural address 2025 solo speech",
        ],
        "start": 120,
        "duration": 30,
    },
}


# Handle audio processing: mono, resample, bandpass, normalize

def _to_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim > 1:
        return data.mean(axis=1)
    return data


def _resample(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return data
    g = gcd(orig_sr, target_sr)
    return resample_poly(data, target_sr // g, orig_sr // g).astype(np.float32)


def _bandpass(data: np.ndarray, sr: int,
              low_hz: int = 80, high_hz: int = 8000) -> np.ndarray:
    nyq = sr / 2
    sos = butter(5, [low_hz / nyq, min(high_hz / nyq, 0.999)], btype="band", output="sos")
    return sosfilt(sos, data).astype(np.float32)


def _normalize(data: np.ndarray) -> np.ndarray:
    peak = np.abs(data).max()
    return (data / peak) if peak > 0 else data


def process_audio(data: np.ndarray, orig_sr: int) -> np.ndarray:
    """mono → resample → bandpass → normalize"""
    data = _to_mono(data)
    data = _resample(data, orig_sr, TARGET_SR)
    data = _bandpass(data, TARGET_SR)
    data = _normalize(data)
    return data


# Handle Decoding audio files

def decode_audio_file(path: str) -> tuple[np.ndarray, int]:
    """Decode any audio file to float32 numpy array using PyAV."""
    container = av.open(path)
    stream = container.streams.audio[0]
    sr = stream.rate or 44100

    frames = []
    for frame in container.decode(audio=0):
        arr = frame.to_ndarray().T
        frames.append(arr)
    container.close()

    data = np.concatenate(frames, axis=0).astype(np.float32)

    if data.max() > 1.0 or data.min() < -1.0:
        data = data / max(abs(data.max()), abs(data.min()))
    return data, sr


# Handle Downloading and preparing reference clips from YouTube (yt-dlp backend)

def _ytdlp_download(url: str, out_path: str) -> None:
    """Download best audio from *url* to *out_path* using yt-dlp."""
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_path,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 30,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def download_and_extract(persona: str, urls: list, start_sec: int, duration: int) -> str:
    """Try each URL/search query in order until one succeeds."""
    os.makedirs(VOICES_DIR, exist_ok=True)
    out_path = os.path.join(VOICES_DIR, f"{persona}_reference.wav")

    last_error = None
    data, orig_sr = None, None

    for i, url in enumerate(urls, 1):
        label = url if not url.startswith("ytsearch") else f"[search] {url[10:]}"
        print(f"\n[Attempt {i}/{len(urls)}] {persona.upper()} — {label}")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                dl_path = os.path.join(tmpdir, "audio")
                print("  Downloading...")
                _ytdlp_download(url, dl_path)

                candidates = [
                    f for f in os.listdir(tmpdir)
                    if os.path.isfile(os.path.join(tmpdir, f))
                ]

                if not candidates:
                    raise FileNotFoundError("yt-dlp produced no output file.")
                
                actual = os.path.join(tmpdir, candidates[0])
                size_mb = os.path.getsize(actual) / 1e6
                print(f"  Downloaded : {size_mb:.1f} MB  ({candidates[0]})")
                print("  Decoding and processing...")
                data, orig_sr = decode_audio_file(actual)

            print(f"  [OK] Got {len(data)/orig_sr:.0f}s of audio at {orig_sr} Hz")
            break 

        except Exception as e:
            last_error = e
            print(f"  [SKIP] {e}")
            continue
    else:
        print(f"\n[WARNING] All {len(urls)} sources failed for {persona.upper()}.")
        print(f"  Last error: {last_error}")
        print(f"\n  Network access to YouTube is unavailable (e.g. on an HPC cluster).")
        print(f"  Generating a silent placeholder WAV so the pipeline can continue.")
        print(f"  Voice cloning will fall back to the default TTS voice.")
        print(f"\n  To supply a real clip later, run:")
        print(f"    python scripts/prepare_voices.py --persona {persona} --mode import --file /path/to/clip.wav")

        # ── Synthetic silent placeholder ──────────────────────────────────────
        os.makedirs(VOICES_DIR, exist_ok=True)
        silence = np.zeros(int(30 * TARGET_SR), dtype=np.float32)
        sf.write(out_path, silence, TARGET_SR)
        print(f"  [PLACEHOLDER] Saved silent reference: {out_path}")
        return out_path

    # Extract time slice -------------------------------------------------------
    start_sample = int(start_sec * orig_sr)
    end_sample   = int((start_sec + duration) * orig_sr)

    if start_sample >= len(data):
        print(f"[WARNING] Start ({start_sec}s) is beyond audio length ({len(data)/orig_sr:.0f}s).")
        print(f"          Using last {duration}s of audio instead.")
        start_sample = max(0, len(data) - duration * orig_sr)
        end_sample   = len(data)

    data = data[start_sample:end_sample]
    data = process_audio(data, orig_sr)

    max_samples = 30 * TARGET_SR
    if len(data) > max_samples:
        data = data[:max_samples]

    sf.write(out_path, data, TARGET_SR)

    final_dur = len(data) / TARGET_SR
    print(f"\n  [OK] Saved: {out_path}")
    print(f"       Duration: {final_dur:.1f}s | Sample rate: {TARGET_SR} Hz | Mono")
    return out_path


def import_clip(persona: str, src_file: str, start_sec: int, duration: int) -> str:
    """Import a locally supplied audio file as the reference clip for *persona*."""
    if not os.path.isfile(src_file):
        print(f"[ERROR] File not found: {src_file}")
        sys.exit(1)

    os.makedirs(VOICES_DIR, exist_ok=True)
    out_path = os.path.join(VOICES_DIR, f"{persona}_reference.wav")

    print(f"\n[Import] {persona.upper()} — {src_file}")
    print("  Decoding and processing...")
    data, orig_sr = decode_audio_file(src_file)
    print(f"  Source  : {len(data)/orig_sr:.0f}s at {orig_sr} Hz")

    start_sample = int(start_sec * orig_sr)
    end_sample   = int((start_sec + duration) * orig_sr)

    if start_sample >= len(data):
        start_sample = 0
        end_sample   = len(data)

    data = data[start_sample:end_sample]
    data = process_audio(data, orig_sr)

    max_samples = 30 * TARGET_SR
    if len(data) > max_samples:
        data = data[:max_samples]

    sf.write(out_path, data, TARGET_SR)
    final_dur = len(data) / TARGET_SR
    print(f"  [OK] Saved: {out_path}")
    print(f"       Duration: {final_dur:.1f}s | Sample rate: {TARGET_SR} Hz | Mono")
    return out_path

# Handle validating the extracted clips (basic sanity checks)

def validate_clip(path: str):
    """Quick sanity check on the reference WAV file."""
    data, sr = sf.read(path)
    duration = len(data) / sr
    print(f"\n[Validation] {os.path.basename(path)}")
    print(f"  Sample rate : {sr} Hz")
    print(f"  Duration    : {duration:.1f} seconds")
    print(f"  Channels    : {'Mono' if data.ndim == 1 else 'Stereo'}")

    if duration < 6:
        print("  [WARNING] Clip < 6s — XTTS-v2 needs at least 6s for good cloning.")
    elif duration > 30:
        print("  [WARNING] Clip > 30s — consider trimming for faster inference.")
    else:
        print("  [OK] Duration looks good for XTTS-v2 cloning.")


# Main function to orchestrate the preparation of reference clips

def main():
    parser = argparse.ArgumentParser(
        description="Download (or import) voice reference clips for XTTS-v2 voice cloning."
    )
    parser.add_argument(
        "--persona", choices=["biden", "trump", "both"], default="both",
        help="Which persona to prepare (default: both)."
    )
    parser.add_argument(
        "--mode", choices=["auto", "import"], default="auto",
        help=(
            "auto   — download automatically from YouTube (default).\n"
            "import — import a manually downloaded audio file (use with --file)."
        ),
    )
    parser.add_argument(
        "--file", default=None,
        help="Path to a local audio file when --mode import is used.",
    )
    parser.add_argument(
        "--start", type=int, default=None,
        help="Start time in seconds (overrides default per-persona offset)."
    )
    parser.add_argument(
        "--duration", type=int, default=None,
        help="Clip duration in seconds (default: 30)."
    )
    args = parser.parse_args()

    # --- import mode ----------------------------------------------------------
    if args.mode == "import":
        if args.persona == "both":
            parser.error("--mode import requires a specific --persona (biden or trump).")
        if not args.file:
            parser.error("--mode import requires --file <path>.")
        cfg      = SOURCES[args.persona]
        start    = args.start    if args.start    is not None else cfg["start"]
        duration = args.duration if args.duration is not None else cfg["duration"]
        out_path = import_clip(args.persona, args.file, start, duration)
        validate_clip(out_path)
        print(f"\n[OK] Reference clip ready: {os.path.abspath(out_path)}")
        return

    # --- auto mode (YouTube download) ----------------------------------------
    personas = ["biden", "trump"] if args.persona == "both" else [args.persona]

    for persona in personas:
        cfg      = SOURCES[persona]
        start    = args.start    if args.start    is not None else cfg["start"]
        duration = args.duration if args.duration is not None else cfg["duration"]

        out_path = download_and_extract(persona, cfg["urls"], start, duration)
        validate_clip(out_path)

    print(f"\n[OK] All reference clips ready in {os.path.abspath(VOICES_DIR)}")
    print("     Run debate.py to start the debate with voice cloning.")


if __name__ == "__main__":
    main()