# ECE 49595NL / ECE 59500NL — Homework 1
## Biden vs. Trump Debate Chatbot

**Course:** Introduction to Natural Language Processing — Spring 2026  
**Due:** 5:00 PM, Friday February 27, 2026  

---

## Overview

This project implements two AI chatbots that emulate **Former President Joseph R. Biden** and **President Donald J. Trump** in a live spoken debate. Each chatbot is powered by a fine-tuned **GPT-2 Medium** language model. Speech-to-text uses the **Azure Cognitive Services Speech SDK**; text-to-speech uses **Coqui XTTS-v2** for zero-shot voice cloning from real reference audio clips of each speaker.

---

## System Architecture

```
Microphone Input
      │
      ▼
Azure STT  ─── stt_utils.py
(fallback: SpeechRecognition / keyboard)
      │
      ▼
GPT-2 Medium — fine-tuned on Biden / Trump speech data
      │
      ▼
Coqui XTTS-v2 voice cloning  ─── voice_synth_utils.py
(fallback: pyttsx3 system TTS)
      │
      ▼
Speaker Output
```

**Two-laptop mode:** Laptop 1 runs the Biden chatbot, Laptop 2 runs Trump. Each listens to the other via microphone and responds.  
**Single-laptop mode:** Both personas run on one machine and alternate automatically.

---

## Project Structure

```
Code/
├── data/
│   ├── raw/
│   │   ├── biden/              # Raw Biden speech transcripts
│   │   └── trump/              # Raw Trump speech transcripts + books
│   ├── processed/
│   │   ├── biden_train.txt     # Cleaned Biden training data
│   │   └── trump_train.txt     # Cleaned Trump training data
│   └── voices/
│       ├── biden_reference.wav # 30-second Biden voice clip (for XTTS-v2)
│       └── trump_reference.wav # 30-second Trump voice clip (for XTTS-v2)
├── models/
│   ├── biden/                  # Fine-tuned Biden GPT-2 weights
│   └── trump/                  # Fine-tuned Trump GPT-2 weights
├── notebooks/
│   ├── train_biden_colab.ipynb # Google Colab training notebook (Biden)
│   └── train_trump_colab.ipynb # Google Colab training notebook (Trump)
├── scripts/
│   ├── collect_data.py         # Scrape and download speech data
│   ├── preprocess_data.py      # Clean and format training data
│   ├── prepare_voices.py       # Download & process voice reference clips
│   ├── train_biden.py          # Fine-tune GPT-2 on Biden data (local)
│   ├── train_trump.py          # Fine-tune GPT-2 on Trump data (local)
│   └── start_debate.py         # Main debate loop (STT → LLM → TTS)
├── utils/
│   ├── stt_utils.py            # Speech-to-Text (Azure / SpeechRecognition)
│   └── voice_synth_utils.py    # Text-to-Voice (Coqui XTTS-v2 / pyttsx3)
├── .env                        # API keys (NOT committed)
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

- **Windows:** `.venv\Scripts\activate`
- **macOS / Linux:** `source .venv/bin/activate`

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

Edit `.env` in the project root:

```env
AZURE_SPEECH_KEY=your_azure_speech_key_here
AZURE_SPEECH_REGION=your_azure_region_here
```

> Only the Azure Cognitive Services **Speech** key is required.  
> Azure OpenAI keys are **not** used — the project runs local GPT-2 models.  
> Keys are distributed via Purdue Filelocker by the course instructor.

### 4. Collect Training Data

```bash
python scripts/collect_data.py
```

Downloads Biden/Trump speeches and transcripts into `data/raw/`.

### 5. Preprocess Training Data

```bash
python scripts/preprocess_data.py
```

Produces `data/processed/biden_train.txt` and `data/processed/trump_train.txt`.

### 6. Prepare Voice Reference Clips

```bash
python scripts/prepare_voices.py
```

Downloads ~30-second audio clips from YouTube, processes them (mono, 22 050 Hz, bandpass, normalised), and saves:
- `data/voices/biden_reference.wav`
- `data/voices/trump_reference.wav`

These clips are used by Coqui XTTS-v2 at debate time for zero-shot voice cloning.

### 7. Fine-Tune the Language Models

#### Option A — Local (GPU recommended)

```bash
python scripts/train_biden.py
python scripts/train_trump.py
```

#### Option B — Google Colab (free T4 GPU)

1. Upload `notebooks/train_biden_colab.ipynb` or `notebooks/train_trump_colab.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. Set the runtime to **GPU (T4)**: *Runtime → Change runtime type → T4 GPU*
3. Run all cells top-to-bottom (training takes ~20–40 min on T4)
4. **Cell 9 (last cell)** automatically finds the highest-numbered `checkpoint-<step>/` folder (= last epoch), zips it into a single file, and triggers a browser download:
   - `biden_checkpoint-<step>.zip` → extract into `models/biden/`
   - `trump_checkpoint-<step>.zip` → extract into `models/trump/`

```
models/
├── biden/
│   ├── config.json
│   ├── model.safetensors      ← ~1.4 GB, the main weights file
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── ...
└── trump/
    └── ...
```

> Model weights are **not** included in this submission. Run the training scripts to replicate them.

### 8. Run the Debate

```bash
python scripts/start_debate.py [OPTIONS]
```

| Argument | Default | Description |
|---|---|---|
| `--mode` | `both` | `both` = single machine; `biden` / `trump` = this machine plays one side |
| `--topic` | `the economy` | Debate topic |
| `--turns` | `5` | Number of exchanges |
| `--max_new` | `150` | Max new tokens per response |
| `--temperature` | `0.85` | Sampling temperature |
| `--top_p` | `0.92` | Nucleus sampling p |
| `--tts` | `auto` | `auto` = Coqui XTTS-v2 voice cloning; `none` = silent |
| `--stt` | `auto` | `auto` = Azure STT → SpeechRecognition fallback; `keyboard` = type input |
| `--trump_model` | `models/trump` | Path to fine-tuned Trump model directory |
| `--biden_model` | `models/biden` | Path to fine-tuned Biden model directory |

#### Single-machine demo (both personas)

```bash
python scripts/start_debate.py --mode both --topic "immigration" --turns 4
```

#### Two-laptop setup

On the Biden laptop:
```bash
python scripts/start_debate.py --mode biden --topic "immigration"
```

On the Trump laptop:
```bash
python scripts/start_debate.py --mode trump --topic "immigration"
```

#### Text-only (no audio)

```bash
python scripts/start_debate.py --tts none --stt keyboard
```

---

## Model Details

| Property | Value |
|---|---|
| Base model | GPT-2 Medium (345 M parameters) |
| Hugging Face ID | `gpt2-medium` |
| Fine-tuning method | Full fine-tuning via Hugging Face `Trainer` |
| Epochs | 8 |
| Batch size | 8 (× gradient accumulation 2 = effective 16) |
| Learning rate | 5 × 10⁻⁵ (cosine schedule) |
| Block size | 512 tokens |
| Mixed precision | FP16 (when CUDA available) |
| Voice cloning model | Coqui XTTS-v2 (`tts_models/multilingual/multi-dataset/xtts_v2`) |
| STT backend | Azure Cognitive Services Speech SDK |

---

## Data Sources

### Trump (19 files)
| File | Source |
|---|---|
| `art_of_the_deal.txt` | PDF — https://ia601405.us.archive.org/19/items/TrumpTheArtOfTheDeal |
| `think_like_a_champion.txt` | PDF — https://www.reboxu.com/uploads/8/6/0/3/86031326/think_like_a_champion.pdf |
| `trump_debate_2020_1.txt` | UCSB Presidency Project (Rev.com fallback) — 1st 2020 Presidential Debate |
| `trump_debate_2020_2.txt` | UCSB Presidency Project (Rev.com fallback) — 2nd 2020 Presidential Debate |
| `trump_rev_speech_00` – `trump_rev_speech_14` | Rev.com speech transcripts — https://www.rev.com/blog/transcripts?s=trump+speech |

### Biden (21 files)
| File | Source |
|---|---|
| `biden_inaugural_2021.txt` | UCSB Presidency Project — Inaugural Address (Jan 20, 2021) |
| `biden_sotu_2022.txt` | UCSB Presidency Project — State of the Union 2022 |
| `biden_sotu_2023.txt` | UCSB Presidency Project — State of the Union 2023 |
| `biden_sotu_2024.txt` | UCSB Presidency Project — State of the Union 2024 |
| `biden_debate_2020_1.txt` | UCSB Presidency Project (Rev.com fallback) — 1st 2020 Presidential Debate |
| `biden_debate_2020_2.txt` | UCSB Presidency Project (Rev.com fallback) — 2nd 2020 Presidential Debate |
| `biden_rev_speech_00` – `biden_rev_speech_14` | Rev.com speech transcripts — https://www.rev.com/blog/transcripts?s=biden+speech |

### Primary Scraping Sources
- UCSB American Presidency Project: https://www.presidency.ucsb.edu/
- Rev.com Transcripts: https://www.rev.com/blog/transcripts/

---

## Code Sources

### Course Reference Implementations
| Script | Reference |
|---|---|
| `utils/stt_utils.py` (Azure STT) | [speech_to_text_microsoft.py](https://github.com/qobi/ece49595nl/blob/main/speech_to_text_microsoft.py) |
| `scripts/start_debate.py` (overall loop) | [spoken_gpt_microsoft.py](https://github.com/qobi/ece49595nl/blob/main/spoken_gpt_microsoft.py) |

### API & Library Documentation
| Library / SDK | Used In | Documentation |
|---|---|---|
| Azure Cognitive Services Speech SDK | `stt_utils.py` | https://learn.microsoft.com/en-us/azure/ai-services/speech-service/get-started-speech-to-text?pivots=programming-language-python |
| Coqui TTS — XTTS-v2 | `voice_synth_utils.py` | https://docs.coqui.ai/en/latest/models/xtts.html |
| Hugging Face Transformers — GPT-2 | `train_biden.py`, `train_trump.py` | https://huggingface.co/docs/transformers/model_doc/gpt2 |
| Hugging Face `Trainer` API | `train_biden.py`, `train_trump.py` | https://huggingface.co/docs/transformers/main_classes/trainer |
| yt-dlp | `prepare_voices.py` | https://github.com/yt-dlp/yt-dlp |
| BeautifulSoup 4 | `collect_data.py` | https://www.crummy.com/software/BeautifulSoup/bs4/doc/ |
| PyPDF2 | `collect_data.py` | https://pypdf2.readthedocs.io/en/latest/ |
| `python-dotenv` | all scripts | https://pypi.org/project/python-dotenv/ |

---

## Requirements

See [requirements.txt](requirements.txt) for the full list of Python dependencies.
