# ECE 49595NL / ECE 59500NL — Homework 1
## Biden vs. Trump Debate Chatbot

**Course:** Introduction to Natural Language Processing — Spring 2026  
**Due:** 5:00 PM, Friday February 27, 2026  

---

## Overview

This project implements two AI chatbots that emulate **Former President Joseph R. Biden** and **President Donald J. Trump** in a live spoken debate. Each chatbot is powered by a fine-tuned GPT-2 Medium model and uses the Microsoft Azure Speech SDK for both speech input (STT) and speech output (TTS).

---

## System Architecture

```
Microphone Input
      │
      ▼
Azure STT (Speech-to-Text)
      │
      ▼
GPT-2 Medium (Fine-tuned on Biden/Trump data)
      │
      ▼
Azure TTS (Text-to-Speech)
      │
      ▼
Speaker Output
```

During the debate (two-laptop mode):
- **Laptop 1** runs the Biden chatbot
- **Laptop 2** runs the Trump chatbot
- Each listens to the other's speech output via microphone and responds accordingly

Alternatively, both chatbots can run on a single laptop (see **Run the Debate** below).

---

## Project Structure

```
Code/
├── data/
│   ├── raw/                    # Raw scraped/downloaded text (not submitted)
│   └── processed/              # Cleaned training data (not submitted)
├── models/
│   ├── fine_tuned_biden/       # Fine-tuned Biden model weights (not submitted)
│   └── fine_tuned_trump/       # Fine-tuned Trump model weights (not submitted)
├── scripts/
│   ├── collect_data.py         # Scrape and download speech data
│   ├── preprocess.py           # Clean and format training data
│   ├── train_biden.py          # Fine-tune GPT-2 on Biden data
│   ├── train_trump.py          # Fine-tune GPT-2 on Trump data
│   ├── debate.py               # Main debate loop (STT → LLM → TTS)
│   └── tts_utils.py            # Azure TTS helper functions
├── .env                        # API keys (NOT committed)
├── .gitignore
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

- **Windows:**
  ```bash
  .venv\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

Create a `.env` file in the project root:

```env
AZURE_SPEECH_KEY=your_azure_speech_key_here
AZURE_SPEECH_REGION=your_azure_region_here
```

> API keys are distributed through Purdue Filelocker by the course instructor.

### 4. Download Training Data

Run the data collection script to download Biden/Trump speeches and transcripts:

```bash
python scripts/collect_data.py
```

Data sources used:
- **Trump:** Art of the Deal (PDF), Think Like a Champion (PDF), 15 Rev.com speech transcripts, 2020 presidential debate transcripts
- **Biden:** Inaugural Address 2021, State of the Union 2022–2024 (UCSB), 15 Rev.com speech transcripts, 2020 presidential debate transcripts
- **Debates:** UCSB American Presidency Project (primary), Rev.com (fallback); transcripts split by speaker

### 5. Preprocess the Data

```bash
python scripts/preprocess.py
```

### 6. Fine-Tune the Models

Fine-tune GPT-2 Medium separately for each persona:

```bash
# Train Biden model
python scripts/train_biden.py

# Train Trump model
python scripts/train_trump.py
```

> **Note:** Model weights are NOT included in this submission. Run the training scripts to replicate them.

### 7. Run the Debate

Two modes are supported:

#### Mode A: Two Laptops (Original Setup)

Each laptop runs one persona. They listen to each other via microphone.

On the Biden laptop:
```bash
python scripts/debate.py --persona biden
```

On the Trump laptop:
```bash
python scripts/debate.py --persona trump
```

#### Mode B: Single Laptop

Both personas run on the same machine. The debate alternates automatically — each model's TTS output is routed back as STT input to the other.

```bash
python scripts/debate.py --mode single
```

You can also specify which persona speaks first (default: `biden`):
```bash
python scripts/debate.py --mode single --first biden
python scripts/debate.py --mode single --first trump
```

---

## Model Details

| Property | Value |
|---|---|
| **Base Model** | GPT-2 Medium (355M parameters) |
| **Hugging Face ID** | `gpt2-medium` |
| **Fine-tuning method** | Full fine-tuning via Hugging Face `Trainer` |
| **Context length** | 1024 tokens |
| **Speech SDK** | Microsoft Azure Cognitive Services Speech |

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

> **Optional:** Place `trump_tweets.csv` (from [Kaggle Trump Twitter Archive](https://www.kaggle.com/datasets/headsortails/trump-twitter-archive)) in `data/raw/trump/` to include tweet data.

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

- ELIZA reference: https://github.com/qobi/ece49595nl/blob/main/eliza.py
- Azure TTS reference: https://github.com/qobi/ece49595nl/blob/main/text_to_speech_microsoft.py
- Azure STT reference: https://github.com/qobi/ece49595nl/blob/main/speech_to_text_microsoft.py
- Spoken GPT reference: https://github.com/qobi/ece49595nl/blob/main/spoken_gpt_microsoft.py
- nanoGPT by Andrej Karpathy: https://github.com/karpathy/nanoGPT

---

## Requirements

See [requirements.txt](requirements.txt) for the full list of Python dependencies.
