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

During the debate:
- **Laptop 1** runs the Biden chatbot
- **Laptop 2** runs the Trump chatbot
- Each listens to the other's speech output via microphone and responds accordingly

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

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```env
AZURE_SPEECH_KEY=your_azure_speech_key_here
AZURE_SPEECH_REGION=your_azure_region_here
```

> API keys are distributed through Purdue Filelocker by the course instructor.

### 3. Download Training Data

Run the data collection script to download Biden/Trump speeches and transcripts:

```bash
python scripts/collect_data.py
```

Data sources used:
- **Trump:** Rev.com debate transcripts, Art of the Deal (PDF), Think Like a Champion (PDF), Trump Twitter Archive
- **Biden:** White House speeches, Rev.com debate transcripts, Promises to Keep (archive.org), Promise Me Dad (archive.org)

### 4. Preprocess the Data

```bash
python scripts/preprocess.py
```

### 5. Fine-Tune the Models

Fine-tune GPT-2 Medium separately for each persona:

```bash
# Train Biden model
python scripts/train_biden.py

# Train Trump model
python scripts/train_trump.py
```

> **Note:** Model weights are NOT included in this submission. Run the training scripts to replicate them.

### 6. Run the Debate

On the Biden laptop:
```bash
python scripts/debate.py --persona biden
```

On the Trump laptop:
```bash
python scripts/debate.py --persona trump
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

### Trump
- Rev.com: https://www.rev.com/blog/transcripts?s=trump
- The Art of the Deal (PDF): https://ia601405.us.archive.org/19/items/TrumpTheArtOfTheDeal
- Think Like a Champion (PDF): https://www.reboxu.com/uploads/8/6/0/3/86031326/think_like_a_champion.pdf
- Trump Twitter Archive: https://www.thetrumparchive.com/
- Kaggle Trump Tweets: https://www.kaggle.com/datasets/headsortails/trump-twitter-archive

### Biden
- White House Speeches: https://www.whitehouse.gov/briefing-room/speeches-remarks/
- Rev.com: https://www.rev.com/blog/transcripts?s=biden
- Promises to Keep: https://archive.org/details/promisestokeepon00joeb
- Promise Me, Dad: https://archive.org/details/promisemedadyear0000bide_j8m5
- Biden Bibliography: https://en.wikipedia.org/wiki/Bibliography_of_Joe_Biden

### Debate Transcripts (Both)
- 2024 Biden vs Trump Debate: https://www.rev.com/blog/transcripts/biden-trump-debate-transcript
- 2020 Debate 1: https://www.rev.com/blog/transcripts/donald-trump-joe-biden-1st-presidential-debate-transcript-2020
- 2020 Debate 2: https://www.rev.com/blog/transcripts/donald-trump-joe-biden-final-presidential-debate-transcript-2020

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
