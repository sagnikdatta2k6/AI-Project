# ReadOrSpeak: Multimodal AI Interview Analysis

**ReadOrSpeak** analyzes a candidate's recorded video interview response and predicts whether the
answer was **spontaneously spoken** or **read from an AI-generated script**. It does this by
running three independent AI "agents" — one per modality (vision, audio, language) — over the
recording and then fusing their individual verdicts into a single explainable decision.

This document describes (1) how the system works end-to-end, (2) the AI/ML theory behind each
component, and (3) how to install and run it locally.

---

## 1. Problem Framing

The task is framed as **binary classification with weak/heuristic supervision**: given a video
clip, predict `label ∈ {Spontaneous (0), Scripted (1)}`.

The key AI idea is that "reading" and "speaking from memory/thought" leave different
**behavioral traces** across multiple channels of communication simultaneously:

| Modality   | Behavioral signal exploited                                   | Underlying theory |
|------------|-----------------------------------------------------------------|--------------------|
| Vision     | Horizontal eye-gaze sweep pattern                               | Eye-tracking / saccadic reading behavior |
| Audio      | Pause structure and pitch (F0) variance                          | Prosody & disfluency in spontaneous speech |
| Language   | Ratio of filler words / disfluencies in the transcript           | Psycholinguistics of spontaneous vs. rehearsed speech |

No single modality is reliable alone (e.g. a person can have low pitch variance simply because
they are calm, not because they are reading). The system therefore treats this as a
**multimodal ensemble / late-fusion problem**, which is the central AI concept demonstrated in
this project.

---

## 2. System Architecture

```
                         ┌──────────────────────────┐
                         │   Uploaded Video (.mp4)  │
                         └────────────┬─────────────┘
                                      │
                     ┌────────────────┼────────────────┐
                     ▼                ▼                ▼
             ┌───────────────┐┌───────────────┐┌─────────────────────┐
             │  VisionAgent  ││  AudioAgent   ││  LinguisticAgent    │
             │ (MediaPipe    ││ (Librosa      ││ (Whisper ASR +      │
             │  Face Mesh)   ││  DSP)         ││  spaCy NLP)         │
             └───────┬───────┘└───────┬───────┘└───────────┬──────────┘
                     │                │                    │
                     ▼                ▼                    ▼
              gaze variance     pause ratio,           transcript,
              (suspicious?)     pitch std              filler ratio
                     │          (suspicious?)          (suspicious?)
                     └────────────────┼────────────────────┘
                                      ▼
                      ┌────────────────────────────────┐
                      │  MultiAgentFusionOrchestrator  │
                      │  (weighted rule-based voting)  │
                      └───────────────┬────────────────┘
                                      ▼
                    Final verdict + confidence score (%) 
                    + natural-language explanation
```

Each "agent" ([`src/agents.py`](src/agents.py)) is a self-contained perception module that turns
raw signal (pixels, waveform samples) into an interpretable numeric **feature** and a **local
verdict**. The [`MultiAgentFusionOrchestrator`](src/orchestrator.py) then performs **decision-level
fusion** across the three local verdicts to produce a final, explainable classification. The
[Streamlit UI](app.py) is a thin presentation layer over this pipeline.

This "specialist sub-agents + central orchestrator" pattern is itself an AI systems concept
worth noting for a class project: it is a lightweight instance of a **multi-agent system (MAS)**
where each agent has a narrow, independent perceptual competency and a coordinator performs
**consensus/fusion**, similar in spirit to ensemble learning and to blackboard-architecture AI
systems (agents contribute partial evidence to a shared decision).

---

## 3. The Three Perceptual Agents — Theory & Implementation

### 3.1 Vision Agent — Gaze-Sweep Detection

**File:** `VisionAgent.analyze()` in [`src/agents.py`](src/agents.py), feature extraction also in
[`src/extract_features.py`](src/extract_features.py).

**Technique:** [MediaPipe Face Mesh](https://ai.google.dev/edge/mediapipe) — a CNN-based dense
facial landmark detector (468 base landmarks + 10 iris landmarks when `refine_landmarks=True`) —
is run on every video frame to locate the iris center and the eye-corner landmarks.

**Feature computed:**
```
gaze_ratio = (iris_x − min(eye_corner_outer_x, eye_corner_inner_x)) / eye_width
```
This normalizes the iris's horizontal position within the eye socket, frame by frame, producing a
gaze-position time series. The agent then computes the **standard deviation of this ratio across
the whole clip** as a proxy for how much the eyes sweep left-to-right.

**AI/theory justification:** This mirrors real **eye-tracking research on reading behavior**.
Reading text (e.g., off-screen teleprompter or a second monitor) produces **regular, repeated
horizontal saccades** as the eyes track lines of text left-to-right. Natural spontaneous speech,
by contrast, tends to involve relatively **fixed gaze on the camera/interviewer**, or gaze
aversion to a single off-axis point while "thinking" — both of which produce **lower horizontal
gaze variance**. This is a classic feature-engineering approach: convert raw video into a
behaviorally meaningful 1-D signal and threshold its variance (`> 0.05` ⇒ "suspicious").

### 3.2 Audio Agent — Prosody Analysis

**File:** `AudioAgent.analyze()` in [`src/agents.py`](src/agents.py).

**Technique:** Digital signal processing via [`librosa`](https://librosa.org/):

1. **Silence/pause detection** — `librosa.effects.split()` uses an energy (dB) threshold to
   segment the waveform into speech vs. silence intervals. `pause_ratio` = fraction of total
   duration that is silence.
2. **Pitch (F0) tracking** — `librosa.pyin()` implements the **probabilistic YIN algorithm**, an
   autocorrelation-based fundamental frequency estimator with a Hidden Markov Model (HMM) applied
   over candidate pitch tracks to pick the most probable, temporally-smooth F0 contour per frame.
   The **standard deviation of F0 across voiced frames** (`pitch_std_hz`) is used as a measure of
   **intonational variety**.

**AI/theory justification:** This draws on **prosody research in psycholinguistics**.
Spontaneous speech is disfluent by nature — speakers pause to retrieve words, plan syntax, and
self-correct, producing a higher pause ratio; it also carries natural pitch inflection tied to
emphasis and thought structure. Reading a pre-written script aloud, in contrast, is fluent
continuous phonation with comparatively **flat/monotone intonation** and few unplanned pauses.
The rule `pause_ratio < 0.15 AND pitch_std < 25 Hz ⇒ suspicious` operationalizes this theory as a
simple two-threshold classifier.

### 3.3 Linguistic Agent — Disfluency Analysis

**File:** `LinguisticAgent.analyze()` in [`src/agents.py`](src/agents.py).

**Technique:**
1. **Automatic Speech Recognition (ASR)** — [OpenAI Whisper](https://github.com/openai/whisper)
   (`tiny` model), a Transformer encoder-decoder trained with weak supervision on ~680k hours of
   multilingual audio, converts the spoken audio into a text transcript.
2. **Tokenization & lexical analysis** — [spaCy](https://spacy.io/) (`en_core_web_sm`) tokenizes
   the transcript, and the agent computes the ratio of **filler words** ("um", "uh", "like",
   "actually", "basically", "so", …) to total tokens.

**AI/theory justification:** This is grounded in **corpus linguistics / speech disfluency
theory** — the well-documented finding that unscripted, spontaneous speech contains measurably
higher rates of filled pauses and discourse markers than planned or written language, because
these fillers serve a real cognitive function (buying time for word/sentence planning). Text
read verbatim from an AI-generated script is, almost by definition, **grammatically "clean"**
written prose — it lacks these disfluency markers. `filler_ratio < 0.02 AND word_count > 20 ⇒
suspicious` operationalizes this.

---

## 4. Decision Fusion — The Orchestrator

**File:** [`src/orchestrator.py`](src/orchestrator.py), class `MultiAgentFusionOrchestrator`.

Rather than trusting any single modality, the orchestrator performs **weighted rule-based
ensemble fusion** (also called *late fusion* / *decision-level fusion* in the multimodal-learning
literature, as opposed to *early fusion*, which would concatenate raw features before a single
model sees them):

```
suspicion_score = 40 × 1[vision suspicious]
                 + 30 × 1[audio suspicious]
                 + 30 × 1[linguistics suspicious]

final_verdict = "Likely Scripted / AI-Read"  if suspicion_score ≥ 50
              = "Likely Spontaneous"          otherwise
```

The vision signal is weighted highest (40%) because gaze-sweep is treated as the strongest single
indicator, while audio and linguistics each contribute 30%. This is a hand-tuned linear
weighted-vote ensemble — conceptually the same family of technique as **weighted majority voting
classifiers** and **soft-voting ensembles** in classical ML, except the "base classifiers" here
are heuristic single-threshold rules over hand-engineered domain features instead of learned
statistical models, and the weights were **chosen by domain reasoning** rather than fit through
cross-validation.

Crucially, the orchestrator also returns an **explanation** (`orchestrator_synthesis`): a
human-readable list of exactly which agents fired and why. This directly supports the AI-safety
/ **Explainable AI (XAI)** principle that a classifier's output should be accompanied by
inspectable evidence, not just a bare label — important in any human-in-the-loop, high-stakes
decision setting such as interview screening, where an opaque verdict would be inappropriate to
act on directly.

---

## 5. The Supervised-Learning Track (Offline Model Training)

While the live app ([`app.py`](app.py)) uses the fixed rule-based orchestrator above, the project
also includes a **learned-model track** demonstrating classical supervised ML on the same
engineered features:

- [`src/extract_features.py`](src/extract_features.py) — batch-extracts the same
  vision/audio-derived numeric features (gaze std, saccade velocity, pause ratio, pauses/min,
  pitch mean/std) from a folder of labeled videos into a tabular dataset
  (`data_features.csv`), where the label is inferred from filename convention
  (`scripted_*` → 1, `spontaneous_*` → 0).
- [`src/generate_mock_data.py`](src/generate_mock_data.py) — generates **synthetic** feature
  rows drawn from two different Gaussian distributions per class (used when real labeled video
  data isn't available, e.g. for demoing the training pipeline).
- [`src/train_baseline.py`](src/train_baseline.py) — trains a **Random Forest classifier**
  (`sklearn.ensemble.RandomForestClassifier`, an ensemble of decision trees using **bagging** —
  bootstrap aggregation — and random feature subsampling to reduce variance/overfitting) on the
  tabular features, evaluated with **Stratified 5-Fold Cross-Validation**
  (`accuracy`, `precision`, `recall`, `F1`, `ROC-AUC`) to get an unbiased estimate of
  generalization performance on the small dataset. It also plots **Gini feature importances**,
  which is itself a model-based interpretability technique — showing which behavioral cues the
  learned model actually leans on, letting you sanity-check the rule-based weights in §4 against
  data-driven evidence.

This second track illustrates the difference between a **hand-crafted expert-rules classifier**
(the live orchestrator) and a **data-driven learned classifier** (Random Forest) operating over
identical engineered features — a useful comparison point for a course covering both
knowledge-based and statistical-learning approaches to AI.

---

## 6. Summary of AI/CS Concepts Demonstrated

| Concept | Where |
|---|---|
| Multi-agent systems / decision fusion | `orchestrator.py` |
| Computer vision — facial landmark detection (CNN-based) | `VisionAgent`, MediaPipe Face Mesh |
| Eye-tracking behavioral inference | `VisionAgent` gaze variance |
| Digital signal processing (energy-based VAD, YIN pitch tracking + HMM smoothing) | `AudioAgent` |
| Automatic Speech Recognition (Transformer seq2seq) | `LinguisticAgent`, Whisper |
| NLP tokenization & lexical/psycholinguistic feature engineering | `LinguisticAgent`, spaCy |
| Weighted ensemble / late fusion voting | `MultiAgentFusionOrchestrator` |
| Explainable AI (XAI) — human-readable rationale | `orchestrator_synthesis` |
| Supervised learning — ensemble trees (bagging) | `train_baseline.py`, RandomForest |
| Model evaluation — stratified k-fold cross-validation | `train_baseline.py` |
| Model interpretability — Gini feature importance | `train_baseline.py` |
| Synthetic data generation for ML demos | `generate_mock_data.py` |

---

## 7. Project Structure

```
AI-Project/
├── app.py                     # Streamlit web UI — upload a video, get a live verdict
├── requirements.txt           # Pinned, mutually-compatible Python dependencies
├── src/
│   ├── agents.py              # VisionAgent, AudioAgent, LinguisticAgent (live inference)
│   ├── orchestrator.py        # MultiAgentFusionOrchestrator — fuses agent verdicts
│   ├── extract_features.py    # Batch feature extraction from a folder of labeled videos
│   ├── generate_mock_data.py  # Synthetic dataset generator for the training pipeline
│   ├── train_baseline.py      # Random Forest training + cross-validation + feature importance
│   ├── text_orcehstrator.py   # Quick smoke-test script for orchestrator wiring
│   └── data_features.csv      # Example extracted/synthetic feature dataset
└── generated-images/          # Output images (git-ignored)
```

---

## 8. Running It Locally

### 8.1 Prerequisites

- **Python 3.12** (mediapipe does not yet support 3.13+; 3.12 is the version this project was
  set up and tested against)
- **FFmpeg** on your system `PATH` (required by MoviePy/Whisper for audio extraction/decoding).
  - Windows: `winget install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org/download.html)
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`

### 8.2 Set up the environment

```bash
# from the project root
python -m venv venv
```

Activate it:

```bash
# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Windows (cmd)
venv\Scripts\activate.bat

# macOS / Linux
source venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> **Note:** if `openai-whisper` fails to build with a `pkg_resources`/`ModuleNotFoundError`
> during install, first run `pip install "setuptools<81" wheel`, then re-run
> `pip install -r requirements.txt`.

### 8.3 Run the interactive web app

```bash
streamlit run app.py
```

This opens a browser tab (default `http://localhost:8501`). Upload an `.mp4`/`.mov`/`.avi`
interview clip in the sidebar and click **"Run Multi-Agent Analysis"** to see the live verdict,
per-agent metrics, and transcript.

### 8.4 (Optional) Run the offline training pipeline

```bash
# 1. Either extract features from your own labeled videos in dataset/raw_videos/
#    (filenames must contain "scripted" or "spontaneous"):
python src/extract_features.py

# 2. ...or generate a synthetic dataset instead:
python src/generate_mock_data.py

# 3. Train and evaluate the Random Forest baseline:
python src/train_baseline.py
```

This prints 5-fold cross-validated accuracy/precision/recall/F1/ROC-AUC, prints feature
importances, and saves a `feature_importance.png` bar chart.

### 8.5 (Optional) Sanity-check the agent/orchestrator wiring without the UI

```bash
cd src
python text_orcehstrator.py
```

---

## 9. Limitations & Honest Caveats (for discussion)

- The live orchestrator's thresholds (`0.05`, `0.15`, `25 Hz`, `0.02`) and fusion weights
  (`40/30/30`) are **hand-tuned heuristics**, not learned from labeled data — they should be
  validated/calibrated against real labeled interview footage before any real-world use.
- Each agent uses a **single scalar threshold** on a **single aggregate statistic** per clip
  (e.g., overall gaze std for the whole video), which discards temporal structure. A stronger
  model could use the full feature *time series* per modality (e.g., an RNN/Transformer over
  frame-level gaze) rather than one summary statistic.
- Whisper's `tiny` model favors speed over transcription accuracy; disfluency counts inherit any
  ASR transcription errors.
- This tool infers *behavioral correlates* of reading vs. speaking — it does not have ground-truth
  access to what the candidate was actually doing, and false positives/negatives are expected;
  it should be treated as a decision-support signal, not an autonomous accusation of misconduct.
