# ASL Sign Language Detector → Text → Speech · 2026 Edition

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Files](#project-files)
- [Methodology](#methodology)
  - [Data Collection and Preprocessing](#data-collection-and-preprocessing)
  - [Feature Extraction](#feature-extraction)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
  - [Real-Time Inference](#real-time-inference)
- [Features](#features)
- [Keyboard Shortcuts](#keyboard-shortcuts)
- [Results and Insights](#results-and-insights)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [Technologies Used](#technologies-used)
- [How to Run the Project Locally](#how-to-run-the-project-locally)
- [Contributor](#contributor)
- [Acknowledgments](#acknowledgments)

## Overview
The **ASL Sign Language Detector – 2026 Edition** is a real-time computer vision project that translates **American Sign Language** hand signs into text and speech using a webcam. It combines **MediaPipe** for hand landmark tracking with **scikit-learn** for classification, wrapped in a futuristic neon HUD interface featuring particle effects, word autocomplete, quiz mode, number mode, and neural AI text-to-speech.

This project was developed as part of the **AI 2** course during the **Master 1** programme at **Djilali Bounaama University of Khemis Miliana (DBKM)**.

## Dataset
The project utilises the well-known ASL Alphabet dataset from Kaggle:
- **Kaggle Dataset:** [ASL Alphabet – Machine Learning from Images](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Description:** The dataset contains labelled images of American Sign Language hand signs for every letter and special gestures.
- **Target Variable:** Hand sign class (A–Z, SPACE, DELETE, NOTHING)
- **Key Features:** 42 normalised hand landmark coordinates (21 landmarks × 2 axes) extracted via MediaPipe
- **Classes:** 29 (26 letters + SPACE + DELETE + NOTHING)
- **Training Samples:** ~9 306 samples after feature extraction
- **Accuracy:** 97.74% (ExtraTrees Classifier)

Alternatively, users can collect their own images via the built-in webcam collection tool.

## Project Files
This repository contains the following key files:

| File | Description |
|------|-------------|
| `main.py` | Menu launcher with ASCII 2026 banner — entry point for the entire pipeline |
| `config.py` | Central configuration (paths, labels, thresholds, colours) |
| `download_dataset.py` | Downloads the Kaggle ASL Alphabet dataset via `kagglehub` |
| `collect_imgs.py` | Captures webcam images for custom dataset collection |
| `create_dataset.py` | Extracts MediaPipe hand landmarks → `data.pickle` |
| `train_classifier.py` | Trains & compares RandomForest and ExtraTrees → `model.p` |
| `inference_classifier.py` | ★ Real-time sign → text detector with 2026 neon HUD |
| `word_completer.py` | Smart prefix-based word autocomplete engine (~500 words) |
| `generate_reference.py` | Generates an ASL reference chart image |
| `model.p` | Saved trained model (ExtraTrees, sklearn 1.8.0) |
| `data.pickle` | Extracted landmark features (9 306 samples, 42 features) |
| `requirements.txt` | Python dependencies |
| `PPT_Presentation/` | Presentation slides (`Ai_MultiModel_2024.pptx`, `Ai_MultiModel_2026.pptx`) |
| `data/` | Image folders (classes 0–28) |
| `exports/` | Exported text files, screenshots, and recordings |

## Methodology

### Data Collection and Preprocessing
- **Data Source:** The ASL Alphabet dataset is downloaded from Kaggle using `kagglehub`, or custom images are collected via the webcam tool.
- **Preprocessing Steps:**
  - Images are organised into class folders (0–28), each corresponding to a sign.
  - A class mapping converts folder indices to labels (A–Z, DEL, NOTHING, SPACE).
  - Non-predictive data (background, lighting variations) is handled by MediaPipe's hand isolation.

### Feature Extraction
- **Hand Landmark Detection:** MediaPipe Hands detects 21 hand landmarks per frame.
- **Normalisation:** Landmark coordinates are normalised relative to the bounding box minimum (x, y), producing 42 features per sample.
- **Output:** Features and labels are saved to `data.pickle` for training.

### Model Training
- **Initialisation:** Two classifiers are trained and compared — RandomForest and ExtraTrees.
- **Automatic Selection:** The model with the highest test accuracy is saved to `model.p`.
- **Best Model:** ExtraTrees Classifier achieved **97.74% accuracy** on the test set.
- **Confidence Smoothing:** A 6-frame temporal smoother (exponential decay = 0.65) averages predictions at inference time, stabilising detections for similar-looking letters.

### Model Evaluation
- **Train/Test Split:** 80% training, 20% testing with stratified sampling.
- **Performance Metric:** Classification accuracy.
- **Results:** ExtraTrees 97.74% vs RandomForest ~96%.

### Real-Time Inference
- **Pipeline:** Webcam frame → MediaPipe hand detection → landmark extraction → model prediction → HUD overlay → display.
- **Stability Logic:** A letter must be detected consistently for 10 consecutive frames before it is confirmed and added to the sentence.
- **Face Rejection Filter:** A 4-check validation system (handedness confidence, bounding box size, aspect ratio, finger spread) filters out false positives caused by face/nose detections.
- **Text-to-Speech:** Neural AI voice via `edge-tts` (Microsoft Azure "en-US-AriaNeural") with `pyttsx3` fallback.

## Features

| Feature | Description |
|---------|-------------|
| **Full A-Z + Gestures** | Recognises all 26 letters plus SPACE, DELETE, and NOTHING (29 classes) |
| **Sign → Text** | Builds words and sentences on screen as you sign |
| **Number Mode (N)** | Maps letter signs to digits 0-9 (D=1, V=2, W=3, B=4, A=5, Y=6, S=7, L=8, F=9, O=0) |
| **Neon Glow Hand Skeleton** | Futuristic hand tracking with glowing landmarks and fingertip highlights |
| **Particle Burst Effects** | Colourful particle explosion when a letter is confirmed |
| **Top-3 Predictions** | Live confidence bars showing the model's top 3 guesses |
| **Smart Word Autocomplete** | Suggests words as you spell — press 1/2/3 or Tab to accept |
| **ASL Reference Guide (G)** | Side-by-side display of the ASL hand-signs chart for beginners |
| **Quiz Mode (M)** | Interactive quiz that challenges you to sign specific letters |
| **Frosted Glass HUD** | Semi-transparent panels with modern glassmorphism design |
| **Session Statistics** | Letters per minute, word count, session timer |
| **Video Recording (R)** | Record signing sessions as AVI video |
| **Screenshot Capture (F)** | Save the current frame with all overlays |
| **Toast Notifications** | Animated pop-up messages for every action |
| **Letter History Trail** | Fading trail of recently typed letters |
| **Stability Ring** | Circular progress indicator showing confirmation progress |
| **Neural AI Voice (S)** | Text-to-speech using Microsoft Edge Neural voice |
| **Export (E)** | Save sentence text to a timestamped file |
| **Face Rejection Filter** | Prevents face/nose from being detected as hand signs |
| **FPS Counter** | Colour-coded live frame-rate display |

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit the application |
| `S` | Speak sentence (Neural AI TTS) |
| `C` | Clear entire sentence |
| `X` / `Backspace` | Delete last letter |
| `E` | Export sentence to `exports/` folder |
| `F` | Take a screenshot |
| `R` | Toggle video recording |
| `G` | Toggle ASL reference guide (side-by-side image) |
| `N` | Toggle Number Mode (0-9) / Skip quiz letter (in Quiz Mode) |
| `M` | Toggle Quiz Mode |
| `Tab` | Accept top autocomplete suggestion |
| `1` / `2` / `3` | Accept autocomplete suggestion by number |

## Results and Insights
- **High Accuracy:** The ExtraTrees Classifier achieved **97.74% accuracy** across 28 classes with only 42 features per sample, demonstrating that hand landmark coordinates are highly discriminative for ASL recognition.
- **Real-Time Performance:** The system runs at 20-30+ FPS on standard hardware, enabling smooth real-time interaction.
- **Confidence Smoothing:** The 6-frame temporal smoother significantly reduces flickering between similar letters (e.g., M/N, U/V), improving the user experience.
- **Practical Usability:** Word autocomplete, toast notifications, and the stability ring create a fluid signing-to-text workflow.

## Limitations and Future Improvements
- **Static Signs Only:** The current model recognises static hand poses; dynamic gestures (J, Z which involve motion) rely on single-frame snapshots.
- **Single Hand:** Only one hand is processed for classification at a time.
- **Lighting Sensitivity:** Performance may degrade in low-light or high-contrast conditions.
- **Future Enhancements:**
  - Add temporal/sequence models (LSTM, Transformer) for dynamic gesture recognition.
  - Extend to two-hand signs and full ASL word-level recognition.
  - Integrate a larger vocabulary for word autocomplete.
  - Add multi-language sign language support (BSL, FSL, etc.).
  - Deploy as a web application for broader accessibility.

## Technologies Used
- **MediaPipe 0.10.18:** For real-time hand landmark detection and tracking.
- **scikit-learn 1.8.0:** For training the ExtraTrees Classifier.
- **OpenCV:** For webcam capture, image processing, and HUD rendering.
- **NumPy:** For numerical operations and feature processing.
- **edge-tts:** For modern neural AI text-to-speech (Microsoft Azure voices).
- **pygame:** For audio playback of TTS output.
- **pyttsx3:** Fallback offline text-to-speech engine.
- **kagglehub:** For automated dataset download from Kaggle.
- **Python 3.12:** Primary programming language.

## How to Run the Project Locally

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/Sign-language-detector-python.git
   cd Sign-language-detector-python
   ```

2. **Create a Virtual Environment (recommended):**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   # source .venv/bin/activate   # Linux/macOS
   ```

3. **Install Required Libraries:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Menu:**
   ```bash
   python main.py
   ```
   From the menu:
   - **Option 1** – Download the ASL Alphabet dataset from Kaggle
   - **Option 2** – Collect your own images via webcam
   - **Option 3** – Extract hand landmark features
   - **Option 4** – Train the classifier
   - **Option 5** – Run the real-time detector and start signing!

5. **Or run the detector directly:**
   ```bash
   python inference_classifier.py
   ```

## Contributor
- **Abdelkarim Douadjia** – Master 1 AI & Big Data, Djilali Bounaama University of Khemis Miliana (DBKM)

## Acknowledgments
- **Mr. Bahloul Djamel** – Course instructor for AI 2, for his guidance and support throughout the project.
- **Kaggle:** For hosting the ASL Alphabet dataset and fostering an active data science community.
- **MediaPipe Team (Google):** For providing an exceptional hand tracking solution.
- **scikit-learn Community:** For robust machine learning tools and documentation.
- **University Faculty at DBKM:** For their academic resources and encouragement.

---

This project demonstrates how computer vision and machine learning can bridge communication gaps by translating sign language into text and speech in real time, making ASL more accessible to everyone.
