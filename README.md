# 🎧 Audio Classification with RAG (Retrieval-Augmented Generation)

An explainable Environmental Audio Classification platform built with **CNN-LSTM & Dense Acoustic Embeddings**, **FAISS Vector Search**, **Acoustic Domain Knowledge Base**, and **Generative AI (Google Gemini / Grounded RAG)** on the **ESC-50 Dataset**.

---

## 🌟 Key Highlights

- **Dual Acoustic & Temporal Analysis:** Generates interactive **time-domain waveforms** and **Log-Mel spectrograms** ($128$ mel bands, calibrated dB scales) along with acoustic KPIs (RMS Energy, Duration, Spectral Centroid).
- **Dense Vector Retrieval (FAISS):** Embeds audio into a 128-dimensional latent space and queries a FAISS vector index of 2,000 reference clips in milliseconds.
- **Audible Exemplar Inspection:** Displays similarity percentage bars and embedded audio players for each top-$k$ retrieved sound so users can hear matching reference sounds.
- **50-Class Domain Knowledge Base:** Curated acoustic metadata across all 50 ESC-50 classes (Animals, Natural Soundscapes, Human Non-Speech, Interior Sounds, Exterior/Urban Noises) including frequency bands and confusable sound cues.
- **Hybrid RAG Reasoning Engine:**
  - **⚡ Cloud LLM Mode:** Uses Google Gemini (`google-genai` SDK) to act as an audio forensic engineer, providing acoustic consensus assessment, sound physics analysis, environmental validation, and edge-case disambiguation.
  - **🖥️ Grounded Offline Mode:** Runs 100% locally with zero latency, synthesizing vector neighbor consensus and acoustic characteristics.

---

## 🏗️ Architecture

```
                        ┌───────────────────────────────┐
                        │      Input Audio Clip         │
                        └──────────────┬────────────────┘
                                       │
                         [Mel-Spectrogram / Preprocessing]
                                       │
                 ┌─────────────────────┴─────────────────────┐
                 │                                           │
                 ▼                                           ▼
   ┌───────────────────────────┐               ┌───────────────────────────┐
   │    CNN-LSTM Classifier    │               │  128-d Acoustic Embedding │
   │   (Softmax Probabilities) │               │   (Normalized L2 Vector)  │
   └─────────────┬─────────────┘               └─────────────┬─────────────┘
                 │                                           │
                 │                                           ▼
                 │                             ┌───────────────────────────┐
                 │                             │   FAISS Vector Index      │
                 │                             │   (Top-K Audio Exemplars) │
                 │                             └─────────────┬─────────────┘
                 │                                           │
                 ▼                                           ▼
   ┌───────────────────────────┐               ┌───────────────────────────┐
   │   Audio Knowledge Base    │               │   Retrieved Audio Clips   │
   │  (50 Classes Domain Info) │               │   (& Similarity Scores)   │
   └─────────────┬─────────────┘               └─────────────┬─────────────┘
                 │                                           │
                 └─────────────────────┬─────────────────────┘
                                       │
                                       ▼
                 ┌───────────────────────────────────────────┐
                 │       RAG Synthesis / Generative AI       │
                 │     (Google Gemini / Grounded Template)   │
                 └─────────────────────┬─────────────────────┘
                                       │
                                       ▼
                 ┌───────────────────────────────────────────┐
                 │   Interactive Streamlit Web Application   │
                 │ • Waveform & Log-Mel Spectrogram          │
                 │ • Top-K Audio Players & Similarity Bars   │
                 │ • Acoustic Forensic Diagnostic Report     │
                 └───────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
├── app.py                      # Main Streamlit web application
├── Audio Classfition .ipynb    # End-to-end model training & experimentation notebook
├── audio_knowledge.csv         # 50-class acoustic domain knowledge base
├── audio_faiss.index           # Precomputed FAISS index for 2,000 ESC-50 exemplars
├── audio_metadata.pkl          # Metadata dictionary corresponding to FAISS index
├── classes.npy                 # 50 sorted ESC-50 category labels
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules (filters large audio binaries)
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/Audio-Classification-RAG.git
cd Audio-Classification-RAG
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Set up Google Gemini API Key
To enable LLM-powered acoustic reasoning:
```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your_api_key_here"

# Linux / macOS
export GEMINI_API_KEY="your_api_key_here"
```
*(You can also enter your API key directly in the Streamlit app sidebar, or run in Offline Mode without an API key).*

### 5. Run the Streamlit Application
```bash
python -m streamlit run app.py
```
Visit `http://localhost:8501` in your browser.

---

## 📊 Dataset Reference

This project utilizes the **ESC-50 (Dataset for Environmental Sound Classification)**:
- 2,000 environmental audio recordings
- 50 semantically distinct classes (40 clips per class)
- Karol J. Piczak. 2015. *ESC: Dataset for Environmental Sound Classification*. ACM Multimedia.

---

## 🛠️ Tech Stack

- **Frontend / UI:** Streamlit
- **Audio Processing:** Librosa, SciPy, SoundFile
- **Vector Retrieval:** FAISS (Facebook AI Similarity Search)
- **Deep Learning:** TensorFlow / Keras (CNN-LSTM)
- **Generative AI:** Google GenAI SDK (Gemini 2.5 Flash)
- **Visualization:** Matplotlib, Seaborn
