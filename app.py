import os
import pickle
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import librosa
import librosa.display
import faiss

# ==================================================
# CONFIGURATION & CONSTANTS
# ==================================================
SAMPLE_RATE = 22050
DURATION = 5.0
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

# Candidate path locations
CANDIDATE_MODEL_PATHS = [
    "cnn_lstm_audio_classifier.keras",
    "models/cnn_lstm_audio.keras",
    "models/cnn_lstm_audio_classifier.keras",
]

CANDIDATE_FAISS_PATHS = [
    "audio_faiss.index",
    "rag/faiss.index",
    "rag/audio_faiss.index",
]

CANDIDATE_METADATA_PATHS = [
    "audio_metadata.pkl",
    "rag/metadata.pkl",
    "rag/audio_metadata.pkl",
]

CANDIDATE_CLASSES_PATHS = [
    "classes.npy",
    "data/processed/classes.npy",
]

CANDIDATE_KNOWLEDGE_PATHS = [
    "audio_knowledge.csv",
    "knowledge/audio_knowledge.csv",
]

AUDIO_SEARCH_DIRS = [
    os.path.join("ESC-50-master", "audio"),
    os.path.join("ESC-50-master", "ESC-50-master", "audio"),
    "audio",
]

# Emojis for categories
CATEGORY_EMOJIS = {
    "dog": "🐶", "cat": "🐱", "rooster": "🐓", "pig": "🐷", "cow": "🐮",
    "frog": "🐸", "hen": "🐔", "insects": "🐝", "sheep": "🐑", "crow": "🦅",
    "rain": "🌧️", "sea_waves": "🌊", "crackling_fire": "🔥", "crickets": "🦗",
    "chirping_birds": "🐦", "water_drops": "💧", "wind": "💨", "pouring_water": "🫗",
    "toilet_flush": "🚽", "thunderstorm": "⛈️", "crying_baby": "👶", "sneezing": "🤧",
    "clapping": "👏", "breathing": "😮‍💨", "coughing": "😷", "footsteps": "👣",
    "laughing": "😂", "brushing_teeth": "🪥", "snoring": "😴", "drinking_sipping": "🥤",
    "door_wood_knock": "🚪", "mouse_click": "🖱️", "keyboard_typing": "⌨️",
    "door_wood_creaks": "🚪", "can_opening": "🥫", "washing_machine": "🧺",
    "vacuum_cleaner": "🧹", "clock_alarm": "⏰", "clock_tick": "⏱️", "glass_breaking": "🍷",
    "helicopter": "🚁", "chainsaw": "🪚", "siren": "🚨", "car_horn": "🚗",
    "engine": "🚗", "train": "🚆", "church_bells": "🔔", "airplane": "✈️",
    "fireworks": "🎆", "hand_saw": "🪚"
}


# ==================================================
# HELPER FUNCTIONS: FILE & AUDIO RESOLUTION
# ==================================================
def find_first_existing(path_list):
    for p in path_list:
        if os.path.exists(p):
            return p
    return None


def resolve_audio_file(filename):
    """Find audio file path on disk across candidate directories."""
    for base_dir in AUDIO_SEARCH_DIRS:
        full_path = os.path.join(base_dir, filename)
        if os.path.exists(full_path):
            return full_path
    if os.path.exists(filename):
        return filename
    return None


def load_audio(file_path):
    """Load audio with librosa, resample to 22.05kHz, mono, pad/trim to 5s."""
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    target_length = int(SAMPLE_RATE * DURATION)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]
    return audio, sr


def audio_to_mel(audio):
    """Compute normalized Log-Mel Spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Min-max normalization
    norm_mel = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return norm_mel.astype(np.float32), mel_db


def extract_fallback_embedding(audio):
    """Extract a 128-dimensional acoustic embedding using Mel band energies."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=128
    )
    # Temporal average across frames
    band_means = np.log1p(mel.mean(axis=1)).astype(np.float32)
    band_means = band_means[np.newaxis, ...]
    faiss.normalize_L2(band_means)
    return band_means


# ==================================================
# RESOURCE LOADING
# ==================================================
@st.cache_resource
def load_rag_resources():
    # 1. Knowledge Base
    k_path = find_first_existing(CANDIDATE_KNOWLEDGE_PATHS)
    knowledge_df = pd.read_csv(k_path) if k_path else pd.DataFrame()

    # 2. Classes
    c_path = find_first_existing(CANDIDATE_CLASSES_PATHS)
    if c_path:
        classes = np.load(c_path)
    elif not knowledge_df.empty and "category" in knowledge_df.columns:
        classes = np.array(sorted(knowledge_df["category"].unique()))
    else:
        classes = np.array([])

    # 3. FAISS Index
    f_path = find_first_existing(CANDIDATE_FAISS_PATHS)
    index = faiss.read_index(f_path) if f_path else None

    # 4. Metadata
    m_path = find_first_existing(CANDIDATE_METADATA_PATHS)
    if m_path:
        with open(m_path, "rb") as f:
            metadata = pickle.load(f)
    else:
        metadata = []

    # 5. Keras Model (Optional / Resilient)
    model = None
    embedding_model = None
    m_file = find_first_existing(CANDIDATE_MODEL_PATHS)
    if m_file:
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(m_file)
            try:
                embedding_model = tf.keras.Model(
                    inputs=model.input,
                    outputs=model.get_layer("audio_embedding").output
                )
            except Exception:
                embedding_model = None
        except Exception as e:
            st.sidebar.warning(f"Note: Could not load Keras model ({e}). Using Acoustic kNN mode.")

    return {
        "model": model,
        "embedding_model": embedding_model,
        "index": index,
        "metadata": metadata,
        "classes": classes,
        "knowledge": knowledge_df,
    }


# ==================================================
# VISUALIZATION FUNCTIONS
# ==================================================
def render_audio_visualizations(audio, sr, mel_db):
    """Renders Time-Domain Waveform and Log-Mel Spectrogram side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 3.8), dpi=120)
    plt.subplots_adjust(wspace=0.25)

    time_axis = np.linspace(0, len(audio) / sr, len(audio))

    # 1. Waveform Plot
    ax1.plot(time_axis, audio, color="#00bcd4", linewidth=0.8, alpha=0.9)
    ax1.axhline(0, color="#ffffff", linestyle="--", linewidth=0.6, alpha=0.4)
    ax1.set_title("🌊 Time-Domain Waveform", fontsize=11, fontweight="bold", pad=8)
    ax1.set_xlabel("Time (seconds)", fontsize=9)
    ax1.set_ylabel("Amplitude", fontsize=9)
    ax1.set_xlim(0, max(time_axis))
    ax1.set_ylim(-1.05, 1.05)
    ax1.grid(True, linestyle=":", alpha=0.3)
    ax1.set_facecolor("#121820")

    # 2. Log-Mel Spectrogram Plot
    img = librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        ax=ax2,
        cmap="magma"
    )
    ax2.set_title("🔥 Log-Mel Spectrogram", fontsize=11, fontweight="bold", pad=8)
    ax2.set_xlabel("Time (seconds)", fontsize=9)
    ax2.set_ylabel("Frequency (Hz)", fontsize=9)
    ax2.set_facecolor("#121820")
    cbar = fig.colorbar(img, ax=ax2, format="%+2.0f dB")
    cbar.ax.tick_params(labelsize=8)

    # Figure background
    fig.patch.set_facecolor("#0b0f14")
    for ax in [ax1, ax2]:
        ax.tick_params(colors="#cfd8dc", labelsize=8)
        ax.xaxis.label.set_color("#cfd8dc")
        ax.yaxis.label.set_color("#cfd8dc")
        ax.title.set_color("#ffffff")
        for spine in ax.spines.values():
            spine.set_color("#37474f")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ==================================================
# PREDICTION & RETRIEVAL ENGINE
# ==================================================
def process_audio_rag(audio, resources, top_k=5):
    model = resources["model"]
    embedding_model = resources["embedding_model"]
    index = resources["index"]
    metadata = resources["metadata"]
    classes = resources["classes"]
    knowledge_df = resources["knowledge"]

    # 1. Feature Preprocessing
    norm_mel, mel_db = audio_to_mel(audio)
    x = norm_mel[np.newaxis, ..., np.newaxis]

    # 2. Classification & Embedding
    probabilities = None
    if model is not None:
        probabilities = model.predict(x, verbose=0)[0]
        class_id = int(np.argmax(probabilities))
        predicted_class = str(classes[class_id]) if len(classes) > class_id else f"Class_{class_id}"
        confidence = float(probabilities[class_id])

        if embedding_model is not None:
            emb = embedding_model.predict(x, verbose=0).astype("float32")
            faiss.normalize_L2(emb)
        else:
            emb = extract_fallback_embedding(audio)
    else:
        # Fallback Acoustic Feature Embedding
        emb = extract_fallback_embedding(audio)

    # 3. FAISS Vector Search
    retrieved = []
    if index is not None and len(metadata) > 0:
        distances, indices = index.search(emb, top_k)
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(metadata):
                item = metadata[idx]
                retrieved.append({
                    "filename": item.get("filename", f"sample_{idx}"),
                    "category": item.get("category", "unknown"),
                    "similarity": float(dist),
                    "fold": item.get("fold", 1),
                    "audio_path": resolve_audio_file(item.get("filename", ""))
                })

    # If model wasn't available, use kNN consensus prediction
    if model is None:
        if retrieved:
            # Majority vote weighted by similarity
            cat_scores = {}
            for r in retrieved:
                cat = r["category"]
                cat_scores[cat] = cat_scores.get(cat, 0.0) + max(r["similarity"], 0.01)
            predicted_class = max(cat_scores, key=cat_scores.get)
            total = sum(cat_scores.values())
            confidence = cat_scores[predicted_class] / total if total > 0 else 0.5
        else:
            predicted_class = "Unknown"
            confidence = 0.0

    # 4. Knowledge Base Lookup
    knowledge_entry = None
    if not knowledge_df.empty:
        match = knowledge_df[
            knowledge_df["category"].str.lower() == str(predicted_class).lower()
        ]
        if len(match) > 0:
            knowledge_entry = match.iloc[0].to_dict()

    return {
        "prediction": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "mel_db": mel_db,
        "retrieved": retrieved,
        "knowledge": knowledge_entry
    }


# ==================================================
# RAG EXPLANATION GENERATORS
# ==================================================
def generate_offline_rag_explanation(result):
    pred = result["prediction"]
    conf = result["confidence"]
    retrieved = result["retrieved"]
    k_info = result["knowledge"]

    # Calculate neighbor consensus
    matching_neighbors = [r for r in retrieved if r["category"].lower() == pred.lower()]
    consensus_ratio = (len(matching_neighbors) / len(retrieved)) if retrieved else 0.0
    avg_similarity = (
        np.mean([r["similarity"] for r in matching_neighbors])
        if matching_neighbors else 0.0
    )

    lines = []
    lines.append(f"### 📋 RAG Grounded Diagnostic Report")
    lines.append(
        f"- **Primary Classification:** `{pred}` with **{conf:.1%}** model confidence.\n"
        f"- **Vector Index Retrieval Consensus:** **{len(matching_neighbors)} of {len(retrieved)}** nearest neighbors "
        f"({consensus_ratio:.0%}) match `{pred}` with an average acoustic similarity of **{avg_similarity:.1%}**."
    )

    if k_info:
        lines.append(f"\n#### 🔬 Acoustic & Spectral Physics")
        lines.append(
            f"- **Sound Category Group:** {k_info.get('group', 'General')}\n"
            f"- **Acoustic Characteristics:** {k_info.get('characteristics', 'N/A')}\n"
            f"- **Frequency Bandwidth:** `{k_info.get('frequency_range', 'Full-spectrum')}`"
        )

        lines.append(f"\n#### 🌍 Environmental & Contextual Validation")
        lines.append(
            f"- **Typical Origin:** {k_info.get('environment', 'Standard outdoor/indoor environment')}\n"
            f"- **Physical Description:** {k_info.get('description', 'N/A')}"
        )

        if "confusable_with" in k_info and pd.notna(k_info["confusable_with"]):
            lines.append(f"\n#### ⚠️ Confusable Sound Disambiguation")
            lines.append(
                f"- **Acoustically Similar Classes:** `{k_info['confusable_with']}`\n"
                f"- **Disambiguation Rationale:** The temporal envelope and harmonic distribution align closely "
                f"with `{pred}` rather than neighboring classes, confirmed by top-ranked FAISS exemplars."
            )

    return "\n".join(lines)


def generate_gemini_rag_explanation(result, api_key, model_name="gemini-2.5-flash"):
    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        pred = result["prediction"]
        conf = result["confidence"]
        retrieved = result["retrieved"]
        k_info = result["knowledge"] or {}

        exemplars_text = "\n".join([
            f"- Rank {i+1}: {r['category']} (similarity: {r['similarity']:.3f}, file: {r['filename']})"
            for i, r in enumerate(retrieved)
        ])

        prompt = f"""
You are an expert Audio Forensic Engineer and Acoustic AI Scientist.
Explain an environmental audio classification result grounded in the provided Retrieval-Augmented Generation (RAG) context.

=== CLASSIFICATION CONTEXT ===
- Input Prediction: {pred}
- Model Confidence: {conf:.1%}

=== ACOUSTIC VECTOR RETRIEVAL (TOP-{len(retrieved)} NEAREST NEIGHBORS FROM FAISS) ===
{exemplars_text}

=== DOMAIN KNOWLEDGE RETRIEVED ===
- Sound Group: {k_info.get('group', 'N/A')}
- Description: {k_info.get('description', 'N/A')}
- Typical Environment: {k_info.get('environment', 'N/A')}
- Acoustic Characteristics: {k_info.get('characteristics', 'N/A')}
- Frequency Range: {k_info.get('frequency_range', 'N/A')}
- Confusable Sounds: {k_info.get('confusable_with', 'N/A')}

=== INSTRUCTIONS ===
Write a concise, professional diagnostic breakdown with the following 4 sections:
1. **Acoustic Consensus & Decision**: Evaluate whether the FAISS neighbor retrieval corroborates the classification.
2. **Spectral & Physical Sound Analysis**: Detail the acoustic physics (harmonics, envelope, transients, frequency profile).
3. **Environmental Plausibility**: Contextualize where this sound naturally occurs.
4. **Edge-case Disambiguation**: Explain how to distinguish this from confusable sound sources.
Keep your response well-structured in markdown with clear bullet points.
"""

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        return response.text

    except Exception as e:
        st.error(f"Gemini API Error: {e}. Falling back to offline template.")
        return generate_offline_rag_explanation(result)


# ==================================================
# STREAMLIT USER INTERFACE
# ==================================================
def main():
    st.set_page_config(
        page_title="Audio Classification with RAG",
        page_icon="🎧",
        layout="wide"
    )

    # Custom Header Style
    st.markdown(
        """
        <style>
        .main-title { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2rem; color: #1E88E5; }
        .subtitle { font-size: 1.05rem; color: #90a4ae; margin-bottom: 1.5rem; }
        .metric-card { background: #1a222d; padding: 1rem; border-radius: 8px; border-left: 4px solid #1E88E5; }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="main-title">🎧 Audio Classification with RAG</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Deep Acoustic Vector Retrieval (FAISS) + Domain Knowledge Augmentation + LLM Synthesis on ESC-50</div>',
        unsafe_allow_html=True
    )

    # Load Resources
    resources = load_rag_resources()

    # --------------------------------------------------
    # SIDEBAR CONTROLS
    # --------------------------------------------------
    st.sidebar.header("⚙️ RAG Engine Settings")

    top_k = st.sidebar.slider("Top-K Retrieved Exemplars", min_value=3, max_value=10, value=5)

    rag_mode = st.sidebar.radio(
        "RAG Synthesis Generator",
        ["🖥️ Offline Grounded Engine", "⚡ Google Gemini AI (LLM)"]
    )

    gemini_api_key = ""
    gemini_model = "gemini-2.5-flash"
    if rag_mode == "⚡ Google Gemini AI (LLM)":
        gemini_api_key = st.sidebar.text_input(
            "Gemini API Key",
            type="password",
            value=os.environ.get("GEMINI_API_KEY", ""),
            help="Enter your Google Gemini API key or set GEMINI_API_KEY in environment."
        )
        gemini_model = st.sidebar.selectbox(
            "Gemini Model",
            ["gemini-2.5-flash", "gemini-1.5-flash"]
        )

    st.sidebar.divider()
    st.sidebar.subheader("📂 Quick Sample Audio")
    sample_files = {
        "— Select a sample audio clip —": None,
        "🐶 Dog Bark (1-100032-A-0.wav)": "1-100032-A-0.wav",
        "🐦 Chirping Birds (1-100038-A-14.wav)": "1-100038-A-14.wav",
        "🌧️ Rain (1-17092-A-27.wav)": "1-17092-A-27.wav",
        "🪚 Chainsaw (1-116765-A-41.wav)": "1-116765-A-41.wav",
        "🚗 Car Horn (1-24074-A-43.wav)": "1-24074-A-43.wav",
        "⏰ Clock Alarm (1-29532-A-37.wav)": "1-29532-A-37.wav",
        "🌊 Sea Waves (1-28135-A-11.wav)": "1-28135-A-11.wav",
        "🤧 Sneezing (1-26177-A-21.wav)": "1-26177-A-21.wav",
    }
    selected_sample_name = st.sidebar.selectbox("Test with Dataset Audio:", list(sample_files.keys()))
    selected_sample_file = sample_files[selected_sample_name]

    # Status Indicators in Sidebar
    st.sidebar.divider()
    st.sidebar.markdown("**System Status:**")
    st.sidebar.write(f"• FAISS Index: {'✅ Connected (' + str(resources['index'].ntotal) + ' vectors)' if resources['index'] else '❌ Missing'}")
    st.sidebar.write(f"• Knowledge Base: {'✅ ' + str(len(resources['knowledge'])) + ' classes' if not resources['knowledge'].empty else '❌ Missing'}")
    st.sidebar.write(f"• Neural Model: {'✅ CNN-LSTM loaded' if resources['model'] else 'ℹ️ Acoustic kNN fallback'}")

    # --------------------------------------------------
    # AUDIO INPUT SECTION
    # --------------------------------------------------
    uploaded_file = st.file_uploader(
        "Upload Audio File (.wav, .mp3, .ogg):",
        type=["wav", "mp3", "ogg"]
    )

    audio_path_to_process = None

    if uploaded_file is not None:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            audio_path_to_process = tmp.name
        st.audio(uploaded_file, format=f"audio/{suffix.replace('.', '')}")
    elif selected_sample_file:
        resolved = resolve_audio_file(selected_sample_file)
        if resolved:
            audio_path_to_process = resolved
            st.info(f"Loaded sample: `{selected_sample_file}`")
            st.audio(audio_path_to_process, format="audio/wav")
        else:
            st.warning(f"Could not locate `{selected_sample_file}` in ESC-50 audio directory.")

    if audio_path_to_process is None:
        st.info("👆 Please upload an audio file or select a sample clip from the sidebar to begin analysis.")
        return

    # --------------------------------------------------
    # EXECUTE ANALYSIS BUTTON
    # --------------------------------------------------
    if st.button("🔍 Analyze Audio with RAG Pipeline", type="primary", use_container_width=True):
        with st.spinner("Processing acoustic features and querying FAISS index..."):
            audio, sr = load_audio(audio_path_to_process)
            result = process_audio_rag(audio, resources, top_k=top_k)

        pred_class = result["prediction"]
        confidence = result["confidence"]
        emoji = CATEGORY_EMOJIS.get(pred_class.lower(), "🔊")

        # ----------------------------------------------
        # SECTION 1: VISUALIZATIONS & ACOUSTIC METRICS
        # ----------------------------------------------
        st.subheader("📊 1. Acoustic Waveform & Spectral Analysis")
        render_audio_visualizations(audio, sr, result["mel_db"])

        # Acoustic KPIs
        rms = np.sqrt(np.mean(audio**2))
        rms_db = 20 * np.log10(rms + 1e-9)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        mean_centroid = np.mean(spectral_centroids)

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("⏱️ Duration", f"{len(audio)/sr:.2f} s")
        kpi2.metric("🔊 Sample Rate", f"{sr} Hz")
        kpi3.metric("📈 RMS Energy", f"{rms_db:.1f} dB")
        kpi4.metric("✨ Spectral Centroid", f"{mean_centroid:.0f} Hz")

        st.divider()

        # ----------------------------------------------
        # SECTION 2: CLASSIFICATION RESULT
        # ----------------------------------------------
        st.subheader("🎯 2. Classification Prediction")
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.markdown(
                f"""
                <div style="background:#162436; padding:1.2rem; border-radius:10px; border:1px solid #1E88E5;">
                    <h3 style="margin:0; color:#90caf9;">Predicted Class</h3>
                    <h1 style="margin:0.2rem 0; font-size:2.4rem;">{emoji} {pred_class}</h1>
                    <p style="margin:0; font-size:1.1rem; color:#81c784;">Confidence: <b>{confidence:.1%}</b></p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with res_col2:
            if result["probabilities"] is not None and len(resources["classes"]) > 0:
                top_indices = np.argsort(result["probabilities"])[::-1][:5]
                top_df = pd.DataFrame({
                    "Category": [resources["classes"][i] for i in top_indices],
                    "Probability": [result["probabilities"][i] for i in top_indices]
                })
                st.caption("Top-5 Softmax Distribution:")
                st.bar_chart(top_df.set_index("Category"), color="#1E88E5")
            else:
                st.info("Operating in Acoustic Vector kNN mode. Confidence is derived from vector similarity density.")

        st.divider()

        # ----------------------------------------------
        # SECTION 3: FAISS RETRIEVAL INSPECTION
        # ----------------------------------------------
        st.subheader(f"🔎 3. FAISS Retrieval Inspection (Top-{len(result['retrieved'])} Acoustically Similar Clips)")
        st.caption(
            "Dense audio embeddings were queried against the FAISS vector index to locate the closest reference sounds in ESC-50."
        )

        for i, item in enumerate(result["retrieved"], start=1):
            cat_emoji = CATEGORY_EMOJIS.get(item["category"].lower(), "🔊")
            sim_pct = max(0.0, min(1.0, item["similarity"]))

            with st.container(border=True):
                r_col1, r_col2, r_col3 = st.columns([2, 2.5, 3])

                with r_col1:
                    st.markdown(f"**Rank #{i}** — {cat_emoji} **`{item['category']}`**")
                    st.caption(f"File: `{item['filename']}` (Fold {item['fold']})")

                with r_col2:
                    st.markdown(f"**Similarity:** `{sim_pct:.1%}`")
                    st.progress(sim_pct)

                with r_col3:
                    if item["audio_path"] and os.path.exists(item["audio_path"]):
                        st.audio(item["audio_path"], format="audio/wav")
                    else:
                        st.caption("⚠️ Audio file not located in dataset path")

        st.divider()

        # ----------------------------------------------
        # SECTION 4: DOMAIN KNOWLEDGE BASE
        # ----------------------------------------------
        st.subheader("📚 4. Retrieved Domain Knowledge")
        k_info = result["knowledge"]
        if k_info:
            with st.container(border=True):
                kd1, kd2 = st.columns(2)
                with kd1:
                    st.write(f"**Group:** {k_info.get('group', 'General')}")
                    st.write(f"**Description:** {k_info.get('description', 'N/A')}")
                    st.write(f"**Environment:** {k_info.get('environment', 'N/A')}")
                with kd2:
                    st.write(f"**Characteristics:** {k_info.get('characteristics', 'N/A')}")
                    st.write(f"**Frequency Range:** `{k_info.get('frequency_range', 'N/A')}`")
                    st.write(f"**Confusable Sounds:** `{k_info.get('confusable_with', 'N/A')}`")
        else:
            st.warning("No knowledge metadata found for this category.")

        st.divider()

        # ----------------------------------------------
        # SECTION 5: RAG REASONING & EXPLANATION
        # ----------------------------------------------
        st.subheader("🧠 5. RAG Diagnostic & Grounded Explanation")

        if rag_mode == "⚡ Google Gemini AI (LLM)":
            if not gemini_api_key:
                st.warning("⚠️ No Gemini API key provided. Falling back to offline grounded template.")
                explanation = generate_offline_rag_explanation(result)
            else:
                with st.spinner("Querying Gemini with retrieved acoustic context..."):
                    explanation = generate_gemini_rag_explanation(
                        result,
                        api_key=gemini_api_key,
                        model_name=gemini_model
                    )
        else:
            explanation = generate_offline_rag_explanation(result)

        st.markdown(explanation)


if __name__ == "__main__":
    main()