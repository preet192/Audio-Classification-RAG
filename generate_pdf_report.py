import os
import sys
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

# ==============================================================================
# NUMBERED CANVAS (Page X of Y with Running Headers)
# ==============================================================================
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_header_footer(self, total_pages):
        self.saveState()
        page_w, page_h = letter

        # Don't draw header on Page 1
        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#1B365D"))
            self.drawString(54, page_h - 36, "NMIMS MSc Data Science — Acoustic AI & Retrieval-Augmented Generation")
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#7F8C8D"))
            self.drawRightString(page_w - 54, page_h - 36, "ESC-50 Audio RAG Project Report")

            self.setStrokeColor(colors.HexColor("#BDC3C7"))
            self.setLineWidth(0.5)
            self.line(54, page_h - 42, page_w - 54, page_h - 42)

        # Footer on all pages
        self.setStrokeColor(colors.HexColor("#BDC3C7"))
        self.setLineWidth(0.5)
        self.line(54, 45, page_w - 54, 45)

        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#7F8C8D"))
        self.drawString(54, 32, "Confidential — Academic & Technical Evaluation Report")
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(page_w - 54, 32, page_str)
        self.restoreState()


# ==============================================================================
# MAIN REPORT GENERATOR
# ==============================================================================
def create_report(output_filename="Audio_Classification_RAG_Report.pdf"):
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom Color Palette
    PRIMARY = colors.HexColor("#1B365D")      # Deep Navy
    SECONDARY = colors.HexColor("#2980B9")    # Deep Cerulean Blue
    TEXT_DARK = colors.HexColor("#2C3E50")    # Charcoal
    TEXT_MUTED = colors.HexColor("#5D6D7E")   # Muted Gray
    ACCENT = colors.HexColor("#16A085")       # Deep Teal
    BG_LIGHT = colors.HexColor("#F8F9FA")     # Off-white
    BORDER_LIGHT = colors.HexColor("#E2E8F0")

    # Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=PRIMARY,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=SECONDARY,
        spaceAfter=14
    )

    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=PRIMARY,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=SECONDARY,
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=TEXT_DARK,
        spaceAfter=7
    )

    bullet_style = ParagraphStyle(
        "BulletDark",
        parent=body_style,
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=4
    )

    caption_style = ParagraphStyle(
        "CaptionStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=10.5,
        textColor=TEXT_MUTED,
        alignment=1, # Center
        spaceBefore=4,
        spaceAfter=10
    )

    callout_style = ParagraphStyle(
        "CalloutStyle",
        parent=body_style,
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=PRIMARY
    )

    story = []

    # ==============================================================================
    # PAGE 1: TITLE & EXECUTIVE SUMMARY
    # ==============================================================================
    story.append(Paragraph("Acoustic Intelligence & Explainable AI", subtitle_style))
    story.append(Paragraph("Environmental Audio Classification Using Retrieval-Augmented Generation (RAG)", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=PRIMARY, spaceBefore=4, spaceAfter=14))

    # Metadata Grid
    meta_data = [
        [
            Paragraph("<b>Author / Researcher:</b> Preet", body_style),
            Paragraph("<b>Program:</b> MSc Data Science (Sem 3)", body_style)
        ],
        [
            Paragraph("<b>Institution:</b> NMIMS University", body_style),
            Paragraph("<b>Dataset:</b> ESC-50 (50 Classes, 2,000 Clips)", body_style)
        ],
        [
            Paragraph("<b>Core Frameworks:</b> CNN-LSTM, FAISS, Librosa, Streamlit", body_style),
            Paragraph("<b>Generative Engine:</b> Google Gemini 2.5 Flash / Offline RAG", body_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[250, 254])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), HexColor("#F1F5F9") if 'HexColor' in locals() else colors.HexColor("#F1F5F9")),
        ('PADDING', (0,0), (-1,-1), 6),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # Executive Summary Box
    exec_summary_text = (
        "<b>Executive Summary:</b> Environmental sound recognition is critical for autonomous robotics, "
        "smart surveillance, and noise monitoring. While conventional neural classifiers yield high accuracy on closed benchmarks, "
        "they suffer from black-box opacity and struggle with confusable acoustic profiles (e.g., rain vs. sea waves, crackling fire vs. water drops). "
        "This project presents an end-to-end <b>Retrieval-Augmented Generation (RAG) framework</b> for environmental audio classification on the ESC-50 dataset. "
        "By coupling a deep <b>CNN-LSTM acoustic feature extractor</b> with a dense <b>FAISS vector database (2,000 exemplars)</b> and an authoritative "
        "<b>50-class Acoustic Knowledge Base</b>, our pipeline combines parametric deep learning with non-parametric exemplar retrieval. "
        "A hybrid generative layer powered by <b>Google Gemini 2.5 Flash</b> synthesizes structured forensic diagnostic reports, validating consensus "
        "across nearest acoustic neighbors and providing physical sound justification in an interactive <b>Streamlit</b> platform."
    )
    exec_table = Table([[Paragraph(exec_summary_text, callout_style)]], colWidths=[504])
    exec_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EBF5FB")),
        ('PADDING', (0,0), (-1,-1), 10),
        ('LINELEFT', (0,0), (-1,-1), 3.5, SECONDARY),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#BDC3C7")),
    ]))
    story.append(exec_table)
    story.append(Spacer(1, 14))

    # Key Architectural Innovations Table
    story.append(Paragraph("Key Technological Innovations in this Pipeline:", h2_style))
    innovations_data = [
        [Paragraph("<b>Component</b>", body_style), Paragraph("<b>Methodology & Role</b>", body_style), Paragraph("<b>Key Metric / Specification</b>", body_style)],
        [
            Paragraph("<b>Acoustic Encoder</b>", body_style),
            Paragraph("CNN-LSTM network modeling spectral hierarchies and temporal recurrence", body_style),
            Paragraph("3 Conv2D blocks + 2 LSTM layers (128-d latent bottleneck)", body_style)
        ],
        [
            Paragraph("<b>Vector Index (FAISS)</b>", body_style),
            Paragraph("Dense L2-normalized vector similarity search across reference audio clips", body_style),
            Paragraph("faiss.IndexFlatIP indexing 2,000 reference clips in sub-millisecond time", body_style)
        ],
        [
            Paragraph("<b>Domain Knowledge Base</b>", body_style),
            Paragraph("Acoustic physics, spectral bandwidths, environments, and confusable sound cues", body_style),
            Paragraph("50 fully characterized classes spanning all 5 ESC-50 domains", body_style)
        ],
        [
            Paragraph("<b>Generative Explainer</b>", body_style),
            Paragraph("Multimodal diagnostic reasoning, consensus evaluation, and disambiguation", body_style),
            Paragraph("Google Gemini 2.5 Flash + Grounded Template offline fallback", body_style)
        ],
    ]
    innovations_table = Table(innovations_data, colWidths=[120, 240, 144])
    innovations_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('PADDING', (0,0), (-1,-1), 5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8F9FA")]),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
    ]))
    story.append(innovations_table)

    story.append(PageBreak())

    # ==============================================================================
    # PAGE 2: PROBLEM STATEMENT & DATA COLLECTION
    # ==============================================================================
    story.append(Paragraph("1. Problem Statement & Motivation", h1_style))
    story.append(Paragraph(
        "Environmental Sound Classification (ESC) presents unique challenges that differentiate it from speech recognition or music information retrieval. "
        "Natural audio scenes are inherently unstructured, non-stationary, and contaminated by background reverberations and environmental noise. "
        "Standard machine learning pipelines approach this as a closed-world classification task ($P(y \\mid x)$), giving rise to three severe operational shortcomings:",
        body_style
    ))
    story.append(Paragraph("• <b>Lack of Explainability:</b> Softmax probabilities indicate confidence magnitude but provide zero causal explanation regarding why an audio clip was categorized as a specific event (e.g., spectral centroid shifts, harmonic formants, or attack envelopes).", bullet_style))
    story.append(Paragraph("• <b>Vulnerability to Confusable Acoustic Signatures:</b> Environmental sounds frequently share near-identical spectral energy profiles (e.g., rain vs. sizzling bacon, chainsaw vs. helicopter blade-slap, footsteps vs. door knocks). Without contextual grounding, pure parametric models frequently produce high-confidence misclassifications.", bullet_style))
    story.append(Paragraph("• <b>Static Knowledge Deprivation:</b> Neural classifiers cannot query external sound physics taxonomies, environmental contexts, or human-curated distinguishing features during inference.", bullet_style))
    story.append(Paragraph(
        "<b>Project Objective:</b> Develop an explainable Audio Classification framework utilizing Retrieval-Augmented Generation (RAG). "
        "By indexing acoustic embeddings within a FAISS vector database and retrieving matching audio exemplars alongside domain acoustic physics, "
        "the system generates grounded forensic diagnostic reports that explain, verify, and substantiate every prediction.",
        body_style
    ))

    story.append(Spacer(1, 10))
    story.append(Paragraph("2. Data Collection & Preprocessing", h1_style))
    story.append(Paragraph(
        "The project is evaluated on the standard <b>ESC-50 (Dataset for Environmental Sound Classification)</b> curated by Karol J. Piczak. "
        "The dataset provides 2,000 labeled environmental recordings evenly distributed across 50 semantically balanced classes (40 recordings per class, 5 seconds each).",
        body_style
    ))

    # Add Domain Distribution Chart
    if os.path.exists("report_assets/dataset_domains.png"):
        img = Image("report_assets/dataset_domains.png", width=490, height=195)
        story.append(img)
        story.append(Paragraph("Figure 1: ESC-50 Dataset Structure across 5 Semantic Domains (50 Classes, 2,000 Total Recordings).", caption_style))

    story.append(Paragraph(
        "<b>Signal Preprocessing Pipeline:</b><br/>"
        "1. <b>Standardized Resampling:</b> Audio waveforms are loaded and resampled to a standardized sampling rate of $f_s = 22,050\\text{ Hz}$ in single-channel mono format.<br/>"
        "2. <b>Temporal Normalization:</b> Audio clips are padded with zeros or truncated to precisely $5.0\\text{ seconds}$ ($N = 110,250\\text{ samples}$).<br/>"
        "3. <b>Time-Frequency Decomposition (STFT):</b> Using an FFT window length of $N_{\\text{fft}} = 2048$ with a hop length of $H = 512$ ($75\\%$ overlap).<br/>"
        "4. <b>Mel Filterbank Projection:</b> Converted to $128$ Mel frequency bands covering $0\\text{ Hz}$ to $11,025\\text{ Hz}$ (Nyquist frequency).<br/>"
        "5. <b>Dynamic Range Compression:</b> Logarithmic decibel conversion ($S_{\\text{dB}} = 10 \\cdot \\log_{10}(S / S_{\\text{ref}})$) followed by min-max normalization into $[0, 1]$, producing a feature tensor of shape $(128, 216, 1)$.",
        body_style
    ))

    story.append(PageBreak())

    # ==============================================================================
    # PAGE 3: EXPLORATORY DATA ANALYSIS & VISUALIZATION
    # ==============================================================================
    story.append(Paragraph("3. Exploratory Data Analysis (EDA) & Acoustic Visualizations", h1_style))
    story.append(Paragraph(
        "Acoustic exploratory analysis was performed across temporal and spectral dimensions to characterize environmental sound signatures. "
        "Unlike speech signals where energy concentrates in lower formant bands (300 Hz - 3.4 kHz), environmental sounds span vastly distinct physical envelopes:",
        body_style
    ))

    eda_table_data = [
        [Paragraph("<b>Category Domain</b>", body_style), Paragraph("<b>Representative Classes</b>", body_style), Paragraph("<b>Acoustic Profile & Frequency Bands</b>", body_style)],
        [
            Paragraph("<b>Animals</b>", body_style),
            Paragraph("Dog, Cat, Rooster, Cow, Frog, Pig", body_style),
            Paragraph("Harmonic formants, rapid frequency modulation, impulsive bursts (250 Hz - 4.5 kHz)", body_style)
        ],
        [
            Paragraph("<b>Natural Soundscapes</b>", body_style),
            Paragraph("Rain, Sea Waves, Thunderstorm, Fire", body_style),
            Paragraph("Continuous broadband noise, sub-bass shockwaves (20 Hz - 8 kHz)", body_style)
        ],
        [
            Paragraph("<b>Human Non-Speech</b>", body_style),
            Paragraph("Crying Baby, Sneezing, Coughing, Clapping", body_style),
            Paragraph("Transient explosive bursts, high-pitch vowel sweeps (300 Hz - 6 kHz)", body_style)
        ],
        [
            Paragraph("<b>Interior Domestic</b>", body_style),
            Paragraph("Clock Tick, Mouse Click, Keyboard, Door Knock", body_style),
            Paragraph("Sparse microsecond percussive clicks, high-frequency transients (1 kHz - 10 kHz)", body_style)
        ],
        [
            Paragraph("<b>Exterior / Urban</b>", body_style),
            Paragraph("Siren, Car Horn, Chainsaw, Helicopter, Train", body_style),
            Paragraph("High-amplitude mechanical roars, cyclic wailing sweeps, blade-slap pulses", body_style)
        ],
    ]
    eda_table = Table(eda_table_data, colWidths=[120, 160, 224])
    eda_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), SECONDARY),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('PADDING', (0,0), (-1,-1), 5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8F9FA")]),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
    ]))
    story.append(eda_table)
    story.append(Spacer(1, 10))

    # Add EDA Waveform and Spectrogram Plots
    if os.path.exists("report_assets/eda_waveform.png") and os.path.exists("report_assets/eda_melspectrogram.png"):
        story.append(Paragraph("Acoustic Signature Comparison: Raw Waveform vs. Log-Mel Spectrogram", h2_style))
        w_img = Image("report_assets/eda_waveform.png", width=490, height=135)
        story.append(w_img)
        story.append(Paragraph("Figure 2: Time-Domain Waveform of Audio Exemplar demonstrating transient attack and decay envelope.", caption_style))

        s_img = Image("report_assets/eda_melspectrogram.png", width=490, height=160)
        story.append(s_img)
        story.append(Paragraph("Figure 3: Log-Mel Spectrogram (128 Mel bands) illustrating spectral energy density across temporal frames.", caption_style))

    story.append(PageBreak())

    # ==============================================================================
    # PAGE 4: MODEL ARCHITECTURE & RAG PIPELINE
    # ==============================================================================
    story.append(Paragraph("4. Model Building & Deep Architecture", h1_style))
    story.append(Paragraph(
        "A hybrid <b>CNN-LSTM deep architecture</b> was constructed to capture both spatial-spectral hierarchies and temporal dependencies across audio frames. "
        "The model is explicitly configured with an intermediate dense bottleneck layer (`audio_embedding`) that generates compact 128-dimensional dense vectors for vector indexing.",
        body_style
    ))

    arch_table_data = [
        [Paragraph("<b>Layer Block</b>", body_style), Paragraph("<b>Layer Operations</b>", body_style), Paragraph("<b>Output Dimensions</b>", body_style), Paragraph("<b>Parameters / Role</b>", body_style)],
        [
            Paragraph("<b>Input</b>", body_style),
            Paragraph("InputLayer(shape=(128, 216, 1))", body_style),
            Paragraph("(None, 128, 216, 1)", body_style),
            Paragraph("Raw Log-Mel Spectrogram", body_style)
        ],
        [
            Paragraph("<b>CNN Block 1</b>", body_style),
            Paragraph("Conv2D(32, 3x3) + BatchNorm + MaxPool(2x2)", body_style),
            Paragraph("(None, 64, 108, 32)", body_style),
            Paragraph("Low-level spectral edge filters", body_style)
        ],
        [
            Paragraph("<b>CNN Block 2</b>", body_style),
            Paragraph("Conv2D(64, 3x3) + BatchNorm + MaxPool(2x2)", body_style),
            Paragraph("(None, 32, 54, 64)", body_style),
            Paragraph("Mid-level harmonic patterns", body_style)
        ],
        [
            Paragraph("<b>CNN Block 3</b>", body_style),
            Paragraph("Conv2D(128, 3x3) + BatchNorm + MaxPool(2x2)", body_style),
            Paragraph("(None, 16, 27, 128)", body_style),
            Paragraph("High-level acoustic textures", body_style)
        ],
        [
            Paragraph("<b>Permute & Reshape</b>", body_style),
            Paragraph("Permute((2, 1, 3)) + TimeDistributed(Flatten)", body_style),
            Paragraph("(None, 27, 2048)", body_style),
            Paragraph("Transition from spatial to temporal sequence", body_style)
        ],
        [
            Paragraph("<b>LSTM Recurrence</b>", body_style),
            Paragraph("LSTM(128, return_seq=True) + Drop(0.3) + LSTM(64)", body_style),
            Paragraph("(None, 64)", body_style),
            Paragraph("Temporal dynamics & sequence memory", body_style)
        ],
        [
            Paragraph("<b>Embedding Bottleneck</b>", body_style),
            Paragraph("Dense(128, relu, name='audio_embedding')", body_style),
            Paragraph("(None, 128)", body_style),
            Paragraph("<b>Dense Acoustic Vector for FAISS</b>", body_style)
        ],
        [
            Paragraph("<b>Classifier Head</b>", body_style),
            Paragraph("Dropout(0.4) + Dense(50, softmax)", body_style),
            Paragraph("(None, 50)", body_style),
            Paragraph("Categorical class distribution", body_style)
        ],
    ]
    arch_table = Table(arch_table_data, colWidths=[100, 180, 110, 114])
    arch_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('PADDING', (0,0), (-1,-1), 4),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8F9FA")]),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
    ]))
    story.append(arch_table)
    story.append(Spacer(1, 10))

    # Add Architecture Diagram
    if os.path.exists("report_assets/rag_architecture.png"):
        story.append(Paragraph("5. Dual-Retrieval Vector Database & RAG Pipeline", h1_style))
        r_img = Image("report_assets/rag_architecture.png", width=490, height=190)
        story.append(r_img)
        story.append(Paragraph("Figure 4: End-to-End Audio Classification RAG System Architecture Flowchart.", caption_style))

    story.append(Paragraph(
        "<b>Dual-Retrieval Mechanism:</b><br/>"
        "• <b>Acoustic Vector Database (FAISS):</b> All 2,000 reference clips in ESC-50 are encoded into 128-dimensional dense vectors and $\\ell_2$-normalized. "
        "A `faiss.IndexFlatIP` index performs exact inner-product (cosine) search in sub-millisecond latency to retrieve the top-$k$ nearest neighbors.<br/>"
        "• <b>Authoritative Knowledge Base:</b> A curated database across all 50 categories storing sound origin, acoustic physics, frequency bandwidths, and confusable sound cues.",
        body_style
    ))

    story.append(PageBreak())

    # ==============================================================================
    # PAGE 5: TRAINING, VALIDATION & TESTING EVALUATION
    # ==============================================================================
    story.append(Paragraph("6. Model Training, Testing & Validation Results", h1_style))
    story.append(Paragraph(
        "The model was trained on the preprocessed ESC-50 dataset using an 80/20 train-validation split. "
        "Optimization was performed with Adam (initial learning rate $\\eta = 0.001$) under Sparse Categorical Cross-Entropy loss over 100 epochs with ModelCheckpoint callbacks.",
        body_style
    ))

    # Loss and Accuracy Plots
    if os.path.exists("report_assets/model_accuracy_curve.png") and os.path.exists("report_assets/model_loss_curve.png"):
        perf_data = [
            [
                Image("report_assets/model_accuracy_curve.png", width=245, height=140),
                Image("report_assets/model_loss_curve.png", width=245, height=140)
            ],
            [
                Paragraph("Figure 5: Training and Validation Accuracy across Epochs.", caption_style),
                Paragraph("Figure 6: Training and Validation Loss across Epochs.", caption_style)
            ]
        ]
        perf_table = Table(perf_data, colWidths=[250, 254])
        perf_table.setStyle(TableStyle([
            ('PADDING', (0,0), (-1,-1), 0),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ]))
        story.append(perf_table)

    story.append(Spacer(1, 8))

    # Confusion Matrix Plot
    if os.path.exists("report_assets/confusion_matrix.png"):
        story.append(Paragraph("Confusion Matrix & Classification Behavior Analysis", h2_style))
        cm_img = Image("report_assets/confusion_matrix.png", width=360, height=270)
        story.append(cm_img)
        story.append(Paragraph("Figure 7: Confusion Matrix Heatmap across all 50 Environmental Sound Classes.", caption_style))

    story.append(Paragraph(
        "<b>Training Observations & The Justification for RAG:</b><br/>"
        "As visible in the training curves and confusion matrix, training a 50-class CNN-LSTM directly from scratch on 2,000 clips (only 40 examples per class) "
        "faces substantial generalization bottlenecks without heavy data augmentation or multi-million audio pretraining. "
        "This empirical reality is precisely why <b>Retrieval-Augmented Generation (RAG) is a game-changer</b>: rather than relying solely on a model's weights, "
        "FAISS retrieves nearest real-world audio exemplars and domain physics, allowing the generative explainer to arbitrate and disambiguate edge cases.",
        body_style
    ))

    story.append(PageBreak())

    # ==============================================================================
    # PAGE 6: CONCLUSION & STRATEGIC RECOMMENDATIONS
    # ==============================================================================
    story.append(Paragraph("7. Streamlit Web Deployment & Generative Reasoning", h1_style))
    story.append(Paragraph(
        "The complete system was deployed as an interactive, production-ready web application in <b>Streamlit (`app.py`)</b>. "
        "The interface enables end users to upload arbitrary audio recordings or select pre-loaded dataset samples to trigger the full RAG pipeline:",
        body_style
    ))
    story.append(Paragraph("• <b>Dual Acoustic Visualizations:</b> Real-time side-by-side rendering of time-domain waveforms and Log-Mel spectrograms with calibrated dB colorbars.", bullet_style))
    story.append(Paragraph("• <b>Audible FAISS Inspection:</b> Retrieved top-$k$ exemplars feature dynamic similarity percentage progress bars and embedded audio players allowing users to listen to the matching reference sounds.", bullet_style))
    story.append(Paragraph("• <b>Hybrid RAG Generation:</b> Supports both <b>Google Gemini 2.5 Flash</b> (via `google-genai` SDK) and an <b>Offline Grounded Engine</b> to generate structured forensic reports evaluating neighbor consensus, frequency physics, and confusable sound cues.", bullet_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph("8. Conclusion & Strategic Recommendations", h1_style))
    story.append(Paragraph(
        "<b>Conclusion:</b> This project demonstrates that combining deep acoustic representations with vector similarity retrieval (FAISS) "
        "and generative domain reasoning (RAG) overcomes the core explainability and disambiguation bottlenecks of traditional environmental sound classification. "
        "The framework achieves three core milestones:",
        body_style
    ))
    story.append(Paragraph("1. <b>Multi-Modal Grounding:</b> Neural predictions are verified against factual reference exemplars in the vector space.", bullet_style))
    story.append(Paragraph("2. <b>Acoustic Physics Explainability:</b> Replaces bare confidence scores with human-interpretable forensic sound reports.", bullet_style))
    story.append(Paragraph("3. <b>Zero-Failure Resilience:</b> The system seamlessly transitions between deep learning inference, kNN vector retrieval, and offline/cloud generative modes.", bullet_style))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Strategic Recommendations for Future Evolution:", h2_style))

    recs_data = [
        [Paragraph("<b>Evolution Area</b>", body_style), Paragraph("<b>Current Architecture</b>", body_style), Paragraph("<b>Recommended Next-Generation Upgrade</b>", body_style)],
        [
            Paragraph("<b>Audio Foundation Embeddings</b>", body_style),
            Paragraph("Custom CNN-LSTM bottleneck (128-d)", body_style),
            Paragraph("Integrate zero-shot pretrained models such as <b>CLAP (Contrastive Language-Audio Pretraining)</b> or <b>AST (Audio Spectrogram Transformer)</b> for generalized 768-d embeddings.", body_style)
        ],
        [
            Paragraph("<b>Vector Index Scaling</b>", body_style),
            Paragraph("FAISS IndexFlatIP (2,000 clips)", body_style),
            Paragraph("Transition to <b>HNSW (Hierarchical Navigable Small World)</b> or <b>IVFFlat</b> to support multi-million sample databases (e.g., AudioSet with 2M+ clips).", body_style)
        ],
        [
            Paragraph("<b>Real-Time Audio Streaming</b>", body_style),
            Paragraph("File-based batch processing (5.0s clips)", body_style),
            Paragraph("Implement sliding-window audio chunking (0.5s hop) with WebSockets to enable live acoustic surveillance and sound event detection.", body_style)
        ],
        [
            Paragraph("<b>Edge Model Quantization</b>", body_style),
            Paragraph("Full float32 Keras model", body_style),
            Paragraph("Quantize the embedding extractor using <b>ONNX Runtime / TFLite (INT8)</b> for edge microcontrollers and smart IoT acoustic sensors.", body_style)
        ],
    ]
    recs_table = Table(recs_data, colWidths=[130, 150, 224])
    recs_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('PADDING', (0,0), (-1,-1), 5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8F9FA")]),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
    ]))
    story.append(recs_table)
    story.append(Spacer(1, 14))

    # Verification Sign-off Box
    signoff_text = (
        "<b>Project Verification:</b> All source code, trained artifacts, vector indexes, and documentation have been tested, "
        "packaged, and pushed to GitHub repository: <b>https://github.com/preet192/Audio-Classification-RAG</b>"
    )
    signoff_table = Table([[Paragraph(signoff_text, callout_style)]], colWidths=[504])
    signoff_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#E8F8F5")),
        ('PADDING', (0,0), (-1,-1), 8),
        ('LINELEFT', (0,0), (-1,-1), 3.5, ACCENT),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#A3E4D7")),
    ]))
    story.append(signoff_table)

    # Build the document with NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF report: {output_filename}")


if __name__ == "__main__":
    create_report()
