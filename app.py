import streamlit as st
import numpy as np
import pickle
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🔮",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #111827 50%, #0d1f2d 100%);
    min-height: 100vh;
}

/* ── Header ── */
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #e879f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    letter-spacing: -1px;
    margin-bottom: 0.2rem;
}
.hero-sub {
    text-align: center;
    color: #64748b;
    font-size: 0.95rem;
    font-weight: 300;
    margin-bottom: 2rem;
    letter-spacing: 0.05em;
}

/* ── Card ── */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 2rem 2.2rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
}

/* ── Section label ── */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #38bdf8;
    margin-bottom: 0.6rem;
}

/* ── Text area override ── */
.stTextArea textarea {
    background: rgba(15, 23, 42, 0.9) !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 1rem !important;
    line-height: 1.7 !important;
    padding: 14px 16px !important;
    caret-color: #38bdf8;
}
.stTextArea textarea:focus {
    border-color: rgba(129, 140, 248, 0.6) !important;
    box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.15) !important;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] {
    padding: 0 !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2rem !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.1s !important;
    box-shadow: 0 4px 24px rgba(99, 102, 241, 0.35) !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Result box ── */
.result-box {
    background: linear-gradient(135deg, rgba(56,189,248,0.07), rgba(232,121,249,0.07));
    border: 1px solid rgba(129, 140, 248, 0.25);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-top: 1.2rem;
}
.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: #818cf8;
    margin-bottom: 0.5rem;
}
.result-text {
    font-size: 1.25rem;
    font-weight: 600;
    color: #f1f5f9;
    line-height: 1.6;
}
.result-new-words {
    color: #e879f9;
    font-weight: 700;
}

/* ── Prediction chips ── */
.chips-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 1rem;
}
.chip {
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.25);
    border-radius: 999px;
    padding: 0.3rem 0.9rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #38bdf8;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
}
.chip .prob {
    color: #64748b;
    font-size: 0.7rem;
}

/* ── Stats bar ── */
.stats-row {
    display: flex;
    gap: 1rem;
    margin-top: 1.2rem;
}
.stat-pill {
    flex: 1;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    text-align: center;
}
.stat-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.2rem;
    font-weight: 700;
    color: #38bdf8;
}
.stat-key {
    font-size: 0.7rem;
    color: #64748b;
    margin-top: 0.15rem;
    letter-spacing: 0.05em;
}

/* ── Info / warning ── */
.stAlert {
    border-radius: 12px !important;
}

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Load model & artifacts ─────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    errors = []

    # Tokenizer
    try:
        with open("tokenizer.pickle", "rb") as f:
            tokenizer = pickle.load(f)
    except FileNotFoundError:
        tokenizer = None
        errors.append("tokenizer.pickle not found")

    # Max len
    try:
        with open("max_len.pkl", "rb") as f:
            max_len = pickle.load(f)
    except FileNotFoundError:
        max_len = 10
        errors.append("max_len.pkl not found — defaulting to 10")

    # Model (lazy import so app still renders without TF)
    model = None
    try:
        from tensorflow.keras.models import load_model
        model = load_model("model.h5")
    except ImportError:
        errors.append("TensorFlow/Keras not installed — run: pip install tensorflow")
    except Exception as e:
        errors.append(f"Could not load model.h5: {e}")

    return tokenizer, max_len, model, errors


def predict_next_words(seed_text, n_words, tokenizer, max_len, model, top_k=5, temperature=1.0):
    """Generate n_words next words and also return top-k candidates for final step."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    generated = []
    current_text = seed_text

    for i in range(n_words):
        token_list = tokenizer.texts_to_sequences([current_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_len - 1, padding="pre")
        predicted_probs = model.predict(token_list, verbose=0)[0]

        # Temperature scaling
        predicted_probs = np.log(predicted_probs + 1e-10) / temperature
        predicted_probs = np.exp(predicted_probs)
        predicted_probs = predicted_probs / predicted_probs.sum()

        # Top-k for final word
        if i == n_words - 1:
            top_indices = np.argsort(predicted_probs)[-top_k:][::-1]
            top_words = []
            for idx in top_indices:
                word = next((w for w, ti in tokenizer.word_index.items() if ti == idx), None)
                if word:
                    top_words.append((word, float(predicted_probs[idx])))
        else:
            top_words = None

        # Sample word
        next_index = np.random.choice(len(predicted_probs), p=predicted_probs)
        next_word = next((w for w, ti in tokenizer.word_index.items() if ti == next_index), "<unk>")
        generated.append(next_word)
        current_text += " " + next_word

    return generated, top_words


# ── UI ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🔮 Next Word Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">LSTM · Deep Language Model</div>', unsafe_allow_html=True)

# Load
with st.spinner("Loading model…"):
    tokenizer, max_len, model, load_errors = load_artifacts()

if load_errors:
    for err in load_errors:
        st.warning(f"⚠️ {err}", icon="⚠️")
    st.info("📂 Place **model.h5**, **tokenizer.pickle**, and **max_len.pkl** in the same folder as this script, then run:\n```bash\nstreamlit run app.py\n```")

# ── Input card ────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Seed Text</div>', unsafe_allow_html=True)
seed_text = st.text_area(
    label="",
    placeholder="Start typing your sentence here…",
    height=110,
    label_visibility="collapsed",
)

col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="section-label">Words to predict</div>', unsafe_allow_html=True)
    n_words = st.slider("", min_value=1, max_value=20, value=3, label_visibility="collapsed")
with col2:
    st.markdown('<div class="section-label">Creativity (temperature)</div>', unsafe_allow_html=True)
    temperature = st.slider("", min_value=0.1, max_value=2.0, value=1.0, step=0.1, label_visibility="collapsed")

predict_btn = st.button("✦ Generate Prediction")
st.markdown('</div>', unsafe_allow_html=True)

# ── Prediction ─────────────────────────────────────────────────────────────
if predict_btn:
    if not seed_text.strip():
        st.error("Please enter some seed text first.")
    elif model is None or tokenizer is None:
        st.error("Model or tokenizer not loaded. Check the warnings above.")
    else:
        with st.spinner("Predicting…"):
            try:
                generated_words, top_candidates = predict_next_words(
                    seed_text.strip(), n_words, tokenizer, max_len, model,
                    top_k=5, temperature=temperature
                )
                full_output = seed_text.strip() + " " + " ".join(generated_words)

                # Result box
                seed_html = seed_text.strip().replace("<", "&lt;").replace(">", "&gt;")
                new_html = " ".join(generated_words).replace("<", "&lt;").replace(">", "&gt;")
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-label">Predicted Output</div>
                    <div class="result-text">
                        {seed_html} <span class="result-new-words">{new_html}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Stats row
                st.markdown(f"""
                <div class="stats-row">
                    <div class="stat-pill">
                        <div class="stat-value">{len(seed_text.split())}</div>
                        <div class="stat-key">Input Words</div>
                    </div>
                    <div class="stat-pill">
                        <div class="stat-value">{n_words}</div>
                        <div class="stat-key">Generated</div>
                    </div>
                    <div class="stat-pill">
                        <div class="stat-value">{max_len}</div>
                        <div class="stat-key">Max Seq Len</div>
                    </div>
                    <div class="stat-pill">
                        <div class="stat-value">{len(tokenizer.word_index):,}</div>
                        <div class="stat-key">Vocab Size</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Top-k candidates
                if top_candidates:
                    st.markdown("---")
                    st.markdown('<div class="section-label" style="margin-top:1rem">Top candidates for final word</div>', unsafe_allow_html=True)
                    chips_html = '<div class="chips-row">'
                    for word, prob in top_candidates:
                        chips_html += f'<div class="chip">{word} <span class="prob">{prob*100:.1f}%</span></div>'
                    chips_html += '</div>'
                    st.markdown(chips_html, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ── Footer hint ────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:3rem;color:#1e293b;font-size:0.75rem;font-family:'Space Mono',monospace;">
    LSTM · Next Word Prediction · Built with Streamlit
</div>
""", unsafe_allow_html=True)
