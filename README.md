# 🔮 Next Word Predictor — Streamlit App

## Setup

1. **Place all files in the same folder:**
   ```
   your-folder/
   ├── app.py
   ├── model.h5
   ├── tokenizer.pickle
   └── max_len.pkl
   ```

2. **Install dependencies:**
   ```bash
   pip install streamlit tensorflow numpy
   ```

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## Features
- 🔤 Enter any seed text and predict the next N words
- 🌡️ Temperature slider to control creativity / randomness
- 🎯 Top-5 candidate words with probabilities shown
- 📊 Stats: vocab size, sequence length, input/output word counts
