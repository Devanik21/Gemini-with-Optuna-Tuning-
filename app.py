import streamlit as st
import google.generativeai as genai
import optuna
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

# --- Sidebar Configuration ---
st.sidebar.title("⚙️ Configuration")
api_key = st.sidebar.text_input("Google AI API Key", type="password", help="Get your key from https://aistudio.google.com/app/apikey")

# --- Sample Prompt Templates ---
prompt_templates = [
    "Predict the next word: '{}'",
    "What comes next after '{}'",
    "Continue the phrase: '{}'",
    "'{}' then?",
    "Next word after '{}':",
]

# --- RNN/LSTM Training Helper ---
def train_rnn_model(text_corpus):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text_corpus])
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    tokens = tokenizer.texts_to_sequences([text_corpus])[0]
    for i in range(1, len(tokens)):
        input_sequences.append(tokens[:i+1])

    max_seq_len = max(len(seq) for seq in input_sequences)
    input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')

    X = input_sequences[:, :-1]
    y = input_sequences[:, -1]
    y = to_categorical(y, num_classes=total_words)

    model = Sequential()
    model.add(Embedding(total_words, 64, input_length=max_seq_len-1))
    model.add(LSTM(64))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X, y, epochs=50, verbose=1)
    return model, tokenizer, max_seq_len

# --- Define Optuna Objective Function ---
def make_objective(text_input, model):
    def objective(trial):
        template = trial.suggest_categorical("template", prompt_templates)
        full_prompt = template.format(text_input)
        try:
            response = model.generate_content(full_prompt)
            output = response.text.strip().split()
            score = len(output[0]) if output else 10
        except Exception:
            score = 10
        return score
    return objective

# --- Main App Logic ---
if api_key:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        st.session_state.model = model
        st.sidebar.success("✨ Gemini model configured!")

        st.title("🔮 Gemini + LSTM Next Word Predictor")
        text_input = st.text_input("Enter partial sentence:", "The stars are")
        corpus_input = st.text_area("Training Corpus (for LSTM)", "The stars are shining bright in the sky. The sky is full of stars.")

        if st.button("🔍 Tune Best Prompt"):
            with st.spinner("Tuning prompt with Optuna..."):
                objective = make_objective(text_input, model)
                study = optuna.create_study(direction="minimize")
                study.optimize(objective, n_trials=10)
                best_template = study.best_params["template"]
                final_prompt = best_template.format(text_input)
                final_output = model.generate_content(final_prompt).text.strip()
                st.success(f"✨ Best Prompt: {best_template}")
                st.markdown(f"**{final_output}**")

        if st.button("🧠 Train LSTM & Predict"):
            with st.spinner("Training LSTM model on custom corpus..."):
                lstm_model, tokenizer, max_seq_len = train_rnn_model(corpus_input)
                seq = tokenizer.texts_to_sequences([text_input])[0]
                seq = pad_sequences([seq], maxlen=max_seq_len-1, padding='pre')
                preds = lstm_model.predict(seq)
                pred_index = np.argmax(preds, axis=-1)[0]
                predicted_word = next((word for word, index in tokenizer.word_index.items() if index == pred_index), "")
                st.success(f"🔡 LSTM Prediction: {text_input} **{predicted_word}**")

    except Exception as e:
        st.sidebar.error(f"Invalid API Key or setup error: {e}")
else:
    st.warning("Please enter your API key to begin.")

st.markdown("---")
st.caption("💖 Gemini + LSTM = next word superpowers~")
