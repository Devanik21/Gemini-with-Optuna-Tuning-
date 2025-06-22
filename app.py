import streamlit as st
import google.generativeai as genai
import optuna
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import logging

# Suppress optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("optuna").setLevel(logging.WARNING)

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
@st.cache_data
def train_rnn_model(text_corpus):
    """Train LSTM model on the given text corpus"""
    try:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts([text_corpus])
        total_words = len(tokenizer.word_index) + 1

        input_sequences = []
        tokens = tokenizer.texts_to_sequences([text_corpus])[0]
        
        if len(tokens) < 2:
            st.error("Text corpus is too short. Please provide more text.")
            return None, None, None
            
        for i in range(1, len(tokens)):
            input_sequences.append(tokens[:i+1])

        if not input_sequences:
            st.error("Could not create input sequences. Please check your text.")
            return None, None, None

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

        # Train with reduced verbosity
        model.fit(X, y, epochs=20, verbose=0)  # Reduced epochs and verbosity for Streamlit
        return model, tokenizer, max_seq_len
    except Exception as e:
        st.error(f"Error training LSTM model: {str(e)}")
        return None, None, None

# --- Define Optuna Objective Function ---
def make_objective(text_input, model):
    def objective(trial):
        template = trial.suggest_categorical("template", prompt_templates)
        full_prompt = template.format(text_input)
        try:
            response = model.generate_content(full_prompt)
            if response and response.text:
                output = response.text.strip().split()
                # Score based on first word length (shorter is better for next word prediction)
                score = len(output[0]) if output else 10
            else:
                score = 10
        except Exception as e:
            st.warning(f"API call failed: {str(e)}")
            score = 10
        return score
    return objective

# --- Main App Logic ---
def main():
    st.title("🔮 Gemini + LSTM Next Word Predictor")
    
    if not api_key:
        st.warning("⚠️ Please enter your Google AI API key in the sidebar to begin.")
        st.info("Get your API key from: https://aistudio.google.com/app/apikey")
        return

    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        st.sidebar.success("✅ Gemini model configured successfully!")

        # Input fields
        text_input = st.text_input(
            "Enter partial sentence:", 
            value="The stars are",
            help="Enter the beginning of a sentence to predict the next word"
        )
        
        corpus_input = st.text_area(
            "Training Corpus (for LSTM)", 
            value="The stars are shining bright in the sky. The sky is full of stars. Stars twinkle in the darkness. The night sky shows many stars.",
            height=100,
            help="Provide text data to train the LSTM model"
        )

        # Create two columns for buttons
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔍 Tune Best Prompt", use_container_width=True):
                if not text_input.strip():
                    st.error("Please enter some text to analyze.")
                    return
                    
                with st.spinner("🔍 Tuning prompt with Optuna..."):
                    try:
                        objective = make_objective(text_input, model)
                        study = optuna.create_study(direction="minimize")
                        study.optimize(objective, n_trials=5, show_progress_bar=False)  # Reduced trials for faster execution
                        
                        best_template = study.best_params["template"]
                        final_prompt = best_template.format(text_input)
                        
                        response = model.generate_content(final_prompt)
                        if response and response.text:
                            final_output = response.text.strip()
                            st.success(f"✨ **Best Prompt:** {best_template}")
                            st.markdown(f"**Gemini Prediction:** {final_output}")
                        else:
                            st.error("No response from Gemini model")
                            
                    except Exception as e:
                        st.error(f"Error during prompt tuning: {str(e)}")

        with col2:
            if st.button("🧠 Train LSTM & Predict", use_container_width=True):
                if not text_input.strip() or not corpus_input.strip():
                    st.error("Please enter both text input and training corpus.")
                    return
                    
                with st.spinner("🧠 Training LSTM model..."):
                    try:
                        lstm_model, tokenizer, max_seq_len = train_rnn_model(corpus_input)
                        
                        if lstm_model is None:
                            return
                            
                        # Make prediction
                        seq = tokenizer.texts_to_sequences([text_input])
                        if not seq or not seq[0]:
                            st.error("Input text not found in training vocabulary. Try different text or expand the corpus.")
                            return
                            
                        seq = pad_sequences(seq, maxlen=max_seq_len-1, padding='pre')
                        preds = lstm_model.predict(seq, verbose=0)
                        pred_index = np.argmax(preds, axis=-1)[0]
                        
                        # Find predicted word
                        predicted_word = ""
                        for word, index in tokenizer.word_index.items():
                            if index == pred_index:
                                predicted_word = word
                                break
                        
                        if predicted_word:
                            st.success(f"🔡 **LSTM Prediction:** {text_input} **{predicted_word}**")
                        else:
                            st.warning("Could not find predicted word in vocabulary.")
                            
                    except Exception as e:
                        st.error(f"Error during LSTM training/prediction: {str(e)}")

        # Add some information
        with st.expander("ℹ️ How it works"):
            st.markdown("""
            **Gemini Prompt Tuning:**
            - Uses Optuna to find the best prompt template
            - Tests different ways to ask Gemini for next word predictions
            - Optimizes based on response quality
            
            **LSTM Training:**
            - Trains a neural network on your custom text corpus
            - Learns patterns in word sequences
            - Predicts the most likely next word based on context
            """)

    except Exception as e:
        st.sidebar.error(f"❌ Configuration error: {str(e)}")
        st.error("Please check your API key and try again.")

if __name__ == "__main__":
    main()

st.markdown("---")
st.caption("💖 Gemini + LSTM = Next word superpowers!")
