import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import optuna
import numpy as np
import pickle
import re
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Tuple
import io
import time

# Set page config
st.set_page_config(
    page_title="Next Word Prediction with Optuna",
    page_icon="🔮",
    layout="wide"
)

class TextDataset(Dataset):
    def __init__(self, sequences, vocab_to_idx, seq_length=10):
        self.sequences = sequences
        self.vocab_to_idx = vocab_to_idx
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.sequences) - self.seq_length
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx:idx + self.seq_length]
        target = self.sequences[idx + self.seq_length]
        
        # Convert to indices
        seq_indices = [self.vocab_to_idx.get(word, 0) for word in sequence]
        target_idx = self.vocab_to_idx.get(target, 0)
        
        return torch.tensor(seq_indices, dtype=torch.long), torch.tensor(target_idx, dtype=torch.long)

class NextWordLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.3):
        super(NextWordLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.dropout(lstm_out[:, -1, :])  # Take last output
        output = self.fc(output)
        return output

def preprocess_text(text: str) -> List[str]:
    """Preprocess text and return list of words"""
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    words = text.split()
    return words

def create_vocabulary(words: List[str], min_freq: int = 2) -> Tuple[Dict, Dict]:
    """Create vocabulary mappings"""
    word_counts = Counter(words)
    vocab = ['<UNK>'] + [word for word, count in word_counts.items() if count >= min_freq]
    
    vocab_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_vocab = {idx: word for word, idx in vocab_to_idx.items()}
    
    return vocab_to_idx, idx_to_vocab

def objective(trial, train_loader, val_loader, vocab_size, device):
    """Optuna objective function"""
    # Hyperparameters to optimize
    embedding_dim = trial.suggest_int('embedding_dim', 50, 200, step=25)
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256, step=32)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    # Create model
    model = NextWordLSTM(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(10):  # Limited epochs for optimization
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx * batch_size >= 500:  # Limit batches for speed
                break
                
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = correct / total
    return accuracy

def train_model(model, train_loader, val_loader, device, epochs=20):
    """Train the model with progress tracking"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Update progress
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    
    return train_losses, val_losses, val_accuracies

def predict_next_word(model, input_text, vocab_to_idx, idx_to_vocab, device, seq_length=10, top_k=5):
    """Predict next word given input text"""
    model.eval()
    
    words = preprocess_text(input_text)
    if len(words) < seq_length:
        words = ['<UNK>'] * (seq_length - len(words)) + words
    else:
        words = words[-seq_length:]
    
    # Convert to indices
    indices = [vocab_to_idx.get(word, 0) for word in words]
    input_tensor = torch.tensor([indices], dtype=torch.long).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        predictions = []
        for i in range(top_k):
            word = idx_to_vocab[top_indices[0][i].item()]
            prob = top_probs[0][i].item()
            predictions.append((word, prob))
    
    return predictions

# Streamlit App
def main():
    st.title("🔮 Next Word Prediction with Optuna Fine-tuning")
    st.markdown("A neural language model with hyperparameter optimization using Optuna")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Initialize session state
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'vocab_to_idx' not in st.session_state:
        st.session_state.vocab_to_idx = None
    if 'idx_to_vocab' not in st.session_state:
        st.session_state.idx_to_vocab = None
    
    # Sample text data
    sample_texts = {
        "Shakespeare": "To be or not to be that is the question whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune or to take arms against a sea of troubles and by opposing end them to die to sleep no more and by a sleep to say we end the heartache and the thousand natural shocks that flesh is heir to",
        "Tech Article": "Machine learning algorithms are revolutionizing the way we process data and make predictions artificial intelligence models can learn from vast amounts of information to identify patterns and generate insights deep learning neural networks have become particularly powerful for tasks like image recognition natural language processing and predictive analytics",
        "News Sample": "The stock market showed mixed results today with technology companies leading gains while energy sectors declined investors remain cautious about economic indicators and federal reserve policy decisions unemployment rates continue to improve but inflation concerns persist"
    }
    
    # Text input
    st.header("1. Data Input")
    text_source = st.selectbox("Choose text source:", ["Sample Text", "Custom Text"])
    
    if text_source == "Sample Text":
        selected_sample = st.selectbox("Select sample text:", list(sample_texts.keys()))
        input_text = sample_texts[selected_sample]
        st.text_area("Text data:", input_text, height=100, disabled=True)
    else:
        input_text = st.text_area("Enter your text data:", height=150, 
                                 placeholder="Enter a large text corpus for training...")
    
    # Hyperparameter optimization
    st.header("2. Hyperparameter Optimization with Optuna")
    
    col1, col2 = st.columns(2)
    with col1:
        n_trials = st.slider("Number of Optuna trials:", 5, 50, 10)
        seq_length = st.slider("Sequence length:", 5, 20, 10)
    
    with col2:
        min_vocab_freq = st.slider("Minimum word frequency:", 1, 5, 2)
        val_split = st.slider("Validation split:", 0.1, 0.3, 0.2)
    
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.info(f"Using device: {device}")
    
    if st.button("🚀 Start Optimization & Training"):
        if not input_text.strip():
            st.error("Please provide text data!")
            return
        
        with st.spinner("Processing text and optimizing hyperparameters..."):
            # Preprocess text
            words = preprocess_text(input_text)
            if len(words) < 100:
                st.error("Text too short! Please provide more text data.")
                return
            
            st.success(f"Processed {len(words)} words")
            
            # Create vocabulary
            vocab_to_idx, idx_to_vocab = create_vocabulary(words, min_vocab_freq)
            vocab_size = len(vocab_to_idx)
            st.info(f"Vocabulary size: {vocab_size}")
            
            # Create dataset
            dataset = TextDataset(words, vocab_to_idx, seq_length)
            
            # Split data
            train_size = int((1 - val_split) * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # Optuna optimization
            st.subheader("Optuna Hyperparameter Optimization")
            study = optuna.create_study(direction='maximize')
            
            progress_container = st.container()
            with progress_container:
                optuna_progress = st.progress(0)
                optuna_status = st.empty()
                
                for i in range(n_trials):
                    study.optimize(lambda trial: objective(trial, train_loader, val_loader, vocab_size, device), n_trials=1)
                    optuna_progress.progress((i + 1) / n_trials)
                    optuna_status.text(f"Trial {i+1}/{n_trials} - Best accuracy: {study.best_value:.4f}")
            
            best_params = study.best_params
            st.success("Optimization completed!")
            
            # Display best parameters
            st.subheader("Best Hyperparameters")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Embedding Dim", best_params['embedding_dim'])
                st.metric("Hidden Dim", best_params['hidden_dim'])
                st.metric("Num Layers", best_params['num_layers'])
            
            with col2:
                st.metric("Dropout", f"{best_params['dropout']:.3f}")
                st.metric("Learning Rate", f"{best_params['learning_rate']:.2e}")
                st.metric("Best Accuracy", f"{study.best_value:.4f}")
            
            # Train final model with best parameters
            st.subheader("Training Final Model")
            final_model = NextWordLSTM(
                vocab_size=vocab_size,
                embedding_dim=best_params['embedding_dim'],
                hidden_dim=best_params['hidden_dim'],
                num_layers=best_params['num_layers'],
                dropout=best_params['dropout']
            ).to(device)
            
            train_losses, val_losses, val_accuracies = train_model(
                final_model, train_loader, val_loader, device, epochs=30
            )
            
            # Store in session state
            st.session_state.model = final_model
            st.session_state.vocab_to_idx = vocab_to_idx
            st.session_state.idx_to_vocab = idx_to_vocab
            st.session_state.model_trained = True
            st.session_state.device = device
            st.session_state.seq_length = seq_length
            
            # Plot training curves
            st.subheader("Training Progress")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=train_losses, name='Training Loss', line=dict(color='red')))
            fig.add_trace(go.Scatter(y=val_losses, name='Validation Loss', line=dict(color='blue')))
            fig.update_layout(title='Training and Validation Loss', xaxis_title='Epoch', yaxis_title='Loss')
            st.plotly_chart(fig, use_container_width=True)
            
            # Accuracy plot
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(y=val_accuracies, name='Validation Accuracy', line=dict(color='green')))
            fig2.update_layout(title='Validation Accuracy', xaxis_title='Epoch', yaxis_title='Accuracy')
            st.plotly_chart(fig2, use_container_width=True)
            
            # Optuna optimization history
            st.subheader("Optuna Optimization History")
            trials_df = study.trials_dataframe()
            fig3 = px.line(trials_df, x='number', y='value', title='Optimization Progress')
            fig3.update_layout(xaxis_title='Trial', yaxis_title='Accuracy')
            st.plotly_chart(fig3, use_container_width=True)
    
    # Prediction section
    if st.session_state.model_trained:
        st.header("3. Next Word Prediction")
        
        input_sequence = st.text_input("Enter text for next word prediction:", 
                                     placeholder="Enter a sequence of words...")
        
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Number of predictions:", 1, 10, 5)
        
        if st.button("🔮 Predict Next Word") and input_sequence:
            predictions = predict_next_word(
                st.session_state.model, 
                input_sequence, 
                st.session_state.vocab_to_idx, 
                st.session_state.idx_to_vocab, 
                st.session_state.device,
                st.session_state.seq_length,
                top_k
            )
            
            st.subheader("Predictions:")
            for i, (word, prob) in enumerate(predictions, 1):
                st.write(f"{i}. **{word}** (confidence: {prob:.3f})")
                st.progress(prob)
    
    # Model information
    if st.session_state.model_trained:
        with st.expander("Model Information"):
            st.write("**Model Architecture:** LSTM-based Neural Language Model")
            st.write("**Optimization:** Optuna hyperparameter tuning")
            st.write("**Features:**")
            st.write("- Embedding layer for word representations")
            st.write("- Multi-layer LSTM for sequence modeling")
            st.write("- Dropout for regularization")
            st.write("- Softmax output for probability distribution")

if __name__ == "__main__":
    main()
