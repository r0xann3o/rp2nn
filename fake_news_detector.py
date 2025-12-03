"""
Fake News Detection using LSTM/GRU Neural Network
This script implements a deep learning model to classify fake vs real news articles.
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import nltk
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

# Download NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# TensorFlow setup
import tensorflow as tf
tf.random.set_seed(42)

# Enable GPU memory growth if GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU detected: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(f"⚠️ Error setting GPU memory growth: {e}")
else:
    print("✓ No GPU detected, using CPU")

print("="*60)
print(f"TensorFlow version: {tf.__version__}")
print(f"Available devices: {[d.name for d in tf.config.list_physical_devices()]}")
print("="*60)

# Configuration
MAX_SEQUENCE_LENGTH = 500
MAX_NB_WORDS = 10000
EMBEDDING_DIM = 128
LSTM_UNITS = 128
GRU_UNITS = 128
BATCH_SIZE = 64
EPOCHS = 3
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1

# -------------------- DATA LOADING AND PREPROCESSING -------------------- #
def load_data():
    print("\n" + "="*60)
    print("STEP 1: Loading datasets...")
    print("="*60)
    
    # Load fake news data
    fake_df = pd.read_csv('Fake.csv')
    fake_df['label'] = 0
    print(f"  ✓ Loaded {len(fake_df)} fake news articles")
    
    # Load real news data
    true_df = pd.read_csv('True.csv')
    true_df['label'] = 1
    print(f"  ✓ Loaded {len(true_df)} real news articles")
    
    # Concatenate fake and real news data AND SHUFFLE 
    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    

    print(f"\nTotal samples: {len(df)}")
    print(f"  - Fake news: {len(df[df['label'] == 0])}")
    print(f"  - Real news: {len(df[df['label'] == 1])}")
    
    return df

def preprocess_text(text):
    if pd.isna(text):
        return ""

    # Preprocess text   
    text = str(text).lower()
    # Remove URLs, email addresses, and non-alphanumeric characters
    text = re.sub(r'http\S+|www.\S+', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def combine_title_text(row):
    title = preprocess_text(row.get('title', ''))
    text = preprocess_text(row.get('text', ''))
    return (title + ' ' + text).strip()


def prepare_data(df):
    print("\n" + "="*60)
    print("STEP 2: Preprocessing text data...")
    print("="*60)
    

    df['combined_text'] = df.apply(combine_title_text, axis=1)
    initial_count = len(df)
    df = df[df['combined_text'].str.len() > 0]
    removed = initial_count - len(df)
    if removed > 0:
        print(f"  ✓ Removed {removed} empty texts")
    
    texts = df['combined_text'].values
    labels = df['label'].values
    print(f"  ✓ Final dataset: {len(texts)} samples")
    
    return texts, labels

# Tokenize and pad sequences
def tokenize_and_pad(texts, tokenizer=None, fit=True):
    if fit:
        print("\n" + "="*60)
        print("STEP 3: Tokenizing and padding sequences...")
        print("="*60)
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS, oov_token='<OOV>')
        tokenizer.fit_on_texts(texts)
        vocab_size = len(tokenizer.word_index) + 1
        print(f"  ✓ Vocabulary size: {vocab_size:,} words")
    else:
        print("Tokenizing texts...")
    
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    print(f"  ✓ Created {len(padded_sequences)} sequences")
    
    return padded_sequences, tokenizer

# -------------------- MODEL BUILDING -------------------- #
def build_lstm_model(vocab_size):
    print("Building LSTM model...")
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
        Dropout(0.3),
        Bidirectional(LSTM(LSTM_UNITS, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(LSTM_UNITS // 2)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'precision', 'recall'])
    return model

def build_gru_model(vocab_size):
    print("Building GRU model...")
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
        Dropout(0.3),
        Bidirectional(GRU(GRU_UNITS, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(GRU(GRU_UNITS // 2)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'precision', 'recall'])
    return model

# -------------------- PLOTTING -------------------- #
def plot_training_history(history, name):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 0].set_title(f'{name} Model Accuracy'); axes[0, 0].legend(); axes[0, 0].grid(True)
    
    axes[0, 1].plot(history.history['loss'], label='Train Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 1].set_title(f'{name} Model Loss'); axes[0, 1].legend(); axes[0, 1].grid(True)
    
    axes[1, 0].plot(history.history['precision'], label='Train Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
    axes[1, 0].set_title(f'{name} Model Precision'); axes[1, 0].legend(); axes[1, 0].grid(True)
    
    axes[1, 1].plot(history.history['recall'], label='Train Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
    axes[1, 1].set_title(f'{name} Model Recall'); axes[1, 1].legend(); axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'training_history_{name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved as 'training_history_{name}.png'")

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake','Real'], yticklabels=['Fake','Real'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{model_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved as 'confusion_matrix_{model_name.lower()}.png'")

# -------------------- MODEL EVALUATION -------------------- #
def evaluate_model(model, X_test, y_test, model_name):
    print(f"\n{'='*60}\nEvaluating {model_name} Model\n{'='*60}")
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nTest Set Metrics:\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Fake','Real'])}")
    plot_confusion_matrix(y_test, y_pred, model_name)
    return accuracy, precision, recall, f1

# -------------------- MAIN PIPELINE -------------------- #
def main():
    df = load_data()
    texts, labels = prepare_data(df)
    
    X_temp, X_test, y_temp, y_test = train_test_split(texts, labels, test_size=TEST_SPLIT, random_state=42, stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=VALIDATION_SPLIT/(1-TEST_SPLIT), random_state=42, stratify=y_temp)
    
    print(f"\n  ✓ Training samples: {len(X_train):,} | Validation samples: {len(X_val):,} | Test samples: {len(X_test):,}")
    
    X_train_padded, tokenizer = tokenize_and_pad(X_train, fit=True)
    X_val_padded, _ = tokenize_and_pad(X_val, tokenizer=tokenizer, fit=False)
    X_test_padded, _ = tokenize_and_pad(X_test, tokenizer=tokenizer, fit=False)
    
    # Save validation split for external use
    val_df = pd.DataFrame({
        'text': X_val,
        'label': y_val
    })
    
    vocab_size = len(tokenizer.word_index) + 1
    print(f"\n  ✓ Final vocabulary size: {vocab_size:,}")
    
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)
    
    # -------------------- LSTM -------------------- #
    print("\n" + "="*60 + "\nBuilding and Training LSTM Model\n" + "="*60)
    lstm_model = build_lstm_model(vocab_size)
    lstm_model.summary()
    
    early_stopping_lstm = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
    model_checkpoint_lstm = ModelCheckpoint('best_lstm_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
    reduce_lr_lstm = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1)
    
    try:
        history_lstm = lstm_model.fit(
            X_train_padded, y_train,
            batch_size=BATCH_SIZE, epochs=EPOCHS,
            validation_data=(X_val_padded, y_val),
            callbacks=[early_stopping_lstm, model_checkpoint_lstm, reduce_lr_lstm],
            verbose=1
        )
        plot_training_history(history_lstm, 'LSTM')
        lstm_model.load_weights('best_lstm_model.h5')
        evaluate_model(lstm_model, X_test_padded, y_test, 'LSTM')
    except Exception as e:
        print(f"❌ Error training LSTM model: {e}")
        raise
    
    # -------------------- GRU -------------------- #
    print("\n" + "="*60 + "\nBuilding and Training GRU Model\n" + "="*60)
    gru_model = build_gru_model(vocab_size)
    gru_model.summary()
    
    early_stopping_gru = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
    model_checkpoint_gru = ModelCheckpoint('best_gru_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
    reduce_lr_gru = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1)
    
    try:
        history_gru = gru_model.fit(
            X_train_padded, y_train,
            batch_size=BATCH_SIZE, epochs=EPOCHS,
            validation_data=(X_val_padded, y_val),
            callbacks=[early_stopping_gru, model_checkpoint_gru, reduce_lr_gru],
            verbose=1
        )
        plot_training_history(history_gru, 'GRU')
        gru_model.load_weights('best_gru_model.h5')
        evaluate_model(gru_model, X_test_padded, y_test, 'GRU')
    except Exception as e:
        print(f"❌ Error training GRU model: {e}")
        raise
    
    # -------------------- SAVE FINAL MODELS -------------------- #
    print("\n" + "="*60 + "\nSaving final models\n" + "="*60)
    lstm_model.save('lstm_model_final.h5')
    gru_model.save('gru_model_final.h5')
    print("Models saved: 'lstm_model_final.h5', 'gru_model_final.h5'")
    
    print("\n🎉 Training Complete! 🎉\nGenerated files:\n  - best_lstm_model.h5\n  - best_gru_model.h5\n  - lstm_model_final.h5\n  - gru_model_final.h5\n  - training_history_LSTM.png\n  - training_history_GRU.png\n  - confusion_matrix_lstm.png\n  - confusion_matrix_gru.png")

if __name__ == "__main__":
    main()
