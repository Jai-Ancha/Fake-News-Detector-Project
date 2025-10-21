import gradio as gr
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Download NLTK data (only need to run this once) ---
print("Downloading NLTK data...")
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
print("NLTK data downloaded.")

# --- 1. Load Model and Tokenizer ---
print("Loading model and tokenizer...")
model = tf.keras.models.load_model('fake_news_bilstm_model.keras')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
print("Model and tokenizer loaded successfully.")

# --- 2. Define Constants and Preprocessing Tools ---
MAX_LEN = 500  # Must be the same as in your training notebook
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_for_prediction(text):
    """Cleans a single raw text string for the model."""
    if not isinstance(text, str):
        text = str(text)
    
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(clean_tokens)

# --- 3. Define the Main Prediction Function ---
def predict_news(text_input):
    """
    Takes a raw text input, preprocesses it, and returns the model's prediction.
    """
    # 1. Preprocess
    clean_text = preprocess_for_prediction(text_input)
    
    # 2. Tokenize and Pad
    sequence = tokenizer.texts_to_sequences([clean_text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # 3. Predict
    prediction_prob = model.predict(padded, verbose=0)[0][0]
    
    # 4. Format output
    if prediction_prob > 0.5:
        return {'Fake News': prediction_prob, 'Real News': 1 - prediction_prob}
    else:
        return {'Real News': 1 - prediction_prob, 'Fake News': prediction_prob}

# --- 4. Launch the Gradio Interface ---
interface = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(lines=10, placeholder="Paste any news article text here..."),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="🤖 Fake News Detection using Bi-LSTM",
    description="An AI model to classify news as Real or Fake. (Model Accuracy: 99.84%)"
)

print("\nGradio app is starting... Go to the local URL (usually http://127.0.0.1:7860)")
interface.launch()