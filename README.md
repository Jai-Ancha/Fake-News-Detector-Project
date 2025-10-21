# 🤖 Fake News Detection using Bi-LSTM

This is a Deep Learning project built for the **AICTE–IBM AIML Internship**. It uses a Bidirectional LSTM (Bi-LSTM) neural network to classify news articles as "Real" or "Fake."

The model was trained on the [Kaggle "Fake and Real News Dataset"](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) and achieved **99.84% accuracy** on the unseen test set.

---

## 🚀 Live Demo

A live, interactive version of this project is deployed on **Hugging Face Spaces**.

**Try the live app here: [https://huggingface.co/spaces/Your-Username/Your-Space-Name](https://huggingface.co/spaces/Your-Username/Your-Space-Name)**

*(**Note:** You'll need to create this Hugging Face Space and replace the link above!)*

### Screenshot

![Gradio App Screenshot](https://user-images.githubusercontent.com/1234567/your-screenshot-link-here.png)

*(**Note:** To add a screenshot, just drag-and-drop your `image_eb7495.png` file into a new "Issue" on GitHub. It will give you a URL. Paste that URL here.)*

---

## 🛠️ Tech Stack

* **Python 3.10+**
* **TensorFlow / Keras:** For building and training the Bi-LSTM model.
* **Gradio:** For creating the interactive web UI.
* **NLTK:** For the core NLP preprocessing (tokenization, stopword removal, lemmatization).
* **Scikit-learn:** For model evaluation (Classification Report, Confusion Matrix).
* **Pandas & NumPy:** For data handling and manipulation.

---

## 💻 How to Run This Project Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Jai-Ancha/Fake-News-Detector-Project.git](https://github.com/Jai-Ancha/Fake-News-Detector-Project.git)
    cd Fake-News-Detector-Project
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Create the environment
    python -m venv venv

    # Activate on Windows
    .\venv\Scripts\activate

    # Activate on Mac/Linux
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Gradio app:**
    ```bash
    python app.py
    ```

5.  Open the local URL in your browser (usually `http://127.0.0.1:7860`).

---

## ⚠️ Important Note: Model Scope & Limitations

This model is a high-accuracy classifier, but it's important to understand its specific "domain."

* **Real News Data:** The training data for "Real News" came **exclusively from "Reuters" articles**.
* **Fake News Data:** The training data for "Fake News" came from sensationalist blogs and conspiracy websites.

Because of this, the model became an expert at detecting **"Reuters-style" vs. "Blog-style"** text.

It may incorrectly classify *real* news from other sources (like the BBC, CNN, or The Hindu) as "Fake" because their writing style does not match the formal, neutral "Reuters" style it was trained on. This is a classic example of **domain mismatch** and is a key finding of this project.

---

## 📁 Project File Structure
├── app.py # The Gradio web application ├── fake_news_bilstm_model.keras # The saved, trained Keras model ├── tokenizer.pickle # The saved Keras tokenizer ├── requirements.txt # List of Python packages to install ├── .gitignore # Tells Git to ignore the 'venv' folder ├── LICENSE # MIT License └── README.md # This file


## 📄 License
This project is licensed under the **MIT License**. See the `LICENSE` file for details.
