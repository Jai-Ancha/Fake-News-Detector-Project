# 🤖 Fake News Detection using Bi-LSTM

This is the final project for my 6-week AICTE-IBM AIML Internship.

The project involves building and training a Bidirectional LSTM (Bi-LSTM) model to classify news articles as "Real" or "Fake". The final model achieved **99.84% accuracy** on the test set.

## Project Details

* **Dataset:** Kaggle - Fake and Real News Dataset
* **Model:** Bidirectional LSTM (Bi-LSTM)
* **Frameworks:** TensorFlow/Keras
* **Deployment:** Gradio Web App

## How to Run This Project Locally

1.  Clone this repository:
    ```bash
    git clone <your-repo-url-here>
    ```
2.  Navigate to the project directory:
    ```bash
    cd Fake_News_App
    ```
3.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
4.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
5.  Run the Gradio app:
    ```bash
    python app.py
    ```
6.  Open the local URL in your browser (e.g., `http://127.0.0.1:7860`).