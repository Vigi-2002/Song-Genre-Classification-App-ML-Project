🎵 ##Song Genre Classification App

Predict the genre of a song based on its lyrics using machine learning.
Built with TF-IDF features, Linear SVM, and deployed using Streamlit.

🌐 Deployment

🎯 Try it here: https://song-genre-classification-app-ml-project-wwv53gfvq94fzdyugxq2i.streamlit.app/

🚀 Features

🎤 Enter song lyrics and get instant genre predictions

🔀 Toggle between bi-gram and tri-gram models

📊 View Top-3 predicted genres with confidence bars

📁 Upload a CSV file to classify multiple lyrics at once

⚖️ Handles class imbalance automatically

🧠 Model Overview

Vectorization: TF-IDF (bi-gram & tri-gram)

Model: LinearSVC with One-vs-Rest Classifier

Feature Engineering:

Average word length

Unique word ratio

Multi-label Support: Using MultiLabelBinarizer

Evaluation Metrics: Exact match accuracy, Precision, Recall, F1-score

🛠️ Tech Stack

Category	Tools

Language	Python

Framework	Streamlit

Libraries	scikit-learn, pandas, numpy, joblib

Deployment	Streamlit Cloud

📂 Project Structure

├── song_genre_app.py               # Streamlit app

├── lyrics_genre_model_bigram.pkl   # Saved bi-gram model

├── lyrics_genre_model_trigram.pkl  # Saved tri-gram model

├── data/

│   └── lyrics_dataset_small.csv    # Dataset

├── requirements.txt

└── README.md

🧪 Example Usage

Input Lyrics:

“We’re up all night to get lucky”

Predicted Genres:
🎶 Pop (0.81)
🎧 Dance (0.72)
💃 Funk (0.65)

💡 Future Improvements

Add deep learning models (BERT / LSTM)

Expand dataset with multilingual lyrics

Add audio-based feature integration

👨‍💻 Author
Vigi-2002
