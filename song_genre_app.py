import streamlit as st
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack

# -------------------------------
# Custom feature extractor
# -------------------------------
def extract_custom_features(texts):
    features = []
    for t in texts:
        words = t.split()
        length = len(words)                  # total words
        num_unique = len(set(words))         # unique words
        features.append([length, num_unique])
    return np.array(features)

# -------------------------------
# Load models
# -------------------------------
bi_model = joblib.load("lyrics_genre_model_bigram.pkl")
tri_model = joblib.load("lyrics_genre_model_trigram.pkl")

# -------------------------------
# Prediction function
# -------------------------------
def predict_genre(lyrics_list, model_type="Bi-gram"):
    if model_type == "Bi-gram":
        model = bi_model
    else:
        model = tri_model

    tfidf = model["tfidf"]
    scaler = model["scaler"]
    classifier = model["classifier"]
    mlb = model["mlb"]

    # TF-IDF features
    tfidf_features = tfidf.transform(lyrics_list)

    # Custom features (2 features)
    custom_features = extract_custom_features(lyrics_list)
    custom_scaled = scaler.transform(custom_features)

    # Combine
    X = hstack([tfidf_features, custom_scaled])

    # Predict
    preds = classifier.predict(X)
    probs = classifier.predict_proba(X)

    # Top-3 predictions
    prob_df = pd.DataFrame(probs, columns=mlb.classes_)
    top3_labels = prob_df.apply(lambda row: row.sort_values(ascending=False).index[:3].tolist(), axis=1)
    top3_probs = prob_df.apply(lambda row: row.sort_values(ascending=False).values[:3], axis=1)

    return mlb.inverse_transform(preds), list(zip(top3_labels[0], top3_probs[0]))

# -------------------------------
# Streamlit App
# -------------------------------
st.title("🎵 Song Lyrics Genre Classifier")
st.write("Paste lyrics manually or upload a CSV file with a 'lyrics' column.")

# Model selection
model_type = st.radio("Choose model:", ["Bi-gram", "Tri-gram"])

# Example lyrics
examples = [
    "Take me home, country roads, to the place I belong",
    "Drop it like it's hot, when the pimp's in the crib ma",
    "We will, we will rock you!"
]

st.subheader("Example Lyrics")
for ex in examples:
    st.code(ex, language="text")

# -------------------------------
# CSV upload
# -------------------------------
uploaded_file = st.file_uploader("Upload CSV file with lyrics", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'lyrics' not in df.columns:
        st.error("CSV must have a column named 'lyrics'.")
    else:
        st.success(f"Found {len(df)} lyrics in CSV. Predicting genres...")
        results = []
        for lyrics in df['lyrics'].astype(str):
            genres, top3 = predict_genre([lyrics], model_type=model_type)
            results.append({
                "lyrics": lyrics,
                "predicted_genres": ', '.join(genres[0]),
                "top3": top3
            })
        st.subheader("Predictions")
        for r in results:
            st.write(f"**Lyrics:** {r['lyrics']}")
            st.write(f"**Predicted Genre(s):** {r['predicted_genres']}")
            st.subheader("Top-3 Predictions with Confidence")
            for label, prob in r['top3']:
                st.write(f"{label}: {prob:.2f}")
                st.progress(prob)
            st.markdown("---")

# -------------------------------
# Manual input
# -------------------------------
lyrics = st.text_area("Or enter lyrics manually here:")
if st.button("Predict Genre (Manual)"):
    if lyrics.strip() == "":
        st.warning("Please enter some lyrics!")
    else:
        genres, top3 = predict_genre([lyrics], model_type=model_type)
        st.success(f"**Predicted Genre(s):** {', '.join(genres[0])}")
        st.subheader("Top-3 Predictions with Confidence")
        for label, prob in top3:
            st.write(f"{label}: {prob:.2f}")
            st.progress(prob)
