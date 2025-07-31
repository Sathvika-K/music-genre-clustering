import streamlit as st
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# ---- PAGE SETUP ---- #
st.set_page_config(page_title="🎵 Music Genre Clustering", layout="centered")
st.title("🎧 Music Genre Clustering & Similar Song Finder")

st.markdown("""
Upload an `.mp3` file to:
- 🧠 Extract 32 audio features (MFCC, Chroma, Contrast)
- 🎯 Predict its cluster using a trained KMeans model
- 🎶 Show similar songs from that cluster
- 📊 Visualize the cluster distribution
""")

# ---- LOAD MODEL & DATA ---- #
model = joblib.load("kmeans_model.pkl")

with open("filenames.txt") as f:
    filenames = f.read().splitlines()

with open("clusters.txt") as f:
    clusters = list(map(int, f.read().splitlines()))

# ---- CLUSTER DISTRIBUTION CHART ---- #
def plot_cluster_distribution(clusters):
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    fig, ax = plt.subplots()
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Number of Songs")
    ax.set_title("🎼 Distribution of Songs in Each Cluster")
    st.pyplot(fig)

# ---- FILE UPLOAD ---- #
st.markdown("### 📤 Upload Your MP3 File")
audio_file = st.file_uploader("Upload an MP3 file", type=["mp3"])

# ---- PROCESSING ---- #
if audio_file is not None:
    try:
        st.audio(audio_file, format="audio/mp3")

        # ---- FEATURE EXTRACTION: 32-DIMENSION ---- #
        y, sr = librosa.load(audio_file, sr=22050, duration=30)

        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)

        features = np.hstack([mfcc, chroma, contrast])  # shape: (32,)

        if features.shape[0] != 32:
            st.error("Feature extraction failed. Expected 32 features but got something else.")
        else:
            features = features.reshape(1, -1).astype(np.float64)

            predicted_cluster = model.predict(features)[0]

            # ---- PREDICTION RESULT ---- #
            st.markdown(f"### 🎯 Predicted Cluster: **Cluster {predicted_cluster}**")
            st.info("This song has been clustered based on MFCC, Chroma, and Spectral Contrast features.")

            # ---- SIMILAR SONGS ---- #
            st.markdown("### 🎧 Similar Songs From This Cluster")
            matched_files = [f for f, c in zip(filenames, clusters) if c == predicted_cluster]
            st.write(f"🔍 Found `{len(matched_files)}` songs in Cluster {predicted_cluster}.")

            if matched_files:
                for i, file in enumerate(matched_files[:10]):
                    display_name = file.replace("/kaggle/input/", "")
                    st.write(f"{i+1}. `{display_name}`")
            else:
                st.info("No similar songs found in this cluster.")

            # ---- CLUSTER DISTRIBUTION ---- #
            st.markdown("###  Cluster Distribution in Full Dataset")
            plot_cluster_distribution(clusters)
            st.markdown("###  What Does This Chart Represent?")
            st.markdown("""
The above bar chart shows the number of songs that were grouped into each cluster by the KMeans algorithm.

**Why is this important?**

- In unsupervised learning, we don't assign labels like 'rock' or 'jazz' — instead, the model finds **natural groupings** in the data based on patterns.
- Each **cluster** represents a group of songs that share similar audio characteristics, such as:
  - **MFCCs**: tonal and timbral qualities
  - **Chroma features**: harmonic content (like chords or pitch classes)
  - **Spectral contrast**: sharpness or texture of the sound

**What can we learn from this?**

- If one cluster has **more songs**, it could mean those musical styles are more common in the dataset.
- If a cluster has **very few songs**, those tracks might be more unique or different in style.

This gives us insight into the **structure of the dataset** and helps in understanding genre trends, audio diversity, or potential outliers.
""")

    except Exception as e:
        st.error(f"⚠️ Error processing file: {str(e)}")
else:
    st.info("Please upload an MP3 file to begin.")
