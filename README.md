#  Music Genre Clustering using KMeans (Unsupervised Learning)

This project was developed as part of my personal learning journey to understand how unsupervised machine learning algorithms — specifically **KMeans clustering** — can be applied to real-world audio data.

The goal was not to build a genre classifier, but to **group songs based on their audio characteristics**, and explore how songs with similar features naturally fall into distinct clusters.

---

## Dataset Used

I used the **FMA (Free Music Archive)** dataset — a popular open-source dataset for music analysis and MIR (Music Information Retrieval) tasks.

- 🔗 Kaggle Link: [Free Music Archive (FMA): Small subset](https://www.kaggle.com/datasets/adarshsng/fma-music-dataset)

> Specifically, I worked with the **fma_small** subset (~8s clips of 8,000 songs in various genres).

---

## Project Structure

| File / Folder           | Description |
|------------------------|-------------|
| `app.py`               | 📱 Streamlit web app to predict cluster from user-uploaded `.mp3` files |
| `kmeans_model.pkl`     | 🎯 Trained KMeans model (32 features) |
| `filenames.txt`        | List of processed filenames used in training |
| `clusters.txt`         | Predicted cluster for each training song |
| `requirements.txt`     | Python dependencies to run the app |
| `music_genre_training.ipynb` | 📓 Kaggle notebook where data was processed & model was trained |

---

##  What I Did in the Jupyter Notebook

In the `music_genre_training.ipynb` file, I:

1. **Extracted audio features** using `librosa` for each `.mp3` file in the dataset:
   - 13 **MFCCs** (representing timbre)
   - 12 **Chroma** features (pitch class)
   - 7 **Spectral contrast** features (tonal texture)
   - ➡️ Combined into a total of **32 features per song**

2. **Preprocessed the data**:
   - Skipped corrupt or unreadable files
   - Normalized inputs and stacked features into a matrix

3. **Trained a KMeans clustering model** using:
   - `KMeans(n_clusters=5, random_state=42)`
   - No labels were used — the model grouped songs purely based on feature similarity

4. **Saved the model and outputs**:
   - Saved the trained model using `joblib`
   - Saved predicted cluster IDs (`clusters.txt`)
   - Saved corresponding filenames (`filenames.txt`)

---

##  What the Streamlit App Does

The deployed web app allows users to:

- **Upload any `.mp3` file**
- The app extracts the same 32 features using `librosa`
- **Predicts which cluster** the uploaded song belongs to using the trained KMeans model
- **Lists similar songs** (based on cluster match) from the training data
- **Displays a cluster distribution chart**
- Provides detailed **explanations and insights** for non-ML viewers

---

##  Run Locally

```bash
git clone https://github.com/Sathvika-k/music-genre-clustering.git
cd music-genre-clustering
pip install -r requirements.txt
streamlit run app.py
 Try the App Live
 Click here to open the app


 Why I Built This:
This project was built entirely for learning purposes.

I wanted to understand:

How raw audio can be converted into numerical features

How unsupervised learning can group data without labels

How to deploy an ML model with a clean web UI using Streamlit

It gave me practical experience with:

Audio processing (librosa)

Clustering (scikit-learn)

Deployment (Streamlit)

End-to-end project workflow

Tech Stack:
Python

NumPy, Pandas

Librosa

scikit-learn

Matplotlib, Seaborn

Streamlit

Contact:

If you'd like to learn more about this project or collaborate on similar work, feel free to connect!

Name: Kambham Sai Sathvika
Email: saisathvika07@gmail.com
GitHub: github.com/Sathvika-k