#  Music Genre Clustering using KMeans (Unsupervised Learning)

This project was developed as part of my personal learning journey to understand how unsupervised machine learning algorithms — specifically **KMeans clustering** — can be applied to real-world audio data.

The goal was not to build a genre classifier, but to **group songs based on their audio characteristics**, and explore how songs with similar features naturally fall into distinct clusters.

---

##  Dataset Used

I used the **FMA (Free Music Archive)** dataset — a popular open-source dataset for music analysis and MIR (Music Information Retrieval) tasks.

- 🔗 Kaggle Link: [Free Music Archive (FMA): Small subset](https://www.kaggle.com/datasets/adarshsng/fma-music-dataset)

> Specifically, I worked with the **fma_small** subset (~8-second clips of 8,000 songs across various genres).

---

##  Project Structure

| File / Folder               | Description |
|----------------------------|-------------|
| `app.py`                   |  Streamlit app for audio upload and clustering prediction |
| `kmeans_model.pkl`         |  Trained KMeans model using 32 audio features |
| `filenames.txt`            |  List of song filenames used during training |
| `clusters.txt`             |  Predicted cluster ID for each training song |
| `requirements.txt`         |  Python dependencies |
| `music_genre_training.ipynb` |  Kaggle notebook for feature extraction & model training |

---

## What I Did in the Jupyter Notebook

In the `music_genre_training.ipynb`, I:

1. **Extracted audio features** using `librosa` from each `.mp3` file:
   - 13 **MFCCs** (for timbre)
   - 12 **Chroma** features (pitch class)
   - 7 **Spectral contrast** features (tonal sharpness)
   - Combined into **32 total features per song**

2. **Preprocessed the data**:
   - Handled broken or unreadable audio files
   - Formatted the features into a clean matrix for clustering

3. **Trained a KMeans model**:
   - `KMeans(n_clusters=5, random_state=42)` using `scikit-learn`
   - No labels used — this was purely **unsupervised learning**

4. **Saved model artifacts**:
   - Exported model using `joblib`
   - Saved predicted clusters and filenames for downstream usage

---

##  What the Streamlit App Does

The Streamlit app is built to:

-  Allow users to upload any `.mp3` file
-  Extract the same 32 features using `librosa`
-  Predict the cluster the uploaded track belongs to using the trained model
-  Show similar tracks from the same cluster
-  Visualize overall cluster distribution
-  Include clear explanations of the model and audio features for any viewer

>  **Note**: Due to `librosa`'s system-level dependencies, this app is designed for **local execution only**. Deployment on Streamlit Cloud was not possible, but the app works fully and reliably on your own machine.

---

##  Run Locally

To use the app locally:

```bash
git clone https://github.com/Sathvika-k/music-genre-clustering.git
cd music-genre-clustering
pip install -r requirements.txt
streamlit run app.py
````

---

##  Why I Built This

This project was created entirely for **learning and hands-on understanding**.

I wanted to explore:

*  How raw audio can be converted into meaningful numerical features
*  How unsupervised learning works without labels
*  How to build an interactive machine learning app with Streamlit

This helped me gain experience in:

* Audio processing (`librosa`)
* Clustering (`scikit-learn`)
* Interface design & integration (`Streamlit`)
* End-to-end ML workflow — from data to user experience

---

##  Tech Stack

* Python
* NumPy, Pandas
* Librosa (feature extraction)
* scikit-learn (KMeans clustering)
* Matplotlib, Seaborn (visualizations)
* Streamlit (web interface)

---

##  Contact

If you'd like to learn more about this project or collaborate on similar work, feel free to connect:

* **Name**: Kambham Sai Sathvika
* **Email**: [saisathvika07@gmail.com](mailto:saisathvika07@gmail.com)
* **GitHub**: [github.com/Sathvika-k](https://github.com/Sathvika-k)

