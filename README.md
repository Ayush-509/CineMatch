# 🎬 CineMatch

A movie recommendation system that suggests similar movies using Machine Learning and NLP. CineMatch uses movie information such as genres, keywords, cast, crew, and overview to find and recommend related movies. The project is built with Streamlit to fetch ratings, and other movie details from cleaned data and uses the TMDB API to show posters, trailers, cast and crew.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green)
![TMDB API](https://img.shields.io/badge/TMDB-API-01b4e4)

---

## ✨ Features

* Movie recommendations based on content similarity
* Smart search with typo correction using RapidFuzz
* Modern and interactive Streamlit interface
* Genres, budget and other details from cleaned data
* Movie posters, trailers, cast and crew from TMDB
* Cast and crew information with profile images
* Watchlist feature to save movies during a session

---

## 🏗️ How It Works

1. Movie data is cleaned and processed.
2. Important information like genres, keywords, cast, crew, and overview are combined into tags.
3. Tags are converted into numerical vectors using CountVectorizer.
4. Cosine Similarity is used to find movies that are most similar.
5. TMDB API provides posters, trailers, ratings, and other movie information.

---

## 📸 Screenshots

### Home Page

<p align="center">
  <img width="90%" src="https://github.com/user-attachments/assets/1dc486ab-6d33-42d5-815e-e57bfbcab79a" />

</p>

*Main dashboard showing movie search, recommendation controls, and project statistics.*

### Movie Details

<p align="center">
  <img width="90%" src="https://github.com/user-attachments/assets/cb6ae49c-a5a4-4b2f-a749-027b010842a5" />
</p>

*Detailed movie information including poster, genres, ratings, runtime, overview, and trailer.*

### Cast & Crew

<p align="center">
  <img width="90%" src="https://github.com/user-attachments/assets/2bf61d79-f8aa-4744-9c6f-5fb1954a44c1" />
</p>

*Cast and crew section with actor profiles and production details fetched using the TMDB API.*

### Recommendations

<p align="center">
  <img width="90%" src="https://github.com/user-attachments/assets/d5d927b5-1970-4b57-bcb1-2bbad6a86ed7" />
</p>

*Recommended movies generated using content-based similarity and NLP processing.*


## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Ayush-509/CineMatch.git
cd CineMatch
```

### 2. Install Required Libraries

```bash
pip install -r requirements.txt
```

If you don't have a requirements file, install these packages:

```bash
pip install streamlit pandas numpy scikit-learn nltk rapidfuzz requests
```

### 3. Download the Dataset

Download the following files from the TMDB 5000 Movie Dataset on Kaggle:

* `tmdb_5000_movies.csv`
* `tmdb_5000_credits.csv`

Place both files in the project's main folder.

Dataset:
[https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

### 4. Generate Required Files

Run all cells in:

```bash
CineMatch.ipynb
```

This will generate:

```bash
movie_list.pkl
similarity.pkl
```

These files are required to run the recommendation system.

### 5. Start the Application

```bash
streamlit run app.py
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* NLTK
* RapidFuzz
* Streamlit
* TMDB API

---

## 📂 Required Files

```bash
CineMatch/
│
├── app.py
├── movie_list.pkl
├── similarity.pkl
├── tmdb_5000_movies.csv
├── tmdb_5000_credits.csv
├── CineMatch.ipynb
└── requirements.txt
```

---

## 🎬 Developed By

**Ayush-509**

**aparna-21-6**
