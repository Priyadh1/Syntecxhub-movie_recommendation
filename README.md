# 🎬 FlickFusion — Movie Recommendation System

**Discover movies that match your taste, mood, and movie night!**

FlickFusion is a Streamlit-based movie recommendation web application that helps users discover movies based on their interests and preferences. It uses content-based filtering with TF-IDF and cosine similarity to identify movies with similar characteristics.

## ✨ Features

- 🎥 **Similar Movies:** Find movies similar to your favourite title.
- 💬 **Vibe Check:** Enter a description of the kind of movie you want to watch.
- 😊 **Emoji Mood Matrix:** Discover movies based on your mood.
- ❤️ **Swipe to Discover:** Like or pass on movies and receive recommendations based on your likes.
- 👥 **Watch Together:** Combine movie preferences for a group movie night.
- 🍿 **Where to Watch:** Open JustWatch to check streaming availability.

## 🧠 Machine Learning Approach

- **Content-Based Filtering** — recommends movies using their content and metadata.
- **TF-IDF Vectorization** — converts movie information into numerical features.
- **Cosine Similarity** — measures similarity between movie feature vectors.
- **Text Processing** — combines movie genres, keywords, cast, director, and overview.

## 📂 Dataset

This project uses the TMDB 5000 Movies and Credits datasets.

Required files:
- `tmdb_5000_movies.csv`
- 'tmdb_5000_credits.csv`

The application can extract the credits CSV from the ZIP file when needed.

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Cosine Similarity

## 🚀 Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/Priyadh1/Syntecxhub-movie_recommendation.git
   ```

2. Open the project directory:

   ```bash
   cd Syntecxhub-movie_recommendation
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the application:

   ```bash
   streamlit run app.py
   ```

## 🌐 Deployment

FlickFusion can be deployed using Streamlit Community Cloud by connecting the GitHub repository and selecting `app.py` as the main application file.

## 📌 Project Note

FlickFusion currently uses metadata-based recommendations. It does not use personal viewing history or a trained rating-prediction model. Streaming availability is checked through an external JustWatch link; live provider availability is not directly integrated.

## 👩‍💻 Author

**Priyadharshini**

GitHub: [Priyadh1](https://github.com/Priyadh1)

---

*FlickFusion — Your mood. Your movies. Your next favourite.*
