import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from movie_profiles import movie_profiles
from datetime import datetime
import streamlit.components.v1 as components
import re

def clean_title(title):

    if ", The" in title:
        title = "The " + title.replace(", The", "")

    elif ", A" in title:
        title = "A " + title.replace(", A", "")

    elif ", An" in title:
        title = "An " + title.replace(", An", "")

    title = re.sub(r"\s*\(\d{4}\)", "", title)

    return title
# -----------------------------
# TMDb API Key
# -----------------------------
API_KEY = "04b9817484edbf1a6cca1fc1d7776184"

# -----------------------------
# Load datasets
# -----------------------------

@st.cache_data
def load_data():

    ratings = pd.read_csv("data/ratings.csv")
    movies = pd.read_csv("data/movies.csv")

    data = pd.merge(ratings, movies, on='movieId')

    return ratings, movies, data

ratings, movies, data = load_data()

# -----------------------------
# Merge datasets
# -----------------------------
data = pd.merge(ratings, movies, on='movieId')

# -----------------------------
# Create user-movie matrix
# -----------------------------
user_movie_matrix = data.pivot_table(
    index='userId',
    columns='title',
    values='rating'
)

# -----------------------------
# Fill missing values
# -----------------------------
matrix_filled = user_movie_matrix.fillna(0)

# -----------------------------
# Cosine similarity
# -----------------------------
@st.cache_resource
def compute_similarity(matrix):

    similarity = cosine_similarity(matrix)

    return pd.DataFrame(
        similarity,
        index=matrix.index,
        columns=matrix.index
    )


similarity_df = compute_similarity(matrix_filled)

# -----------------------------
# Movie statistics
# -----------------------------
movie_stats = data.groupby('title').agg({
    'rating': ['mean', 'count']
})

movie_stats.columns = ['avg_rating', 'rating_count']

movie_stats = movie_stats.reset_index()


# -----------------------------
# Content-Based Filtering
# -----------------------------

movies['genres'] = movies['genres'].fillna('')

tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(
    movies['genres']
)

content_similarity = linear_kernel(
    tfidf_matrix,
    tfidf_matrix
)

indices = pd.Series(
    movies.index,
    index=movies['title']
).drop_duplicates()

# -----------------------------
# Fetch movie poster from TMDb
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_poster(movie_title):

    try:

        clean_title = movie_title.split('(')[0].strip()

        url = (
            f"https://api.themoviedb.org/3/search/movie"
            f"?api_key={API_KEY}&query={clean_title}"
        )

        response = requests.get(url)

        data = response.json()

        if data['results']:

            poster_path = data['results'][0].get('poster_path')

            if poster_path:

                return (
                    f"https://image.tmdb.org/t/p/w500"
                    f"{poster_path}"
                )

    except Exception as e:
        st.write(e)

    return "https://via.placeholder.com/150x220?text=No+Image"

def fetch_backdrop(movie_title):

    try:

        clean_title = movie_title.split('(')[0].strip()

        url = (
            f"https://api.themoviedb.org/3/search/movie"
            f"?api_key={API_KEY}&query={clean_title}"
        )

        response = requests.get(url)

        data = response.json()

        if data['results']:

            backdrop_path = (
                data['results'][0]
                .get('backdrop_path')
            )

            if backdrop_path:

                return (
                    "https://image.tmdb.org/t/p/original"
                    f"{backdrop_path}"
                )

    except:
        pass

    return None
st.markdown(
                        """
                        <style>

                            .movie-card {
                                transition:
                                    transform 0.3s ease,
                                    box-shadow 0.3s ease;
                            }

                            .movie-card:hover {
                                transform: translateY(-10px) scale(1.03);

                                box-shadow:
                                    0 12px 30px rgba(0,0,0,0.5);

                                border:
                                    1px solid rgba(255,255,255,0.15);
                            }

                        </style>
                        """,
                        unsafe_allow_html=True
                    )
def fetch_trailer(movie_title):

    try:

        clean_title = (
            movie_title.split('(')[0]
            .strip()
        )

        search_url = (
            f"https://api.themoviedb.org/3/search/movie"
            f"?api_key={API_KEY}"
            f"&query={clean_title}"
        )

        response = requests.get(search_url)

        movie_data = response.json()

        if movie_data['results']:

            movie_id = movie_data['results'][0]['id']

            video_url = (
                f"https://api.themoviedb.org/3/movie/"
                f"{movie_id}/videos"
                f"?api_key={API_KEY}"
            )

            video_response = requests.get(video_url)

            video_data = video_response.json()

            for video in video_data['results']:

                if (
                    video['site'] == "YouTube"
                    and
                    (
                        video['type'] == "Trailer"
                        or
                        video['type'] == "Teaser"
                    )
                ):

                    return video['key']

    except Exception as e:
        print(e)

    return None
#---------------------------------------
#if completely new user
#---------------------------------------
def get_popular_movies(num_movies=5):

    popular = movie_stats.merge(
        movies[['title', 'genres']],
        on='title'
    )

    popular = popular[
        popular['rating_count'] > 200
    ]

    popular = popular.sort_values(
        by=['avg_rating', 'rating_count'],
        ascending=False
    )

    return popular.head(num_movies)


def get_content_recommendations(title, num_recommendations=10):

    if title not in indices:
        return pd.DataFrame()

    idx = indices[title]

    similarity_scores = list(
    enumerate(content_similarity[idx].flatten())
)

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )[1:11]

    similarity_scores = similarity_scores[1:num_recommendations+1]

    movie_indices = [
        i[0] for i in similarity_scores
    ]

    recommendations = movies.iloc[
        movie_indices
    ][['title', 'genres']]

    recommendations = recommendations.merge(
        movie_stats,
        on='title'
    )

    return recommendations


def analyze_prompt(prompt):

    detected_mood = "Any"
    detected_genre = "All"

    prompt = prompt.lower()

    # Mood Detection

    if (
        "funny" in prompt or
        "comedy" in prompt or
        "laugh" in prompt
    ):

        detected_mood = "funny"
        detected_genre = "Comedy"

    elif (
        "dark" in prompt or
        "serious" in prompt or
        "crime" in prompt
    ):

        detected_mood = "dark"
        detected_genre = "Crime"

    elif (
        "mind" in prompt or
        "intelligent" in prompt or
        "psychological" in prompt
    ):

        detected_mood = "mind-bending"
        detected_genre = "Sci-Fi"

    elif (
        "romantic" in prompt or
        "love" in prompt or
        "emotional" in prompt
    ):

        detected_mood = "romantic"
        detected_genre = "Romance"

    elif (
        "epic" in prompt or
        "adventure" in prompt
    ):

        detected_mood = "epic"
        detected_genre = "Adventure"

    return detected_mood, detected_genre

def advanced_ai_understanding(query):

    query = query.lower()

    detected = {
        "genres": [],
        "moods": [],
        "keywords": []
    }

    genre_map = {
        "sci-fi": "Sci-Fi",
        "space": "Sci-Fi",
        "romance": "Romance",
        "love": "Romance",
        "crime": "Crime",
        "detective": "Mystery",
        "funny": "Comedy",
        "comedy": "Comedy",
        "scary": "Horror",
        "horror": "Horror",
        "action": "Action",
        "war": "Action",
        "fantasy": "Fantasy",
        "magic": "Fantasy"
    }

    mood_map = {
        "dark": "dark",
        "emotional": "romantic",
        "mind-bending": "mind-bending",
        "psychological": "psychological",
        "epic": "epic",
        "happy": "happy"
    }

    for word, genre in genre_map.items():

        if word in query:

            detected["genres"].append(
                genre
            )

    for word, mood in mood_map.items():

        if word in query:

            detected["moods"].append(
                mood
            )

    return detected

def build_user_profile(user_id):

    user_data = data[
        data['userId'] == user_id
    ]

    genre_scores = {}

    for _, row in user_data.iterrows():

        genres = row['genres'].split('|')

        rating = row['rating']

        for genre in genres:

            if genre not in genre_scores:

                genre_scores[genre] = []

            genre_scores[genre].append(rating)

    profile = {}

    for genre, ratings_list in genre_scores.items():

        profile[genre] = (
    np.mean(ratings_list)
    *
    np.log(len(ratings_list) + 1)
)

    return profile
current_hour = datetime.now().hour

time_preferences = {
    "Morning": [
        "Animation",
        "Family",
        "Comedy"
    ],

    "Afternoon": [
        "Adventure",
        "Action",
        "Fantasy"
    ],

    "Evening": [
        "Drama",
        "Romance",
        "Thriller"
    ],

    "Night": [
        "Crime",
        "Mystery",
        "Sci-Fi",
        "Horror"
    ]
}
# -----------------------------
# Recommendation Function
# -----------------------------
def recommend_movies(
        user_id,
        num_recommendations=10,
        genre=None,
        cinematic_mood="Any",
        time_context="evening"
):

    if user_id not in user_movie_matrix.index:
        return get_popular_movies(num_recommendations)

    similar_users = similarity_df[user_id] \
                        .sort_values(ascending=False)[1:11]

    watched_movies = set(
        user_movie_matrix.loc[user_id]
        .dropna()
        .index
    )

    weighted_scores = pd.Series(dtype=float)
    current_period = "Evening"

    for similar_user, similarity_score in similar_users.items():

        similar_user_ratings = user_movie_matrix.loc[similar_user].dropna()

        for movie, rating in similar_user_ratings.items():

            if movie not in watched_movies:

                if movie not in weighted_scores.index:
                    weighted_scores[movie] = 0.0

                weighted_scores[movie] += rating * similarity_score
                

    # ---------------------------------
# Hybrid Recommendation Logic
# ---------------------------------

    content_scores = pd.Series(dtype=float)

    for watched_movie in watched_movies:

        if watched_movie in indices:

            idx = indices[watched_movie]

            if isinstance(idx, pd.Series):
                idx = idx.iloc[0]

            similarity_scores = list(
            enumerate(content_similarity[idx].flatten())
            )

            similarity_scores = sorted(
            similarity_scores,
            key=lambda x: x[1],
            reverse=True
            )[1:11]

            for movie_idx, sim_score in similarity_scores:

                movie_title = movies.iloc[movie_idx]['title']

                if movie_title not in watched_movies:

                    if movie_title not in content_scores.index:
                        content_scores[movie_title] = 0.0

                    content_scores[movie_title] += sim_score

# Normalize scores
    if len(weighted_scores) > 0:
        weighted_scores = (
        weighted_scores / weighted_scores.max()
    )

    if len(content_scores) > 0:
        content_scores = (
        content_scores / content_scores.max()
    )

# Combine both systems
    hybrid_scores = (
    weighted_scores * 0.7
).add(
    content_scores * 0.3,
    fill_value=0
)
    # ---------------------------------
# Cinematic Mood Score Boost
# ---------------------------------

    if cinematic_mood != "Any":

        preferred_genres = mood_genre_map[
        cinematic_mood
    ]

        for movie in hybrid_scores.index:

            movie_row = movies[
            movies['title'] == movie
        ]

            if not movie_row.empty:

                movie_genres = (
                movie_row.iloc[0]['genres']
            )

                for mood_genre in preferred_genres:

                    if mood_genre in movie_genres:

                        hybrid_scores[movie] *= 1.5
                        break
        # ---------------------------------
# Time-Aware Recommendation Boost
# ---------------------------------
    if current_hour < 12:

        current_period = "Morning"

    elif current_hour < 17:

        current_period = "Afternoon"

    elif current_hour < 22:

        current_period = "Evening"

    else:

        current_period = "Night"

    if cinematic_mood == "Any":
        preferred_genres = time_preferences[current_period]
    else:
        preferred_genres = []
        

    for movie in hybrid_scores.index:

        movie_genres = movies[
        movies['title'] == movie
    ]['genres']

        if len(movie_genres) > 0:

            movie_genres = movie_genres.iloc[0]

            for genre_name in preferred_genres:

                if genre_name in movie_genres:

                    hybrid_scores[movie] *= 2.2
                    break
    
    recommendations = hybrid_scores.sort_values(
    ascending=False
)

    recommendations_df = recommendations.reset_index()

    recommendations_df.columns = ['title', 'score']
    max_score = recommendations_df['score'].max()

    recommendations_df['match_percent'] = (
    recommendations_df['score']
    / max_score
) * 100
    recommendations_df = recommendations_df.merge(
        movie_stats,
        on='title'
    )

    recommendations_df = recommendations_df.merge(
        movies[['title', 'genres']],
        on='title'
    )

    recommendations_df = recommendations_df[
        recommendations_df['rating_count'] > 50
    ]

    if genre and genre != "All":

        recommendations_df = recommendations_df[
            recommendations_df['genres'].str.contains(
                genre,
                case=False,
                na=False
            )
        ]
    
    explanations = []

    for recommended_movie in recommendations_df['title']:

        best_match = None
        best_score = -1

        if recommended_movie in indices:

            rec_idx = indices[recommended_movie]

            if isinstance(rec_idx, pd.Series):
                rec_idx = rec_idx.iloc[0]

            for watched_movie in watched_movies:

                if watched_movie in indices:

                    watched_idx = indices[watched_movie]

                    if isinstance(watched_idx, pd.Series):
                        watched_idx = watched_idx.iloc[0]

                    similarity = content_similarity[
                    rec_idx,
                    watched_idx
                    ]

                    if similarity > best_score:

                        best_score = similarity
                        best_match = watched_movie

        explanations.append(best_match)

    recommendations_df['because_you_watched'] = explanations
    
    return (
    recommendations_df.head(
        num_recommendations
    ),
    current_period
)
current_hour = datetime.now().hour
time_context = ""

if 5 <= current_hour < 12:

    time_context = "morning"

elif 12 <= current_hour < 17:

    time_context = "afternoon"

elif 17 <= current_hour < 22:

    time_context = "evening"

else:

    time_context = "late night"

def recommendation_quality(recommendations):

    avg_rating_score = recommendations[
        'avg_rating'
    ].mean()

    popularity_score = recommendations[
        'rating_count'
    ].mean()

    normalized_popularity = min(
        popularity_score / 500,
        5
    )

    quality_score = (
        avg_rating_score * 0.8
        +
        normalized_popularity * 0.2
    )

    return round(quality_score, 2)
mood_genre_map = {

    "thoughtful": [
        "Sci-Fi",
        "Drama"
    ],

    "mind-bending": [
        "Sci-Fi",
        "Mystery",
        "Thriller"
    ],

    "dark": [
        "Crime",
        "Thriller",
        "Drama"
    ],

    "happy": [
        "Comedy",
        "Animation"
    ],

    "romantic": [
        "Romance",
        "Drama"
    ],

    "psychological": [
        "Thriller",
        "Mystery"
    ],

    "funny": [
        "Comedy"
    ],

    "epic": [
        "Adventure",
        "Action",
        "Fantasy"
    ]
}
# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    layout="wide"
)
st.set_page_config(
    page_title="Movie Recommendation System",
    layout="wide"
)

st.markdown(
    """
    <style>

    .trailer-container {
        background: transparent !important;
        padding: 0px !important;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .trailer-title {
        color: white;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 20px;
        text-align: center;
    }

    iframe {
        border-radius: 20px !important;
        overflow: hidden !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>

    div[data-baseweb="select"] > div {
        cursor: pointer;
    }

    input {
        cursor: pointer;
    }

    button {
        cursor: pointer;
    }

    </style>
    """,
    unsafe_allow_html=True
)
popular_movies = get_popular_movies()

st.title("🎬 CineMatch AI")

st.markdown("""
##### Hybrid Movie Recommendation System
""")
featured_movie = popular_movies.iloc[0]['title']
# -----------------------------
# Session State Initialization
# -----------------------------
if "show_modal" not in st.session_state:
    st.session_state.show_modal = False

if "modal_trailer_url" not in st.session_state:
    st.session_state.modal_trailer_url = None

if "saved_recommendations" not in st.session_state:
    st.session_state.saved_recommendations = None

if "saved_current_period" not in st.session_state:
    st.session_state.saved_current_period = None
# -----------------------------
# Trailer Player
# -----------------------------
if (
    st.session_state.show_modal
    and
    st.session_state.modal_trailer_url
):

    video_id = st.session_state.modal_trailer_url

    embed_url = (
        f"https://www.youtube.com/embed/{video_id}"
        f"?autoplay=1"
    )

    st.markdown(
        """
        <style>

        .video-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(0,0,0,0.80);
            backdrop-filter: blur(12px);
            z-index: 9999;

            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }
        iframe {
        position: fixed !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) !important;
        z-index: 999999 !important;
    }

        </style>
        """,
        unsafe_allow_html=True
    )

    video_html = f"""
<div style="
    position:fixed;
    top:0;
    left:0;
    width:100vw;
    height:100vh;
    background:rgba(0,0,0,0.85);
    backdrop-filter:blur(10px);
    z-index:999999;

    display:flex;
    justify-content:center;
    align-items:center;
">

    <div style="
        position:relative;
        width:75%;
        max-width:1200px;
    ">

    </a>

        <iframe
            width="100%"
            height="650"
            src="{embed_url}"
            title="YouTube video player"
            frameborder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowfullscreen
            style="
                border:none;
                border-radius:20px;
                box-shadow:0 0 40px rgba(0,0,0,0.7);
            ">
        </iframe>

    </div>

</div>
"""
    
   
    

    

backdrop_url = fetch_backdrop(
    featured_movie
)
if backdrop_url:

    st.image(
        backdrop_url,
        use_container_width=True
    )

    st.markdown(
        """
        <style>
        .hero-text {
            margin-top: -320px;
            padding-left: 50px;
            padding-bottom: 100px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="hero-text">

        <h1 style="
            color:white;
            font-size:40px;
            margin-bottom:10px;
            text-shadow:2px 2px 10px black;
        ">
            🎬 Discover Your Next Favorite Movie
        </h1>

        <h3 style="
            color:#dddddd;
            font-weight:400;
            line-height:1.5;
            text-shadow:2px 2px 8px black;
        ">
            Personalized movie experiences
            powered by cinematic intelligence
        </h3>

        <p style="
            color:#cccccc;
            font-size:20px;
            margin-top:20px;
            text-shadow:2px 2px 8px black;
        ">
            🌟 Featured Tonight:
            {clean_title(featured_movie)}
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🎬 Movies", len(movies))

    with col2:
        st.metric("👥 Users", ratings['userId'].nunique())

    with col3:
        st.metric("⭐ Ratings", len(ratings))

    with col4:
        st.metric("🧠 Engine", "Hybrid AI")
st.subheader("🔥 Popular Movies")


popular_cols = st.columns(5)

for idx, (_, row) in enumerate(
    popular_movies.iterrows()
):

    with popular_cols[idx]:

        poster = fetch_poster(row['title'])

        genres_clean = row['genres'].replace(
            "|",
            " • "
        )

        popular_card = f"""
        <div class="movie-card" style="
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(10px);
            border-radius: 18px;
            padding: 12px;
            margin-bottom: 20px;
            min-height: 500px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        ">

        <img src="{poster}"
            style="
                width:100%;
                height:300px;
                object-fit:cover;
                border-radius:15px;
                margin-bottom:12px;
        ">

        <h4 style="
            color:white;
            height:70px;
            overflow:hidden;
            margin-bottom:10px;
        ">
            {clean_title(row['title'])}
        </h4>

        <p style="
            color:#ffd700;
            font-size:16px;
            margin-bottom:10px;
        ">
            ⭐ {round(row['avg_rating'], 1)}
        </p>

        <p style="
            color:#cccccc;
            font-size:14px;
            height:45px;
            overflow:hidden;
        ">
            🎭 {genres_clean}
        </p>
        

        </div>
        """

        st.markdown(
            popular_card,
            unsafe_allow_html=True
        )
        if st.button(
    "▶ Watch Trailer",
    key=f"popular_{idx}",
    use_container_width=True
):

            trailer = fetch_trailer(
        row['title']
    )

            if trailer:

                st.session_state.modal_trailer_url = trailer
                st.session_state.show_modal = True

                st.rerun()
        

        
with st.expander("🧠 How Cinematic AI Works"):

    st.markdown("""
    ### Recommendation Pipeline

    ✅ Collaborative Filtering

    Finds users with similar tastes.

    ✅ Content-Based Filtering

    Matches movie genres and metadata.

    ✅ Mood Intelligence

    Adapts recommendations based on cinematic mood.

    ✅ Time-Aware Ranking

    Suggests different content for morning, evening and night.

    ✅ Natural Language Understanding

    Understands prompts like:
    * "I want a dark crime thriller"
    * "Give me something mind-bending"
    * "Funny movies for tonight"
    """)

st.write(
    "Get personalized movie recommendations "
    
)

st.subheader("🤖 AI Movie Concierge")

ai_query = st.text_area(
    "Describe your perfect movie experience"
)
movie_list = sorted(movies['title'].unique())

if "search_reset_counter" not in st.session_state:
    st.session_state.search_reset_counter = 0

search_movie = st.selectbox(
    "🔍 Search Movie",
    [""] + movie_list,
    index=0,
    key=f"search_movie_{st.session_state.search_reset_counter}"
)
search_clicked = st.button("Search Movie")

if search_clicked and search_movie != "":

    st.session_state.selected_movie = search_movie

    st.session_state.show_modal = False
    st.session_state.modal_trailer_url = None
    st.rerun()
if "selected_movie" in st.session_state:

    search_movie = st.session_state.selected_movie

    movie_data = movies[
        movies['title'] == search_movie
    ].iloc[0]

    poster = fetch_poster(search_movie)

    st.subheader("🎬 Movie Details")

    col1, col2 = st.columns([1,3])

    with col1:
        st.image(poster, width=220)

    with col2:

        st.markdown(
            f"# {clean_title(search_movie)}"
        )

        genres_clean = movie_data['genres'].replace(
            "|",
            ", "
        )

        st.write(f"🎭 Genres: {genres_clean}")

        movie_rating = movie_stats[
            movie_stats['title'] == search_movie
        ]

        if not movie_rating.empty:

            st.write(
                f"⭐ Average Rating: "
                f"{round(movie_rating.iloc[0]['avg_rating'],2)}"
            )

            st.write(
                f"👥 Rating Count: "
                f"{int(movie_rating.iloc[0]['rating_count'])}"
            )

        trailer_key = fetch_trailer(search_movie)

            

        if st.button(
        "▶ Watch Trailer",
        key="search_trailer"
    ):

            st.session_state.modal_trailer_url = trailer_key
            st.session_state.show_modal = True

            st.rerun()

    st.markdown("---")

# -----------------------------
# User Input
# -----------------------------
# -----------------------------
# User Mode Selection
# -----------------------------
cinematic_mood = "Any"
mode = st.radio(
    "Choose User Type",
    ["Existing User", "New User"]
)

# -----------------------------
# Existing User
# -----------------------------
st.sidebar.header("User Options")
if mode == "Existing User":

    user_id = st.sidebar.selectbox(
        "Select User ID",
        user_movie_matrix.index
    )

    genre = st.sidebar.selectbox(
        "Select Genre",
        [
            "All",
            "Action",
            "Adventure",
            "Animation",
            "Comedy",
            "Crime",
            "Drama",
            "Fantasy",
            "Horror",
            "Romance",
            "Sci-Fi",
            "Thriller"
        ]
    )
    cinematic_mood = st.selectbox(
    "🎭 Choose Cinematic Mood",
    [
        "Any",
        "thoughtful",
        "mind-bending",
        "dark",
        "happy",
        "romantic",
        "psychological",
        "funny",
        "epic"
    ],
    disabled=(ai_query.strip() != "")
)

# -----------------------------
# New User
# -----------------------------
else:

    genre = st.sidebar.selectbox(
        "Choose Favorite Genre",
        [
            "Action",
            "Adventure",
            "Animation",
            "Comedy",
            "Crime",
            "Drama",
            "Fantasy",
            "Horror",
            "Romance",
            "Sci-Fi",
            "Thriller"
        ]
    )

# -----------------------------
# Recommendation Button
# -----------------------------





generate_clicked = st.button(
    "Get Recommendations"
)
if generate_clicked:

    with st.spinner("🎬 AI is crafting your cinematic experience..."):
        if ai_query.strip() != "":

    # AI query overrides manual filters
            genre = "All"
            cinematic_mood = "Any"

            ai_result = advanced_ai_understanding(ai_query)

            if len(ai_result["genres"]) > 0:
                genre = ai_result["genres"][0]

            if len(ai_result["moods"]) > 0:
                cinematic_mood = ai_result["moods"][0]

        if mode == "Existing User":

            user_profile = build_user_profile(
                user_id
            )

            recommendations, current_period = recommend_movies(
                user_id=user_id,
                genre=genre,
                cinematic_mood=cinematic_mood,
                time_context=time_context
            )

        else:

            recommendations = movie_stats.merge(
                movies[['title', 'genres']],
                on='title'
            )

            recommendations = recommendations[
                recommendations['genres'].str.contains(
                    genre,
                    case=False,
                    na=False
                )
            ]

            recommendations = recommendations[
                recommendations['rating_count'] > 100
            ]

            recommendations = recommendations.sort_values(
                by=['avg_rating', 'rating_count'],
                ascending=False
            ).head(10)
            recommendations['match_percent'] = (
                recommendations['avg_rating']
                    / recommendations['avg_rating'].max()
                    ) * 100

            current_period = "Popular Picks"
        st.session_state["saved_mood"] = cinematic_mood
        st.session_state["saved_genre"] = genre
        st.session_state["saved_mode"] = mode
        st.session_state["saved_recommendations"] = (
            recommendations.copy()
        )

        st.session_state["saved_current_period"] = (
            current_period
        )


    
if st.session_state.saved_recommendations is not None:

    recommendations = st.session_state[
        "saved_recommendations"
    ]

    current_period = st.session_state[
        "saved_current_period"
    ]

    st.subheader(
        f"🌙 {current_period} Cinematic Experience"
    )
    quality = recommendation_quality(recommendations)

    st.success(
    f"🎯 Recommendation Quality Score: {quality}/5"
)

    cols = st.columns(5)
    saved_mood = st.session_state.get(
    "saved_mood",
    "Any"
)

    saved_genre = st.session_state.get(
    "saved_genre",
    "All"
)

    saved_mode = st.session_state.get(
    "saved_mode",
    mode
)
    if ai_query.strip() != "":
        st.info(
        "🤖 AI Concierge is controlling recommendation filters."
    )
    
    for idx, (_, row) in enumerate(
        recommendations.iterrows()
    ):

        with cols[idx % 5]:

            poster = fetch_poster(row['title'])

            genres_clean = row['genres'].replace(
                "|",
                " • "
            )
            if ai_query != "":
                reason_text = "AI analyzed your request and selected this movie."

            elif saved_mood != "Any":
                reason_text = f"Matches your {saved_mood} cinematic mood."

            elif saved_genre != "All":
                reason_text = f"Highly rated in the {saved_genre} genre."

            else:
                reason_text = "Recommended from similar viewers."
            if row['match_percent'] >= 90:
                match_badge = "🔥 Excellent Match"

            elif row['match_percent'] >= 75:
                match_badge = "⭐ Strong Match"

            else:
                match_badge = "👍 Good Match"
            st.markdown(
                f"""
                <div class="movie-card" style="
                    background: rgba(255,255,255,0.05);
                    backdrop-filter: blur(10px);
                    border-radius: 18px;
                    padding: 12px;
                    margin-bottom: 20px;
                    min-height: 520px;
                    overflow: hidden;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
                ">
                

                <img src="{poster}"
                    style="
                        width:100%;
                        height:300px;
                        object-fit:cover;
                        border-radius:15px;
                        margin-bottom:12px;
                ">

                <h4 style="
                    color:white;
                    height:78px;
                    overflow:hidden;
                    margin-bottom:10px;
                    font-size: 28px;
                ">
                    {clean_title(row['title'])}
                </h4>

                <p style="
                    color:#ffd700;
                    font-size:16px;
                    margin-bottom:6px;
                ">
                    ⭐ IMDB {round(row['avg_rating'], 1)}
                </p>
                
                <p style="
                    color:#00ffcc;
                    font-size:14px;
                    margin-bottom:6px;
                ">
                {match_badge}
                </p>

                <p style="
                    color:#00ffcc;
                    font-size:15px;
                    margin-bottom:10px;
                ">
                🎯 {round(row['match_percent'],1)}%
                </p>
                

                <p style="
                    color:#cccccc;
                    font-size:14px;
                    height:45px;
                    overflow:hidden;
                ">
                    🎭 {genres_clean}
                </p>
                
                </div>
                """,
                
                unsafe_allow_html=True
            )
            st.markdown(f"""
<div class="recommendation-reason" style="
    color:#d9d9d9;
    font-size:14px;
    margin-top:8px;
">
    <b>Why Recommended?</b><br>
     {reason_text}
</div>
""", unsafe_allow_html=True)
            

          
            st.markdown(
    """
    <style>

    .movie-card {
        transition:
            transform 0.35s ease,
            box-shadow 0.35s ease,
            border 0.35s ease;
    }

    .movie-card:hover {

        transform:
            translateY(-12px)
            scale(1.04);

        box-shadow:
            0 18px 40px rgba(0,0,0,0.65);

        border:
            1px solid rgba(255,255,255,0.18);

        background:
            rgba(255,255,255,0.08);
    }

    .movie-card img {
        transition:
            transform 0.35s ease;
    }

    .movie-card:hover img {

        transform: scale(1.03);

    }

    </style>
    """,
    unsafe_allow_html=True
)
            
            if st.button(
    "▶ Watch Trailer",
    key=f"rec_{idx}",
    use_container_width=True
):

                trailer = fetch_trailer(
        row['title']
    )

                if trailer:

                    st.session_state.modal_trailer_url = trailer
                    st.session_state.show_modal = True

                    st.rerun()
# ==========================
# TRAILER DIALOG
# ==========================
st.markdown("""
<style>
button[aria-label="Close"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

if (
    st.session_state.show_modal
    and
    st.session_state.modal_trailer_url
):

    trailer_url = st.session_state.modal_trailer_url

    video_id = trailer_url

    embed_url = (
        f"https://www.youtube.com/embed/{video_id}"
        "?autoplay=1&mute=0&controls=1&rel=0"
    )

    @st.dialog("🎬 Trailer", width="large")
    def show_trailer():

        components.iframe(
            embed_url,
            height=600,
            scrolling=False
        )

    show_trailer()
    
    st.markdown("---")
        
st.markdown("""
---
<center>
            
**Version 1.0**
             
Built By Kanishka Sharma

**Tech Stack** : 
Python • Streamlit • Scikit-Learn • TMDb API

**Recommendation Techniques** :
Collaborative Filtering  •  Content-Based Filtering  •  Hybrid Recommendation Engine  •  Mood-Aware Ranking  
•  Time-Aware Recommendations  •  Natural Language Query Understanding
</center>        
""",unsafe_allow_html=True)
    
    