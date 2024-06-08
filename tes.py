import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# Initialize IBM NLU
api_key = 'sSXHNVTN-iWR7PhHP3t2lLFr2k7oYNn4YXYXBM99wesI'
service_url = 'https://api.au-syd.natural-language-understanding.watson.cloud.ibm.com/instances/5cb19ae3-01a5-4a79-9e70-dc2d7a42af02'

authenticator = IAMAuthenticator(api_key)
nlu = NaturalLanguageUnderstandingV1(
    version='2021-08-01',
    authenticator=authenticator
)

# Set service URL
nlu.set_service_url(service_url, verify=False)

# Function to analyze feedback using IBM NLU
def analyze_feedback(feedback_text):
    try:
        response = nlu.analyze(
            text=feedback_text,
            features=Features(sentiment=SentimentOptions())
        ).get_result()
        sentiment = response['sentiment']['document']['label']
        return sentiment
    except Exception as e:
        st.error(f"Error analyzing feedback: {e}")
        return None

# Load dataset
df_buku = pd.read_csv('books.csv')
df_buku['description'] = df_buku['description'].fillna('')

# TF-IDF and cosine similarity
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df_buku['description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get popular title by category
def get_popular_title(df, category):
    category_df = df[df['categories'] == category]
    popular_book = category_df.sort_values(by='average_rating', ascending=False).iloc[0]
    return popular_book['title']

# Popular genres
genre_popular = ['Fiction', 'Juvenile Fiction', 'Biography & Autobiography', 'Young Adult Fiction', 'Drama',
                 'Art museum curators', 'Juvenile Nonfiction', 'Comics & Graphic Novels', 'History', 'Boys', 'Self-Help',
                 'Blind', 'Philosophy', 'Business & Economics', 'Science', 'Religion', 'Fantasy fiction',
                 'Education', 'American fiction', 'Labrador retriever', 'Poetry', 'Art', 'Sports & Recreation']

# Get popular titles
popular_titles = {category: get_popular_title(df_buku, category) for category in genre_popular}

# Function to get genre recommendations
def get_genre_recommendations(selected_genres, cosine_sim=cosine_sim, df=df_buku):
    recommendations_by_genre = {}
    genre_sim_scores = {}
    for genre in selected_genres:
        indices = df.index[df['categories'] == genre].tolist()
        if not indices:
            continue
        avg_sim_scores = np.mean(cosine_sim[indices], axis=0)
        sim_scores = list(enumerate(avg_sim_scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[:6]  # Get top 5 recommendations (skipping the first element)
        book_indices = [i[0] for i in sim_scores]
        recommended_books = df.iloc[book_indices][['title', 'authors', 'categories']]
        recommendations_by_genre[genre] = recommended_books
        genre_sim_scores[genre] = sim_scores
    return recommendations_by_genre, genre_sim_scores

# Function to get genre recommendations with user preferences
def get_genre_recommendations_with_preferences(selected_genres, reading_type, popularity, rating_influence, selected_book, cosine_sim=cosine_sim, df=df_buku):
    all_recommended_books = []
    all_sim_scores = []

    for genre in selected_genres:
        indices = df.index[df['categories'] == genre].tolist()
        if not indices:
            continue

        avg_sim_scores = np.mean(cosine_sim[indices], axis=0)
        sim_scores = list(enumerate(avg_sim_scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[:6]
        book_indices = [i[0] for i in sim_scores]
        recommended_books = df.iloc[book_indices][['title', 'authors', 'average_rating', 'num_pages', 'categories']]

        # Apply user preferences
        if reading_type == 'Menyelesaikan buku dalam sekali duduk':
            recommended_books = recommended_books[recommended_books['num_pages'] <= 100]  # Example threshold, adjust as needed

        if popularity == 'Ya':
            popular_titles_list = df.sort_values(by='average_rating', ascending=True)['title'].head(10).tolist()
            popular_books_df = df[df['title'].isin(popular_titles_list)]
            recommended_books = pd.concat([recommended_books, popular_books_df])

        if rating_influence == 'Ya' and selected_book:
            similar_books, _ = get_genre_recommendations(selected_genres, cosine_sim=cosine_sim, df=df)
            similar_books_list = []
            for genre_books in similar_books.values():
                similar_books_list.append(genre_books)
            similar_books_df = pd.concat(similar_books_list)
            recommended_books = pd.concat([recommended_books, similar_books_df])

        # Remove duplicates and sort by overall_sim_scores
        recommended_books = recommended_books.drop_duplicates()
        new_indices = recommended_books.index.tolist()
        avg_new_sim_scores = np.mean(cosine_sim[new_indices], axis=0)
        new_sim_scores = list(enumerate(avg_new_sim_scores))
        new_sim_scores = sorted(new_sim_scores, key=lambda x: x[1], reverse=True)
        new_sim_scores = new_sim_scores[:6]

        all_recommended_books.append(recommended_books)
        all_sim_scores.extend(new_sim_scores)

    return pd.concat(all_recommended_books), all_sim_scores

# Function to get popular titles by genre
def get_popular_titles_by_genre(selected_genres, df=df_buku):
    popular_titles_by_genre = []
    for genre in selected_genres:
        category_df = df[df['categories'] == genre]
        if not category_df.empty:
            popular_book = category_df.sort_values(by='average_rating', ascending=False).iloc[0]
            popular_titles_by_genre.append(popular_book['title'])
    return popular_titles_by_genre

# Function to get user feedback
def get_user_feedback():
    feedback = st.text_input("Masukkan feedback Anda di sini:")
    if feedback:
        sentiment = analyze_feedback(feedback)
        if sentiment:
            st.write(f"Sentimen dari umpan balik Anda adalah: {sentiment}")
    return feedback

# Main function
def main():
    st.title("Bookverse: Website yang memberikan rekomendasi buku bacaan untuk Anda")
    st.markdown("---")
    st.write("Halo! Saya di sini untuk membantu Anda menemukan rekomendasi buku berdasarkan preferensi Anda.")

    name = st.text_input("Silakan masukkan nama Anda:")

    if name:
        st.write(f"Hai {name}! Siap untuk menemukan buku-buku keren sesuai selera Anda? Ayo kita mulai!")

        top_25_genre = df_buku['categories'].value_counts().index[:25]
        selected_genres = st.multiselect("Pilih genre/jenis buku favoritmu (top 25 genre)", top_25_genre)

        if selected_genres:
            reading_type = st.radio("Manakah tipe kamu saat membaca buku?", ('Menyelesaikan buku dalam sekali duduk', 'Santai dalam membaca'), index=None)

            if reading_type:
                popularity = st.radio("Apakah tingkat kepopuleran buku mempengaruhi keputusan Anda dalam memilih buku?", ('Ya', 'Tidak'), index=None)

                if popularity:
                    rating_influence = st.radio("Apakah rating buku mempengaruhi keputusan Anda dalam memilih buku?", ('Ya', 'Tidak'), index=None)

                    if rating_influence:
                        popular_titles = get_popular_titles_by_genre(selected_genres)

                        st.write("Pilih judul buku yang membuatmu tertarik:")
                        popular_titles.insert(0, "Pilih buku...")  # Add placeholder
                        selected_book = st.selectbox("Pilih buku:", popular_titles, index=0)

                        if selected_book != "Pilih buku...":
                            st.write("Yuk, kita cek apa saja pilihan kamu:")
                            st.write(f"Nama                : {name}")
                            st.write(f"Genre Favorit       : {', '.join(selected_genres)}")
                            st.write(f"Tipe Membaca        : {reading_type}")
                            st.write(f"Pengaruh Kepopuleran: {popularity}")
                            st.write(f"Pengaruh Rating     : {rating_influence}")
                            st.write(f"Judul buku yang tertarik   : {selected_book}")

                            if 'confirm' not in st.session_state:
                                st.session_state.confirm = False

                            confirm = st.button("Konfirmasi Pilihan")
                            if confirm or st.session_state.confirm:
                                st.session_state.confirm = True
                                st.write("Berikut beberapa buku yang mungkin menarik bagi Anda:")
                                recommendations_by_genre, genre_sim_scores = get_genre_recommendations(selected_genres)
                                recommended_books, overall_sim_scores = get_genre_recommendations_with_preferences(selected_genres, reading_type, popularity, rating_influence, selected_book)

                                if not recommendations_by_genre:
                                    st.write("Maaf, kami tidak dapat menemukan rekomendasi berdasarkan genre yang dipilih.")
                                else:
                                    for genre, recommendations in recommendations_by_genre.items():
                                        st.write(f"Rekomendasi berdasarkan genre {genre}:")
                                        st.dataframe(recommendations)

                                        st.write(f"Similarity Scores untuk rekomendasi berdasarkan genre {genre}:")
                                        st.write(genre_sim_scores[genre])

                                    st.write("Rekomendasi berdasarkan keseluruhan:")
                                    st.dataframe(recommended_books[['title', 'authors', 'categories']].head(15))

                                    st.write("Similarity Scores untuk rekomendasi berdasarkan keseluruhan:")
                                    st.write(overall_sim_scores[:15])

                                feedback = get_user_feedback()
                                if feedback:
                                    st.write("Terima kasih atas umpan balik Anda!")
                                else:
                                    st.write("Kami menunggu masukan dari Anda! Silakan isi feedback Anda di atas untuk membantu kami memberikan rekomendasi yang lebih baik.")

if __name__ == "__main__":
    main()
