import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    indices = df.index[df['categories'].isin(selected_genres)].tolist()
    if not indices:
        return pd.DataFrame(), []

    avg_sim_scores = np.mean(cosine_sim[indices], axis=0)
    sim_scores = list(enumerate(avg_sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    book_indices = [i[0] for i in sim_scores]
    recommended_books = df.iloc[book_indices][['title', 'authors', 'categories']]
    return recommended_books, sim_scores

# Function to get genre recommendations with user preferences
def get_genre_recommendations_with_preferences(selected_genres, reading_type, popularity, rating_influence, selected_book, cosine_sim=cosine_sim, df=df_buku):
    indices = df.index[df['categories'].isin(selected_genres)].tolist()
    if not indices:
        return pd.DataFrame(), []

    avg_sim_scores = np.mean(cosine_sim[indices], axis=0)
    sim_scores = list(enumerate(avg_sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    book_indices = [i[0] for i in sim_scores]
    recommended_books = df.iloc[book_indices][['title', 'authors', 'average_rating', 'num_pages']]

    st.write("Books before preferences applied:", recommended_books)  # Debugging

    # Apply user preferences
    if reading_type == 'Menyelesaikan buku dalam sekali duduk':
        recommended_books = recommended_books[recommended_books['num_pages'] <= 100]  # Example threshold, adjust as needed

    st.write("Books after reading_type applied:", recommended_books)  # Debugging

    if popularity == 'Ya':
        popular_titles = df.sort_values(by='average_rating', ascending=False)['title'].head(10).tolist()
        popular_books_df = df[df['title'].isin(popular_titles)]
        recommended_books = pd.concat([recommended_books, popular_books_df])

    st.write("Books after popularity applied:", recommended_books)  # Debugging

    if rating_influence == 'Ya' and selected_book:
        similar_books, _ = get_genre_recommendations([selected_book], cosine_sim=cosine_sim, df=df)
        recommended_books = pd.concat([recommended_books, similar_books])

    st.write("Books after rating_influence applied:", recommended_books)  # Debugging

    return recommended_books.drop_duplicates(), sim_scores


# Function to get user feedback
def get_user_feedback():
    feedback = st.text_input("Masukkan feedback Anda di sini:")
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
                        st.write("Pilih judul buku yang membuatmu tertarik:")
                        popular_titles = df_buku.sort_values(by='average_rating', ascending=False)['title'].head(10).tolist()
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

                            confirm = st.button("Konfirmasi Pilihan")
                            if confirm:
                                st.write("Berikut beberapa buku yang mungkin menarik bagi Anda:")
                                recommendations, genre_sim_scores = get_genre_recommendations(selected_genres)
                                recommended_books, overall_sim_scores = get_genre_recommendations_with_preferences(selected_genres, reading_type, popularity, rating_influence, selected_book)

                                if recommendations.empty:
                                    st.write("Maaf, kami tidak dapat menemukan rekomendasi berdasarkan genre yang dipilih.")
                                else:
                                    st.write("Rekomendasi berdasarkan genre:")
                                    st.dataframe(recommendations)

                                    st.write("Similarity Scores untuk rekomendasi berdasarkan genre:")
                                    st.write(genre_sim_scores)

                                    st.write("Rekomendasi berdasarkan keseluruhan:")
                                    st.dataframe(recommended_books[['title', 'authors', 'categories']], use_container_width=True)

                                    st.write("Similarity Scores untuk rekomendasi berdasarkan keseluruhan:")
                                    st.write(overall_sim_scores)

                                feedback = get_user_feedback()
                                if feedback:
                                    st.write("Terima kasih atas umpan balik Anda!")
                                else:
                                    st.write("Kami menunggu masukan dari Anda! Silakan isi feedback Anda di atas untuk membantu kami memberikan rekomendasi yang lebih baik.")

if __name__ == "__main__":
    main()
