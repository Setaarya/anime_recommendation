# -*- coding: utf-8 -*-
"""content_based_anime_recommendation.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1cU4e50S9p89pBJgJpkKRZ2p-8UOauhwj

# **LOAD DATA**

Mengimpor modul files dari google.colab dan menjalankan fungsi files.upload() untuk memungkinkan pengguna mengunggah file lokal (misalnya, kaggle.json) ke lingkungan Google Colab.
"""

from google.colab import files
files.upload()  # Upload kaggle.json di sini

"""Membuat direktori untuk menyimpan file konfigurasi API Kaggle, menyalin kredensial API, mengunduh dataset "anime-recommendations-database" dari Kaggle, dan mengekstraknya untuk digunakan dalam analisis."""

# Persiapan kredensial Kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download dataset dari Kaggle
!kaggle datasets download -d cooperunion/anime-recommendations-database

# Ekstrak file zip
!unzip anime-recommendations-database.zip

"""# **IMPORT LIBRARY**

Mengimport library yang dibutuhkan untuk membuat sistem rekomendasi
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

"""# **DATA UNDERSTANDING**

Menampilkan preview data dari anime
"""

anime = pd.read_csv('/content/anime.csv')
anime.head()

"""Menampilkan jumlah kolom/fitur, banyak data, dan tipe data dari data anime. Dataset anime ini terdiri dari 12.294 entri yang mencakup informasi seperti judul, genre, tipe, jumlah episode, rating, dan jumlah anggota yang menonton. Beberapa kolom memiliki data yang hilang, yakni kolom genre (62 data kosong), type (25 data kosong), dan rating (230 data kosong)."""

anime.info()

"""Menampilkan jumlah data di data anime berdasarkan anime_id yang unik"""

print('Jumlah data anime : ', len(anime.anime_id.unique()))

"""Mengecek apakah ada nilai NaN di fitur name dan menampilkan jumlah data yang unik pada kolom fitur name, dari eksplorasi ada 2 data yang duplikat."""

# Cek apakah ada nilai NaN di kolom 'name' dan tampilkan jumlah NaN
na_count = anime['name'].isnull().sum()
if na_count > 0:
    print(f"Terdapat {na_count} nilai NaN di kolom 'name'")
else:
    print("Tidak ada nilai NaN di kolom 'name'")

# Tampilkan jumlah dan nama anime yang unik (tanpa NaN)
print('Banyak data anime: ', len(anime['name'].dropna().unique()))
print('Judul-judul anime: ', anime['name'].dropna().unique())

"""Menampilkan data yang duplikat pada kolom fitur name, nantinya ini akan kita drop karena dapat menyebabkan kerancuan data."""

# Cek apakah ada data duplikat pada kolom 'name'
duplicate_names = anime['name'].duplicated().sum()
if duplicate_names > 0:
    print(f"Terdapat {duplicate_names} duplikat pada kolom 'name'")
    print("Contoh data duplikat:")
    print(anime[anime['name'].duplicated(keep=False)])
else:
    print("Tidak ada duplikat pada kolom 'name'")

"""Mengecek dan menampilkan data yang unik sekaligus apakah ada nilai NaN di kolom fitur genre. Dan ternyata ada nilai NaN, dan 3264 genre beberapa diantaranya seperti : Genre-genre anime:  ['Drama, Romance, School, Supernatural'
 'Action, Adventure, Drama, Fantasy, Magic, Military, Shounen'
 'Action, Comedy, Historical, Parody, Samurai, Sci-Fi, Shounen' ...
 'Hentai, Sports' 'Drama, Romance, School, Yuri' 'Hentai, Slice of Life']
"""

# Cek apakah ada nilai NaN di kolom 'genre' dan tampilkan jumlah NaN
na_count = anime['genre'].isnull().sum()  # Menghitung jumlah NaN
if na_count > 0:
    print(f"Terdapat {na_count} nilai NaN di kolom 'genre'")
else:
    print("Tidak ada nilai NaN di kolom 'genre'")

# Tampilkan jumlah dan jenis genre anime yang unik (tanpa NaN)
print('Banyak genre anime: ', len(anime['genre'].dropna().unique()))
print('Genre-genre anime: ', anime['genre'].dropna().unique())

"""Menampilkan nilai nilai data di kolom fitur type dan mengecek apakah ada nilai NaN di kolom fitur type. Dan hasilnya menunjukkan bahwa ada nilai NaN di kolom fitur type. Dan ini merupakan nilai-nilai data di kolom fitur type, Tipe-tipe anime:  ['Movie' 'TV' 'OVA' 'Special' 'Music' 'ONA']"""

# Cek apakah ada nilai NaN di kolom 'type' dan tampilkan jumlah NaN
na_count = anime['type'].isnull().sum()  # Menghitung jumlah NaN
if na_count > 0:
    print(f"Terdapat {na_count} nilai NaN di kolom 'type'")
else:
    print("Tidak ada nilai NaN di kolom 'type'")

# Tampilkan jumlah dan tipe-tipe anime yang unik (tanpa NaN)
print('Banyak tipe anime: ', len(anime['type'].dropna().unique()))
print('Tipe-tipe anime: ', anime['type'].dropna().unique())

"""Menampilkan nilai nilai data di kolom fitur episodes dan mengecek apakah ada nilai NaN di kolom fitur episodes. Dan hasilnya menunjukkan bahwa tidak ada nilai NaN di kolom fitur episodes, tetapi ada nilai unknown di kolom fitur ini. Dan ini merupakan nilai-nilai data di kolom fitur episodes, Episode-episode anime:  ['1' '64' '51' '24' '10' '148' '110' '13' '201' '25' '22' '75' '4' '26'
 '12' '27' '43' '74' '37' '2' '11' '99' 'Unknown' '39' '101' '47' '50']
"""

# Cek apakah ada nilai NaN di kolom 'episodes'
na_count = anime['episodes'].isnull().sum()
if na_count > 0:
    print(f"Terdapat {na_count} nilai NaN di kolom 'episodes'")
else:
    print("Tidak ada nilai NaN di kolom 'episodes'")

# Tampilkan jumlah dan nilai unik episode anime (tanpa NaN)
print('Banyak episode anime: ', len(anime['episodes'].dropna().unique()))
print('Episode-episode anime: ', anime['episodes'].dropna().unique())

"""Menampilkan jumlah baris data dengan nilai "Unknown" pada kolom  episodes"""

# Cek jumlah nilai 'Unknown' di kolom 'episodes'
unknown_count = (anime['episodes'] == 'Unknown').sum()
print(f"Terdapat {unknown_count} nilai 'Unknown' di kolom 'episodes'")

"""Menampilkan nilai nilai data unik di kolom fitur rating dan mengecek apakah ada nilai NaN di kolom fitur rating. Dan hasilnya menunjukkan bahwa  ada nilai NaN di kolom fitur rating. Nilai data dari kolom fitur rating berkisar antara -1 sampai dengan 10. -1 jika ditonton tetapi tidak diber rating"""

# Cek apakah ada nilai NaN di kolom 'rating' dan tampilkan jumlah NaN
na_count = anime['rating'].isnull().sum()
if na_count > 0:
    print(f"Terdapat {na_count} nilai NaN di kolom 'rating'")
else:
    print("Tidak ada nilai NaN di kolom 'rating'")

# Tampilkan jumlah dan nilai unik rating anime (tanpa NaN)
print('Banyak nilai rating anime: ', len(anime['rating'].dropna().unique()))
print('Nilai-nilai rating anime: ', anime['rating'].dropna().unique())

"""Menampilkan nilai nilai data di kolom fitur members dan mengecek apakah ada nilai NaN di kolom fitur members. Dan hasilnya menunjukkan bahwa tidak ada nilai NaN di kolom fitur members. Nilai-nilai members anime:  [200630 793665 114262 ...  27411  57355    652]"""

# Cek apakah ada nilai NaN di kolom 'rating'
if anime['members'].isnull().any():
    print("Terdapat nilai NaN di kolom 'members'")
else:
    print("Tidak ada nilai NaN di kolom 'members'")

# Tampilkan jumlah dan nilai unik rating anime (tanpa NaN)
print('Banyak nilai members anime: ', len(anime['members'].dropna().unique()))
print('Nilai-nilai members anime: ', anime['members'].dropna().unique())

"""# **DATA PREPARATION**

Dari proses data understanding langkah-langkah yang akan kita lakukan adalah:


1.   Drop duplikat data pada kolom name
2.   Drop nilai NaN pada kolom genre
3.   Drop nilai NaN pada kolom type
4.   Drop nilai Unknown pada kolom episodes
5.   Drop nilai NaN pada kolom rating

Drop duplikat data pada kolom name
"""

# 1. Drop duplikat berdasarkan kolom 'name'
anime_cleaned = anime.drop_duplicates(subset='name')

"""Drop nilai NaN pada kolom genre, type, dan rating"""

# 2. Drop data NaN pada kolom 'genre', 'type', dan 'rating'
anime_cleaned = anime_cleaned.dropna(subset=['genre', 'type', 'rating'])

"""Drop nilai unknown pada kolom episodes"""

# 3. Drop baris dengan nilai 'Unknown' pada kolom 'episodes'
anime_cleaned = anime_cleaned[anime_cleaned['episodes'].str.lower() != 'unknown']

"""Mereset index DataFrame setelah pembersihan data"""

# 4. Reset index setelah pembersihan
anime_cleaned.reset_index(drop=True, inplace=True)

"""Cek data hasil pembersihan. Data yang didapatkan setelah pembersihan sebanyak 11.828 data anime"""

# Cek hasil akhir
print("Jumlah data setelah dibersihkan:", len(anime_cleaned))
print(anime_cleaned.head())

"""# **CONTENT BASED FILTERING**

Mengubah data genre anime menjadi representasi numerik menggunakan TF-IDF Vectorizer dengan menghapus kata-kata umum (stop words), lalu menghasilkan vektor berdasarkan kata-kata unik dalam genre untuk digunakan dalam perhitungan kemiripan. Hasilnya adalah daftar fitur (genre) unik yang menjadi dimensi dari vektor TF-IDF ['action', 'adventure', 'ai', 'arts', 'cars', 'comedy', 'dementia',
       'demons', 'drama', 'ecchi', 'fantasy', 'fi', 'game', 'harem',
       'hentai', 'historical', 'horror', 'josei', 'kids', 'life', 'magic',
       'martial', 'mecha', 'military', 'music', 'mystery', 'parody',
       'police', 'power', 'psychological', 'romance', 'samurai', 'school',
       'sci', 'seinen', 'shoujo', 'shounen', 'slice', 'space', 'sports',
       'super', 'supernatural', 'thriller', 'vampire', 'yaoi', 'yuri']
"""

# Inisialisasi TF-IDF Vectorizer dengan menghapus stop words (kata umum dalam bahasa Inggris)
tfidf = TfidfVectorizer(stop_words="english")

# Mengubah kolom 'genre' menjadi representasi numerik berbasis TF-IDF
# Setiap genre akan direpresentasikan sebagai vektor berdasarkan kata-kata unik
tfidf_matrix = tfidf.fit_transform(anime_cleaned['genre'])

# Menampilkan daftar fitur/kata unik yang diambil dari kolom 'genre'
# Ini adalah nama-nama genre individual yang digunakan sebagai dimensi vektor
tfidf.get_feature_names_out()

"""Melihat ukuran (dimensi) dari matriks TF-IDF yang dihasilkan, yang menunjukkan jumlah anime (baris) dan jumlah fitur unik dari genre (kolom). 11.828 baris dan 46 kolom."""

# Melihat ukuran matrix tfidf
tfidf_matrix.shape

"""Mengonversi matriks TF-IDF dari bentuk sparse menjadi dense array (NumPy array) agar lebih mudah dilihat, dianalisis, atau digunakan dalam proses komputasi selanjutnya."""

# Mengubah matriks TF-IDF dari bentuk sparse matrix ke dense array (NumPy array)
# Tujuannya agar lebih mudah dilihat atau diproses lebih lanjut
tfidf_array = tfidf_matrix.toarray()

"""Membuat DataFrame dari matriks TF-IDF dengan baris sebagai judul anime dan kolom sebagai genre unik, lalu menampilkan secara acak 21 genre dan 10 anime untuk memberikan gambaran isi dari representasi TF-IDF dalam bentuk yang lebih terbaca."""

# Membuat dataframe untuk melihat tf-idf matrix
# Kolom diisi dengan genre anime
# Baris diisi dengan judul anime

pd.DataFrame(
    tfidf_matrix.todense(),
    columns=tfidf.get_feature_names_out(),
    index=anime_cleaned.name
).sample(21, axis=1).sample(10, axis=0)

"""Menghitung skor kemiripan antar semua anime berdasarkan genre menggunakan cosine similarity terhadap vektor TF-IDF, lalu menghasilkan matriks kemiripan di mana nilai mendekati 1 menunjukkan genre yang sangat mirip antar anime."""

# Menghitung kemiripan antar semua anime berdasarkan TF-IDF dari deskripsi/fitur teks
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Menampilkan matriks kemiripan cosine (nilai antara 0 dan 1, semakin tinggi semakin mirip)
cosine_sim

"""Membuat DataFrame dari matriks cosine similarity dengan baris dan kolom berisi judul anime, sehingga memudahkan pencarian kemiripan antar anime. Kemudian, sampel 5 kolom dan 10 baris ditampilkan secara acak untuk melihat sebagian isi matriks kemiripan tersebut."""

# Membuat dataframe dari variabel cosine_sim dengan baris dan kolom berupa judul anime
cosine_sim_df = pd.DataFrame(cosine_sim, index=anime_cleaned['name'], columns=anime_cleaned['name'])
print('Shape:', cosine_sim_df.shape)

# Melihat similarity matrix pada setiap judul anime
cosine_sim_df.sample(5, axis=1).sample(10, axis=0)

"""Fungsi anime_recommendations akan memberikan rekomendasi anime berdasarkan judul yang diberikan, dengan mencari anime yang paling mirip menggunakan cosine similarity. Fungsi ini akan mengembalikan daftar k anime yang paling mirip, mengabaikan judul anime yang diminta, dan menampilkan informasi genre dari anime tersebut."""

def anime_recommendations(anime_title, similarity_data=cosine_sim_df, items=anime_cleaned[['name', 'genre']], k=50):
    # Mengambil indeks anime yang paling mirip berdasarkan nilai cosine similarity tertinggi
    index = similarity_data.loc[:, anime_title].to_numpy().argpartition(
        range(-1, -k, -1))

    # Mengambil nama-nama anime yang paling mirip
    most_similar = similarity_data.columns[index[-1:-(k+2):-1]]

    # Menghapus judul anime yang diminta dari daftar hasil rekomendasi
    most_similar = most_similar.drop(anime_title, errors='ignore')

    # Menggabungkan hasil dengan data asli untuk mendapatkan informasi genre
    return pd.DataFrame(most_similar).merge(items).head(k)

"""Menampilkan 50 rekomendasi teratas berdasarkan judul anime"""

# Mendapatkan rekomendasi anime yang mirip dengan Gintama
anime_recommendations('Gintama')

"""Fungsi get_recommendations_by_genre memberikan rekomendasi anime berdasarkan genre yang diminta. Fungsi ini pertama-tama menyaring anime berdasarkan genre yang diberikan, kemudian memilih satu anime sebagai referensi dan menggunakan fungsi anime_recommendations untuk memberikan rekomendasi berdasarkan anime tersebut. Jika tidak ada anime yang sesuai dengan genre, fungsi ini akan mengembalikan pesan error. Hasilnya ditampilkan dalam bentuk tabel yang mencakup rekomendasi anime dan genre terkait."""

def get_recommendations_by_genre(genre, similarity_data=cosine_sim_df, items=anime_cleaned[['name', 'genre']], k=50):
    # Filter anime berdasarkan genre yang diberikan
    filtered_items = items[items['genre'].str.contains(genre, case=False)]

    # Jika tidak ada anime dengan genre tersebut, kembalikan pesan error
    if filtered_items.empty:
        return "Tidak ada anime dengan genre tersebut."

    # Ambil satu anime dari hasil filter untuk dijadikan acuan rekomendasi
    anime_title = filtered_items['name'].iloc[0]

    # Dapatkan rekomendasi anime berdasarkan anime acuan
    recommendations = anime_recommendations(anime_title, similarity_data, items, k)

    # Tampilkan hasil rekomendasi dalam tabel
    recommendations = pd.DataFrame(recommendations)
    recommendations = recommendations.rename(columns={'name': 'Anime Recommendations'})
    recommendations = recommendations[['Anime Recommendations', 'genre']]

    # Styling tabel
    recommendations = recommendations.style.set_properties(**{'text-align': 'left'})
    recommendations = recommendations.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])

    return recommendations

"""Menampilkan 50 rekomendasi teratas berdasarkan genre anime"""

# Mendapatkan rekomendasi anime dengan genre "Action"
recommendations = get_recommendations_by_genre("Action")

# Menampilkan hasil rekomendasi
display(recommendations)

"""# **EVALUATION**

Cosine Similarity Score Average (Internal Metric)

Mengukur seberapa "relevan" rekomendasi berdasarkan rata-rata kemiripan antara anime target dan anime lain yang paling mirip. Menghitung rata-rata skor kemiripan antara suatu anime (diberikan melalui anime_title) dan k anime lainnya yang paling mirip berdasarkan data kemiripan yang disediakan.
"""

def average_similarity_score(anime_title, similarity_data, k=10):
    # Ambil nilai similarity terhadap anime_title
    sim_scores = similarity_data[anime_title].drop(index=anime_title)
    # Ambil k skor tertinggi
    top_k_scores = sim_scores.sort_values(ascending=False).head(k)
    return top_k_scores.mean()

"""Mengukur seberapa mirip rata-rata dari 10 rekomendasi teratas terhadap anime referensi, yaitu "Naruto", berdasarkan skor cosine similarity."""

score = average_similarity_score("Naruto", cosine_sim_df, k=10)
print(f"Rata-rata skor kemiripan untuk rekomendasi 'Naruto' (Top-10): {score}")

"""Precision@k

Precision@k menunjukkan proporsi dari item rekomendasi yang benar-benar relevan (sesuai dengan minat pengguna) dari k item teratas yang direkomendasikan.
"""

def precision_at_k(recommended, relevant, k=10):
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    recommended_set = set(recommended_k)
    true_positives = recommended_set.intersection(relevant_set)
    return len(true_positives) / k

"""Mengukur kualitas rekomendasi sistem berbasis konten (Content-Based Filtering) dengan metrik Precision@k, menggunakan anime dengan genre yang sama sebagai ground truth (acuan kebenaran). Pada kode ini, ground truth (daftar anime yang dianggap relevan) diambil langsung dari data anime_cleaned berdasarkan genre yang sama dengan anime referensi, yaitu "Gintama"."""

# Ambil genre dari anime referensi (misalnya 'Naruto')
target_genre = anime_cleaned[anime_cleaned['name'] == 'Gintama']['genre'].values[0]

# Filter anime dengan genre yang sama dan bukan 'Naruto'
relevant_df = anime_cleaned[
    (anime_cleaned['genre'] == target_genre) &
    (anime_cleaned['name'] != 'Gintama')
]

# Ambil 10 judul sebagai ground truth
relevant = relevant_df['name'].head(10).tolist()

# Rekomendasi sistem
recommended_df = anime_recommendations('Gintama')
recommended = recommended_df['name'].tolist()

# Hitung precision@k
precision = precision_at_k(recommended, relevant, k=10)
print("Relevant (Ground Truth):", relevant)
print("Recommended:", recommended[:5])
print(f"Precision@5: {precision}")