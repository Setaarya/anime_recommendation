# Laporan Proyek Machine Learning - Yulianto Aryaseta

## Domain Proyek

Permasalahan dalam menemukan anime yang sesuai dengan preferensi individu masih menjadi tantangan signifikan di berbagai platform streaming dan komunitas penggemar anime. Banyaknya judul yang tersedia, genre yang beragam, serta perbedaan selera pengguna membuat proses pencarian anime yang cocok seringkali memakan waktu dan tidak efisien. Penelitian yang dilakukan oleh Jannach et al. (2020) menunjukkan bahwa rata-rata pengguna platform hiburan online menghabiskan 15-25% waktu mereka hanya untuk proses pemilihan konten, fenomena yang dikenal sebagai "choice overload" atau kelebihan pilihan [1]. Masalah ini bersifat kompleks karena dipengaruhi oleh preferensi personal yang unik dari setiap pengguna, seperti genre favorit, gaya animasi yang disukai, tema cerita, hingga rating dan popularitas. Oleh karena itu, dibutuhkan pendekatan berbasis data untuk membantu pengguna menemukan rekomendasi anime yang relevan dan personalisasi dengan lebih cepat dan akurat.

Dalam proyek ini, dibangun sebuah sistem rekomendasi anime menggunakan metode Content-Based Filtering. Pendekatan ini bekerja dengan cara menganalisis fitur-fitur dari anime yang pernah ditonton atau disukai pengguna, seperti genre, tipe penayangan, jumlah episode, jumlah member komunitas, dan skor rating, untuk merekomendasikan anime lain dengan karakteristik serupa. Model ini memanfaatkan vectorization dari data teks (judul dan tag genre) menggunakan teknik seperti TF-IDF, serta menghitung kemiripan antar konten menggunakan metrik seperti cosine similarity.

Tujuan utama dari sistem ini adalah untuk merekomendasikan anime yang paling sesuai dengan selera pengguna berdasarkan konten yang telah mereka nikmati sebelumnya. Dengan sistem ini, platform streaming atau aplikasi anime tracker dapat memberikan saran tayangan yang lebih akurat dan meningkatkan kepuasan pengguna dalam eksplorasi anime.

Beberapa penelitian terdahulu juga telah menunjukkan efektivitas sistem rekomendasi berbasis konten dalam domain hiburan. Penelitian oleh Lops et al. (2011) menunjukkan bahwa content-based filtering efektif dalam memberikan rekomendasi yang konsisten terhadap preferensi pengguna [2].

Dengan pendekatan ini, diharapkan sistem rekomendasi anime berbasis content-based filtering dapat menjadi solusi praktis bagi penggemar anime dalam menemukan tontonan yang sesuai, mengurangi waktu pencarian, dan meningkatkan pengalaman menikmati anime secara keseluruhan.

## Business Understanding

Dalam industri hiburan, khususnya dunia anime, memberikan pengalaman menonton yang personal dan memuaskan kepada pengguna merupakan tantangan utama bagi platform penyedia layanan streaming dan komunitas penggemar. Banyaknya pilihan judul anime yang tersedia justru dapat membuat pengguna kesulitan dalam menemukan tontonan yang sesuai dengan preferensi mereka, yang berujung pada kebingungan atau bahkan kehilangan minat untuk menonton.

Untuk mengatasi masalah tersebut, dibutuhkan sebuah sistem yang mampu memahami selera pengguna berdasarkan konten anime yang pernah mereka tonton atau sukai. Dengan menyediakan rekomendasi yang relevan dan personal, platform tidak hanya meningkatkan kepuasan pengguna, tetapi juga dapat memperpanjang durasi keterlibatan (engagement) dan loyalitas pengguna terhadap layanan. Sistem ini juga berperan penting dalam membantu pengguna mengeksplorasi lebih banyak judul anime yang mungkin tidak populer, namun memiliki kualitas konten yang sesuai dengan minat mereka.

### Problem Statements

1. Bagaimana mengurangi waktu yang dihabiskan pengguna dalam memilih anime yang sesuai dengan preferensi mereka?

   Banyak pengguna merasa kesulitan dan menghabiskan waktu lama untuk menemukan tontonan yang cocok karena banyaknya pilihan dan kurangnya rekomendasi yang personal.

2. Mengapa sistem rekomendasi yang ada seringkali tidak sesuai dengan selera pengguna?

   Sistem rekomendasi seringkali terlalu bergantung pada popularitas atau riwayat global tanpa memahami karakteristik konten seperti genre dan judul.

3. Bagaimana cara memanfaatkan fitur konten anime (judul dan genre) secara optimal dalam membangun sistem rekomendasi yang akurat?

   Tanpa teknik pemrosesan teks dan pengukuran kesamaan yang tepat, fitur-fitur ini belum dimanfaatkan secara maksimal.

### Goals

1. Membangun sistem rekomendasi anime berbasis Content-Based Filtering yang menggunakan judul dan genre sebagai fitur utama.
2. Memberikan rekomendasi anime yang relevan dan personal berdasarkan kesamaan konten.
3. Mengurangi waktu pencarian pengguna dalam memilih tontonan.
4. Membantu pengguna mengeksplorasi anime yang belum populer namun sesuai selera.
5. Meningkatkan keterlibatan dan kepuasan pengguna terhadap platform.

### Solution statements

1. Mengembangkan pipeline ekstraksi fitur dari kolom title dan genre menggunakan teknik TF-IDF vectorization.
2. Membangun vektor representasi teks dari setiap anime dan menghitung kemiripan antar-anime menggunakan cosine similarity.
3. Menyediakan sistem rekomendasi yang mampu memberikan daftar anime serupa berdasarkan judul dan genre input dari pengguna.
4. Menguji kualitas rekomendasi melalui uji coba dengan pengguna dan validasi subjektif berbasis kesesuaian preferensi.

## Data Understanding

Dataset yang digunakan dalam proyek ini berasal dari [Anime Recommendation Database di Kaggle](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database/data). Dataset ini memuat data preferensi pengguna terhadap berbagai judul anime, yang diambil dari situs populer MyAnimeList. Tujuan utama dari dataset ini adalah mendukung pembangunan sistem rekomendasi anime yang lebih personal, berdasarkan preferensi nyata pengguna terhadap ribuan judul anime yang tersedia. Dataset ini terdiri dari dua file utama: `anime.csv` dan `rating.csv`.

### Data

1. anime.csv

Berisi metadata dari 12.294 anime, mencakup berbagai informasi penting seperti:
- anime_id: ID unik dari setiap anime (mengacu pada ID dari myanimelist.net)
- name: Judul lengkap anime
- genre: Daftar genre yang dimiliki setiap anime, dipisahkan dengan koma (contoh: Action, Adventure, Comedy)
- type: Tipe anime (contoh: TV, Movie, OVA, dll.)
- episodes: Jumlah episode anime (1 jika berupa film)
- rating: Rata-rata rating dari komunitas pengguna (skala 1–10)
- members: Jumlah pengguna yang menambahkan anime tersebut ke daftar mereka

2. rating.csv

Berisi informasi penilaian (rating) yang diberikan oleh 73.516 pengguna terhadap berbagai judul anime:
- user_id: ID unik dari pengguna (anonim, diacak)
- anime_id: ID anime yang dinilai oleh pengguna
- rating: Skor yang diberikan pengguna terhadap anime (skala 1–10, atau -1 jika ditonton tetapi tidak diberi rating)

### Exploratory Data Analisis

Pertama kita melakukan analisis terhadap banyak kolom dan jenis data. Dataset yang digunakan terdiri dari 7 kolom dengan total 12,294 entri. Setelah itu, kita juga melakukan pengecekan nilai NaN dan duplikasi data. Hasilnya diperoleh bahwa ada nilai NaN di kolom genre, type, dan rating. Selain itu, juga ditemukan bahwa ada nilai 'Unknown' di kolom episodes dan ada 2 data duplikat di kolom name. Untuk lebih detailnya, berikut adalah tabel hasil analisis:

| No | Kolom     | Non-Null Count | Tipe Data | Nilai NaN | Nilai `'Unknown'` | Duplikat |
| -- | --------- | -------------- | --------- | --------- | ----------------- | -------- |
| 1  | anime\_id | 12,294         | int64     | 0         | -                 | -        |
| 2  | name      | 12,294         | object    | 0         | -                 | 2        |
| 3  | genre     | 12,232         | object    | 62        | -                 | -        |
| 4  | type      | 12,269         | object    | 25        | -                 | -        |
| 5  | episodes  | 12,294         | object    | 0         | 340               | -        |
| 6  | rating    | 12,064         | float64   | 230       | -                 | -        |
| 7  | members   | 12,294         | int64     | 0         | -                 | -        |

## Data Preparation

Dalam proses data preparation yang dilakukan, terdapat beberapa tahapan penting yang diterapkan untuk memastikan bahwa data yang digunakan bersih, relevan, dan siap untuk digunakan. Berikut merupakan tahapan-tahapan yang dilakukan dalam proses data preparation:

1. Menghapus Duplikasi pada Kolom name
   - Proses: Menghapus entri duplikat berdasarkan kolom name untuk memastikan setiap anime hanya muncul satu kali dalam dataset.
   - Alasan: Duplikasi dapat menyebabkan bias dalam analisis dan model prediktif, karena data yang sama akan dihitung lebih dari sekali dan mempengaruhi distribusi nilai.

2. Menghapus Nilai Kosong (NaN) pada Kolom genre
   - Proses: Baris dengan nilai NaN pada kolom genre dihapus dari dataset.
   - Alasan: Genre merupakan fitur penting yang dapat menggambarkan karakteristik konten dari anime. Nilai kosong pada kolom ini akan mengurangi informasi yang tersedia bagi model atau
     analisis eksploratif.

3. Menghapus Nilai Kosong (NaN) pada Kolom type
   - Proses: Baris dengan nilai NaN pada kolom type dihapus.
   - Alasan: Tipe anime (TV, Movie, OVA, dll.) adalah informasi kategorikal yang esensial dalam segmentasi data. Nilai kosong pada fitur ini akan mengganggu proses klasifikasi atau
     pengelompokan.

4. Menghapus Nilai "Unknown" pada Kolom episodes
   - Proses: Baris dengan nilai "Unknown" pada kolom episodes dihapus dari dataset.
   - Alasan: Kolom episodes berisi jumlah episode yang merupakan data numerik. Nilai "Unknown" tidak dapat diproses secara numerik dan dapat mengganggu proses analisis atau pelatihan
     model prediktif.

5. Menghapus Nilai Kosong (NaN) pada Kolom rating
   - Proses: Baris dengan nilai NaN pada kolom rating dihapus.
   - Alasan: Kolom rating mencerminkan penilaian komunitas terhadap anime dan menjadi indikator penting dalam model rekomendasi. Nilai kosong di kolom ini berarti tidak ada masukan dari
     pengguna dan sebaiknya dihilangkan agar tidak mengganggu analisis statistik.

## Modeling

Dalam pembuatan sistem content based filtering digunakan TF-IDF sebagai metode untuk merepresentasikan data teks (dalam hal ini genre) ke dalam bentuk vektor numerik dan Cosine Similarity sebagai metode untuk mengukur kemiripan antar vektor genre dari setiap anime.

### TF-IDF
TF-IDF (Term Frequency-Inverse Document Frequency) adalah metode statistik yang mengevaluasi pentingnya suatu kata dalam dokumen relatif terhadap kumpulan dokumen. Dalam konteks sistem rekomendasi anime, dokumen adalah genre anime dan kata-kata adalah komponen genre individu.

- Cara kerja TF-IDF:
  
   1). Term Frequency (TF): Menghitung frekuensi kemunculan sebuah kata dalam dokumen. Semakin sering kata muncul dalam dokumen, semakin tinggi nilai TF-nya.
   
      `TF(t, d) = (Jumlah kemunculan term t dalam dokumen d) / (Total term dalam dokumen d)`
      
   2). Inverse Document Frequency (IDF): Mengukur seberapa penting sebuah kata dengan mempertimbangkan kejarangannya di seluruh koleksi dokumen. Kata yang jarang muncul di banyak dokumen akan memiliki nilai IDF tinggi.
   
      `IDF(t) = log_e(Total dokumen / Jumlah dokumen yang mengandung term t)`
     
   3). TF-IDF: Hasil perkalian dari TF dan IDF, memberikan nilai yang tinggi pada kata yang sering muncul dalam dokumen tertentu tetapi jarang dalam koleksi dokumen.
   
      `TF-IDF(t, d) = TF(t, d) × IDF(t)`
      
- Kelebihan TF-IDF:
   - Mampu menangkap informasi yang penting dan spesifik dalam sebuah dokumen
   - Mengurangi pengaruh kata-kata umum yang tidak memberi nilai informasi tinggi
   - Relatif sederhana dan efisien dalam komputasi
   - Mudah diimplementasikan dan intuitif untuk dipahami

- Kekurangan TF-IDF:
   - Tidak memperhitungkan semantik atau konteks kata
   - Tidak mempertimbangkan urutan kata dalam dokumen
   - Rentan terhadap masalah dimensionalitas tinggi pada dataset besar
   - Tidak dapat menangkap hubungan antar kata (misalnya sinonim)

### Cosine Similarity
Cosine Similarity adalah metode pengukuran kemiripan berbasis sudut antara dua vektor. Cosine Similarity mengukur kemiripan dua vektor dengan menghitung kosinus sudut di antara keduanya. Nilai cosine similarity berkisar antara -1 hingga 1, di mana 1 berarti vektor identik, 0 berarti tidak berkorelasi, dan -1 berarti berlawanan arah.

- Cara kerja Cosine Similarity:

   1). Representasi Vektor: Setiap anime direpresentasikan sebagai vektor multi-dimensi (hasil dari TF-IDF)
  
   2). Perhitungan Dot Product: Menghitung dot product dari dua vektor

   3). Normalisasi: Membagi dot product dengan hasil kali panjang (magnitude) kedua vektor
      
      Rumus Cosine Similarity:
      
      `Cosine Similarity(A, B) = (A · B) / (||A|| × ||B||)`
      
      Di mana:
      - A · B adalah dot product dari vektor A dan B
      - ||A|| adalah panjang (magnitude) vektor A
      - ||B|| adalah panjang (magnitude) vektor B

- Kelebihan Cosine Similarity:
   - Tidak sensitif terhadap ukuran dokumen, hanya mempertimbangkan orientasi vektor
   - Baik untuk data sparse (seperti matriks TF-IDF)
   - Efisien untuk perhitungan kemiripan dalam ruang dimensi tinggi
   - Mudah diinterpretasikan: nilai mendekati 1 berarti sangat mirip

- Kekurangan Cosine Similarity:
   - Tidak mempertimbangkan perbedaan besaran nilai jika orientasi vektor sama
   - Kurang efektif untuk data dengan distribusi yang tidak merata
   - Bisa menyebabkan bias terhadap data yang memiliki nilai nol pada banyak dimensi
   - Tidak memperhitungkan korelasi non-linear antara fitur

### Langkah-Langkah Implementasi:

1. Preprocessing Genre
   
   Menggunakan TfidfVectorizer untuk mengubah kolom genre menjadi representasi numerik berbasis TF-IDF, yang menekankan kata-kata unik dalam genre setiap anime.
   - Setiap anime memiliki daftar genre (misalnya "Action, Adventure, Comedy")
   - TfidfVectorizer memecah genre menjadi komponen individual (misalnya "Action", "Adventure", "Comedy")
   - Setiap komponen genre diberi bobot berdasarkan frekuensi kemunculannya dalam anime tersebut (TF) dan keunikannya di seluruh dataset (IDF)
   - Hasil akhir adalah matriks di mana setiap baris mewakili anime dan setiap kolom mewakili komponen genre dengan nilai TF-IDF

2. Menghitung Kemiripan
   
   Menggunakan cosine_similarity untuk mengukur kedekatan antar anime berdasarkan genre-nya.
   - Cosine similarity menghitung tingkat kemiripan antar semua pasangan anime
   - Nilai mendekati 1 menunjukkan genre yang sangat mirip
   - Nilai mendekati 0 menunjukkan genre yang sangat berbeda
   - Hasil akhir berupa matriks kemiripan dengan ukuran n×n, di mana n adalah jumlah anime

3. Fungsi Rekomendasi dengan Input Judul Anime
   
   Fungsi anime_recommendations(anime_title) akan menghasilkan daftar anime yang paling mirip dengan judul input.
   - Mencari anime dengan nilai cosine similarity tertinggi terhadap anime input
   - Menggunakan algoritma partisi untuk mendapatkan k anime paling mirip secara efisien
   - Menghapus anime input dari hasil rekomendasi
   - Menggabungkan hasil dengan data asli untuk mendapatkan informasi genre

4. Fungsi Rekomendasi dengan Input Genre Anime
   
   Fungsi get_recommendations_by_genre(genre) akan mencari anime dengan genre tersebut dan merekomendasikan anime yang mirip dengan salah satu judulnya.
   - Memfilter dataset untuk mencari anime yang mengandung genre yang diminta
   - Memilih satu anime dari hasil filter sebagai titik acuan
   - Menggunakan fungsi anime_recommendations untuk mendapatkan anime yang mirip
   - Memformat hasil dalam bentuk tabel yang mudah dibaca

### Top 10 Recommendations
1. Top 10 Rekomendasi Anime Mirip dengan Gintama

| No | Nama Anime | Genre |
|----|-----------|-------|
| 1 | Gintama: Jump Festa 2014 Special | Action, Comedy, Historical, Parody, Samurai, Shounen |
| 2 | Gintama° | Action, Comedy, Historical, Parody, Samurai, Shounen |
| 3 | Gintama' | Action, Comedy, Historical, Parody, Samurai, Shounen |
| 4 | Gintama Movie: Shinyaku Benizakura-hen | Action, Comedy, Historical, Parody, Samurai, Shounen |
| 5 | Gintama: Shinyaku Benizakura-hen | Action, Comedy, Historical, Parody, Samurai, Shounen |
| 6 | Gintama: Yorinuki Gintama-san on Theater 2D | Action, Comedy, Historical, Parody, Samurai, Shounen |
| 7 | Gintama Movie: Kanketsu-hen - Yorozuya yo Eien... | Action, Comedy, Historical, Parody, Samurai, Shounen |
| 8 | Gintama': Enchousen | Action, Comedy, Historical, Parody, Samurai, Shounen |
| 9 | Gintama: Nanigoto mo Saiyo ga Kanjin nano de T... | Action, Comedy, Historical, Mecha, Parody, Samurai, Shounen |
| 10 | Gintama: Jump Festa 2015 Special | Action, Comedy, Historical, Parody, Samurai, Shounen |

2. Top 10 Rekomendasi Anime dengan Genre Action

| No | Nama Anime | Genre |
|----|-----------|-------|
| 1 | Fullmetal Alchemist | Action, Adventure, Comedy, Drama, Fantasy, Magic, Military, Shounen |
| 2 | Fullmetal Alchemist: The Sacred Star of Milos | Action, Adventure, Comedy, Drama, Fantasy, Magic, Military, Shounen |
| 3 | Fullmetal Alchemist: Brotherhood Specials | Adventure, Drama, Fantasy, Magic, Military, Shounen |
| 4 | Tales of Vesperia: The First Strike | Action, Adventure, Fantasy, Magic, Military |
| 5 | Tide-Line Blue | Action, Adventure, Drama, Military, Shounen |
| 6 | Fullmetal Alchemist: Reflections | Adventure, Comedy, Drama, Fantasy, Military, Shounen |
| 7 | Meoteoldosawa Ttomae | Action, Adventure, Fantasy, Magic, Shounen |
| 8 | Log Horizon Recap | Action, Adventure, Fantasy, Magic, Shounen |
| 9 | Dragon Quest: Dai no Daibouken Tachiagare!! Aban no Shito | Action, Adventure, Fantasy, Magic, Shounen |
| 10 | Magi: Sinbad no Bouken (TV) | Action, Adventure, Fantasy, Magic, Shounen |

## Evaluation

Pada proyek ini digunakan metrik evaluasi Cosine Similarity Score Average (Internal Metric) dan Precision@k.

### Cosine Similarity Score Average (Internal Metric)

Cosine Similarity Score Average adalah metrik internal yang menghitung rata-rata nilai kemiripan (similarity) antara item referensi dengan top-k item yang direkomendasikan oleh sistem. Metrik ini mengukur seberapa mirip konten yang direkomendasikan dengan konten referensi berdasarkan representasi vektor fitur-fiturnya. Semakin tinggi nilai rata-rata similarity, semakin baik sistem dalam merekomendasikan item yang memiliki karakteristik serupa dengan item referensi.

Secara matematis, Cosine Similarity Score Average dapat diformulasikan sebagai berikut:

![Cosine Similarity Score Average](https://github.com/user-attachments/assets/2f62ce5b-8097-4b4a-9bdb-3338d31a28f5)

Keterangan:

- CSSA: Cosine Similarity Score Average
- k: Jumlah item rekomendasi teratas yang dievaluasi
- 𝑣𝑟⃗: Vektor representasi item referensi (misalnya, vektor TF-IDF dari anime referensi)
- 𝑣𝑖⃗: Vektor representasi item rekomendasi ke-i
- cos(𝑣𝑟⃗,𝑣𝑖⃗): Nilai cosine similarity antara 𝑣𝑟⃗ dan 𝑣𝑖⃗
​
### Precision@k

Precision@k adalah metrik evaluasi berbasis relevansi yang mengukur proporsi item yang relevan dari k item teratas yang direkomendasikan. Metrik ini mengevaluasi ketepatan sistem dalam merekomendasikan item yang benar-benar relevan dengan preferensi pengguna. Dalam konteks sistem rekomendasi anime, item yang relevan didefinisikan sebagai anime yang memiliki genre sama dengan anime referensi.

Rumus Matematis Precision@k

![Precision@k](https://github.com/user-attachments/assets/6890b785-5265-40b9-a256-25189c4a0629)
 
Keterangan:

- Precision@k: Ukuran ketepatan sistem rekomendasi dalam menampilkan item relevan pada daftar teratas sebanyak 𝑘 item
- ∣{item relevan}∩{item rekomendasi top-𝑘}∣: Jumlah item yang relevan dan juga direkomendasikan dalam top-
- 𝑘: Jumlah rekomendasi teratas yang dievaluasi

### Hasil evaluasi

Dari kedua matriks evaluasi diatas didapatkan hasil sebagai berikut:

Cosine Similarity Score Average

`Rata-rata skor kemiripan untuk rekomendasi 'Naruto' (Top-10): 0.9875639023932294`

Precision@k

`Relevant (Ground Truth): ['Gintama°', 'Gintama&#039;', 'Gintama Movie: Kanketsu-hen - Yorozuya yo Eien Nare', 'Gintama&#039;: Enchousen', 'Gintama: Yorinuki Gintama-san on Theater 2D', 'Gintama Movie: Shinyaku Benizakura-hen', 'Gintama: Shinyaku Benizakura-hen', 'Gintama: Jump Festa 2014 Special']`

`Recommended: ['Gintama: Jump Festa 2014 Special', 'Gintama°', 'Gintama&#039;', 'Gintama Movie: Shinyaku Benizakura-hen', 'Gintama: Shinyaku Benizakura-hen']`

`Precision@5: 0.8`

Nilai Cosine Similarity Score Average (CSSA) sebesar 0.9876 menunjukkan bahwa sistem rekomendasi yang dikembangkan mampu memberikan daftar tontonan dengan konten yang sangat mirip dengan anime referensi dari segi genre dan judul. Selain itu, nilai Precision@5 sebesar 0.8 mengindikasikan bahwa 80% dari rekomendasi yang dihasilkan berhasil sesuai dengan daftar anime relevan yang berbagi genre maupun franchise yang sama. Hasil ini menegaskan bahwa pendekatan content-based filtering yang diterapkan cukup efektif dalam memahami kesamaan antar-anime secara semantik.

Temuan tersebut memiliki keterkaitan yang erat dengan problem statements yang telah diidentifikasi sebelumnya. Permasalahan umum seperti lamanya waktu yang dihabiskan pengguna dalam memilih anime dapat diatasi melalui sistem rekomendasi ini, karena sistem secara otomatis mengidentifikasi kemiripan konten dan menyajikan daftar tontonan yang relevan, sehingga pengguna tidak perlu mencari secara manual. Selain itu, kelemahan sistem rekomendasi konvensional yang cenderung bergantung pada popularitas atau riwayat tontonan juga berhasil diatasi, karena pendekatan ini lebih berfokus pada pemahaman konten melalui fitur judul dan genre, bukan hanya sekadar popularitas.

Dengan memanfaatkan teknik TF-IDF vectorization untuk mengekstraksi fitur dari teks dan cosine similarity untuk menghitung tingkat kemiripan antar-anime, sistem ini mampu menyajikan rekomendasi yang lebih personal dan sesuai dengan preferensi pengguna. Pencapaian ini secara langsung mendukung tujuan utama proyek, yaitu memberikan rekomendasi yang relevan dan personal, mengurangi waktu pencarian, memungkinkan eksplorasi anime yang belum populer, serta meningkatkan keterlibatan dan kepuasan pengguna terhadap platform rekomendasi.

## Daftar Pustaka

[1] Jannach, Dietmar & Resnick, Paul & Tuzhilin, Alexander & Zanker, Markus. (2016). Recommender systems---: beyond matrix completion. Communications of the ACM. 59. 94-102. 10.1145/2891406. https://www.researchgate.net/publication/309600906_Recommender_systems---_beyond_matrix_completion

[2] Lops, Pasquale & de Gemmis, Marco & Semeraro, Giovanni. (2011). Content-based Recommender Systems: State of the Art and Trends. 10.1007/978-0-387-85820-3_3. https://www.researchgate.net/publication/226098747_Content-based_Recommender_Systems_State_of_the_Art_and_Trends
