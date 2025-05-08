# Laporan Proyek Machine Learning - Yulianto Aryaseta

## Domain Proyek

Permasalahan dalam menemukan anime yang sesuai dengan preferensi individu masih menjadi tantangan di berbagai platform streaming dan komunitas penggemar anime. Banyaknya judul yang tersedia, genre yang beragam, serta perbedaan selera pengguna membuat proses pencarian anime yang cocok seringkali memakan waktu dan tidak efisien. Berdasarkan laporan MyAnimeList (2021), rata-rata pengguna menghabiskan lebih dari 30 menit hanya untuk memilih anime yang ingin ditonton, terutama di musim rilis baru yang penuh pilihan [1].

Masalah ini bersifat kompleks karena dipengaruhi oleh preferensi personal yang unik dari setiap pengguna, seperti genre favorit, gaya animasi yang disukai, tema cerita, hingga rating dan popularitas. Oleh karena itu, dibutuhkan pendekatan berbasis data untuk membantu pengguna menemukan rekomendasi anime yang relevan dan personalisasi dengan lebih cepat dan akurat.

Dalam proyek ini, dibangun sebuah sistem rekomendasi anime menggunakan metode Content-Based Filtering. Pendekatan ini bekerja dengan cara menganalisis fitur-fitur dari anime yang pernah ditonton atau disukai pengguna, seperti genre, tipe penayangan, jumlah episode, jumlah member komunitas, dan skor rating, untuk merekomendasikan anime lain dengan karakteristik serupa. Model ini memanfaatkan vectorization dari data teks (judul dan tag genre) menggunakan teknik seperti TF-IDF, serta menghitung kemiripan antar konten menggunakan metrik seperti cosine similarity.

Tujuan utama dari sistem ini adalah untuk merekomendasikan anime yang paling sesuai dengan selera pengguna berdasarkan konten yang telah mereka nikmati sebelumnya. Dengan sistem ini, platform streaming atau aplikasi anime tracker dapat memberikan saran tayangan yang lebih akurat dan meningkatkan kepuasan pengguna dalam eksplorasi anime.

Beberapa penelitian terdahulu juga telah menunjukkan efektivitas sistem rekomendasi berbasis konten dalam domain hiburan. Penelitian oleh Lops et al. (2019) menunjukkan bahwa content-based filtering efektif dalam memberikan rekomendasi yang konsisten terhadap preferensi pengguna [2]. Studi lain oleh Iqbal et al. (2021) menyatakan bahwa sinopsis dan tag genre menjadi fitur paling relevan dalam sistem rekomendasi anime yang personalisasi [3].

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



## Data Preparation

Dalam proses data preparation yang dilakukan, terdapat beberapa tahapan penting yang diterapkan untuk memastikan bahwa data yang digunakan bersih, relevan, dan siap untuk digunakan. Berikut merupakan tahapan-tahapan yang dilakukan dalam proses data preparation:

1. Menghapus Duplikasi pada Kolom name
   - Proses: Menghapus entri duplikat berdasarkan kolom name untuk memastikan setiap anime hanya muncul satu kali dalam dataset.
   - Alasan: Duplikasi dapat menyebabkan bias dalam analisis dan model prediktif, karena data yang sama akan dihitung lebih dari sekali dan mempengaruhi distribusi nilai.

2. Menghapus Nilai Kosong (NaN) pada Kolom genre
   - Proses: Baris dengan nilai NaN pada kolom genre dihapus dari dataset.
Alasan: Genre merupakan fitur penting yang dapat menggambarkan karakteristik konten dari anime. Nilai kosong pada kolom ini akan mengurangi informasi yang tersedia bagi model atau analisis eksploratif.

Menghapus Nilai Kosong (NaN) pada Kolom type
Proses: Baris dengan nilai NaN pada kolom type dihapus.
Alasan: Tipe anime (TV, Movie, OVA, dll.) adalah informasi kategorikal yang esensial dalam segmentasi data. Nilai kosong pada fitur ini akan mengganggu proses klasifikasi atau pengelompokan.

Menghapus Nilai "Unknown" pada Kolom episodes
Proses: Baris dengan nilai "Unknown" pada kolom episodes dihapus dari dataset.
Alasan: Kolom episodes berisi jumlah episode yang merupakan data numerik. Nilai "Unknown" tidak dapat diproses secara numerik dan dapat mengganggu proses analisis atau pelatihan model prediktif.

Menghapus Nilai Kosong (NaN) pada Kolom rating
Proses: Baris dengan nilai NaN pada kolom rating dihapus.
Alasan: Kolom rating mencerminkan penilaian komunitas terhadap anime dan menjadi indikator penting dalam model rekomendasi. Nilai kosong di kolom ini berarti tidak ada masukan dari pengguna dan sebaiknya dihilangkan agar tidak mengganggu analisis statistik.

## Modeling

Pada tahap pemodelan ini, kami menggunakan XGBoost (Extreme Gradient Boosting) dengan algoritma XGBClassifier untuk menyelesaikan permasalahan klasifikasi pada dataset. Berikut adalah penjelasan terkait tahapan, parameter yang digunakan, serta proses evaluasi model:

XGBClassifier adalah sebuah algoritma machine learning yang menggunakan model XGBoost (Extreme Gradient Boosting) untuk masalah klasifikasi. XGBoost adalah metode yang berbasis pada teknik Gradient Boosting, yang merupakan algoritma ensemble learning yang menggabungkan beberapa model prediksi (decision trees) untuk menghasilkan prediksi yang lebih kuat.

Cara Kerja:

Gradient Boosting bekerja dengan membangun pohon keputusan (decision trees) secara bertahap. Setiap pohon yang baru dibangun akan berfokus untuk memperbaiki kesalahan yang dilakukan oleh pohon-pohon sebelumnya.

XGBClassifier menggunakan pendekatan boosting di mana setiap pohon yang dibuat mencoba untuk mengurangi kesalahan dari pohon sebelumnya dengan menghitung gradien dari kesalahan tersebut.

Model ini juga menggunakan berbagai teknik optimasi untuk meningkatkan kecepatan pelatihan dan mengurangi overfitting, seperti regularization, column subsampling, dan row subsampling.

Parameter dan Nilai Parameter
XGBClassifier dalam proyek ini masih menggunakan parameter default. Berikut adalah beberapa parameter default dalam XGBClassifier:

- learning_rate: Default = 0.3
  Menentukan ukuran langkah yang digunakan untuk memperbarui bobot di setiap iterasi. Nilai yang lebih rendah bisa meningkatkan ketelitian model tetapi membutuhkan lebih banyak pohon (iteration) untuk mencapai konvergensi.

- n_estimators: Default = 100
  Merupakan jumlah maksimum pohon keputusan yang akan dibuat oleh model. Setiap pohon memperbaiki kesalahan yang dilakukan oleh pohon sebelumnya.

- max_depth: Default = 6
  Menentukan kedalaman maksimal pohon keputusan. Semakin dalam pohon, semakin kompleks modelnya. Nilai yang lebih tinggi dapat menyebabkan overfitting jika tidak diatur dengan hati-hati.

- subsample: Default = 1
  Menentukan proporsi data yang digunakan untuk membangun setiap pohon keputusan. Pengaturan nilai lebih rendah dapat mencegah overfitting dengan melakukan subsampling pada data pelatihan.

- colsample_bytree: Default = 1
  Menentukan proporsi fitur yang digunakan untuk setiap pohon keputusan. Dengan menurunkan nilai ini, kita mengurangi kompleksitas model dan membantu generalisasi.

- objective: Default = 'binary:logistic'
  Menentukan jenis masalah yang ingin diselesaikan. Dalam hal ini, digunakan untuk masalah klasifikasi biner, di mana model akan mengoutputkan probabilitas untuk dua kelas.

- booster: Default = 'gbtree'
  Menentukan jenis model boosting yang digunakan. 'gbtree' mengindikasikan penggunaan pohon keputusan sebagai estimator dasar, yang umum digunakan untuk klasifikasi dan regresi.

### Kelebihan dan Kekurangan XGBoost

Kelebihan dan Kekurangan XGBClassifier

Kelebihan:
- Akurasi tinggi: XGBoost dikenal memiliki akurasi yang sangat baik, bahkan pada dataset besar dengan banyak fitur.
- Kemampuan menangani missing values dan outliers: Model ini dapat menangani nilai yang hilang dan outliers dengan baik tanpa memerlukan preprocessing yang berlebihan.
- Kecepatan dan Efisiensi: XGBoost sangat cepat dalam pelatihan dan prediksi karena optimisasi yang dilakukan pada level pohon keputusan.

Kekurangan:
- Sensitif terhadap parameter: Meskipun secara default sangat kuat, model ini sangat bergantung pada pengaturan parameter untuk mendapatkan performa terbaik.
- Kesulitan dengan data yang sangat tidak seimbang: Meskipun SMOTE digunakan untuk penyeimbangan kelas, XGBoost mungkin masih mengalami kesulitan dalam menangani ketidakseimbangan kelas yang ekstrem tanpa penyesuaian lebih lanjut.

## Evaluation

Pada proyek ini, kita menggunakan beberapa metrik evaluasi untuk menilai kinerja model klasifikasi yang diterapkan pada data siswa. Metrik yang digunakan meliputi akurasi, precision, recall, dan F1 score. Masing-masing metrik ini memiliki tujuan yang berbeda, yang membantu kita memahami kinerja model secara lebih menyeluruh.

- Akurasi (Accuracy):

  Akurasi mengukur seberapa sering model melakukan prediksi yang benar. Metrik ini berguna untuk memberikan gambaran umum tentang kinerja model pada dataset yang seimbang. Namun, akurasi bisa menjadi metrik yang menyesatkan jika data tidak seimbang (misalnya, jika satu kelas lebih dominan daripada yang lain).

- Precision (Presisi)

  Precision mengukur seberapa tepat model dalam memprediksi kelas positif. Metrik ini penting jika kita ingin meminimalkan jumlah kesalahan tipe I (false positives). Dalam konteks ini, precision mengukur seberapa banyak prediksi untuk setiap kelas benar.

- Recall (Sensitivitas)

  Recall mengukur kemampuan model untuk menangkap seluruh kelas positif yang ada di dalam data. Recall sangat penting jika kita ingin meminimalkan kesalahan tipe II (false negatives), yang berarti kita ingin memastikan sebanyak mungkin kelas positif terdeteksi.

- F1 Score

  F1 score adalah rata-rata harmonis dari precision dan recall. Metrik ini berguna ketika kita ingin keseimbangan antara precision dan recall, terutama dalam kasus di mana keduanya sangat penting. F1 score memberikan gambaran yang lebih baik daripada akurasi dalam konteks data yang tidak seimbang.

### Analisis Pemilihan Model Terbaik:

![barplot](https://github.com/user-attachments/assets/c6d1e12a-92de-4340-8128-272097436883)

Dari hasil evaluasi, model XGBoost dipilih sebagai model terbaik untuk solusi ini dengan alasan sebagai berikut:

- Akurasi Tertinggi:

  XGBoost menghasilkan akurasi tertinggi yaitu 78.10%, mengungguli model lain seperti Random Forest (76.98%), Logistic Regression (75.17%), dan KNN (65.01%).

- Kinerja yang Konsisten di Semua Kelas:

   Pada kelas 1 (kelas minoritas), XGBoost menunjukkan hasil yang lebih baik dibanding model lain dengan recall 0.51 dan precision 0.50. Meskipun belum ideal, ini tetap menjadi yang terbaik di antara seluruh model yang diuji.

- Untuk kelas 0 dan kelas 2, XGBoost menunjukkan kinerja yang sangat baik:

   Kelas 0: precision 0.84, recall 0.75

   Kelas 2: precision 0.85, recall 0.90

- Rata-Rata Makro dan Tertimbang yang Seimbang:

  XGBoost memiliki rata-rata macro precision (0.73), recall (0.72), dan F1-score (0.72) yang seimbang, menunjukkan kemampuan model dalam menjaga performa di seluruh kelas, termasuk minoritas.

Kelemahan Model Lain:

- Random Forest hanya sedikit di bawah XGBoost, namun performa pada kelas 1 masih kurang optimal (precision 0.47, recall 0.50).

- Logistic Regression mengalami penurunan akurasi dan performa pada kelas 0 dan 1 lebih buruk, meskipun kinerja pada kelas 2 cukup baik.

- KNN menunjukkan performa terendah di semua metrik, dengan akurasi 65.01% dan macro F1-score 0.62, menjadikannya model yang tidak direkomendasikan.

### Conclusion

Proyek ini secara langsung menjawab masalah yang dihadapi oleh institusi pendidikan dalam mempertahankan tingkat kelulusan mahasiswa dan meminimalkan risiko dropout. Dengan menggunakan model klasifikasi berbasis XGBoost, kita dapat mengidentifikasi mahasiswa yang berisiko tinggi mengalami dropout berdasarkan data historis yang tersedia. 

*Identifikasi Mahasiswa yang Berisiko Dropout:*

Model XGBoost yang telah dievaluasi berhasil memberikan prediksi yang akurat dengan akurasi tertinggi sebesar 78.10%, diikuti oleh performa yang cukup baik pada kelas-kelas minoritas (seperti kelas 1 yang berisiko tinggi). Hal ini sesuai dengan kebutuhan untuk mengidentifikasi mahasiswa yang berisiko tinggi gagal, berdasarkan data akademik dan administrasi mereka.

*Meningkatkan Kualitas Layanan Akademik:*

Dengan menggunakan hasil dari model XGBoost, institusi dapat mengintegrasikan sistem prediksi ini ke dalam sistem pengelolaan akademik mereka. Misalnya, pihak universitas bisa melakukan intervensi dini kepada mahasiswa yang diprediksi berisiko, seperti memberikan bimbingan atau konseling untuk membantu mahasiswa agar tetap pada jalur kelulusan.

*Identifikasi Fitur-fitur Penting:*

Dari model XGBoost, analisis fitur penting menunjukkan bahwa faktor-faktor seperti penyelesaian mata kuliah semester genap, pembayaran biaya kuliah tepat waktu, dan beban mata kuliah pada semester pertama dan kedua sangat berpengaruh terhadap keberhasilan akademik mahasiswa. Hal ini memberikan wawasan yang berguna bagi universitas untuk merancang kebijakan yang lebih efektif, seperti memberikan beasiswa, mengelola beban studi, atau menyediakan pendampingan keuangan.

*Membangun Model Prediksi Dropout Mahasiswa:*

Model XGBoost telah berhasil dibangun dan dievaluasi dengan hasil yang memuaskan, mencapai akurasi sebesar 78.10% dan performa yang baik pada beberapa metrik lainnya. Ini menunjukkan bahwa model tersebut dapat diandalkan untuk prediksi dropout mahasiswa berdasarkan data yang ada.

*Menyediakan Sistem Klasifikasi untuk Intervensi Dini:*

Output dari model ini memberikan informasi yang dapat digunakan oleh dosen pembimbing, bagian kemahasiswaan, atau pusat layanan akademik untuk melakukan intervensi dini. Dengan mengetahui mahasiswa yang berisiko, mereka dapat merencanakan langkah-langkah yang diperlukan untuk meningkatkan peluang kelulusan mahasiswa tersebut.

*Mengidentifikasi Fitur yang Mempengaruhi Keberhasilan Studi Mahasiswa:*

Dengan menggunakan analisis fitur penting, hasil dari model XGBoost menunjukkan faktor-faktor yang paling berpengaruh terhadap keberhasilan akademik mahasiswa, seperti penyelesaian mata kuliah dan pembayaran biaya kuliah tepat waktu. Ini memberi petunjuk bagi universitas untuk merancang program-program pendukung yang lebih tepat sasaran.

*Menggunakan Algoritma XGBoost sebagai Model Prediktif:*

Penggunaan XGBoost terbukti efektif, dengan akurasi dan fitur interpretabilitas yang sangat membantu dalam memahami faktor-faktor yang berpengaruh terhadap keberhasilan akademik mahasiswa. Model ini memberikan solusi yang sangat berdampak karena dapat dipakai dalam aplikasi dunia nyata untuk mendeteksi mahasiswa yang berisiko tinggi gagal dan memerlukan intervensi.

*Mengevaluasi Model dengan Algoritma Alternatif:*

Evaluasi terhadap algoritma alternatif (Random Forest dan Logistic Regression) memberikan wawasan bahwa meskipun model lain memberikan hasil yang cukup baik, XGBoost tetap menjadi pilihan terbaik dengan akurasi tertinggi dan kemampuan untuk menjaga keseimbangan kinerja antar kelas. Pemilihan model ini berkontribusi pada pemilihan solusi terbaik yang efektif untuk aplikasi pendidikan tinggi.

Dengan menggunakan model XGBoost untuk memprediksi risiko dropout mahasiswa, proyek ini memberikan dampak yang signifikan terhadap pemahaman institusi mengenai faktor-faktor yang memengaruhi keberhasilan akademik. Model ini tidak hanya memberikan prediksi yang akurat, tetapi juga memungkinkan pengambilan keputusan yang lebih berbasis data untuk membantu mahasiswa yang berisiko tinggi. Implementasi sistem prediksi ini dapat meningkatkan kualitas layanan akademik dan membantu meminimalkan angka dropout di masa depan, yang sejalan dengan tujuan strategis institusi pendidikan.

## Daftar Pustaka

[1] World Bank. (2021). Learning Poverty in the Time of COVID-19: A crisis within a crisis. https://www.worldbank.org/en/topic/education/publication/learning-poverty-in-the-time-of-covid-19

[2] Castro, M., Oliveira, M., & Silva, A. (2020). Early prediction of student dropout and academic failure using machine learning: A case study with Portuguese higher education data. Education and Information Technologies, 25, 4745–4763. https://doi.org/10.1007/s10639-020-10183-w

[3] Umer, S. R., Sherin, S., & Ahmad, M. (2022). A predictive model for student dropout using supervised machine learning techniques. Computers & Education: Artificial Intelligence, 3, 100076. https://doi.org/10.1016/j.caeai.2022.100076
