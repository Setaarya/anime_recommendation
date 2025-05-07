# Laporan Proyek Machine Learning - Yulianto Aryaseta

## Domain Proyek

Permasalahan dropout mahasiswa masih menjadi tantangan serius di berbagai institusi pendidikan tinggi. Tingkat kelulusan yang rendah tidak hanya mencerminkan efektivitas sistem pembelajaran, tetapi juga berdampak pada reputasi institusi dan beban biaya yang ditanggung mahasiswa serta keluarganya. Menurut laporan World Bank (2021), satu dari lima mahasiswa di negara berkembang berisiko tidak menyelesaikan studi tepat waktu, atau bahkan putus kuliah sebelum meraih gelar [1].

Masalah ini sangat kompleks karena dipengaruhi oleh kombinasi faktor akademik dan non-akademik, termasuk latar belakang keluarga, kondisi ekonomi, kinerja akademik, serta faktor psikososial. Oleh karena itu, pendekatan prediktif berbasis data sangat dibutuhkan untuk membantu institusi pendidikan dalam melakukan deteksi dini terhadap mahasiswa yang berpotensi mengalami kegagalan studi.

Dalam proyek ini, dibangun sebuah sistem klasifikasi berbasis machine learning yang memanfaatkan algoritma XGBoost Classifier. XGBoost dikenal sebagai salah satu algoritma klasifikasi berbasis ensemble yang sangat efisien dan memiliki akurasi tinggi dalam banyak kasus prediktif. Model ini dilatih menggunakan dataset yang berisi 37 fitur, di antaranya: status pernikahan, usia saat masuk kuliah, gender, nilai ujian, jumlah mata kuliah yang diambil dan disetujui, hingga indikator ekonomi seperti inflasi dan PDB. Label target dari data ini diklasifikasikan menjadi dua kelas utama: dropout dan graduate success.

Tujuan utama dari sistem ini adalah untuk mengklasifikasikan mahasiswa berdasarkan kemungkinan mereka mengalami dropout atau berhasil menyelesaikan studinya. Dengan sistem ini, pihak kampus dapat mengidentifikasi kelompok risiko lebih awal dan menyusun strategi intervensi yang lebih efektif, seperti pendampingan akademik, beasiswa tambahan, atau konseling psikologis.

Beberapa penelitian terdahulu juga telah membuktikan efektivitas model klasifikasi dalam kasus serupa. Penelitian oleh Castro et al. (2020) menggunakan XGBoost untuk memprediksi dropout dan berhasil mencapai akurasi lebih dari 90% [2]. Studi lain oleh Umer et al. (2022) menekankan bahwa faktor akademik semester awal, seperti jumlah evaluasi dan nilai mata kuliah, sangat berkontribusi dalam model prediktif [3].

Dengan pendekatan ini, diharapkan sistem prediksi berbasis machine learning dapat menjadi alat bantu pengambilan keputusan yang berdampak nyata dalam meningkatkan kualitas pendidikan tinggi dan mengurangi angka kegagalan studi khususnya di Indonesia.

## Business Understanding

Dalam dunia pendidikan tinggi, menjaga tingkat retensi mahasiswa merupakan tantangan utama yang harus dihadapi oleh institusi akademik. Tingginya angka mahasiswa yang mengalami dropout atau tidak menyelesaikan studi tepat waktu dapat berdampak buruk terhadap reputasi universitas, efisiensi pengelolaan sumber daya, dan masa depan akademik mahasiswa itu sendiri. Untuk itu, dibutuhkan sebuah sistem yang mampu mengidentifikasi mahasiswa yang berisiko tinggi agar dapat diberikan intervensi sedini mungkin.

### Problem Statements

1. Bagaimana cara mengidentifikasi mahasiswa yang berpotensi mengalami dropout berdasarkan data historis yang tersedia?
   - Banyak institusi kesulitan dalam melakukan prediksi dropout karena kurangnya alat prediktif yang mampu menangkap berbagai faktor kompleks seperti prestasi akademik, kondisi sosial ekonomi, dan faktor pribadi lainnya.

2. Bagaimana meningkatkan kualitas layanan akademik dengan pemanfaatan data secara efektif?
   - Informasi yang tersebar di berbagai sistem informasi akademik sering kali belum digunakan secara optimal untuk mendukung pengambilan keputusan yang berbasis data.

3. Apa saja fitur yang paling berpengaruh terhadap keberhasilan studi mahasiswa?
   - Memahami fitur penting akan membantu universitas dalam merancang program intervensi yang lebih tepat sasaran, seperti pemberian beasiswa, konseling, atau penyesuaian kurikulum.

### Goals

1. Membangun model prediksi dropout mahasiswa menggunakan algoritma machine learning berbasis XGBoost.
   - Model ini akan dilatih menggunakan data yang mencakup aspek akademik dan non-akademik mahasiswa dengan target prediksi berupa status kelulusan.

2. Memberikan sistem klasifikasi yang dapat diintegrasikan ke dalam sistem pengelolaan akademik untuk mendukung intervensi dini.
   - Output dari model prediksi dapat digunakan oleh dosen pembimbing, bagian kemahasiswaan, atau pusat layanan akademik sebagai dasar tindakan.

3. Mengidentifikasi fitur-fitur penting (feature importance) dalam menentukan keberhasilan studi mahasiswa.
   - Hasil analisis feature importance dari model XGBoost dapat dimanfaatkan untuk menyusun kebijakan atau program pendukung mahasiswa yang lebih efektif.

### Solution statements

1. Menggunakan algoritma XGBoost Classifier sebagai baseline model prediktif.
   - XGBoost dikenal karena keunggulannya dalam menangani data tabular, akurasi yang tinggi, serta interpretabilitas terhadap kontribusi fitur melalui feature importance. Model akan dievaluasi menggunakan metrik seperti accuracy, precision, recall, dan F1-score.

2. Mengevaluasi dan membandingkan hasil prediksi dengan algoritma alternatif seperti Random Forest dan Logistic Regression.
   - Dengan membandingkan performa beberapa model, kita dapat memastikan bahwa solusi terbaik telah dipilih untuk diadopsi dalam praktek nyata.

## Data Understanding

Dataset yang digunakan dalam proyek ini berasal dari Kaggle, yaitu [Student Dropout and Academic Success](https://www.kaggle.com/datasets/adilshamim8/predict-students-dropout-and-academic-success). Dataset ini dikembangkan sebagai bagian dari proyek nasional di Portugal yang bertujuan untuk mengurangi angka dropout dan kegagalan akademik di institusi pendidikan tinggi.

Dataset ini berisi informasi lengkap mengenai 4.424 mahasiswa dari 8 program studi yang berbeda, seperti Agronomi, Desain, Pendidikan, Keperawatan, Jurnalisme, Manajemen, Pelayanan Sosial, dan Teknologi. Tujuan utama dari dataset ini adalah untuk mendukung sistem intervensi dini dengan memprediksi kemungkinan hasil akademik mahasiswa, yaitu:

- Dropout (Berhenti Studi)
- Enrolled (Masih Terdaftar)
- Graduate (Lulus)

Masalah ini diformulasikan sebagai klasifikasi tiga kelas (multiclass classification) dengan tantangan ketidakseimbangan kelas (class imbalance), menjadikannya skenario yang realistis untuk penerapan machine learning di bidang edukasi.

Ringkasan Dataset:
- Jumlah Observasi (Rows): 4.424 mahasiswa
- Jumlah Fitur (Columns): 37 kolom, terdiri dari 36 fitur input dan 1 fitur target
- Jenis Data: Numerik (integer & float) dan kategorikal
- Fitur Target: 'Target' (kategori: Dropout, Enrolled, Graduate)

### Variabel-variabel pada Student Dropout and Academic Success:

1. Demografis & Sosial Ekonomi
   - Marital status: Status pernikahan mahasiswa
   - Application mode: Mode pendaftaran mahasiswa ke universitas
   - Application order: Urutan pilihan program studi saat mendaftar
   - Course: Program studi mahasiswa
   - Daytime/evening attendance: Jadwal kuliah (pagi/sore)
   - Previous qualification: Pendidikan terakhir sebelum masuk universitas
   - Previous qualification grade: Nilai pendidikan sebelumnya (0–200)
   - Nationality: Kewarganegaraan
   - Mother's qualification: Pendidikan terakhir ibu
   - Father's qualification: Pendidikan terakhir ayah
   - Mother's occupation: Pekerjaan ibu
   - Father's occupation: Pekerjaan ayah
   - Displaced: Apakah mahasiswa berasal dari luar kota/tempat tinggal utama
   - Educational special needs: Apakah mahasiswa memiliki kebutuhan khusus
   - Debtor: Apakah mahasiswa memiliki utang akademik
   - Tuition fees up to date: Apakah pembayaran kuliah lancar
   - Gender: Jenis kelamin
   - Scholarship holder: Penerima beasiswa
   - Age at enrollment: Usia saat mendaftar kuliah
   - International: Apakah mahasiswa merupakan mahasiswa internasional

2. Riwayat Akademik
   - Curricular units 1st sem (credited, enrolled, evaluated, approved, grade, without evaluations): Informasi akademik semester 1
   - Curricular units 2nd sem (credited, enrolled, evaluated, approved, grade, without evaluations): Informasi akademik semester 2
   - Unemployment rate: Tingkat pengangguran saat tahun masuk kuliah
   - Inflation rate: Tingkat inflasi saat tahun masuk kuliah
   - GDP: Produk domestik bruto saat tahun masuk kuliah
   - Target: Label/kelas output (Dropout, Enrolled, Graduate)

### Exploratory Data Analisis

![histogram](https://github.com/user-attachments/assets/be451cfd-ace7-46e5-b6bc-1feb35ac4e08)

Untuk memahami distribusi awal dari setiap fitur dalam dataset, dilakukan visualisasi menggunakan histogram. Gambar di atas menggambarkan distribusi dari 36 fitur numerik dan kategorikal dalam dataset. Berikut beberapa temuan penting dari hasil eksplorasi data:

1. Distribusi Biner dan Kategorikal
   - Gender, Displaced, Educational special needs, Scholarship holder, International, Debtor, dan Tuition fees up to date menunjukkan distribusi yang sangat tidak seimbang. Contohnya, sebagian besar mahasiswa tidak memiliki kebutuhan khusus dan bukan penerima beasiswa.
   - Daytime/evening attendance juga didominasi oleh salah satu kelas (kemungkinan mahasiswa pagi).
   - Fitur kategorikal seperti Marital status, Application mode, Course, Mother's/Father's occupation, dan Previous qualification menunjukkan bahwa beberapa kategori sangat dominan, sementara kategori lain sangat jarang muncul, yang bisa berdampak pada generalisasi model.

2. Distribusi Numerik
   - Fitur seperti Age at enrollment menunjukkan distribusi right-skewed di mana mayoritas mahasiswa berusia antara 17–25 tahun, namun ada outlier dengan usia lebih dari 60 tahun.
   - Admission grade dan Previous qualification grade memiliki distribusi normal dengan sedikit skewness ke kiri, berkisar antara 80 hingga 200 poin.
   - Curricular units di semester 1 dan 2 seperti approved, grade, enrolled, dan evaluations menunjukkan distribusi yang sangat skewed ke kanan dengan banyak mahasiswa memiliki nilai/aktivitas rendah.
   - Fitur ekonomi makro seperti GDP, Unemployment rate, dan Inflation rate memiliki variabilitas cukup tinggi dan distribusi yang terfragmentasi, kemungkinan karena hanya mencerminkan beberapa tahun akademik tertentu.

3. Potensi Masalah
   - Beberapa fitur memiliki outlier ekstrem, seperti pada Age at enrollment, Previous qualification grade, dan GDP, yang perlu dipertimbangkan dalam proses praproses data.
   - Terdapat ketidakseimbangan kelas yang jelas dalam beberapa fitur biner dan kategorikal, yang bisa memengaruhi performa model jika tidak diatasi dengan teknik tertentu seperti oversampling atau class weight.

## Data Preparation

Dalam proses data preparation yang dilakukan, beberapa tahapan penting diterapkan untuk memastikan data siap digunakan dalam model machine learning. Berikut adalah penjelasan rinci mengenai setiap tahapan yang dilakukan, alasan mengapa tahapan tersebut diperlukan, dan teknik yang digunakan:

1. Penanganan Outliers (Nilai Pencilan)
   - Proses: Menggunakan Interquartile Range (IQR) untuk mengidentifikasi dan membatasi (cap) outliers pada kolom numerik. Outliers adalah nilai yang sangat jauh dari nilai lainnya dan bisa menyebabkan model gagal mempelajari pola yang sebenarnya.
   - Alasan: Outliers dapat mengganggu performa model, terutama pada model berbasis jarak. Dengan membatasi outliers, kami memastikan bahwa model tidak terpengaruh oleh data ekstrem yang tidak representatif.

2. Pemisaan Kolom Kategorikal dan Numerikal
   - Pada tahap ini, dilakukan pemisahan antara kolom numerik dan kolom kategorikal. Kolom bertipe data object dan category dikategorikan sebagai kolom kategorikal, sedangkan kolom bertipe int64 dan float64 dianggap sebagai kolom numerik.

3. Penghapusan Kategori Langka pada Kolom Kategorikal
   - Proses: Pada kolom kategorikal, kami mengidentifikasi kategori langka (kategori yang memiliki frekuensi < 1%) dan mengganti kategori tersebut dengan label 'Other'. Hal ini dilakukan untuk mengurangi jumlah kategori yang sangat sedikit dan kemungkinan besar tidak memberikan informasi signifikan.
   - Alasan: Kategori langka bisa menambah kompleksitas model tanpa memberikan kontribusi berarti. Menghapus atau menggabungkannya menjadi 'Other' membuat model lebih sederhana dan lebih efisien.

4. Normalisasi Data pada Kolom Numerikall
   - Proses: Menggunakan StandardScaler untuk melakukan normalisasi pada fitur numerik. Teknik ini mengubah data sehingga memiliki distribusi dengan rata-rata 0 dan standar deviasi 1. Ini membantu model machine learning bekerja dengan lebih efisien, terutama pada algoritma yang sensitif terhadap skala fitur.
   - Alasan: Dengan melakukan normalisasi, kita memastikan bahwa setiap fitur memiliki kontribusi yang setara dalam pelatihan model.

5. Persiapan Fitur dan Target (Label Encoding pada Target)
   - Proses: Pada tahap ini, fitur prediktor (X) disiapkan dengan menghapus kolom 'target' dari DataFrame. Kemudian, kolom 'target' sebagai variabel yang akan diprediksi diubah menjadi bentuk numerik menggunakan LabelEncoder. Proses ini mengonversi kelas target dari bentuk kategorikal (string) menjadi bilangan bulat (misalnya: 'Graduate', 'Dropout', 'Enrolled' menjadi 0, 1, 2).
   - Alasan: Algoritma machine learning umumnya hanya dapat bekerja dengan data numerik. Oleh karena itu, encoding diperlukan untuk merepresentasikan kelas target dalam format yang bisa diproses oleh model secara efisien dan konsisten.

6. Pembagian Data Menjadi Data Latih dan Data Uji
   - Proses: Dataset dibagi menjadi dua bagian, yaitu 90% untuk data latih (X_train, y_train) dan 10% untuk data uji (X_test, y_test) menggunakan fungsi train_test_split. Parameter stratify=y digunakan untuk memastikan proporsi kelas target tetap seimbang di kedua subset, dan random_state=42 digunakan untuk menjaga konsistensi hasil pembagian.
   - Alasan: Pembagian ini penting agar model dapat dilatih pada sebagian data dan diuji pada data yang belum pernah dilihat sebelumnya, sehingga kita bisa mengevaluasi kinerja model secara obyektif. Stratifikasi digunakan untuk menghindari bias akibat distribusi kelas yang tidak merata.

7. Penyeimbangan Kelas (Balancing the Classes)
   - Proses: Menggunakan SMOTE (Synthetic Minority Over-sampling Technique) untuk menyeimbangkan jumlah sampel antara kelas mayoritas dan minoritas pada data pelatihan. Teknik ini menghasilkan data sintetis untuk kelas minoritas sehingga distribusi kelas menjadi lebih seimbang.
   - Alasan: Pada dataset dengan ketidakseimbangan kelas yang besar, model cenderung lebih memprediksi kelas mayoritas. Dengan menyeimbangkan kelas, kita dapat meningkatkan kemampuan model dalam memprediksi kelas minoritas dengan lebih akurat.

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
