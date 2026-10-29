// =====================================================================
// Laporan Tugas — Workshop Deep Learning (Teknik Telekomunikasi ITERA)
// Studi Ablasi RNN, LSTM, GRU vs ANN untuk Prediksi Trafik Seluler Milan
//
// Teks MERAH [ISI: ...] = hanya yang tidak ada di file/README (cari "ISI").
// =====================================================================
#import "@preview/charged-ieee:0.1.4": ieee

#set text(lang: "id")
#let isi(x) = text(fill: red)[\[ISI: #x\]]

#show: ieee.with(
  title: [Studi Ablasi RNN, LSTM, dan GRU dibandingkan ANN untuk Prediksi Trafik Seluler Kota Milan],
  abstract: [
    Operator seluler membutuhkan ramalan trafik per sel untuk mengatur kapasitas jaringan. Studi ini membandingkan ANN dengan tiga model sekuensial (RNN, LSTM, GRU) untuk memprediksi trafik internet satu jam ke depan pada data Telecom Italia di kotak Duomo dan Bandara Linate (kotak 4489). Kami melakukan studi ablasi dengan mengubah satu faktor setiap kali: panjang jendela, ukuran _hidden_, jumlah lapisan, _learning rate_, fitur, _dropout_, dan fungsi _loss_. Setiap konfigurasi dilatih dengan tiga _seed_ dan dilaporkan sebagai rata-rata ± simpangan baku. _Learning rate_ adalah faktor paling berpengaruh, sedangkan _dropout_ justru memperburuk semua model. Di Duomo, konfigurasi terbaik (dipilih dengan val MAE) menghasilkan test MAE 216,2 ± 11,3 untuk GRU, 224,5 ± 11,7 untuk RNN, 254,4 ± 3,2 untuk ANN, dan 270,7 ± 8,9 untuk LSTM, atau 42% hingga 54% lebih rendah daripada baseline naif terbaik (467,6). Di Linate, RNN dan GRU terbaik (7,4 dan 7,5) terhadap baseline naif 10,1, tetapi pada minggu Natal baseline nilai terakhir (7,1) masih mengungguli semua model. Model rekuren sederhana cukup untuk prediksi satu jam, dan LSTM tidak memberi keuntungan pada data yang tersedia.
  ],
  authors: (
    (name: "Rafael Christito", department: [NIM 124400041], organization: [Teknik Telekomunikasi, ITERA], email: "email@student.itera.ac.id"),
    (name: "Rizky Adishawa Albarkah", department: [NIM 124400035], organization: [Teknik Telekomunikasi, ITERA], email: "email@student.itera.ac.id"),
  ),
  index-terms: ("prediksi trafik seluler", "RNN", "LSTM", "GRU", "studi ablasi"),
  bibliography: bibliography("references.bib", title: [Pustaka]),
  figure-supplement: [Gambar],
)

= Pendahuluan
Prediksi trafik per sel memungkinkan operator menyiapkan kapasitas, menghemat energi pada jam sepi, dan mengantisipasi lonjakan seperti hari libur. Trafik seluler bersifat deret waktu dengan pola harian dan mingguan, sehingga model sekuensial seperti RNN, LSTM @hochreiter1997lstm, dan GRU @cho2014gru secara alami dibandingkan dengan jaringan _feed-forward_ (ANN) yang memperlakukan jendela masukan sebagai vektor datar.

Perbandingan antar arsitektur saja tidak cukup, karena hasilnya sangat bergantung pada _hyperparameter_. Studi ablasi, yaitu mengubah satu faktor setiap kali sambil menahan faktor lain, memisahkan pengaruh tiap faktor sehingga kesimpulan tidak tercampur. Kontribusi laporan ini: (1) perbandingan empat model pada konfigurasi acuan yang sama dengan tiga _seed_; (2) ablasi lima faktor wajib dan dua faktor pilihan (_dropout_ dan fungsi _loss_); (3) uji generalisasi ke lokasi kelompok, Bandara Linate.

= Data dan Metode

== Dataset
Kami memakai Telecom Italia Big Data Challenge @barlacchi2015milan dalam versi per jam dari @zhang2021fedda. Fitur yang tersedia adalah aktivitas internet (net), panggilan (call), dan SMS (sms) per kotak. Eksperimen utama dilakukan pada kotak Duomo (5060), pusat kota Milan dengan trafik tinggi. Lokasi kelompok kami adalah lokasi tugas nomor 5, Bandara Linate, kotak 4489, ditentukan dari digit terakhir NIM terkecil (124400035). @fig:peta menunjukkan posisi lokasi-lokasi tersebut.

#figure(
  image("peta_lokasi_tugas.png", width: 85%),
  caption: [Rata-rata aktivitas internet per kotak di Milan. Lingkaran kuning: lokasi tugas (kode 0–9); lingkaran toska: Duomo.],
  placement: top,
) <fig:peta>

== Pembagian Data dan Praproses
Data mencakup periode November hingga Desember 2013 dengan resolusi per jam. Pembagian dilakukan menurut urutan waktu menjadi empat bagian: 4 minggu untuk latih (672 jam), 1 minggu untuk validasi (168 jam), 1 minggu untuk uji (168 jam), dan 1 minggu Natal (168 jam) sebagai uji tambahan pada periode libur. Data dibagi berdasarkan urutan waktu, bukan diacak, karena pengacakan membuat sampel latih memuat informasi dari masa setelah sampel uji (kebocoran waktu) dan menghasilkan estimasi galat yang terlalu optimistis. Standarisasi memakai rata-rata dan simpangan baku dari data sebelum minggu validasi, lalu diterapkan ke seluruh bagian. Data kemudian dipotong menjadi jendela (_windowing_) sepanjang $W$ jam dengan target satu jam setelahnya. Data dibagi berdasarkan urutan waktu, bukan diacak, karena pengacakan membuat sampel latih memuat informasi dari masa setelah sampel uji (kebocoran waktu) dan menghasilkan estimasi galat yang terlalu optimistis. Standarisasi memakai rata-rata dan simpangan baku dari data sebelum minggu validasi, lalu diterapkan ke seluruh bagian. Data kemudian dipotong menjadi jendela (_windowing_) sepanjang $W$ jam dengan target satu jam setelahnya.

== Model dan Konfigurasi Acuan
ANN adalah _multilayer perceptron_ yang menerima jendela sebagai vektor datar. RNN, LSTM, dan GRU memproses jendela jam demi jam; keadaan _hidden_ terakhir diteruskan ke lapisan keluaran linear. LSTM memakai sel memori dengan tiga gerbang, GRU menyederhanakannya menjadi dua gerbang, dan RNN sederhana tidak memiliki gerbang. Ukuran _hidden_ acuan dipilih agar jumlah parameter keempat model sebanding, yaitu 5.305 hingga 6.497 (@tab:acuan).

#figure(
  caption: [Konfigurasi acuan],
  placement: top,
  table(
    columns: 2,
    align: left,
    stroke: none,
    table.hline(),
    table.header([*Hyperparameter*], [*Nilai acuan*]),
    table.hline(stroke: 0.5pt),
    [Panjang jendela $W$], [24 jam],
    [Horizon], [1 jam],
    [Ukuran hidden], [ANN 32, RNN 68, LSTM 33, GRU 38],
    [Jumlah lapisan], [ANN 2, rekuren 1],
    [Fitur], [net, call, sms, jam, hari (7)],
    [Optimizer, _learning rate_], [Adam, $10^(-3)$],
    [_Batch size_], [64],
    [_Early stopping_], [_patience_ 30, maks. 300 epoch],
    table.hline(),
  ),
) <tab:acuan>

== Protokol Ablasi dan Evaluasi
Lima faktor wajib diubah satu per satu: A1 panjang jendela $W$ (6, 12, 24, 48, 168 jam), A2 ukuran _hidden_ (16, 32, 64, 128), A3 jumlah lapisan (1, 2, 3), A4 _learning rate_ ($10^(-4)$, $10^(-3)$, $10^(-2)$), dan A5 set fitur (net; net+call+sms; net+waktu; semua). Dua faktor pilihan kami adalah B2, yaitu _dropout_ (0; 0,2; 0,5), dan B4, yaitu fungsi _loss_ (MSE, MAE, Huber). B2b mengulang B2 dengan dua lapisan rekuren, karena _dropout_ antar lapisan hanya berlaku bila lapisan lebih dari satu. Hipotesis awal kami: untuk B2, _dropout_ moderat (0,2) mengurangi _overfitting_ sehingga menurunkan galat, sedangkan 0,5 terlalu kuat dan menurunkan kapasitas model; untuk B4, _loss_ MAE dan Huber lebih baik daripada MSE karena metrik evaluasi adalah MAE dan kedua _loss_ itu lebih tahan terhadap lonjakan trafik.

Setiap konfigurasi dilatih dengan tiga _seed_ dan dilaporkan sebagai rata-rata ± simpangan baku. Metrik adalah MAE, RMSE, dan MAPE pada skala asli (setelah dikembalikan dari standarisasi). Konfigurasi terbaik dipilih dengan val MAE, bukan test MAE, agar data uji tidak ikut menentukan pilihan. Seluruh eksperimen berjalan di Google Colab dengan total sekitar 18 menit untuk 297 konfigurasi unik (396 baris di `ablasi.csv`, sebagian disalin dari run yang sama tanpa dilatih ulang).

= Hasil

== Baseline Naif dan Acuan
Baseline naif terbaik di Duomo adalah memakai nilai pada minggu lalu, dengan test MAE 467,6. Pada konfigurasi acuan, keempat model sudah jauh lebih baik: RNN 229,6 ± 5,1, GRU 237,2 ± 9,3, LSTM 273,6 ± 25,4, dan ANN 309,9 ± 5,9. Urutan RNN, GRU, LSTM, ANN ini juga terlihat pada val MAE (206,1; 207,5; 230,5; 248,9). Di minggu Natal galat semua model naik tajam (421,2 untuk RNN hingga 680,1 untuk ANN) karena pola trafik hari libur berbeda dari hari biasa.

== Ablasi Wajib
#figure(
  grid(
    columns: 2,
    gutter: 4pt,
    image("A1_W.png"), image("A2_hidden.png"),
    image("A3_layers.png"), image("A4_lr.png"),
  ),
  caption: [Ablasi wajib terhadap test MAE (rata-rata ± simpangan baku, 3 seed): panjang jendela $W$ (A1), ukuran _hidden_ (A2), jumlah lapisan (A3), dan _learning rate_ (A4).],
  placement: top,
) <fig:ablasi>

*A1, panjang jendela.* Pengaruh $W$ sangat berbeda antar model (@fig:ablasi, kiri atas). ANN paling sensitif: test MAE naik dari 268,9 pada $W$=6 menjadi 402,0 pada $W$=168, karena jumlah masukan membesar sementara data latih hanya empat minggu sehingga ANN cenderung _overfit_. Model rekuren hampir tidak terpengaruh: RNN berkisar 224,5 hingga 233,5 dan GRU 237,1 hingga 249,8. Jendela pendek tidak merugikan RNN dan LSTM, jadi informasi beberapa jam terakhir sudah memuat sebagian besar sinyal untuk prediksi satu jam. Dengan val MAE, $W$ terbaik adalah 6 untuk ANN dan LSTM serta 48 untuk GRU dan RNN.

*A2, ukuran hidden.* Ukuran _hidden_ yang lebih besar membantu GRU (test MAE 246,9 pada 16 menjadi 220,3 pada 128) dan LSTM (295,3 menjadi 245,6). Pada RNN pengaruhnya kecil (253,3 pada 16 dan 234,6 pada 128, sementara acuan 68 mencapai 229,6), dan pada ANN selisihnya masih dalam simpangan baku (296,7 hingga 316,2). Model bergerbang tampaknya lebih membutuhkan kapasitas, sedangkan RNN dan ANN sudah cukup dengan ukuran kecil.

*A3, jumlah lapisan.* Menambah lapisan merugikan model rekuren. LSTM paling terpukul: 273,6 (1 lapisan), 308,9 (2), dan 412,6 (3). GRU memburuk pada 3 lapisan (264,2 dari 237,2) dan RNN pada 3 lapisan (245,5 dari 229,6). ANN tidak berubah berarti (300,7 hingga 311,2). Dengan data sedikit, model yang lebih dalam lebih sulit dilatih dan mudah _overfit_.

*A4, _learning rate_.* Faktor ini paling berpengaruh. Nilai $10^(-4)$ terlalu kecil untuk semua model: GRU 356,3 dan LSTM 373,5, jauh di atas acuan, karena pelatihan belum konvergen sebelum batas epoch. Nilai $10^(-2)$ justru memperbaiki GRU (215,7) dan LSTM (225,8), tetapi memperburuk RNN (240,6) dan membuat ANN sangat tidak stabil (simpangan baku 53,2).

*A5, fitur.* Fitur tambahan membantu model rekuren. Memakai net saja memberi test MAE 270,5 (GRU) dan 254,1 (RNN), turun menjadi 237,2 dan 229,6 dengan semua fitur. Penambahan call dan sms saja sudah menurunkan RNN ke 227,6, sedangkan fitur waktu (jam dan hari) membantu val MAE paling besar pada GRU (264,6 menjadi 213,2). ANN kebalikannya: net saja terbaik (271,7) dan semua fitur terburuk kedua (309,9), karena dimensi masukannya bertambah tanpa tambahan data latih. @tab:rentang merangkum rentang test MAE tiap faktor.

#figure(
  image("A5_fitur.png", width: 80%),
  caption: [Pengaruh set fitur masukan terhadap test MAE (A5).],
  placement: top,
) <fig:fitur>

#figure(
  caption: [Rentang test MAE (minimum–maksimum antar nilai faktor) di Duomo. Rentang lebar berarti faktor itu berpengaruh besar.],
  placement: top,
  table(
    columns: 5,
    align: (left, center, center, center, center),
    stroke: none,
    table.hline(),
    table.header([*Faktor*], [*ANN*], [*RNN*], [*LSTM*], [*GRU*]),
    table.hline(stroke: 0.5pt),
    [A1 jendela], [269–402], [225–234], [274–304], [237–250],
    [A2 hidden], [297–316], [235–253], [246–295], [220–247],
    [A3 lapisan], [301–311], [228–246], [274–413], [237–264],
    [A4 _lr_], [309–340], [230–288], [226–374], [216–356],
    [A5 fitur], [272–316], [228–254], [263–293], [237–271],
    [B2 _dropout_], [310–368], [230–273], [274–300], [237–259],
    [B4 _loss_], [289–314], [211–230], [266–274], [224–237],
    table.hline(),
  ),
) <tab:rentang>

== Ablasi Pilihan
#figure(
  grid(
    columns: 3,
    gutter: 4pt,
    image("B2_dropout.png"), image("B2b_dropout_l2.png"), image("B4_loss.png"),
  ),
  caption: [Ablasi pilihan: _dropout_ (B2), _dropout_ dengan dua lapisan rekuren (B2b), dan fungsi _loss_ (B4).],
  placement: top,
) <fig:pilihan>

*B2, _dropout_.* @fig:pilihan menunjukkan bahwa _dropout_ tidak membantu. Test MAE naik seiring _dropout_: ANN 309,9 menjadi 332,4 (0,2) dan 368,2 (0,5), RNN 229,6 menjadi 250,7 dan 273,1. Untuk GRU dan LSTM, nilai 0,2 masih setara dengan tanpa _dropout_ (237,8 dan 275,0), tetapi 0,5 memperburuk (259,4 dan 299,7). Pada B2b dengan dua lapisan rekuren tren sama: GRU 239,2 menjadi 262,0 dan 299,9, LSTM 308,9 menjadi 350,0 dan 373,7. Model kami sudah kecil dan tidak jelas _overfit_, sehingga regularisasi tambahan hanya mengurangi kapasitas. Hipotesis awal: Hipotesis awal kami bahwa _dropout_ 0,2 menurunkan galat tidak terbukti: pada ANN dan RNN justru memperburuk, sedangkan pada GRU dan LSTM tidak berbeda berarti dari tanpa _dropout_.

*B4, fungsi _loss_.* _Loss_ MAE paling baik di test untuk keempat model: ANN 309,9 menjadi 288,5, GRU 237,2 menjadi 224,4, RNN 229,6 menjadi 210,7, dan LSTM 273,6 menjadi 266,2 (selisih LSTM masih dalam simpangan baku). Huber hampir sama dengan MSE. Hal ini wajar karena metrik evaluasi adalah MAE sehingga melatih langsung dengan MAE menyelaraskan tujuan. Namun pada val MAE perbaikannya tidak seragam (RNN 206,1 dengan MSE dan 211,5 dengan MAE), sehingga pemilihan dengan val MAE tetap mempertahankan MSE untuk RNN. Hipotesis awal: Hipotesis awal kami bahwa MAE lebih baik daripada MSE terbukti pada test untuk ANN, GRU, dan RNN, tetapi tidak untuk LSTM (selisih dalam simpangan baku). Dugaan bahwa Huber lebih baik daripada MSE tidak terbukti karena hasilnya hampir sama.

== Konfigurasi Terbaik
Konfigurasi terbaik tiap model dipilih dengan val MAE (percobaan `C-coba1`) dari hasil ablasi. ANN memakai $W$=6, _hidden_ 128, 2 lapisan; GRU $W$=24, _hidden_ 128, 1 lapisan, _loss_ MAE; LSTM $W$=6, _hidden_ 128, 1 lapisan; dan RNN $W$=48 dengan _hidden_ 68. Lainnya sama dengan acuan. @tab:terbaik merangkum hasilnya dan @fig:terbaik membandingkannya dengan acuan. GRU terbaik dengan test MAE 216,2 ± 11,3, disusul RNN (224,5 ± 11,7), ANN (254,4 ± 3,2), dan LSTM (270,7 ± 8,9). Dibanding baseline naif (467,6), penurunannya 53,8% (GRU), 52,0% (RNN), 45,6% (ANN), dan 42,1% (LSTM). Perbaikan terbesar dari acuan terjadi pada GRU (237,2 menjadi 216,2, turun 8,9%) dan ANN (309,9 menjadi 254,4, turun 17,9%), sedangkan RNN (229,6 menjadi 224,5) dan LSTM (273,6 menjadi 270,7) hampir tidak berubah. Pada minggu Natal, RNN terbaik (404,5) dan GRU kedua (430,6).

#figure(
  caption: [Test MAE dan Natal MAE di Duomo pada konfigurasi acuan dan terbaik (rata-rata ± simpangan baku, 3 seed).],
  placement: top,
  table(
    columns: 5,
    align: (left, center, center, center, center),
    stroke: none,
    table.hline(),
    table.header([*Model*], [*Test acuan*], [*Test terbaik*], [*Natal acuan*], [*Natal terbaik*]),
    table.hline(stroke: 0.5pt),
    [ANN], [309,9 ± 5,9], [254,4 ± 3,2], [680,1 ± 52,0], [534,7 ± 29,7],
    [RNN], [229,6 ± 5,1], [224,5 ± 11,7], [421,2 ± 2,4], [*404,5 ± 9,9*],
    [LSTM], [273,6 ± 25,4], [270,7 ± 8,9], [586,0 ± 56,8], [546,5 ± 43,9],
    [GRU], [237,2 ± 9,3], [*216,2 ± 11,3*], [468,2 ± 21,8], [430,6 ± 9,2],
    table.hline(),
  ),
) <tab:terbaik>

#figure(
  image("C_terbaik_vs_acuan.png", width: 100%),
  caption: [Konfigurasi terbaik dibandingkan konfigurasi acuan di Duomo.],
  placement: top,
) <fig:terbaik>

== Uji Generalisasi
Kami menguji konfigurasi acuan dan konfigurasi terbaik Duomo di Bandara Linate (@tab:linate). Urutan model berubah dibanding Duomo: RNN terbaik (test MAE 7,4 ± 0,1), GRU hampir setara (7,5 ± 0,2), sedangkan ANN (8,0 ± 0,4) dan LSTM (8,0 ± 0,3) tertinggal. Dibanding baseline nilai terakhir (10,1), RNN lebih rendah 26,7%. Manfaat konfigurasi terbaik Duomo di Linate kecil pada test: ANN membaik dari 8,8 menjadi 8,0, sedangkan model lain berubah kurang dari 0,3. Pada minggu Natal manfaatnya lebih terlihat untuk ANN (14,4 menjadi 11,8) dan GRU (11,4 menjadi 9,7). Namun pada Natal, baseline nilai terakhir (7,1) masih lebih baik daripada semua model, sehingga model belajar dari pola hari biasa dan kurang menangkap perubahan pada hari libur.

Skala trafik Linate jauh lebih kecil daripada Duomo (val MAE sekitar 8 berbanding 200), tetapi MAPE test keduanya sebanding (8,2% hingga 10,3% di Linate; 6,5% hingga 11,3% di Duomo), sehingga galat relatifnya tidak jauh berbeda. @fig:pola membandingkan pola trafik kedua lokasi. Di Duomo, trafik memiliki siklus harian yang teratur dengan amplitudo besar, sedangkan di Linate siklus harian tetap ada tetapi amplitudonya jauh lebih kecil dan kurva tampak lebih bergerigi, sehingga selisih absolut antar model otomatis kecil.

#figure(
  caption: [Test MAE dan Natal MAE di Bandara Linate (kotak 4489), konfigurasi acuan dan terbaik. Baris naif memakai nilai terakhir.],
  placement: top,
  table(
    columns: 5,
    align: (left, center, center, center, center),
    stroke: none,
    table.hline(),
    table.header([*Model*], [*Test acuan*], [*Test terbaik*], [*Natal acuan*], [*Natal terbaik*]),
    table.hline(stroke: 0.5pt),
    [ANN], [8,8 ± 0,7], [8,0 ± 0,4], [14,4 ± 0,5], [11,8 ± 1,5],
    [RNN], [7,5 ± 0,1], [*7,4 ± 0,1*], [10,8 ± 0,1], [10,7 ± 1,2],
    [LSTM], [8,2 ± 0,2], [8,0 ± 0,3], [12,8 ± 0,8], [12,8 ± 0,2],
    [GRU], [7,5 ± 0,2], [7,5 ± 0,2], [11,4 ± 0,9], [9,7 ± 0,5],
    [Naif], [10,1], [10,1], [7,1], [7,1],
    table.hline(),
  ),
) <tab:linate>

#figure(
  image("D_pola_lokasi.png", width: 100%),
  caption: [Pola trafik di lokasi kelompok dibandingkan Duomo.],
  placement: top,
) <fig:pola>

= Pembahasan
Perbedaan antar model di Duomo (216,2 hingga 270,7) lebih besar daripada simpangan baku antar _seed_ (3,2 hingga 11,7) untuk pasangan yang berjauhan, misalnya GRU dengan LSTM atau ANN. Namun selisih GRU dan RNN (8,3) lebih kecil daripada simpangan baku keduanya (sekitar 11), dan RNN justru lebih baik pada val MAE (200,1 berbanding 197,5 hampir setara) serta pada Natal (404,5 berbanding 430,6). Jadi kami hanya dapat menyimpulkan bahwa GRU dan RNN lebih baik daripada ANN dan LSTM, bukan bahwa GRU lebih baik daripada RNN.

Faktor yang paling berpengaruh adalah _learning rate_ (A4): rentang test MAE mencapai 140 pada GRU dan 148 pada LSTM, dan nilai terbaik berbeda antar model, sehingga _learning rate_ perlu disetel per model. Berikutnya jumlah lapisan untuk LSTM (rentang 139) dan panjang jendela untuk ANN (rentang 133). Faktor yang paling kecil pengaruhnya atau merugikan adalah _dropout_ dan, untuk model rekuren, panjang jendela. Pola ini konsisten dengan data yang kecil: model besar atau dalam sulit dilatih (A3), masukan panjang memperburuk ANN (A1), sedangkan regularisasi tidak membantu karena model sudah kecil (B2).

LSTM menjadi yang terburuk walau paling ekspresif. Dugaan kami, data latih empat minggu terlalu pendek untuk melatih gerbang tambahannya dan LSTM sangat peka terhadap _learning rate_ serta kedalaman (A3, A4), tetapi dugaan ini belum diuji langsung. Di Linate, perbedaan antar model kecil (7,4 hingga 8,0) dan konfigurasi Duomo tidak banyak membantu, sehingga _hyperparameter_ terbaik tampak bergantung pada lokasi.

Keterbatasan studi: hanya satu kota, data latih hanya empat minggu, satu kali pembagian waktu, dan hanya tiga _seed_ sehingga simpangan baku sendiri kurang stabil. Faktor diubah satu per satu, sehingga interaksi antar faktor, misalnya _learning rate_ dengan ukuran _hidden_, tidak tertangkap. Konfigurasi terbaik juga disusun dari pilihan terbaik tiap faktor sehingga tidak dijamin optimal bersama.

= Kesimpulan
Pada prediksi trafik internet satu jam ke depan di Duomo, GRU (test MAE 216,2 ± 11,3) dan RNN (224,5 ± 11,7) mengungguli ANN (254,4 ± 3,2) dan LSTM (270,7 ± 8,9), dan semuanya 42% hingga 54% lebih baik daripada baseline naif (467,6). _Learning rate_ adalah faktor paling berpengaruh, sedangkan _dropout_ merugikan semua model dan lapisan tambahan merugikan model rekuren, terutama LSTM. Di Bandara Linate, RNN (7,4) dan GRU (7,5) kembali terbaik terhadap baseline naif 10,1, tetapi manfaat konfigurasi Duomo kecil dan pada minggu Natal baseline nilai terakhir (7,1) masih lebih baik daripada semua model.

#heading(numbering: none)[Kontribusi Anggota]
#table(
  columns: (auto, 1fr),
  stroke: none,
    [Rafael Christito], [Bagian A dan B (ablasi wajib dan ablasi pilihan), serta penyusunan laporan Typst],
  [Rizky Adishawa Albarkah], [Bagian C dan D (konfigurasi terbaik dan uji generalisasi), serta penyusunan laporan Typst],
)

#heading(numbering: none)[Pernyataan Penggunaan AI]
Claude (Anthropic) digunakan untuk menyusun kerangka laporan dalam Typst, meringkas hasil ablasi dari berkas CSV, dan merapikan bahasa. Seluruh angka berasal dari run kami sendiri dan telah kami cocokkan dengan `ablasi.csv`.

#heading(numbering: none)[Kode dan Data]
Kode, notebook beserta output, dan seluruh hasil run tersedia di #link("https://github.com/rafayelll14/dlw-tugas-RA-124400035-dan-124400041").