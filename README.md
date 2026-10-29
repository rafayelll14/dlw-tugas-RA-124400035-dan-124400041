# Studi Ablasi RNN, LSTM, GRU vs ANN untuk Prediksi Trafik Seluler Milan

Tugas Workshop Deep Learning — Teknik Telekomunikasi ITERA.

## Kelompok

| Peran | Nama | NIM |
|---|---|---|
| Ketua | Rafael Christito | 124400041 |
| Anggota | Rizky Adishawa Albarkah | 124400035 |

- **Kelas:** RA
- **Lokasi kelompok:** 5 — Bandara Linate, kotak 4489 (digit terakhir NIM terkecil, 124400035)
- **Laporan:** `laporan-latex/main.pdf` atau `laporan-typst/main.pdf` (menyusul)

## Cara Menjalankan

```bash
uv sync
uv run jupyter lab
```

Buka `tugas_ablasi.ipynb`, isi `LOKASI_KELOMPOK`, lalu jalankan semua sel. Semua run tercatat di `hasil/ablasi.csv`.

Perangkat yang kami pakai: Google Colab, total waktu training: ± 18 menit (297 konfigurasi unik; 396 baris run di `hasil/ablasi.csv`, sebagian disalin tanpa dilatih ulang).

## Ringkasan Hasil

Di Duomo, konfigurasi terbaik (dipilih dengan val MAE, bagian `C-coba1`) memberi test MAE: GRU 216,2 ± 11,3; RNN 224,5 ± 11,7; ANN 254,4 ± 3,2; LSTM 270,7 ± 8,9 — semuanya lebih baik daripada baseline naif terbaik (minggu lalu, 467,6), dan ANN turun paling banyak dari acuan (309,9). Di lokasi kelompok (Bandara Linate), test MAE terbaik: RNN 7,4 ± 0,1; GRU 7,5 ± 0,2; ANN 8,0 ± 0,4; LSTM 8,0 ± 0,3, sedangkan baseline naif terbaik (nilai terakhir) 10,1. Manfaat konfigurasi terbaik Duomo di Linate kecil, kecuali ANN dan GRU di minggu Natal.

## Struktur

```
tugas_ablasi.ipynb      notebook eksperimen (di-commit beserta output)
data/                   data trafik per jam (Duomo + 10 lokasi tugas)
hasil/                  CSV seluruh run, tabel, dan grafik
laporan-latex/ atau laporan-typst/   sumber laporan + PDF
```

## Data

Telecom Italia Big Data Challenge (Barlacchi dkk., 2015, *Scientific Data*, lisensi ODbL), versi per jam dari Zhang dkk. (INFOCOM 2021). Disiapkan oleh pengajar di repo [mctosima/dl-workshop](https://github.com/mctosima/dl-workshop).
