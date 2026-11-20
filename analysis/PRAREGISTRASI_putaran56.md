# PRA-REGISTRASI PUTARAN 56 (referensi acak berukuran setara kontrol; ditulis SEBELUM menghitung; 2026-11-13)

Status analisis: **lanjutan** (sesudah putaran 3--55 diketahui). Perintah pemilik 2026-11-13 atas ulasan `review pak dwi/REVIEW_EEG_COVARIATE_DAN_PILIHAN_JURNAL_2026-11-13.pdf`, butir M3; pemilik memilih "M3 referensi acak setara" (hanya analisis ini).

## 0. Yang SUDAH dilihat sebelum dokumen ini ditulis

1. Putaran 29 (H1, 100 replikasi, benih 29100+r): C1 50.6, V0 56.0, V1 56.1, V2 55.7, V3 52.5, V4 50.4 (AUC gabungan); V0 − C1 +5.4 poin (SD 2.5).
2. Putaran 55 (simulasi): kebocoran referensi kontrol bergantung pada hubungan label–kovariat, mengecil perlahan dengan ukuran referensi (+4.2 poin pada 1000 kontrol), dan tidak memerlukan sel jarang.
3. **Tidak ada** AUC dengan referensi acak dari semua pasien latih yang pernah dihitung atau dilihat.

## 1. Pertanyaan

V0 dan V4 berbeda dalam komposisi label referensi, jumlah pasien, dukungan era, dan leverage desain. Apakah kebocoran V0 berasal dari pemilihan referensi menurut label (hanya kontrol), atau dari ukuran/ketidaksamaan referensi dengan pasien latih pengklasifikasi, yang juga dimiliki referensi acak berukuran sama?

## 2. Metode

Data, pengacakan, lipatan, dan pengklasifikasi identik dengan putaran 29 (H1: `round28_analisis.shuffle_cols` pada BP, kelompok semua pasien, benih `29100 + r`, r = 0..99; 50 lipatan kanonik; `evalkit.nested_lr`; tanpa usia/jenis kelamin di masukan; penyesuaian `round29.fit_norm` dengan desain `D_I2`).

Varian (pengklasifikasi selalu dilatih pada semua pasien latih lipatan, seperti V0):
- **C1**: tanpa penyesuaian (reproduksi putaran 29);
- **V0**: referensi = kontrol latih (reproduksi putaran 29);
- **VR**: referensi = subset acak pasien latih (tanpa memandang label) berukuran sama dengan jumlah kontrol latih lipatan itu; subset tetap per lipatan (benih `56000 + lipatan`), sama di semua replikasi, seperti referensi V0 yang tetap per lipatan;
- **V4**: referensi = semua pasien latih (reproduksi putaran 29).

**Diagnostik desain per lipatan (tanpa label hasil):** rank desain pada referensi, jumlah pasien referensi per era (minimum), leverage maksimum $h_{ii}$ referensi, proporsi epilepsi dalam referensi; untuk V0 dan VR.

## 3. Titik akhir

**Kontras utama (satu, dibekukan):** selisih berpasangan VR − V0 dalam AUC gabungan atas 100 replikasi H1 (permutasi identik), uji t satu sampel dua sisi, 95% CI. Tidak ada koreksi keluarga (satu uji).

**Sekunder (deskriptif):** keputusan aturan berpasangan untuk VR − C1 dan aturan mutlak untuk VR (ambang asli, `round49_kalibrasi.decide`); diagnostik desain.

**Pemeriksaan reproduksi:** C1, V0, V4 per replikasi identik dengan `hasil29/A_kebocoran_replikasi.csv` (|selisih| < 1e-9); bila gagal, run dihentikan dan dicatat.

**Prediksi (ditulis sebelum run):** VR juga bocor (VR − C1 "kebocoran" menurut aturan berpasangan), dan |VR − V0| < 2 poin, karena menurut putaran 55 kebocoran berasal dari ketidaksamaan pasien pemasang referensi dengan pasien latih pengklasifikasi, bukan dari label kontrol saja.

**Tafsir yang ditetapkan sebelumnya:** bila VR bocor setara V0, pemilihan kontrol bukan penyebab tunggal; klaim (ii) dirumuskan sebagai "referensi yang tidak sama dengan pasien latih pengklasifikasi" dengan referensi kontrol sebagai kasus nyata. Bila VR tidak bocor (aturan berpasangan "tidak ada bukti") dan VR − V0 negatif bermakna, pemilihan menurut label adalah sumber utama. Hasil lain dilaporkan apa adanya tanpa atribusi sebab.

## 4. Batas

Satu arsip, satu representasi (I2, kolom diacak), satu pengklasifikasi linear; referensi acak satu per lipatan.
