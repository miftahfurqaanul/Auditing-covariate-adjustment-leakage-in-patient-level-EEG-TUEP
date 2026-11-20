# PRA-REGISTRASI PUTARAN 55 (simulasi referensi kontrol untuk klaim ii; ditulis SEBELUM menghitung; 2026-11-13)

Status analisis: **lanjutan** (sesudah putaran 3--54 diketahui). Perintah pemilik 2026-11-13: menjalankan semua perintah dalam `review pak dwi/TELAAH_NASKAH_EEG_LEAKAGE_2026-11-13.pdf`. Simulasi ini disebut namanya oleh pemilik ("Jalankan simulasi"; langkah 4 telaah). ComBat tidak termasuk.

## 0. Yang SUDAH dilihat sebelum dokumen ini ditulis

1. Hasil nyata penyesuaian berbasis kontrol (V0) pada I2 TUEP: AUC nol gabungan 56.4 (audit terencana) dan 56.0 (seri varian), selisih berpasangan V0 − C1 +5.4 poin (SD 2.5, 100 replikasi); V3 tanpa era 52.5; V4 50.4 (putaran 29, 46).
2. Kohort TUEP: era paling awal (≤2009) berisi 25 pasien epilepsi dan 6 non-epilepsi; sekitar 80 kontrol latih per lipatan.
3. Diagnostik eksploratif putaran 51--52 (tidak dilaporkan): kebocoran = injeksi galat estimasi koefisien yang dipakai bersama; penyusutan EB menghilangkannya. Dugaan "fitting asymmetry" di Pembahasan naskah.
4. **Tidak ada** hasil simulasi dengan referensi kontrol yang pernah dihitung atau dilihat.

## 1. Pertanyaan

Apakah kebocoran penyesuaian berbasis kontrol ke pengklasifikasi linear tingkat pasien (a) mengecil bila jumlah kontrol referensi bertambah, (b) membesar bila sel desain kontrol jarang, dan (c) membesar dengan kekuatan hubungan label–kovariat, ketika kebenarannya diketahui (fitur tanpa informasi label maupun kovariat)?

## 2. Metode

**Data per replikasi.** Label seimbang: N/2 kasus, N/2 kontrol. Kovariat (usia, jenis kelamin, kelompok era ≤2009/2010/2011/2012/2013) diambil ulang (dengan pengembalian) dari baris pasien TUEP (`round4`: AGE, MALE, YEAR_G). Dengan peluang s sebuah baris diambil dari kelas sumber pasien itu, selain itu dari ke-200 pasien gabungan; s ∈ {0, 0.5, 1}. Fitur: 114 kolom normal multivariat dengan korelasi empiris fitur I2 (`round4.BP`), independen dari label dan kovariat (kebenaran: tidak ada informasi).

**Faktor (30 sel).**
- Jumlah kontrol referensi per lipatan latih n_ref ∈ {20, 40, 80, 200, 1000}; N = 2.5 n_ref (50, 100, 200, 500, 2500), sehingga 5 lipatan berstrata memberi n_ref kontrol latih.
- Desain sel: **jarang** (kelas sumber kontrol = non-epilepsi TUEP, sehingga kontrol berisi sekitar 6% era ≤2009, seperti TUEP) atau **cermin** (kelas sumber ditukar: kontrol dari baris epilepsi, kasus dari baris non-epilepsi; besar hubungan sama, tetapi sel era terkecil kontrol sekitar 15%).
- s ∈ {0, 0.5, 1}.

**Pipeline (sama persis dengan I2 audit, tanpa usia/jenis kelamin di masukan):** C1 tanpa penyesuaian; CR = penyesuaian berbasis kontrol (OLS fitur pada [1, usia, jenis kelamin, indikator era] pada kontrol latih, residu dibagi SD residu latih dengan ddof = rank; identik dengan `round4.normative_z`/`round29.fit_norm`); AP = penyesuaian semua pasien latih (sama, pada semua pasien latih). Pengklasifikasi `evalkit.nested_lr` (StandardScaler + LogisticRegressionCV L2, 9 nilai C, balanced, AUC dalam 5 lipatan). Satu pembagian 5 lipatan berstrata per replikasi (seperti simulasi dan plasmode naskah).

**Replikasi:** 100 per sel. Benih bergantung pada (sel, replikasi), tidak pada jumlah pekerja.

**Metrik:** AUC gabungan (200-skor setara: semua skor luar-lipatan satu pembagian) sebagai metrik primer, seperti Tabel kebocoran naskah; AUC per lipatan dicatat. Kebocoran = selisih berpasangan CR − C1 (poin) pada data dan lipatan yang sama.

## 3. Titik akhir dan uji

**Pemeriksaan reproduksi (bukan hipotesis):** sel n_ref = 80, jarang, s = 1 meniru TUEP; prediksi: aturan berpasangan "kebocoran" (selisih > 3 poin).

**Keluarga primer (Holm, 3 uji Welch satu sisi, α = 0.05):**
- P1 (ukuran referensi): kebocoran pada n_ref = 80 > pada n_ref = 1000 (jarang, s = 1).
- P2 (sel jarang): kebocoran jarang > cermin (n_ref = 80, s = 1).
- P3 (hubungan): kebocoran s = 1 > s = 0 (n_ref = 80, jarang).

**Sekunder (deskriptif):** keputusan aturan mutlak (AUC CR) dan berpasangan (CR − C1) per sel dengan ambang asli (`round49_kalibrasi.decide`, THR asli); AP − C1 per sel; kemiringan kebocoran terhadap log n_ref.

**Prediksi (ditulis sebelum run):** P1, P2, P3 terpenuhi; AP "tidak ada bukti" (aturan berpasangan) di semua sel atau hampir semua; pada s = 0 kebocoran CR dekat nol; pada n_ref = 1000 jarang s = 1 kebocoran < 2 poin.

**Pemeriksaan validitas:** sel s = 0: CR − C1 "tidak ada bukti" pada sebagian besar sel; AP − C1 rata-rata |selisih| < 1 poin.

## 4. Batas

Fitur normal multivariat (fitur I2 di-log, sehingga mendekati normal); hanya pengklasifikasi linear tingkat pasien; hanya desain OLS naskah; hasil tidak mencakup model normatif heteroskedastik, ComBat, atau referensi eksternal. Naskah hanya memuat titik akhir di atas.
