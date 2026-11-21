# PRA-REGISTRASI PUTARAN 61 (validasi dalam bersarang ketat untuk kontras referensi; ditulis SEBELUM menghitung; 2026-11-20)

Status analisis: **lanjutan** (sesudah putaran 3--60 diketahui). Perintah pemilik 2026-11-20 atas `review pak dwi/Laporan_Review_Editorial_JBI.pdf`, butir mayor 1. Pemilik memilih "Butir 1 saja" (tanpa referensi acak seimbang era).

## 0. Yang SUDAH dilihat sebelum dokumen ini ditulis

1. Putaran 29/56 (H1, 100 replikasi, benih 29100+r, AUC gabungan): C1 50.6, V0 56.0, VR 52.8, V4 50.4; V0 − C1 +5.4 poin (SD 2.5, R2 "kebocoran"); VR − C1 +2.2 (R2 "tidak pasti"); V4 − C1 −0.2 (R2 "tidak ada bukti"); VR − V0 −3.25 (95% CI −3.59 sampai −2.91).
2. Putaran 49 (nesting ketat model lain, label asli): AUC per lipatan setiap model berubah paling banyak 0.3 poin.
3. **Tidak ada** AUC V0, VR, V4, atau C1 di bawah nesting ketat yang pernah dihitung atau dilihat.

## 1. Pertanyaan

Di putaran 29/56, referensi penyesuaian dan `StandardScaler` dipasang pada seluruh pasien latih lipatan luar sebelum C dipilih oleh CV dalam. Untuk V0, referensi dipilih menurut label (kontrol latih), sehingga label pasien validasi dalam ikut menentukan referensi yang dipakai saat memilih C. Pasien uji luar tetap terpisah. Seberapa besar perubahan kontras V0 − C1 dan V0 − VR, dan keputusan aturan, bila pemilihan C dibuat bersih?

## 2. Metode

Data, pengacakan, lipatan luar, dan model identik dengan putaran 56 (H1: `round28_analisis.shuffle_cols` pada BP, kelompok semua pasien, benih `29100 + r`, r = 0..99; 50 lipatan kanonik; regresi logistik L2 berbobot kelas seimbang, C dari `evalkit.CS` (9 nilai); tanpa usia/jenis kelamin di masukan; penyesuaian `round29.fit_norm` dengan desain `D_I2`).

**Nesting ketat:** CV dalam StratifiedKFold(5) tanpa acak pada pasien latih luar (lipatan dalam sama dengan `LogisticRegressionCV`). Di tiap lipatan dalam k, referensi penyesuaian dan `StandardScaler` dipasang ulang hanya dari pasien latih-dalam:
- **C1**: tanpa penyesuaian (hanya scaler yang dipasang ulang);
- **V0**: referensi = kontrol latih-dalam;
- **VR**: referensi = subset acak pasien latih-dalam (tanpa memandang label) berukuran sama dengan jumlah kontrol latih-dalam, benih `61000 + 10·lipatan + k`, tetap di semua replikasi (seperti VR putaran 56 yang tetap per lipatan);
- **V4**: referensi = semua pasien latih-dalam.

C dipilih sebagai argmaks rata-rata AUC dalam (jalur C menaik dengan inisialisasi hangat, seperti `round49_nesting_ketat.nested`). Model akhir dipasang pada semua pasien latih luar dengan referensi luar putaran 56 (V0: kontrol latih; VR: `round56.ref_random(tr, f)`; V4: semua pasien latih) dan scaler luar.

**Mode longgar dengan kode yang sama:** semua 100 replikasi juga dihitung di mode longgar (referensi dan scaler luar, jalur C dan refit yang sama). Pembanding "asli" untuk kontras utama adalah mode longgar ini, agar selisih implementasi (refit `LogisticRegressionCV` dengan inisialisasi hangat; di `--check` replikasi 0: C1 0.0021, V0 0.00003, VR 0.0002, V4 0.0002) tidak tercampur ke dalam efek nesting. Angka putaran 56 dilaporkan berdampingan sebagai rujukan naskah.

**Pemeriksaan sebelum run (`--check`):** (a) mode longgar (referensi dan scaler luar, jalur C yang sama) mereproduksi AUC gabungan replikasi 0 putaran 56 untuk keempat varian (|selisih| < 0.005); (b) di mode ketat, referensi dan scaler tiap lipatan dalam tidak memuat pasien validasi dalam; (c) referensi VR dalam berukuran sama dengan jumlah kontrol latih-dalam. AUC mode ketat tidak dicetak di `--check`.

## 3. Titik akhir

**Kontras utama (dua, dibekukan, Holm atas 2):** perubahan akibat nesting ketat, berpasangan atas 100 replikasi:
- Δ(V0 − C1) = (V0 − C1)_ketat − (V0 − C1)_longgar;
- Δ(V0 − VR) = (V0 − VR)_ketat − (V0 − VR)_longgar.
Masing-masing uji t satu sampel dua sisi dan 95% CI.

**Dilaporkan (deskriptif, dengan 95% CI berpasangan atas replikasi):** kontras ketat V0 − C1, V0 − VR, VR − C1, V4 − C1; AUC gabungan dan per lipatan keempat varian; keputusan R1 (mutlak, per varian) dan R2 (berpasangan, terhadap C1) dengan ambang asli `round49_kalibrasi`, sebelum (putaran 56 dan mode longgar) dan sesudah (mode ketat); sebaran C terpilih.

**Prediksi (ditulis sebelum run):** |Δ| < 1 poin untuk kedua kontras; V0 − C1 tetap "kebocoran" menurut R2; V0 − VR tetap positif dengan CI tidak memuat 0.

**Tafsir yang ditetapkan sebelumnya:** naskah melaporkan kedua estimasi (asli dan ketat) di suplemen beserta keputusan sebelum/sesudah. Bila salah satu kontras utama berubah lebih dari 0.1 poin, angka klaim (ii) di Abstrak, Hasil, Pembahasan, Kesimpulan, dan surat pengantar diganti dengan estimasi ketat. Bila keputusan berubah, klaim (ii) dirumuskan ulang mengikuti keputusan ketat. Hasil lain dilaporkan apa adanya.

## 4. Batas

Satu arsip, satu representasi (I2, kolom diacak), satu pengklasifikasi linear; lipatan dalam tidak diacak; satu referensi acak per lipatan dalam.
