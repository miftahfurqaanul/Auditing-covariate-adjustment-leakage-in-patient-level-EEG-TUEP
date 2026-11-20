# PRA-REGISTRASI PUTARAN 54 (uji proposisi ketidakcocokan pemasangan pada regresi logistik tingkat jendela; ditulis SEBELUM menghitung; 2026-11-12)

Status analisis: **lanjutan** (sesudah putaran 3--53 diketahui). Perintah pemilik 2026-11-12 ("iyaa silahkan anda perbaiki semua") atas telaah sejawat butir M1, yaitu proposisi yang menyatukan kebocoran tingkat jendela dan kebocoran berbasis kontrol, beserta satu uji kecil.

## 0. Yang SUDAH dilihat sebelum dokumen ini ditulis

1. Hasil putaran 53 di naskah (Tabel S2.12, Gambar 3), pada gamma 1:
   - LRw (regresi logistik tingkat jendela, fitur jendela Rohira mentah, penyesuaian per pasien) bocor +6.6 poin di TUEP dan +1.2 poin di Emotiv;
   - LRp (median per pasien) +0.2 di TUEP dan +1.1 di Emotiv;
   - symlog, log, dan rank pada LRw TUEP +0.7 sampai +1.0.
2. Tabel 4 dan S3.3 (V0--V4), beserta rencana putaran 29. Pada V1, separuh kontrol latih hanya dipakai memasang B dan tidak ikut melatih LR.
3. Kode `round42_common.adjust_units` dan `round53_lr_mentah.lrw_clf`:
   - penyesuaian "pat" adalah OLS tanpa bobot atas rata-rata fitur per pasien, dipasang pada **semua** pasien latih lipatan (tr);
   - LRw dilatih pada **80% pasien latih** (`split_val`, sama dengan RF), dengan setiap **jendela berbobot sama**, sehingga pasien berbobot sebanding jumlah jendelanya (TUEP 12--411 per pasien);
   - jadi sampel dan bobot pengklasifikasi berbeda dari sampel dan bobot pemasangan penyesuaian.
4. Baris LRw "raw-win" putaran 53 (penyesuaian OLS per jendela) dihitung dan dicetak di `hasil53/ringkas.log`. Penulis rencana ini **tidak** membacanya, dan baris itu tidak ada di naskah.
5. **Belum ada** AUC putaran 54 yang dihitung.

## 1. Proposisi dan pertanyaan

**Proposisi (aljabar OLS).**
- Misalkan B-hat dipasang dengan kuadrat terkecil pada himpunan pemasangan F dengan bobot w. Maka residual z_q = x_q - B-hat' d_q memenuhi sum_{q in F} w_q z_q d_q' = 0.
- Pengklasifikasi linear yang dilatih pada himpunan T dengan bobot v hanya "melihat" informasi kovariat linear lewat sum_{q in T} v_q z_q d_q'.
- Besaran itu nol bila (T, v) = (F, w), dan pada umumnya tidak nol bila berbeda.
- Pada uji nol (fitur tanpa hubungan dengan kovariat), arah kovariat yang tersisa dalam data latih berkorelasi dengan label melalui hubungan label--kovariat, dan arah yang sama (B-hat yang sama) ada pada data uji.

**Prediksi proposisi terhadap hasil yang sudah ada.**
- V0 (F = kontrol, T = semua pasien latih), V1 (F terpisah dari T), dan V2 (cross-fitting) tidak cocok, dan ketiganya bocor.
- V4 (F = T, bobot sama) cocok, dan tidak bocor.
- LRp (F = T = pasien latih, bobot sama) cocok, dan tidak bocor.
- LRw tidak cocok dalam sampel (80% lawan 100%) dan dalam bobot (jendela lawan pasien), dan bocor +6.6 poin di TUEP.

**Pertanyaan putaran ini.** Apakah kebocoran LRw di TUEP hilang bila sampel dan bobot pengklasifikasi diselaraskan dengan pemasangan penyesuaian? Alternatifnya, kebocoran tetap ada karena sebab lain (perata-rataan probabilitas jendela yang nonlinear, atau ekor berat yang memberi titik pengaruh besar pada fungsi rugi logistik).

## 2. Metode

Kode putaran 42/50/53 diimpor tanpa diubah. Label, pertukaran H1, lipatan, transformasi "raw", penyesuaian per pasien (`adjust_units(..., "pat")`), benih, dan lipatan dalam berkelompok per pasien sama dengan LRw putaran 53.

Empat lengan LRw (StandardScaler + LogisticRegressionCV L2, `Cs = evalkit.CS`, `class_weight="balanced"`, skor patien = rata-rata probabilitas jendela uji), masing-masing untuk raw-none dan raw-pat:

| Lengan | Pasien latih pengklasifikasi | Bobot jendela |
|---|---|---|
| A0 | 80% (`split_val`, = putaran 53) | sama (= putaran 53) |
| W | 80% | 1/n_p per jendela (setiap pasien berbobot total sama) |
| S | semua pasien latih lipatan (= himpunan pemasangan penyesuaian) | sama |
| WS | semua pasien latih lipatan | 1/n_p |

Rincian:
- Bobot dinormalkan agar rata-ratanya 1 atas jendela latih, supaya skala penalti C sebanding.
- Lipatan dalam untuk memilih C tetap berkelompok per pasien atas pasien latih lengan itu.
- Benih lipatan dalam sama dengan putaran 53 (`lrw_inner`).
- Pada WS, untuk raw-pat, sum_p (1/n_p) sum_w z_pw d_p' = sum_p zbar_p d_p' = 0 atas tr, karena penyesuaian per pasien dipasang pada rata-rata pasien tr. Kondisi ini diperiksa numerik di mode --check.

Lengan A0 harus mereproduksi AUC per lipatan LRw raw-none dan raw-pat putaran 53 untuk setiap unit (selisih maksimum < 1e-9).

Replikasi (H1 saja):
- TUEP: replikasi 0--119 pada gamma 0 dan 1;
- Emotiv: replikasi 0--29 pada gamma 0 dan 1.

Metrik: AUC per lipatan (rata-rata 5 lipatan), dan selisih berpasangan raw-pat minus raw-none per replikasi, dalam poin.

## 3. Titik akhir dan uji

- **P1 (primer, keputusan aturan):** pada lengan WS, TUEP, gamma 1: selisih raw-pat minus raw-none (rata-rata, SD, CI 95% t) dan keputusan R2 (`round49_kalibrasi.dec_pool`, ambang asli).
- **P2 (famili primer inferensial, Holm dua uji, satu sisi):**
  - (a) TUEP, gamma 1: kebocoran A0 minus kebocoran WS > 0, berpasangan per replikasi (n = 120);
  - (b) Emotiv, gamma 1: hal yang sama (n = 30).
- **Sekunder (deskriptif):**
  - kebocoran lengan W dan S (dekomposisi: bobot saja, sampel saja), dengan keputusan R2, di kedua kohort;
  - pada mode --check: besar sum_p v_p zbar_p d_p' (norma Frobenius dibagi jumlah bobot) untuk A0, W, S, dan WS pada satu replikasi.
- **Kontrol:** pada gamma 0, keempat lengan raw-pat tidak berkeputusan R2 "kebocoran" di kedua kohort.

## 4. Aturan tafsir (ditetapkan sekarang)

- **Bila WS TUEP berkeputusan R2 "tidak ada bukti" dan P2(a) lolos Holm < 0.05:** proposisi didukung untuk LRw. Naskah boleh menyatakan bahwa kebocoran pengklasifikasi linear pada uji nol muncul ketika sampel atau bobot pengklasifikasi berbeda dari pemasangan penyesuaian, dan bahwa penyelarasan menghilangkannya di TUEP. Rekomendasi (1) dan (2) di Pembahasan diperbarui.
- **Bila WS TUEP berkeputusan R2 "kebocoran":** proposisi tidak cukup menjelaskan kebocoran LRw. Naskah mempertahankan dugaan perata-rataan atau ekor berat dan melaporkan hasil ini sebagai bukti yang menolak penjelasan ketidakcocokan saja.
- **Bila WS "tidak pasti":** naskah melaporkan penurunan (P2) apa adanya tanpa menyatakan kebocoran hilang.
- **Bila salah satu kontrol gamma 0 gagal:** kesimpulan untuk lengan itu dibatasi seperti putaran 50.
- Emotiv hanya sekunder, karena kebocoran A0 di sana kecil (+1.2 poin); daya P2(b) rendah.

## 5. Prediksi (peluang subjektif)

- P54-1: WS TUEP gamma 1 berkeputusan R2 "tidak ada bukti" (55%).
- P54-2: P2(a) lolos Holm < 0.05 (80%).
- P54-3: kebocoran lengan W (bobot saja) TUEP lebih kecil daripada kebocoran lengan S (sampel saja) (55%).
- P54-4: kontrol gamma 0 lulus untuk keempat lengan di kedua kohort (85%).
- P54-5: |kebocoran WS Emotiv| <= 1.0 poin (65%).

## 6. Pemeriksaan sebelum run (mode --check; tanpa mencetak AUC)

1. A0 mereproduksi LRw raw-none dan raw-pat putaran 53, replikasi 0, gamma 0, kedua kohort (selisih maksimum < 1e-9 pada AUC per lipatan).
2. Pada WS, raw-pat: norma sum_p zbar_p d_p' atas tr relatif terhadap norma sum_p xbar_p d_p' < 1e-8. Norma relatif yang sama untuk A0, W, dan S dicetak, dan diharapkan tidak nol.
3. Invariansi terhadap label uji: membalik label pasien uji tidak mengubah prediksi keempat lengan.
4. Lipatan dalam berkelompok per pasien: tidak ada pasien yang sama di latih dan validasi dalam.
5. Bobot per pasien pada W dan WS berjumlah sama untuk setiap pasien (sebelum normalisasi global).
6. Waktu satu unit (dicetak detik saja).

## 7. Komputasi

- CPU saja (CUDA disembunyikan), satu proses, 8 pekerja joblib (mesin dipakai bersama).
- Keluaran `hasil54/LRw4_tuep.csv` dan `hasil54/LRw4_emotiv.csv` (dapat dilanjutkan).
- Ringkasan `round54_ringkas.py` di-hash sebelum run.
- Pemeriksaan independen: reproduksi A0 terhadap seluruh unit putaran 53 yang dipasangkan, dihitung di dalam `round54_ringkas.py` dari CSV tersimpan.
- Hasil masuk naskah sesuai aturan tafsir bagian 4; pemilik sudah menyetujui bahwa uji ini dilaporkan (2026-11-12).

## Adendum 1 (ditulis SESUDAH mode --check pertama dan SEBELUM run; belum ada AUC putaran 54 yang dilihat; 2026-11-12)

**Temuan.** Pemeriksaan 1 (reproduksi A0 < 1e-9 terhadap CSV putaran 53) gagal untuk TUEP: selisih maksimum 0.0025 pada satu AUC lipatan. Emotiv lolos dengan selisih 0.

**Diagnosis.** Fungsi asli `round53_lr_mentah.unit("tuep", 0, 0.0)`, dijalankan sekarang dengan satu utas BLAS, juga tidak mereproduksi `hasil53/LRw_tuep.csv` secara bit-identik:
- raw-none identik;
- raw-pat berselisih −0.0025 pada lipatan 4;
- raw-win berselisih 0.0025 dan 0.005 pada lipatan 3 dan 4.

Satu langkah AUC pada lipatan 40 pasien adalah 1/400 = 0.0025. Jadi selisih itu setara satu atau dua pasang pasien yang urutannya bertukar. Lengan A0 identik dengan fungsi putaran 53 yang dijalankan sekarang. Sumber ketidakidentikan ada pada nilai TUEP putaran 53 yang tersimpan (lingkungan run sebelumnya), bukan pada kode putaran 54.

**Perubahan (tidak ada titik akhir, uji, ambang, atau data yang diubah):**
1. Pemeriksaan 1 kini mensyaratkan dua hal:
   - (a) A0 identik (< 1e-12) dengan `round53_lr_mentah.unit` yang dijalankan sekarang;
   - (b) selisih A0 terhadap CSV putaran 53 < 0.01 pada setiap AUC lipatan.
2. Semua titik akhir memakai A0 yang dihitung ulang dalam run ini, bukan nilai tersimpan putaran 53.
3. `round54_ringkas.py` melaporkan reproduksi A0 terhadap CSV putaran 53 untuk semua unit secara deskriptif: selisih maksimum, jumlah AUC lipatan yang berbeda, dan kebocoran raw-pat TUEP gamma 1 dari kedua sumber. Syaratnya selisih per lipatan < 0.01.
4. Bila kebocoran raw-pat TUEP gamma 1 hasil hitung ulang berbeda dari +6.6 poin setelah pembulatan, naskah memakai nilai tersimpan putaran 53 untuk putaran 53. Selisihnya dicatat di S2.
