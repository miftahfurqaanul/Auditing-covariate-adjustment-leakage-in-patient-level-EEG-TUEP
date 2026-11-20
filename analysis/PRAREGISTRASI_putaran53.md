# PRA-REGISTRASI PUTARAN 53 (regresi logistik pada fitur mentah yang sama dengan pohon; ditulis SEBELUM menghitung; 2026-11-08)

Status analisis: **lanjutan** (sesudah putaran 3--52 diketahui). Perintah pemilik 2026-11-08 ("iyaa lakukan dengan teliti dan detail") atas usulan butir 1 dan 2 dari telaah klaim naskah.

## 0. Yang SUDAH dilihat sebelum dokumen ini ditulis

1. Seluruh hasil plasmode putaran 42 dan 50 di naskah, termasuk: kebocoran penyesuaian mentah per pasien pada pohon (TUEP RF +11.1, XGB +11.6 poin; Emotiv RF +12.7, XGB +11.6), sisa log/rank (TUEP 0.7--0.9; Emotiv 1.0--2.7), LR tingkat pasien I2 (raw-pat +1.0 poin; rank-pat $-$0.3) dan LR tingkat subjek Emotiv (paling banyak 0.3 poin).
2. Telaah klaim 2026-11-08 menemukan bahwa di kedua kohort LR memakai fitur yang **sudah di-log** (TUEP: I2, log daya relatif; Emotiv: median log daya per subjek), sedangkan RF/XGB memakai daya pita **mentah** jendela Rohira. Perbandingan "linear vs nonlinear" pada data nyata karena itu tercampur dengan transformasi; hanya simulasi (LR maksimum +2.6 poin pada 134 sel) memakai fitur yang sama.
3. Hasil putaran 51--52 (penawar ash) ada tetapi tidak relevan untuk pertanyaan ini dan tidak dipakai.
4. **Tidak ada** AUC LR pada fitur mentah jendela Rohira (TUEP atau Emotiv) yang pernah dihitung atau dilihat.

## 1. Pertanyaan

Apakah kebocoran kecil pada regresi logistik di plasmode berasal dari linearitas pengklasifikasi atau dari transformasi log fiturnya? Dijawab dengan menjalankan LR pada **fitur mentah yang identik** dengan RF/XGB, dengan label sintetis, pertukaran, lipatan, pasien latih, dan penyesuaian yang identik per replikasi, sehingga hanya pengklasifikasi yang berbeda.

## 2. Metode

Kode putaran 42 (`round42_common`, `round42_cpu`) dan 50 (`round50_emotiv`) diimpor tanpa diubah; label (`gen_labels`), pertukaran H1 (`gen_perm`), lipatan (`folds_for`), pasien latih 80% (`split_val` dengan benih `plasmode_clf`/`emotiv_clf` yang sama dengan RF), transformasi (`make_transform`, dipasang pada jendela latih) dan penyesuaian (`adjust_units`) sama persis dengan RF.

**LRw (primer; tingkat jendela, sama dengan RF):** fitur jendela Rohira mentah (TUEP 48; Emotiv 60), pipeline raw-none, raw-win, raw-pat, symlog-pat, log-pat, rank-pat. Pengklasifikasi: StandardScaler + LogisticRegressionCV L2 (`Cs = evalkit.CS`, 9 nilai $10^{-3}$--$10^{1}$; `class_weight="balanced"`; `scoring="roc_auc"`; `max_iter=5000`), dipasang pada jendela pasien latih 80% yang sama dengan RF; C dipilih dengan validasi silang dalam 5 lipatan **berkelompok per pasien** (StratifiedGroupKFold, benih `lrw_inner|rep|gamma|lipatan`). Skor pasien = rata-rata probabilitas jendela (sama dengan RF).

**LRp (sekunder; tingkat pasien):** median fitur jendela Rohira mentah per pasien (sama dengan cara I2 meringkas jendela), lalu transformasi (raw/log/rank, dipasang pada pasien latih) dan penyesuaian per pasien (none/pat); `evalkit.nested_lr` pada semua pasien latih (sama dengan LR putaran 42). Pipeline raw-none, raw-pat, log-none, log-pat, rank-none, rank-pat.

**Replikasi (H1 saja):** TUEP: replikasi 0--119 untuk gamma 0 dan 1 (dipasangkan dengan 120 replikasi RF/XGB putaran 42+50A), 0--29 untuk gamma 0.5. Emotiv: replikasi 0--29 untuk gamma 0, 0.5, 1 (dipasangkan dengan RF 30 dan XGB 20 putaran 50B).

**Metrik:** AUC per lipatan (rata-rata 5 lipatan), seperti naskah. Selisih berpasangan per replikasi dalam poin.

**Aturan keputusan:** R1 dan R2 dengan ambang asli, memakai `round49_kalibrasi.dec_pool` (tidak diubah).

## 3. Titik akhir dan uji

- **E1 (primer, deskriptif dengan aturan):** selisih LRw raw-pat minus raw-none pada gamma 1, per kohort: rata-rata, SD, CI 95% (t), keputusan R2; keputusan R1 untuk LRw raw-pat.
- **E2 (famili primer inferensial, Holm dua uji):** interaksi I = (RF raw-pat minus raw-none) minus (LRw raw-pat minus raw-none) pada gamma 1, berpasangan per replikasi; uji t satu sampel **satu sisi** (H1: I > 0); TUEP n = 120, Emotiv n = 30.
- **Sekunder (Holm dua uji):** interaksi yang sama dengan XGB (TUEP n = 120, Emotiv n = 20).
- **Sekunder deskriptif:** LRp raw-pat minus raw-none, log-pat minus log-none, rank-pat minus rank-none (gamma 1, kedua kohort); LRw symlog/log/rank-pat minus raw-none dan minus transformasi yang sama tanpa penyesuaian (tidak dijalankan untuk LRw, jadi hanya terhadap raw-none); raw-win; gamma 0.5.
- **Kontrol:** pada gamma 0, LRw raw-pat dan LRp raw-pat tidak berkeputusan R2 "kebocoran" di kedua kohort.

## 4. Aturan tafsir (ditetapkan sekarang)

- Bila E1 bukan "kebocoran" (R2) di kedua kohort **dan** E2 lolos Holm < 0.05 di kedua kohort: klaim "bergantung pada pengklasifikasi" didukung pada fitur yang identik; naskah boleh menyebut LR pada fitur mentah yang sama.
- Bila LRw berkeputusan R2 "kebocoran" di salah satu kohort: klaim "LR paling banyak 1.0 poin" dan "mitigasi bergantung pada pengklasifikasi" harus dibatasi; perlindungan linear tidak berlaku bila skor pasien adalah rata-rata probabilitas jendela.
- Bila LRp raw-pat berkeputusan R2 "kebocoran" sedangkan LR I2 (log) tidak: kebocoran kecil LR I2 dijelaskan oleh transformasi log, bukan oleh linearitas.
- Hasil putaran ini masuk naskah hanya bila menambah ketepatan klaim; keputusan pada pemilik.

## 5. Prediksi (peluang subjektif)

- P53-1: LRw TUEP raw-pat pada gamma 1 tidak berkeputusan R2 "kebocoran" (55%).
- P53-2: LRw Emotiv raw-pat pada gamma 1 tidak berkeputusan R2 "kebocoran" (55%).
- P53-3: E2 lolos Holm < 0.05 di kedua kohort (80%).
- P53-4: LRp raw-pat TUEP |selisih| <= 1.0 poin atau R2 "tidak ada bukti"/"tidak pasti" (70%).
- P53-5: kontrol gamma 0 lulus di kedua kohort untuk LRw dan LRp (85%).

## 6. Pemeriksaan sebelum run (mode --check; tanpa mencetak AUC)

1. Reproduksi RF putaran 42 replikasi 0, H1, gamma 0 dengan kode yang diimpor (selisih maksimum < 1e-9), dan RF Emotiv putaran 50B replikasi 0, gamma 0.
2. Label, pertukaran, dan lipatan LRw identik dengan RF untuk replikasi uji.
3. Invariansi terhadap label uji (membalik label pasien uji tidak mengubah prediksi LRw dan LRp).
4. Lipatan dalam LRw tidak memuat pasien yang sama di latih dan validasi dalam.
5. Waktu satu unit LRw (dicetak detik saja).

## 7. Komputasi

CPU saja (CUDA disembunyikan), 12 pekerja (joblib), satu proses. Keluaran `hasil53/LRw_tuep.csv`, `LRw_emotiv.csv`, `LRp_tuep.csv`, `LRp_emotiv.csv` (dapat dilanjutkan). Ringkasan `round53_ringkas.py` dan pemeriksaan independen `round53_cek_independen.py` ditulis dan di-hash SEBELUM keluaran dilihat.

## Adendum 1 (ditulis SESUDAH keluaran pertama `round53_ringkas.py` terlihat; 2026-11-08)

Keluaran pertama memperlihatkan E1 dan E2, lalu berhenti di bagian prediksi. Dua bug skrip ringkasan diperbaiki agar sesuai rencana bagian 2--3; tidak ada titik akhir, uji, ambang, atau data yang diubah:
1. `prep()` membuang duplikat dengan kunci (rep, gamma, pipe) tanpa `cond`. Karena `hasil42/RF.csv` dan `XGB.csv` memuat H1, H2a, dan nyata, baris H1 replikasi awal (RF 0--29, XGB 0--19) tertimpa sebelum disaring, sehingga E2 TUEP memakai n = 90 (RF) dan 100 (XGB) alih-alih 120 yang direncanakan. Kunci kini memuat `cond`, dan jumlah pasangan per sel diperiksa (assert 120/120/30/20). E1 (berkas LRw/LRp, hanya H1) tidak terdampak.
2. `E1.pipe` adalah metode pandas, bukan kolom (kesalahan yang sama dengan putaran 50); kini `E1["pipe"]`.
Salinan sebelum perbaikan: `hasil53/round53_ringkas.sebelum_perbaikan.py`; keluaran pertama: `hasil53/ringkas.log`.

## Adendum 2 (PENYAJIAN, DI LUAR rencana; ditulis SESUDAH keluaran putaran 53 terlihat; 2026-11-08)

Pemilik memutuskan hasil putaran 53 masuk naskah (pelaporan jujur 53 putaran; LR ditambahkan ke Gambar 4). `round53_bahan_naskah.py` (di-hash SEBELUM dijalankan) menulis baris Tabel plasmode, tabel interaksi S2.17, dan data Gambar 4 dari CSV tersimpan, dengan pembulatan setengah ke atas; tidak ada model yang dilatih dan tidak ada angka yang berubah.
