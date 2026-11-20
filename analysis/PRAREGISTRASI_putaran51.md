# PRA-REGISTRASI PUTARAN 51 (butir 5(c): mekanisme kebocoran penyesuaian kovariat dan penawarnya; ditulis SEBELUM menghitung keluaran utama; 2026-11-03)

Status analisis: **lanjutan** (sesudah seluruh hasil putaran 3--50 diketahui). Arah disetujui pemilik 2026-11-03 ("iyaa setuju, kerjakan dengan profesional dan teliti").

## 0. Yang SUDAH dilihat sebelum dokumen ini ditulis

1. Seluruh hasil plasmode putaran 42 dan 50, termasuk:
   - kebocoran penyesuaian mentah pada pohon (TUEP RF +11.1, XGB +11.6; Emotiv RF +12.7, XGB +11.6);
   - sisa log/rank (TUEP 0.7--0.9 poin; Emotiv 1.0--2.7 poin);
   - LR Emotiv tanpa kebocoran (raw-pat +0.27, p = 0.59).
2. **Diagnostik eksploratif** (skrip di luar proyek, `scratchpad/diag_injeksi.py`; TUEP, H1, 4 replikasi x 5 lipatan; bukan bagian rencana ini dan tidak dilaporkan sebagai bukti). Ukurannya rho Spearman antara prediksi RF-regresor jendela -> ETA dan ETA pada pasien uji:

   | Pipeline | rho |
   |---|---|
   | raw-none | -0.05 |
   | raw-OLS | 0.945 |
   | raw-LAD | 0.81 |
   | raw-EB (versi skala-mentah) | 0.45 |
   | log-OLS | 0.06 |
   | rank-OLS | 0.08 |
   | log-EB | -0.05 |
   | rank-EB | -0.04 |

   Besar suntikan median 18 x MAD inti pada fitur mentah. Kurtosis median fitur jendela mentah TUEP 8486 (log: 3.0).
3. Statistik t OLS kovariat (tanpa label):
   - BP TUEP (fitur I2/N1; lipatan luar 0, 160 pasien): usia 5/114 fitur |t| > 2, maks 2.25, SD(t) 1.06; jenis kelamin 9/114, maks 2.68; era maks 2.49.
   - Faktor ash median ~0 untuk semua kolom (lipatan 0); EB normal tunggal lambda usia 0.14.
   - Log rata-rata-per-pasien fitur jendela TUEP (semua 200 pasien): usia maks |t| 1.4; dua kolom era 24/48 dan 17/48 fitur |t| > 3.
   - Emotiv: negara 58/60 fitur |t| > 3 (maks 21.9); hari z 25/60.
4. Saat menguji `round51_ringkas.py` dengan prediksi palsu, bagian E3b versi awal (kovariat D lengkap) menghitung model lama dengan prediksi nyata.
   - Delta R2 (p permutasi): Floor 0.396, C1 0.156, I2 0.146, P3 0.119, NM 0.037 (p 0.20), F2 0.107, N0 0.523, N1 0.138.
   - Karena usia dan jenis kelamin adalah fitur masukan AS model-model ini, uji itu trivial. E3b kini dibatasi ke dummy era dan model tanpa ACQ (bagian 5). Nilai era untuk model lama BELUM dilihat.
5. Pemeriksaan kode (`--check`) lulus. Reproduksi pipeline lama identik:
   - RF TUEP rep 0 vs `hasil42/RF.csv` dan RF Emotiv vs `hasil50/B_RF.csv`: 5.6e-17;
   - Bi-LSTM rep 0 raw-none/raw-pat vs `hasil42/Tang.csv`: 5.6e-17.

   Tidak ada AUC plasmode putaran 51 atau AUC label nyata dengan penyesuaian ash yang dihitung atau dilihat.

## 1. Celah dan hipotesis mekanisme

Naskah putaran 50 menunjukkan bahwa penyesuaian kovariat pada fitur mentah menciptakan diskriminasi palsu pada model pohon, dan bahwa log/rank-normal hampir menghilangkannya, tetapi tidak menjelaskan mengapa. Hamdan dkk. (2023) menyebut fitur miring dan presisi terbatas sebagai mekanisme dan tidak menemukan penawar yang tuntas.

**Hipotesis H-mek (suntikan galat estimasi bersama).** Penyesuaian per pasien mengurangkan D_p B-hat dari semua jendela pasien p. Galat (B-hat - B) adalah fungsi linear kovariat yang sama untuk semua jendela pasien itu. Ini analog dengan galat estimasi batch yang "affects all values within the same batch in the same manner" (Nygaard dkk. 2016, Biostatistics; genomik). Di bawah H1 (B = 0) seluruh D_p B-hat adalah suntikan.

Pada fitur berekor berat, B-hat ditarik ekor sehingga suntikan besar relatif terhadap sebaran inti tempat pohon membelah. Model linear didominasi varians ekor. Akibatnya label yang bergantung pada kovariat dapat diprediksi dari fitur yang "disesuaikan". Zindler dkk. (2020, BMC Bioinformatics; metilasi DNA) melaporkan positif palsu ComBat pada data tanpa sinyal yang berkurang dengan n dan bertambah dengan jumlah level faktor, sesuai mekanisme ini.

**Penawar yang diuji.** Koefisien penyesuaian disusutkan dengan empirical Bayes adaptif sebelum dikurangkan. Bila efek kovariat sebenarnya tidak terdeteksi, koefisien menuju 0 sehingga tidak ada suntikan; efek kuat dipertahankan.

## 2. Metode penyesuaian (semua dipasang HANYA pada pasien latih; `round51_common.py`)

Notasi: M = rata-rata fitur per pasien latih (n x p, setelah transformasi), D_t = desain pasien latih (kolom 0 intersep).

- **pat (OLS)**: B = lstsq(D_t, M). Identik dengan putaran 42/50 (diperiksa).
- **eb (normal tunggal, pembanding)**: untuk tiap kolom kovariat k >= 1:
  - beta*_kj = B_kj / sigma_j, dengan sigma_j = SD residu OLS (derajat bebas n - rank);
  - V = (D_t' D_t)^-1;
  - tau2_k = max(0, rata-rata_j beta*_kj^2 - V_kk);
  - B_kj <- tau2_k/(tau2_k + V_kk) B_kj.
- **ash (utama)**: seperti eb, tetapi tiap beta*_kj disusutkan ke rataan posteriornya di bawah prior campuran normal berpusat 0 (Stephens 2017, Biostatistics; default paket ashr, dibaca dari kode sumbernya):
  - massa titik di 0 + N(0, s_m^2);
  - grid s: sigmaamin = se/10; sigmaamax = 2 sqrt(max(beta*^2 - se^2)) (atau 8 sigmaamin); rasio sqrt(2);
  - se = sqrt(V_kk);
  - bobot campuran dengan EM berpenalti Dirichlet, nullweight 10 pada komponen nol;
  - rataan posterior sum_m w_m(beta*) s_m^2/(s_m^2 + se^2) beta*.
- Pada eb dan ash, kolom tak teridentifikasi (V_kk = 0) diberi koefisien 0, dan intersep dipasang ulang sehingga rata-rata residu pasien latih = 0.

Ash dipilih sebagai utama SEBELUM hasil karena eb normal tunggal merata-ratakan varians prior lintas fitur, sehingga menyusutkan berlebihan fitur dengan efek kuat bila sebagian besar fitur tanpa efek (uji kode: galat koefisien ash 0.047 vs eb 0.082 vs OLS 0.084 pada efek jarang yang disimulasikan).

## 3. E2: plasmode dengan penawar (`round51_plasmode.py`, `round51_tang.py`)

**Kode lama tidak diubah.** Label sintetis (gamma * ETA + logistik), permutasi, lipatan StratifiedKFold(5), pembagian validasi 20%, RF (100 pohon) dan XGB (GPU), skor pasien = rata-rata probabilitas jendela, dan semua benih diimpor apa adanya dari putaran 42 (TUEP) dan 50B (Emotiv).

**Kondisi:**
- **H1**: blok jendela ditukar antar-pasien (fitur independen dari kovariat; B = 0).
- **E (baru; efek kovariat sebenarnya diketahui)**: H1 dengan permutasi yang sama, lalu jendela pasien target p dikalikan exp((D_p - rata-rata D) B_T).
  - B_T = OLS log rata-rata-per-pasien fitur jendela NYATA pada D, semua pasien (tanpa label).
  - Efek ini aditif dalam skala log dan berukuran nyata: TUEP efek era sedang; Emotiv efek negara besar.
  - Penyesuaian yang benar membawa AUC ke ~0.50. Kekurangan penyesuaian (penyusutan berlebihan) dan suntikan sama-sama menaikkan AUC.
- **H2a (TUEP)**: penukaran di dalam era (hubungan era-fitur alami dipertahankan).

**Pipeline (9):**
- raw-none;
- raw-pat, log-pat, rank-pat;
- raw-ash, log-ash, rank-ash;
- log-eb, rank-eb.

**Replikasi:**
- TUEP RF dan XGB: H1 dan E rep 0--119, gamma {0, 1}; H2a rep 0--29, gamma {0, 1}.
- Emotiv RF dan XGB: H1 dan E rep 0--119, gamma {0, 1}.

**Bi-LSTM (Tang; sekunder, dijalankan sesudah rantai XGB bila waktu GPU memungkinkan):** H1, rep 0--19, gamma {0, 1}; pipeline raw-none, raw-pat, rank-pat, raw-ash, rank-ash.

**Metrik:** AUC rata-rata 5 lipatan per replikasi. Aturan keputusan R1/R2 dengan ambang asli putaran 42 (tidak diubah); R2 hanya pada H1 (selisih terhadap raw-none).

**Keluarga primer (per kohort, Holm atas 4 uji; t satu sisi atas selisih per replikasi pada gamma = 1):**
- P-a (RF), P-b (XGB): H1, d = AUC(rank-pat) - AUC(rank-ash) > 0. Penawar menghapus sisa kebocoran rank-normal.
- P-c (RF), P-d (XGB): E, non-inferioritas, H0: AUC(rank-ash) - AUC(rank-pat) >= 0.01 (margin 1 poin AUC) melawan < 0.01. Penawar tidak kurang menyesuaikan secara bermakna.

Klaim "ash menghapus sisa kebocoran tanpa kekurangan penyesuaian" untuk satu kohort dan satu famili hanya bila uji H1 dan E famili itu keduanya lulus (Holm < 0.05).

**Sekunder (Holm per kohort x famili, 4 uji, satu sisi):**
- H1 raw-pat - raw-ash > 0;
- H1 log-pat - log-ash > 0;
- E rank-eb - rank-ash > 0 (normal tunggal kurang menyesuaikan dibanding ash);
- H1 rank-eb - rank-ash > 0.

Semua sel lain (H2a, gamma 0, AUC level, R1/R2) deskriptif.

**Kontrol:**
- K1: pada gamma = 0, >= 95% sel tersesuaikan tidak berkeputusan R1 "kebocoran".
- K2 (kontrol positif kondisi E): raw-none pada E, gamma = 1, AUC > 0.55. Bila gagal, kondisi E dinyatakan tidak informatif untuk kohort itu dan P-c/P-d tidak ditafsirkan.

## 4. E1: hukum keterpulihan kovariat (tanpa label; `round51_plasmode.py --rho`)

Indeks: untuk rep 0--29 dan setiap kondisi (TUEP: H1, E, H2a; Emotiv: H1, E) dan pipeline, RF-regresor (100 pohon, min_samples_leaf 5) dilatih pada jendela pasien latih untuk memprediksi ETA (prediktor linear kovariat). rho = Spearman antara prediksi out-of-fold rata-rata per pasien dan ETA (KFold(5), benih `kseed("r51_rho", kohort, rep)`). Label tidak dipakai.

Uji:
- Spearman antar-sel (kondisi x pipeline) antara rata-rata rho dan rata-rata AUC gamma = 1, per kohort x famili;
- AUC maksimum di antara sel dengan rho <= 0.10.

## 5. E3: label nyata (`round51_nyata.py`, `round51_ringkas.py`)

50 lipatan kanonis, nesting ketat seperti putaran 49A (pemilihan C dengan CV dalam; semua transformasi dipasang ulang di tiap lipatan dalam). Penyesuaian OLS diganti ash:
- I2eb = [ash-v4(BP), AS];
- N1eb = [ACQ, AS, ash-v4(BP)];
- P3eb = PCA lalu ash;
- F2eb = (I2eb + P3eb + NM ketat)/3. NM tidak dilatih ulang (GPU); keterbatasan ini dilaporkan.

**Keluarga primer (Holm, 2 uji; t terkoreksi Nadeau-Bengio atas 50 AUC lipatan, dua sisi):**
- F2eb - F2 (ketat);
- N1eb - N0.

**Deskriptif:**
- AUC dengan CI untuk I2eb, P3eb, N1eb, F2eb;
- kontras I2eb - I2, P3eb - P3, N1eb - N1, I2eb - C1, F2eb - Floor, F2eb - C1;
- faktor penyusutan ash per lipatan.

**E3b (deskriptif):** uji confounder parsial (Spisak 2022, GigaScience; H0: skor independen dari kovariat bersyarat label).
- Karena label biner, permutasi baris kovariat di dalam kelas label eksak (Berrett dkk. 2020, bagian 2.1).
- Kovariat: empat dummy era. Usia dan jenis kelamin adalah fitur masukan AS.
- Model tanpa ACQ: C1, I2, I2eb, P3, P3eb, NM, F2, F2eb.
- Statistik: R2([1, y, era]) - R2([1, y]) atas skor pasien (rata-rata 10 ulangan); 10,000 permutasi; Holm.
- **Catatan Nygaard:** penyesuaian yang dipasang tanpa label pada desain tak seimbang dapat menyisakan atau menimbulkan ketergantungan kovariat bersyarat label walau benar. Sinyal EEG sah yang berkorelasi dengan era juga menolak H0. E3b tidak dipakai sebagai bukti kebocoran.

**Penegasan kejujuran:** menghapus kebocoran tidak menaikkan AUC yang sah. Kenaikan AUC label nyata dengan ash yang disertai kenaikan ketergantungan era (E3b) TIDAK ditafsirkan sebagai perbaikan sinyal EEG.

## 6. Prediksi (probabilitas subjektif)

- P51-1: TUEP P-a dan P-b lulus (rank-ash menghapus sisa 0.8--0.9 poin) (65%).
- P51-2: TUEP P-c dan P-d lulus (non-inferioritas pada efek era nyata) (55%).
- P51-3: Emotiv P-a dan P-b lulus (60%); Emotiv P-c dan P-d lulus (55%; efek negara besar dipertahankan ash, efek hari sedang berisiko disusutkan).
- P51-4: raw-ash pada H1, gamma 1, R1 bukan "kebocoran" untuk RF dan XGB di kedua kohort (65%). Pada E, raw-ash berkeputusan "kebocoran" karena efek multiplikatif disesuaikan secara aditif dalam skala mentah (60%).
- P51-5: E1 Spearman antar-sel >= 0.8 di keempat kohort x famili (70%); semua sel dengan rho <= 0.10 memiliki AUC <= 0.53 (65%).
- P51-6: E3 keluarga primer tidak bermakna (F2eb - F2: 85%; N1eb - N0 Holm >= 0.05: 75%). |I2eb - C1| < 1 poin (70%; efek kovariat BP tidak terdeteksi sehingga ash hampir mematikan penyesuaian).
- P51-7: K1 lulus di semua kohort x famili (85%); K2 lulus di TUEP (70%) dan Emotiv (85%).
- P51-8 (Bi-LSTM, sekunder): rank-ash dan raw-ash tidak lebih bocor daripada rank-pat (60%).

## 7. Pelaksanaan dan penyimpangan

- Urutan: RF TUEP dan Emotiv (CPU, NJOBS 10 karena server dipakai bersama), indeks rho (CPU), E3 (CPU); XGB TUEP lalu XGB Emotiv lalu Bi-LSTM (satu proses GPU).
- Skrip ringkasan `round51_ringkas.py` ditulis dan di-hash bersama dokumen ini sebelum keluaran utama ada (diuji hanya dengan data palsu di scratchpad, kecuali catatan bagian 0.4).
- Setiap perubahan sesudah hash dicatat sebagai adendum bertanggal dengan hash baru.
- Hasil negatif dilaporkan. Bila ash gagal pada E (kekurangan penyesuaian), naskah menyatakan pertukaran antara suntikan dan kekurangan penyesuaian, bukan penawar tuntas.

## Rujukan yang dibaca untuk putaran ini

- Hamdan dkk. 2023 (PMC10541796).
- Nygaard, Rodland & Hovig 2016, Biostatistics 17:29.
- Zindler dkk. 2020, BMC Bioinformatics 21:271 (PMC7328269).
- Stephens 2017, Biostatistics 18:275 (PMC5379932), dan kode sumber ashr (stephens999/ashr: ash.R, mix_opt.R, normalmix.R).
- Berrett dkk. 2020, JRSS-B (arXiv 1807.05405).
- Spisak 2022, GigaScience 11:giac082.
- Johndrow & Lum 2019 (arXiv 1703.04957), sebagai pembanding penyesuaian distribusional, tidak diuji.
- Gandy 2009, tidak dipakai: hanya untuk keluaran Bernoulli.
