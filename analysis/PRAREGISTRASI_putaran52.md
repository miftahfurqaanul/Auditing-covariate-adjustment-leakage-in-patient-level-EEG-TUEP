# PRA-REGISTRASI PUTARAN 52 (perluasan penawar ash ke komponen F2 yang lain dan ke pembanding; ditulis SEBELUM menghitung; 2026-11-03)

Status analisis: **lanjutan** (sesudah putaran 3--50 dan SEBELUM keluaran putaran 51 diringkas). Perintah pemilik 2026-11-03: setelah putaran 51 selesai seluruhnya, jalankan putaran 52 dengan CPU dan GPU semaksimal mungkin. Putaran 52 memakai metode ash dan seluruh definisi putaran 51 (`round51_common.py`, `PRAREGISTRASI_putaran51.md` bagian 2) tanpa perubahan.

## 0. Yang SUDAH dilihat sebelum dokumen ini ditulis

1. Semua hasil putaran lama yang relevan, termasuk (dibaca ulang hari ini untuk kalibrasi prediksi):
   - plasmode NM putaran 42 (H1; 20 replikasi): gamma 1, raw-V4 menambah +0.37 poin (SE 0.86) dan rank-V4 -0.25 poin (SE 0.85) terhadap raw-N0; keputusan R1 dan R2 "tidak ada bukti";
   - kondisi nol pembanding putaran 33 (AUC gabungan; DNM 20 replikasi, Rohira 10): Rohira H1 mentah 0.503 (SD 0.053), V4 **0.573** (SD 0.026); H2a mentah 0.540, V4 0.562; DNM H1 mentah 0.504, V4 0.500; H2a 0.521 dan 0.514;
   - kondisi nol embedding putaran 43 (fold_mean, 20 replikasi, V4): semua model 0.49--0.51.
2. Keluaran putaran 51 (`hasil51/`): hanya diperiksa kemajuan baris dan log; **tidak ada AUC** plasmode atau label nyata putaran 51 yang dilihat. `nyata_prediksi.npz` putaran 51 sudah ada tetapi belum diringkas.
3. Pemeriksaan kode (CPU) lulus: prediksi V4 putaran 33 (DNM-V4, Rohira-V4+AS) dan putaran 43 (LaBraM-eq AS) tereproduksi pada lipatan 0 dan 17 dengan selisih < 1e-8 oleh salinan kode putaran 52. Pemeriksaan GPU (reproduksi plasmode NM putaran 42) dijalankan sesudah putaran 51 selesai dan sebelum run (hasilnya dicatat di hash log).

Tidak ada AUC putaran 52 yang dihitung atau dilihat.

## 1. Pertanyaan

Putaran 51 menguji penawar ash pada I2, P3, dan model pohon/Bi-LSTM. Putaran 52 menjawab: (a) apakah NM dengan ash dan F2 yang seluruh komponennya dengan ash (F2ash) berbeda dari F2 ketat; (b) apakah NM dan pembanding yang memakai penyesuaian V4 (DNM, Rohira, embedding beku) berubah dengan ash; (c) apakah ash mengurangi kebocoran kondisi nol Rohira (satu-satunya kasus lama dengan sinyal kebocoran pada kondisi nol: V4 0.573 vs mentah 0.503).

**Tidak diuji (alasan):**
- Tang-F pada label nyata: 50 lipatan x 100 epoch terlalu mahal; plasmode Tang sudah di putaran 51.
- LaBraM-FT dan CBraMod-FT: tidak memakai penyesuaian kovariat.
- DNM-F asli dan Rohira-F asli (tanpa V4): tidak memakai penyesuaian kovariat.

## 2. Metode

**ash**: persis `round51_common.ash_coef` (rataan posterior prior campuran normal berpusat 0, grid dan penalti default ashr; koefisien dipasang hanya pada pasien latih).

**NM-ash (label nyata, `round52_nm.py`):**
- nesting ketat seperti `round49_nm_ketat.py` (statistik hanya dari pasien latih tanpa validasi; 20% validasi untuk grid dan early stopping; 3 seed; grid konfigurasi sama; seed `1000 r + 10 k` dan seterusnya identik);
- satu-satunya perubahan: koefisien penyesuaian segmen dari ash atas **rata-rata segmen per pasien latih** (V4 putaran 49 memakai OLS tertimbang segmen). Perbedaan ini (rata-rata per pasien vs tertimbang segmen) tidak dipisahkan dari perbedaan OLS vs ash dan dicatat sebagai keterbatasan;
- F2ash = rata-rata (I2eb + P3eb + NM-ash) dengan I2eb dan P3eb dari putaran 51.

**NM plasmode H1 (`round52_nm.py`):** salinan `round42_gpu_nn.nm_unit`; pipeline raw-ash dan rank-ash; rep 0--19, gamma {0, 1}; pembanding raw-N0, raw-V4, rank-V4 dari `hasil42/NM.csv` (seed sama, tidak dihitung ulang). Koefisien dari semua pasien latih (sama dengan `M.tokens(..., "V4", no_as=True)`).

**Pembanding (`round52_cpu.py`; salinan fungsi putaran 33 dan 43, V4 diganti ash):**
- DNM-ash: tingkat pasien, `nested_lr`, ditambah usia/jenis kelamin.
- Rohira-ash (+usia/jenis kelamin; dan tanpa usia/jenis kelamin): RF jendela; koefisien ash atas rata-rata per pasien latih (V4 putaran 33 memakai OLS atas semua jendela).
- Embedding beku (LaBraM, CBraMod lapisan 12, BENDR, dan BP-eq): PCA 50, rank-normal, ash, usia/jenis kelamin, LR dengan C dari CV dalam. Pembanding: baris `*-eq__AS` pada `hasil43/A_prediksi.npz`.

**Kondisi nol (`round52_cpu.py --run null`):**
- embedding: H1 dan H2a, rep 0--19, benih dan permutasi sama dengan `round43_fm_kompresi.run_C`; pasangan per replikasi dengan `hasil43/C_null_replikasi.csv`;
- DNM: 20 replikasi; Rohira: 10 replikasi; benih 33000 + r; AUC gabungan. Hanya perbandingan deskriptif dengan ringkasan `hasil33/pembanding_null.csv`, karena berkas itu tidak menyimpan nilai per replikasi.

## 3. Analisis

**Keluarga primer (label nyata; Holm, 2 uji; t terkoreksi Nadeau--Bengio atas 50 AUC lipatan, dua sisi):**
- P1: AUC(F2ash) - AUC(F2 ketat).
- P2: AUC(NM-ash) - AUC(NM ketat).

**Sekunder (Holm, 6 uji, label nyata):** AUC(ash) - AUC(V4) untuk DNM, Rohira+usia/jenis kelamin, LaBraM-eq, CBraMod-eq, BENDR-eq, BP-eq.

**Plasmode NM (primer, Holm, 2 uji, satu sisi):** gamma 1, AUC(rank-V4) - AUC(rank-ash) > 0 dan AUC(raw-V4) - AUC(raw-ash) > 0 (t satu sampel atas selisih per replikasi). Aturan R1/R2: ambang asli putaran 42.

**Deskriptif:**
- AUC dengan CI semua model;
- F2ash - Floor, F2ash - C1, F2ash - (I2eb + P3eb)/2, F2ash - N0;
- kondisi nol: embedding (delta ash - V4 per replikasi, keputusan R1), DNM dan Rohira (rata-rata, SD, persentil 95);
- uji confounder parsial (empat dummy era, permutasi dalam kelas label, 10,000 permutasi, benih 51003) untuk C1, I2, I2eb, P3, P3eb, NM, NM-ash, F2, F2eb, F2ash. Catatan Nygaard dan penegasan kejujuran putaran 51 (bagian 5) berlaku.

**Kontrol:**
- K1: NM-ash pada gamma 0 tidak berkeputusan R1 "kebocoran" pada kedua pipeline.
- Rujukan reproduksi: lulus pemeriksaan kode (bagian 0.3).

## 4. Prediksi (probabilitas subjektif)

- P52-1: P1 dan P2 tidak bermakna (Holm >= 0.05) (80%). |AUC(F2ash) - AUC(F2)| < 1.5 poin (75%).
- P52-2: NM plasmode: kedua uji primer tidak bermakna dan semua sel "tidak ada bukti" atau "tidak pasti" (75%); NM tidak menunjukkan kebocoran pada penyesuaian apa pun (sesuai putaran 42).
- P52-3: Rohira-ash pada kondisi nol H1: AUC gabungan rata-rata <= 0.53 (V4: 0.573) (65%); pada H2a <= 0.55 (55%).
- P52-4: Rohira-ash+usia/jenis kelamin pada label nyata: perbedaan dari V4 tidak bermakna (Holm >= 0.05) (70%).
- P52-5: embedding beku dan DNM: perbedaan ash - V4 pada label nyata tidak bermakna di keenam uji (80%); kondisi nol embedding: |delta| < 0.01 pada semua model dan skenario (80%).
- P52-6: uji confounder parsial: NM-ash dan F2ash tidak menyingkirkan ketergantungan era yang ada pada F2 (p tidak lebih besar secara bermakna) (60%).

## 5. Pelaksanaan

- Setelah putaran 51 selesai (semua log berakhir "selesai", berkas shard digabung dengan `round51_shard.py merge`, `round51_ringkas.py` dijalankan dan keluarannya disimpan): periksa ulang kelengkapan, lalu jalankan:
  - GPU: pemeriksaan `round52_nm.py --check`; lalu NM real 3 shard dan NM plasmode 2 shard (lima proses GPU);
  - CPU: `round52_cpu.py --run real`, lalu `--run null` (NJOBS 18).
- Skrip ringkasan `round52_ringkas.py` ditulis dan di-hash bersama dokumen ini sebelum keluaran utama ada.
- Setiap perubahan sesudah hash dicatat sebagai adendum bertanggal.
- Hasil negatif dilaporkan. Bila ash tidak mengubah NM dan F2 (prediksi P52-1), naskah menyatakan bahwa penawar tidak merugikan komponen yang tidak bocor.
