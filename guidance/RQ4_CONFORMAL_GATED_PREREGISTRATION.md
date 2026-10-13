# Pra-registrasi: RQ4 — Conformal-Gated Guidance

**Ditulis:** 2026-10-13, SEBELUM kode ditulis. **Status:** desain belum lengkap — satu gap nyata ditemukan
di bawah dan harus diselesaikan dengan pengecekan murah (CPU, menit) sebelum komitmen GPU apa pun dibuat.

---

## 1. Rencana asli dan dua prediksi yang berlawanan

TASKS.md: terapkan gradien guidance ATAU terima kandidat rejection-sampling HANYA di tempat interval
conformal surrogate cukup sempit. Dua prediksi ditulis sebelumnya:
- **M1** (surrogate tidak informatif) memprediksi rute **gradien** TIDAK akan terbantu — menggerbangi
  kekuatan guidance berdasar lebar interval tidak memperbaiki ARAH gradien yang memang tidak membawa
  sinyal; confidence yang tinggi di prediksi TITIK tidak membuat arah turunannya jadi benar.
- **M3** (objektif bisa dicurangi) memprediksi rute **rejection** AKAN terbantu — kandidat yang confident
  (interval sempit) lebih kecil kemungkinannya hasil eksploitasi out-of-distribution yang membuat
  objektif gameable.

Satu eksperimen yang mengkonfirmasi satu prediksi dan membantah satunya = bukti kuat untuk seluruh
dekomposisi M1/M2/M3. Ini tetap berlaku dan tetap bernilai setelah RQ2 — RQ2 menguji "apakah reranking
dengan sinyal GENUIN (Vina asli) membantu", RQ4 menguji mekanisme berbeda: "apakah menggerbangi
berdasarkan CONFIDENCE surrogate kita sendiri (yang sudah terbukti tidak membaca antarmuka) cukup untuk
meniru sebagian manfaat itu, tanpa sinyal yang lebih baik."

## 2. Gap desain yang ditemukan — HARUS diselesaikan sebelum kode guidance ditulis

**A1d's conformal quantile `q` itu GLOBAL, bukan per-titik.** Dibaca langsung dari
`guidance/uncertainty_a1/conformal_prediction.py`: `conformal_quantile(cal_scores, alpha)` menghitung
SATU `q` (=2,1960) dari seluruh himpunan kalibrasi, lalu `covered = |y_true - y_pred| <= q` dipakai
SERAGAM untuk setiap titik test. **Setiap prediksi punya lebar interval yang SAMA PERSIS.** "Terima
kandidat HANYA di tempat interval cukup sempit" tidak bermakna kalau lebar interval tidak pernah
bervariasi antar molekul — tidak ada apa pun untuk digerbangi.

**Implikasi yang tidak nyaman dari temuan kita SENDIRI**: dua kandidat paling alami untuk membuat lebar
interval bervariasi per-titik, keduanya sudah terbukti TIDAK INFORMATIF di proyek ini:
1. **Spread ensemble (A1c)** sebagai proksi ketidakpastian lokal -- tapi `a1c_results.json` sudah
   mencatat spearman(sigma, error) = 0,0084, CI memuat nol. Kalau sigma ensemble tidak berkorelasi
   dengan error sama sekali, menormalisasi residual dengannya (conformal ternormalisasi, konvensi
   Lei & Wasserman) kemungkinan besar TIDAK menghasilkan lebar interval yang bermakna berbeda --
   cuma noise yang diberi baju "adaptif".
2. **Novelty tier (item 3)** sebagai kategori Mondrian -- tapi item 3 sendiri menemukan 0 dari 4 tier
   yang coverage-nya berbeda signifikan dari global. Kalau coverage sudah seragam lintas tier, memaksa
   `q` berbeda per tier kemungkinan besar juga tidak menangkap apa pun yang nyata.

**Jadi sebelum menulis satu baris kode guidance pun, pertanyaan yang harus dijawab dulu**: apakah ADA
konstruksi conformal lokal-adaptif di data kita yang lebar intervalnya benar-benar bervariasi secara
bermakna (bukan cuma noise resampling)? Kalau tidak ada, seluruh rute REJECTION (prediksi M3) tidak
punya sinyal gerbang yang bisa diuji -- bukan karena M3 salah, tapi karena kita belum punya alat ukur
confidence yang terbukti bekerja.

## 3. Langkah PERTAMA, murah, CPU saja -- sebelum GPU apa pun disentuh

Uji langsung, pada val/test yang sudah ada (tidak perlu sampling generatif baru):
1. Hitung `q` TER-NORMALISASI (conformal ternormalisasi): skor kalibrasi `|y-pred| / sigma_ensemble`
   (bukan `|y-pred|` mentah), lalu `q_norm` dari skor itu. Interval per-titik = `pred +- q_norm * sigma_i`
   -- lebar BERVARIASI per titik kalau `sigma_i` bervariasi (selalu benar secara aritmetika; pertanyaannya
   apakah variasi itu BERMAKNA, bukan apakah ada).
2. Ukur: rasio p90/p10 lebar interval across test set. Kalau mendekati 1,0 (hampir konstan), konstruksi
   ini TIDAK memberi sinyal gerbang yang berguna -- dicatat sebagai hasil negatif untuk rute ini, bukan
   dipaksa dipakai.
3. Ukur yang lebih penting: apakah lebar interval (bukan cuma bervariasi, tapi) BERKORELASI dengan error
   aktual (`|y_true-y_pred|`) pada test set -- kalau korelasinya tidak beda dari nol (konsisten dengan
   spearman sigma-error A1c yang sudah 0,0084), maka menggerbangi berdasarkan lebar ini SETARA dengan
   menggerbangi acak, dan baik M1 maupun M3 tidak bisa diuji secara bermakna lewat jalur ini.
4. **Kalau langkah 1-3 menunjukkan tidak ada sinyal gerbang yang bermakna**: ini sendiri adalah temuan
   substantif untuk dilaporkan -- "tidak cuma guidance gradien yang gagal (M1), tapi bahkan confidence
   surrogate kita tidak cukup terkalibrasi secara lokal untuk mendukung desain conformal-gated apa pun
   (RQ4 secara keseluruhan tidak bisa diuji dengan infrastruktur UQ yang kita punya)". Itu bukan
   kegagalan pra-registrasi, itu hasil yang jujur dan konsisten dengan temuan UQ-suite (5/6 metode gagal).

**Biaya langkah ini: menit, CPU saja, memakai cache A1c yang sudah ada.** Tidak ada komitmen GPU sampai
hasil langkah ini diketahui.

## 4. KALAU langkah 3 menunjukkan ada sinyal (skenario optimis) -- desain lanjutannya

Reuse infrastruktur yang SUDAH ada, bukan membangun baru:
- **Rute rejection**: pola identik `run_task_f.py`'s `vina_rerank` (oversample pool tanpa guidance,
  seleksi, evaluasi independen) -- tapi seleksi berdasar "prediksi surrogate bagus DAN interval lokal
  sempit", bukan skor Vina. Variant baru `conformal_gated_rejection` di file yang sama.
- **Rute gradien**: `guided_sampling.py` sudah menerima `lambda_affinity` per-batch; perlu ekstensi agar
  lambda bisa nol-kan PER MOLEKUL berdasar interval lokal di x0-hat setiap langkah -- perubahan lebih
  dalam, perlu desain terpisah kalau sampai ke titik ini.
- Pocket/skala: samakan dengan RQ2 (20 pocket, pola pilot-dulu-skala-penuh) untuk komparabilitas
  langsung dengan hasil `baseline`/`affinity_only`/`vina_rerank` yang sudah ada.

## 5. KALAU langkah 3 menunjukkan TIDAK ada sinyal (skenario realistis berdasar prior kita)

Laporkan sebagai temuan, bukan jalan buntu: RQ4 dikoreksi dari "conformal-gated guidance gagal" menjadi
"RQ4 tidak bisa diuji secara bermakna dengan UQ yang kita punya -- yang itu sendiri konsisten dengan dan
memperkuat temuan UQ-suite (hanya coverage MARGINAL yang valid, tidak ada metode yang memberi sinyal
PER-TITIK yang informatif)". Ini menutup RQ4 dengan kesimpulan yang jujur tanpa perlu GPU sama sekali.

## 6. Keputusan yang diperlukan

Langkah 3 (CPU, menit) dijalankan lebih dulu, hasilnya menentukan apakah lanjut ke §4 (butuh scope GPU
terpisah, komitmen besar lagi) atau berhenti di §5 (selesai, murah). **Tidak ada kode guidance baru
ditulis sampai keputusan ini diketahui.**
