# Pra-registrasi: RQ2 — Kontrol Positif untuk Guidance

**Ditulis:** 2026-10-12, SEBELUM dijalankan. **Status run:** belum dimulai — menunggu keputusan skala (lihat §6).
**Tidak ada kode pra-registrasi ini yang mengubah hasil `baseline`/`affinity_only` yang sudah ada di `guidance/task_f_results/`.** Variant baru ditambahkan secara aditif.

---

## 1. Latar belakang — kenapa ini wajib ada sebelum tesis bisa diklaim selesai

`guidance/task_f_results/` sudah punya perbandingan `baseline` (tanpa guidance) vs `affinity_only` (guidance gradien dari surrogate EGNN terlatih), 20 pocket × 3 seed × 10 molekul = 600 percobaan per variant, dievaluasi lewat `eval/honest_eval.py` (Vina docking nyata mode `dock`, exhaustiveness=16, plus PoseBusters):

| metrik | baseline | affinity_only | Wilcoxon berpasangan p |
|---|---|---|---|
| Vina Dock (mean±sd) | −6,813 ± 10,079 | −7,251 ± 2,409 | 0,546 |
| QED | 0,449 | 0,452 | 0,648 |
| SA | 0,601 | 0,600 | 0,277 |
| Ligand efficiency | −0,340 | −0,372 | 0,330 |
| honest_success_rate | 0,0302 | 0,0268 | — |

**Tidak ada perbedaan signifikan di metrik apa pun** (n_pockets_paired=20). Ini konsisten dengan M1 (surrogate tidak membaca antarmuka — §2.6 di `LAPORAN_RISET_LENGKAP.md`) tapi juga konsisten dengan kemungkinan lain yang jauh lebih mengkhawatirkan: **harness ini (generator + pocket + jalur evaluasi) mungkin tidak bisa mendeteksi perbaikan apa pun, terlepas dari seberapa baik sinyal pemandunya.** Tanpa kontrol positif, dua kemungkinan itu tidak bisa dibedakan, dan penguji berhak bertanya "bagaimana saya tahu alat ukurmu bukan yang rusak?"

## 2. Hipotesis dan prediksi — ditulis sebelum satu baris hasil pun ada

Tiga mekanisme kegagalan yang sudah dirumuskan di `guidance/RESEARCH_PROGRAM.md` Part II:
- **M1** — surrogate tidak informatif (gradiennya tidak membawa sinyal nyata tentang interaksi). Bukti tak langsung sudah ada: §2.6 ablasi faktorial menunjukkan model 4,5× lebih sensitif ke jumlah atom protein daripada ke identitas/posisi interaksinya.
- **M2** — gradien informatif tapi salah arah/tidak sampai (masalah pada mekanisme guidance itu sendiri: skala, titik aplikasi, dsb).
- **M3** — objektifnya bisa "dicurangi" generator tanpa benar-benar memperbaiki ikatan (size-inflation, dsb — `analyze_task_f.py` sudah punya `vina_hacking_check` untuk ini).

**Prediksi, dicatat sebelum data:**
- Kalau sinyal yang BUKAN surrogate-terlatih (Vina docking asli — zero-shot, tidak pernah dilatih di label pK kita) **berhasil** menggeser distribusi molekul yang dihasilkan ke arah Vina Dock yang lebih baik (diukur independen, bukan sirkular — lihat §3): **harness valid**, dan kegagalan `affinity_only` murni disebabkan surrogate EGNN-nya (M1 terkonfirmasi sebagai sebab tunggal yang cukup).
- Kalau sinyal Vina asli **juga gagal** menggeser distribusi: ini temuan yang LEBIH signifikan dan lebih mengkhawatirkan — menyiratkan hambatannya bukan di kualitas surrogate, tapi di generator/jalur sampling itu sendiri (M2) atau di ruang molekul yang bisa dijangkau generator pada budget sampling ini (keterbatasan ekspresivitas). Ini harus dilaporkan selengkap kegagalan yang sudah ada, bukan disembunyikan.

## 3. Desain — kenapa bukan backprop lewat Vina, dan kenapa bukan sirkular

Vina bukan fungsi yang bisa diturunkan (ia menjalankan pencarian pose internal, bukan evaluasi differentiable atas posisi atom kontinu), jadi "guidance gradien dari Vina" secara literal tidak mungkin. Desain yang dipakai di sini — **best-of-N reranking** — adalah satu-satunya cara metodologis bersih memakai Vina sebagai sinyal pemandu tanpa melatihnya di data kita:

1. **Perbesar pool**: untuk setiap (pocket, seed), sampel `K × samples_per_pocket` molekul TANPA guidance (jalur `baseline` yang sudah ada, tidak ada kode sampling baru).
2. **Seleksi murah**: skor seluruh pool dengan Vina mode `score_only` (~0,57 detik/molekul, terukur di `eval/honest_eval.py`'s docstring — murah, bukan pencarian pose penuh).
3. **Simpan `samples_per_pocket` terbaik** menurut skor seleksi itu.
4. **Evaluasi independen**: jalankan himpunan yang disimpan lewat `evaluate_molecules(..., docking_mode='vina_dock', exhaustiveness=16)` — **mode yang SAMA dan exhaustiveness yang SAMA dipakai untuk `baseline`/`affinity_only`**, tapi ini adalah pass Vina yang BERBEDA (pencarian pose penuh, bukan skor cepat) dari yang dipakai untuk menyeleksi.

**Kenapa ini tidak sirkular**: skor seleksi (`score_only`, cepat, tanpa pencarian) dan skor evaluasi (`dock`, pencarian pose penuh) adalah DUA proses Vina yang berbeda atas pose yang (mungkin) berubah setelah pencarian. Kalau top-k menurut skor seleksi TIDAK lebih baik di evaluasi independen, itu artinya sinyal seleksinya tidak cukup stabil untuk ditransfer — persis logika uji reprodusibilitas score-rescore yang standar di virtual screening. Kalau top-k-nya LEBIH baik di evaluasi independen, itu bukti genuin bahwa ada sinyal yang bisa dieksploitasi di pool molekul generator ini, dan harness mampu mendeteksinya.

**Kenapa bukan PLIP untuk desain utama**: PLIP butuh pose yang sudah didok/diminimalkan untuk menghitung interaksi bertipe secara bermakna — menjalankannya di pose mentah hasil generator (sebelum minimisasi) akan menghasilkan interaksi palsu dari clash geometri yang belum direlaksasi. Vina `score_only` tidak punya masalah ini sampai taraf yang sama (skornya dirancang untuk menilai pose "apa adanya"). PLIP-based reranking disimpan sebagai **arm sekunder opsional** (§7), bukan bagian dari komitmen utama — menambahnya tanpa keputusan scoping baru akan memperluas lingkup tanpa alasan yang cukup kuat saat ini.

## 4. Variant baru: `vina_rerank`

Ditambahkan secara ADITIF ke `guidance/run_task_f.py` (fungsi baru, tidak mengubah `run_triple` yang dipakai `baseline`/`affinity_only` — nol risiko regresi ke hasil yang sudah ada):

```
run_triple_vina_rerank(model, test_set, pocket, seed, samples_per_pocket, oversample_k,
                       num_steps, batch_size, device, out_dir, vina_select_exhaustiveness=1)
```

Alur: sampel `oversample_k * samples_per_pocket` tanpa guidance → rekonstruksi SDF → skor `score_only` tiap molekul → ambil `samples_per_pocket` terbaik → `evaluate_molecules` mode `vina_dock` exhaustiveness=16 pada himpunan terpilih → tulis `honest_eval.csv` dengan kolom tambahan `vina_select_score` (skor seleksi, untuk audit post-hoc apakah seleksinya benar-benar menyaring) dan `pool_size`.

Pocket yang dipakai: **20 pocket yang SAMA** dari `guidance/task_f_pockets.json` (yang sudah dipakai `baseline`/`affinity_only`) — supaya perbandingan tiga-variant memakai `n_pockets_paired` yang identik tanpa pocket baru yang belum teruji.

## 5. Kriteria keputusan — ditulis sebelum melihat angka

Dihitung dengan infrastruktur statistik yang sudah ada (`stats_common.cluster_bootstrap_ci`, pasangan per-pocket, Benjamini-Hochberg untuk 5 metrik: Vina Dock, QED, SA, ligand efficiency, honest_success_rate):

- **Harness tervalidasi (hasil utama yang diharapkan)**: Vina Dock `vina_rerank` berbeda signifikan (q<0,05 setelah BH) dari `baseline`, DAN **arah perbaikannya konsisten dengan skor seleksi** (yaitu, `vina_select_score` pool yang disimpan memang lebih baik dari pool yang dibuang — pemeriksaan sanity, bukan uji utama).
- **Harness tidak valid / tidak bisa mendeteksi apa pun**: `vina_rerank` TIDAK berbeda signifikan dari `baseline` di Vina Dock. Ini dilaporkan selengkap dan sejujur hasil `affinity_only` yang sudah null — **bukan dianggap kegagalan eksperimen**, tapi temuan substantif tentang batas generator/sampling pada budget ini.
- **Vina-hacking check** (`analyze_task_f.py` sudah punya fungsinya): kalau Vina Dock membaik tapi PoseBusters pass rate turun atau heavy_atoms naik tajam, itu sinyal size-inflation/pose-invalid, bukan perbaikan ikatan genuin — dilaporkan sebagai temuan M3, bukan disembunyikan di balik angka Vina Dock yang membaik.

## 6. Skala dan biaya — perlu keputusan sebelum dijalankan

Waktu per triple (10 molekul) terukur dari log yang sudah ada (`run_log.jsonl`, 120 triple baseline): **median 134 detik, mean 292 detik** (variansi besar antar-pocket — ukuran protein/ligan berbeda; maksimum pernah 1.849 detik). Pass seleksi (`score_only`) murah (~0,57 detik/molekul) dan diabaikan dalam estimasi. Estimasi total untuk oversampling faktor K, N pocket, S seed (memakai MEAN sebagai basis konservatif, bukan median, supaya estimasi tidak terlalu optimis):

| skala | K | pocket × seed | estimasi GPU-wall-time |
|---|---|---|---|
| **Pilot (direkomendasikan, mulai di sini)** | 2 | 6 × 1 = 6 triple | ≈ 6 × (2×292 + 104) ≈ **1,1 jam** |
| Skala penuh, hemat | 2 | 20 × 3 = 60 triple | ≈ 60 × 688 ≈ **11,5 jam** |
| Skala penuh, sesuai niat literal (4×) | 4 | 20 × 3 = 60 triple | ≈ 60 × (4×292+104) ≈ **21,5 jam** |

**Rekomendasi:** jalankan **pilot (≈1,1 jam)** dulu lewat `scripts/queued_run.sh` (thermal-gated, bukan langsung di shell) pada 6 pocket yang dipilih stratified dari 20 yang ada (variasi ukuran pocket/ligand), 1 seed, K=2. Ini sudah cukup untuk: (a) memverifikasi implementasinya benar secara mekanis pada skala nyata (bukan cuma smoke test trivial), (b) memberi pembacaan awal arah efek sebelum komit ke skala penuh. **Keputusan skala penuh (11,5–21,5 jam) menunggu hasil pilot dan persetujuan eksplisit Anda** — ini komitmen GPU/waktu besar sesuai definisi yang sudah disepakati, jadi tidak saya jalankan sendiri.

## 7. Arm sekunder opsional — TIDAK dikerjakan sekarang, dicatat supaya tidak hilang

Kalau hasil Vina-rerank ambigu atau menarik cukup untuk digali lebih jauh: **PLIP-count reranking** (seleksi berdasar jumlah interaksi bertipe — h-bond/hidrofobik/π-stacking/jembatan garam/halogen dari `PlipLabelStore` yang sudah ada di `guidance/track_e/core.py`, dihitung pada pose setelah minimisasi Vina bukan pose mentah) sebagai variant keempat. Ini menambah lingkup baru, jadi butuh keputusan scoping terpisah setelah hasil primer ada — bukan dibundel ke commitment ini.

## 8. Yang TIDAK diubah oleh pra-registrasi ini

- `baseline` dan `affinity_only` di `guidance/task_f_results/` — tidak disentuh, tidak dijalankan ulang.
- `run_triple()` yang sudah ada — tidak diedit, hanya ditambah fungsi sibling baru.
- Tidak ada perubahan ke `guided_sampling.py`, `affinity_guidance.py`, atau core diffusion model manapun.
