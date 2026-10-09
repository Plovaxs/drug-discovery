# To-do

## CARA LANJUT SETELAH LAPTOP DI-SHUTDOWN (ditulis 2026-10-09 06:47, untuk sesi berikutnya)
Semua training di project ini resumable: `last.pt` disimpan SETIAP epoch validasi, jadi shutdown paling buruk cuma kehilangan epoch yang sedang berjalan. Semua perintah dijalankan dari `~/research/drug-discovery/targetdiff`, pakai `PYTHONPATH=.` dan python env `~/miniconda3/envs/drugdisc/bin/python`.

**Status saat shutdown:**
- QAT seed 2021: SELESAI. Test R2=0,3239, Pearson=0,6132 (vs FP32 0,342/0,609) — setara baseline.
- QAT seed 2022: sedang jalan saat catatan ini ditulis (best di epoch 1, val loss 1,981; `last.pt` di epoch 5). Kalau mati di tengah, resume (lihat di bawah). Kalau sudah selesai sendiri, langsung eval test.
- QAT seed 2023: BELUM dijalankan. Diminta tetap dijalankan setelah cooling.

**1. Resume seed 2022 kalau terputus di tengah:**
```
PYTHONPATH=. ~/miniconda3/envs/drugdisc/bin/python guidance/uncertainty_a1/train_egnn_qat_ternary.py \
  configs/prop/crossdocked_affinity_egnn.yml --seed 2022 --skip_test_logging \
  --resume ./logs_a1g_qat_ternary/crossdocked_affinity_egnn_2026_10_09__05_38_*_a1g_s2022/checkpoints/last.pt
```
(cek dulu `ls logs_a1g_qat_ternary/` untuk nama folder persisnya; `--resume` otomatis pakai folder log yang sama, tidak bikin baru)

**2. Evaluasi test set sebuah seed QAT yang sudah selesai (~5 menit GPU):**
```
PYTHONPATH=. ~/miniconda3/envs/drugdisc/bin/python guidance/uncertainty_a1/eval_qat_test.py --seed 2022
```

**3. Jalankan seed 2023 (fresh, ~1,5-2 jam GPU):**
```
PYTHONPATH=. ~/miniconda3/envs/drugdisc/bin/python guidance/uncertainty_a1/train_egnn_qat_ternary.py \
  configs/prop/crossdocked_affinity_egnn.yml --seed 2023 --skip_test_logging \
  --logdir ./logs_a1g_qat_ternary --tag a1g_s2023
```

**Catatan**: ada file baru yang BELUM di-commit (stats_common.py + test-nya, eval_qat_test.py, hasil replikasi A1f seed 2022/2023, TASKS.md ini). Aman — tetap ada di disk setelah shutdown. Commit hanya kalau user minta.

## ROADMAP yang disepakati (2026-10-09, urutan dari user)
1. **Selesaikan proses yang sedang jalan** (QAT seed 2022; seed 2023 opsional/ditunda).
2. **ISTIRAHAT — cooling system.** GPU sudah jalan hampir nonstop sejak 2026-10-07 20:57. Tidak ada compute berat (GPU MAUPUN CPU — satu thermal envelope di laptop) selama periode ini. Kerjaan ringan saja: nulis kode/script tanpa dieksekusi, update dokumen.
3. **Surrogate data besar**: training EGNN/GIGN pakai 125.238 baris BindingNet yang sudah dipasangkan struktur (`bindingnet_v1_paired.csv`), dibandingkan ke Stage 0 di LP test yang sama. **Ekstraksi pocket10 SUDAH DIBANGUN + TERVALIDASI** (`guidance/surrogate_data/extract_bindingnet_pockets.py`) — lihat bagian "Ekstraksi pocket10 BindingNet" di bawah.
4. **Gate DIAG1 + Vina-hacking** pada surrogate baru itu (pre-registrasi kriteria SEBELUM lihat hasil, sama seperti A1-A1g).
5. **Integrasi TNN/QAT ke arsitektur** surrogate yang sudah lolos gate (mekanisme `guidance/tnn_qat.py` sudah siap dan terbukti jalan di A1g).
6. Setelah semua itu: tentukan langkah berikutnya bersama.


Terakhir diperbarui: 2026-10-07. Tag: [wajib] = dibutuhkan untuk thesis, [berguna] = meningkatkan kualitas, [opsional] = overengineering / pengembangan lanjut.

## Selesai (ringkasan)
- [x] Build ulang thesis: 198 halaman, 59 referensi, tidak ada referensi tak terpakai
- [x] Cek dropout: EGNN Stage 0 tidak punya dropout; GIGN punya p = 0,1 di HIL
- [x] A1 (GIGN, MC Dropout): gate gagal. Spearman 0,090 (p_BH 0,34), rasio kalibrasi 9,7
- [x] Training A1b (EGNN + dropout node-update 0,1): best epoch 8, early stop epoch 13
- [x] A1b analisis: gate gagal. Spearman 0,002 (p_BH 0,98), rasio kalibrasi 64,8. Analisis sempat terhenti di 11.500 dan dilanjutkan dari partial.npz
- [x] Unduh LP-PDBBind (metadata, BDB2020+, LICENSE): data/lp_pdbbind/
- [x] Unduh BindingDB TSV (594 MB, zip valid): data/bindingdb/
- [x] Unduh BindingNet v1 arsip utama (312 MB, lengkap): data/bindingnet_v1/
- [x] Cek overlap PDB-ID dan UniProt: BindingDB dan LP-PDBBind tumpang tindih besar dengan split kita
- [x] A1c training: 5 anggota ensemble EGNN selesai semua (2026-10-08 06:34 WIB) — seed 2021 (reuse, epoch4 val_loss 1,780), 2022 (epoch11, 1,9264), 2023 (epoch8, 1,9159), 2024 (epoch3, 1,9285), 2025 (epoch7, 1,9036). Resume/save system (`--resume` di `train_egnn_stage0.py`, driver `run_a1c_ensemble.py`) dibuat di tengah jalan tapi TIDAK dipakai (seed 2025 sudah start sebelum fix tersimpan) — tidak masalah karena semua seed selesai natural, tidak ada yang mati di tengah.
- [x] Probe kelayakan TNN (`guidance/tnn_feasibility_probe.py`): ternarisasi post-hoc TWN (Li et al. 2016, tanpa retraining) pada checkpoint Stage 0 EGNN asli. **Hasil: GAGAL TOTAL.** FP32 R2=0,411 (n=1.500 subsample val) -> ternary R2=-1,497 (Pearson 0,650->0,563, RMSE 1,32->2,72, sparsity 37%). Ini SESUAI prediksi dari literatur terverifikasi (Rasool et al. 2025, J. Cheminformatics: kuantisasi 2-bit saja sudah "severely degrades performance" utk molecular GNN) — ternary (~1,58-bit) lebih ekstrem lagi. **Kesimpulan: TNN tanpa QAT/STE tidak viable di model ini, bukan prioritas.** Hasil: `guidance/tnn_feasibility_probe_results.json`.
- [x] BindingDB x train-overlap (`guidance/surrogate_data/bindingdb_train_overlap.py`): **854.823 baris BindingDB** (dari 2.840.436 bersih) match protein yang SUDAH punya struktur pocket di train kita (499 dari 720 protein train yang ke-resolve UniProt) — bisa jadi data pelatihan tambahan TANPA unduh struktur baru sama sekali. 37 protein train gagal resolve UniProt (kebanyakan non-human/MYCTU, kemungkinan nama entry UniProt yang sudah usang). Hasil: `guidance/surrogate_data/bindingdb_train_reusable.csv`.

## Sedang / menunggu
- [ ] Binding MOAD: situs memblokir akses otomatis (403) dan sertifikat HTTPS kedaluwarsa. Perlu unduh manual dari browser, lalu taruh di `data/binding_moad/`. Lisensi belum terverifikasi.
- [x] Crystal_Templates diekstrak (7,8GB, 17.179 template) + dipasangkan ke `bindingnet_v1_index_filtered2.csv` (`guidance/surrogate_data/pair_bindingnet_structures.py`): **125.238 / 125.243 baris (99,996%) dapat struktur reseptor**, cuma 3 template yang hilang. Sanity check coordinate-frame (jarak minimum atom ligan-reseptor, 100 sampel acak): rata-rata 1,73A, maks 2,25A, 0 yang >50A — persis seperti kontak nyata, bukan frame yang salah. Hasil: `guidance/surrogate_data/bindingnet_v1_paired.csv`. **Ini dataset BindingNet yang sudah lengkap: pocket + ligand + afinitas + bersih dari leakage val/test** — siap buat tahap ekstraksi pocket10 dan training surrogate.

## Keputusan yang perlu diambil (riset)
- [x] Cabang ketidakpastian: **diputuskan 2026-10-07 — lanjut ke deep ensemble** (5 seed GIGN, sekitar 12 jam GPU), bukan tutup dengan hasil negatif.
- [x] Peran surrogate: **diputuskan 2026-10-07 — (a) pengganti docking di guidance loop**, bukan (b) model pelatihan tambahan. Semua item "Surrogate" di bawah harus dibaca dengan peran (a).

## Evaluasi dan rigor
- [x] CI Stage 0: `guidance/lp_split/stage0_cluster_bootstrap_ci.py` (CPU-only, reuse `guidance/track_e/_stage0_test_preds.npz`, B=2.000 seed 20260925 — konstanta diimpor langsung dari `mc_dropout_calibration.py` biar tidak drift dari A1/A1b/A1c). **Temuan penting**: CI lama (per kompleks, i.i.d.) jauh lebih sempit dari yang seharusnya. R2 = 0,342, CI i.i.d. [0,322, 0,361] (lebar 0,039) vs CI cluster-per-target yang benar [0,160, 0,468] (lebar 0,308 — 7,9x lebih lebar). Pearson 0,609: CI i.i.d. [0,597, 0,621] vs cluster [0,513, 0,694] (7,6x lebih lebar). Harus ganti semua angka CI Stage 0 di thesis dengan versi cluster ini. Hasil: `guidance/lp_split/stage0_cluster_bootstrap_ci_results.json`.
- [ ] [wajib] Catat di thesis: overlap PDB ID train/test **sudah diverifikasi ulang dari data mentah** (bukan cuma dikutip) — detailnya lebih presisi dari catatan sebelumnya: 2 PDB ID reseptor tumpang tindih (`1rdt`, `3a9e`), memengaruhi 7 dari 11.855 kompleks test (0,06%), SEMUA di target test `RXRA_HUMAN_222_461_ligBind_0`. **Penting**: ini bukan duplikasi pocket test yang identik — `1rdt` dan `3a9e` adalah struktur kristal heterodimer nuclear-receptor multi-chain: `1rdt` chain A = RXRA (masuk test), chain D = PPARG (masuk train, target `PPARG_HUMAN_229_505_0`); `3a9e` chain A = RXRA (test), chain B = RARA (train, target `RARA_HUMAN_173_420_0`). Jadi yang overlap adalah ID PDB-nya (satu entri kristalografi berisi beberapa protein berbeda), bukan pocket protein yang sama persis — tapi RXRA/PPARG/RARA adalah paralog nuclear-receptor LBD yang homolog secara struktural, jadi tetap ada risiko leakage ringan (bukan exact match). Overlap ID PDB sumber-ligan (struktur asal pose ligand yang didocking ulang) = 0, bersih. Belum ditulis ke teks thesis — masuk ke antrian rebuild thesis (lihat bawah).
- [ ] [berguna] Cek family holdout: `guidance/lp_split/build_family_holdout_split.py` belum saya baca.
- [ ] [berguna] Jalankan ulang kontrol confound ukuran pocket (`analyze_pocket_size_confound.py`) untuk model baru.
- [ ] [berguna] Pre-registrasi tertulis untuk setiap uji baru, sebelum melihat hasil.
- [ ] [berguna] Unit test untuk fungsi statistik (bootstrap cluster, BH, partial Spearman) dengan nilai referensi yang diketahui.

## EKSPERIMEN SURROGATE — rencana dicicil per arm (disepakati 2026-10-09)
Dikerjakan **satu arm per sesi**, rest/cooldown di antaranya (laptop, GPU sempat 74 C). Semua resumable:
`last.pt` disimpan tiap epoch validasi, `--resume <path last.pt>` lanjut persis dari epoch terakhir.
Script: `guidance/surrogate_data/train_surrogate_arm.py` (pre-registrasi lengkap ada di docstring-nya).

**Desain (pre-registered, JANGAN diubah setelah lihat hasil):** val & test SELALU LP split murni
CrossDocked, tidak pernah BN, tidak pernah di-subsample — identik di semua arm.

| Arm | Komposisi train | Perannya |
|---|---|---|
| 0 | 6.000 CD | Anchor. Reproduksi budget data Stage 0 di batch size baru, supaya perubahan batch size bisa disingkirkan sebagai penyebab beda hasil nanti. Termurah -> dijalankan pertama sebagai sanity check. |
| B | 6.000 CD + 40.964 BN | **Tes.** Total sama dengan A. Membawa sinyal abort: kalau hancur, jalur BN berhenti di sini. |
| A | 46.964 CD | **Kontrol jumlah data.** Wajib untuk MENAFSIRKAN B (kalau B naik, itu karena BN atau karena 47K>6K?). |
| C | 46.964 CD + BN (komposisi & filter dipandu hasil A/B) | Run produksi, bukan eksperimen terkontrol. Yang dipakai untuk gate DIAG1 + integrasi TNN. |

**Urutan: tes batch size -> 0 -> B -> A -> baca hasil -> C.** B sebelum A karena B yang memberi sinyal
berhenti lebih awal; tapi kesimpulan apa pun butuh 0, A, DAN B selesai.

**Perintah (3 seed per arm: 2021, 2022, 2023; ganti `--seed`):**
```
# 0. tes batch size dulu (sekali saja, cari yang muat di 4GB)
PYTHONPATH=. ~/miniconda3/envs/drugdisc/bin/python guidance/surrogate_data/train_surrogate_arm.py \
  configs/prop/crossdocked_affinity_egnn.yml --arm_tag bstest --cd_rows 2000 --batch_size 8 --seed 2021

# Arm 0 (termurah, ~1 jam utk 3 seed kalau batch size jalan)
... --arm_tag arm0 --cd_rows 6000              --batch_size <BS> --seed 2021
# Arm B
... --arm_tag armB --cd_rows 6000 --bn_rows 40964 --batch_size <BS> --seed 2021
# Arm A
... --arm_tag armA --cd_rows 46964             --batch_size <BS> --seed 2021
# resume kalau terputus:
... --arm_tag <sama> --resume ./logs_surrogate_arms/<folder>/checkpoints/last.pt
```

**Catatan yang sudah diputuskan:**
- Baris BN `censored` (2.772, afinitasnya "<nilai") **dibuang default** — melatihnya sebagai nilai eksak
  menyuntik label noise berarah. `--bn_keep_censored` kalau mau dipertahankan.
- Filter kualitas `--bn_max_core_rmsd` / `--bn_min_similarity` tersedia, **default mati** untuk Arm B
  (biar B jadi tes BN apa adanya); dipakai nanti di Arm C kalau hasil A/B membenarkan.
- Batas yang diakui: pose BN = hasil pemodelan template, LP test = pose docking CrossDocked. Setiap arm
  ber-BN menyeberangi batas domain yang tidak dilalui arm CD-saja. Arm B sengaja menahan 6.000 baris CD
  (bukan swap murni) supaya asimetri domain ini berkurang.
- Estimasi total: **~31 jam kalau batch_size=8 muat, ~189 jam kalau tetap batch_size=1.** Selisih 6x itu
  sebabnya tes batch size didahulukan.

## Ekstraksi pocket10 BindingNet (2026-10-09 sore)
`guidance/surrogate_data/extract_bindingnet_pockets.py` — menghasilkan layout on-disk yang SAMA dengan pocket10 CrossDocked (dir pocket PDB + ligand SDF + `index.pkl`), supaya `PocketLigandPairDataset` bisa memprosesnya ke LMDB tanpa loader baru dan featurization yang sudah tervalidasi dipakai apa adanya. Output: `data/bindingnet_pocket10/` + `labels.csv` (idx -> pk).

**3 temuan yang kalau kelewat akan membatalkan perbandingan dengan Stage 0 (semua diverifikasi empiris, bukan diasumsikan):**
1. **Hidrogen WAJIB dibuang.** Dicek langsung dari LMDB yang model Stage 0 benar-benar latih: atom protein HANYA Z=6,7,8,16 (C/N/O/S), NOL Z=1, rata-rata 447,6 atom/pocket. Receptor BindingNet (`rec_h_opt.pdb`) terprotonasi (~6,5k atom, H dinamai `1H`/`2H`). Kalau dibiarkan: jumlah atom hampir dobel dan dengan `knn=48` di encoder, hidrogen akan menggusur atom berat yang membawa sinyal — distribusi yang belum pernah dilihat checkpoint Stage 0. H dibuang dari teks PDB SEBELUM `PDBProtein` parse, jadi center-of-mass residu (yang menentukan seleksi pocket) juga heavy-atom.
2. **Nama residu gaya AMBER.** Scan 300 template: ada `HID`, `HIE` (varian protonasi histidin), `CYM` (sistein tiolat) — tidak ada di `PDBProtein.AA_NAME_NUMBER` (hanya 20 nama standar) sehingga KeyError. Dinormalisasi ke residu induk (HIS/CYS, plus set AMBER standar lain: HIP/CYX/ASH/GLH/LYN/ARN/TYM). Lossless karena variannya cuma beda hidrogen/protonasi yang toh dibuang. Nama tak dikenal di luar daftar itu di-skip + dilaporkan, bukan bikin crash.
3. **Kolom element (77-78) kosong** di PDB BindingNet, jadi `PDBProtein` jatuh ke fallback `line[13:14]`. Kebetulan benar di sini karena semua elemen satu huruf (C/N/O/S/H) dan field atom-name ter-pad sehingga posisi itu tepat kena huruf elemen (termasuk H berprefiks digit seperti `1H`). Filter H saya meniru logika inferensi elemen yang sama persis supaya tidak bisa berbeda pendapat dengan `PDBProtein`.

**Validasi pilot (200 baris) sebelum jalan penuh:** 200/200 berhasil; elemen pocket C/O/N/S saja (nol H); 322-484 atom (mean 383) vs CrossDocked 447,6 — sebanding; `PocketLigandPairDataset` memproses ke LMDB dengan sukses dan `protein_atom_feature` keluar 27 dimensi = `FeaturizeProteinAtom.feature_dim` persis. Jadi formatnya benar-benar kompatibel, bukan cuma "kelihatan jalan".

**BUG BLOKIR yang ditemukan & diperbaiki:** `utils/data.py` dan `datasets/protein_ligand.py` masih pakai alias NumPy yang DIHAPUS di NumPy 2.0 (`np.compat.long`, `np.long`, `np.int`, `np.bool`) sementara env ini NumPy 2.2.6 — artinya `parse_sdf_file`/`PDBProtein.to_dict_atom` RUSAK untuk pemrosesan data baru apa pun. LMDB CrossDocked yang ada dibuat sebelum env di-upgrade, itu sebabnya training lama tetap jalan (baca LMDB jadi, tidak pernah panggil fungsi ini). Diganti ke `np.int64`/`bool` — behavior-preserving di 64-bit Linux (alias lama memang int64/bool). Diverifikasi: jalur LMDB existing tetap memuat dtype identik (int64/bool). Backup pra-edit ada di scratchpad sesi.

## TNN (QAT, diputuskan 2026-10-08 — setelah A1f jadi kandidat paling menjanjikan, lalu dibantah replikasi)
- [x] Mekanisme QAT ternary (STE, `guidance/tnn_qat.py`) selesai ditulis + ditest di CPU: 34 `nn.Linear` berhasil dikonversi ke `TernaryLinear`, warm-start dari checkpoint Stage 0 FP32 jalan normal, gradien mengalir lewat trik STE (dikonfirmasi non-zero), tidak NaN/meledak di 15 step, sparsity ~35-45% per layer (cocok dengan probe post-hoc sebelumnya ~37%).
- [x] Replikasi A1f (evidential) seed 2022, 2023 — **hasil: ranking signal seed2021 (p_BH=0,042) TIDAK replikasi** (seed2022 p_BH=0,61, seed2023 p_BH=0,26), arah kalibrasi malah terbalik antar seed. Lihat update di bagian A1f di atas — kesimpulan final 5 dari 5 metode point-wise gagal.
- [x] **A1g (QAT ternary) seed 2021 SELESAI — hasil kuat, bukan "melebihi" FP32 seperti kelihatannya di val.** Val R2 sempat ke 0,370 di epoch 6 (early-stop di epoch 11, checkpoint terbaik = epoch 6 val loss 1,852), tapi **di TEST set (baru dicek sekarang, bukan dilihat selama training)**: R2=0,3239, Pearson=0,6132 — vs baseline FP32 R2=0,342, Pearson=0,609. Jadi SETARA baseline (R2 sedikit lebih rendah, Pearson sedikit lebih tinggi), bukan "lebih baik" seperti narasi val tadi. **Tetap hasil kuat**: ternarisasi naif post-hoc GAGAL TOTAL (R2=-1,497), tapi quantization-aware training (STE) menyelamatkan ke level setara FP32 penuh. Koreksi diri: jangan percaya angka val sebagai klaim final, selalu cek test. Hasil: `guidance/uncertainty_a1/a1g_test_preds_s2021.npz`.
- [x] **Replikasi A1g seed 2022 SELESAI — REPLIKASI BERSIH.** Test R2=0,3201, Pearson=0,5975 (seed 2021: R2=0,3239, Pearson=0,6132). Dua seed beda cuma 0,004 di R2 — kebalikan dari kasus evidential yang runtuh saat direplikasi. **Koreksi klaim**: dengan 1 seed, 0,324 vs FP32 0,342 kelihatan noise ("setara"); dengan 2 seed yang dua-duanya ~0,32, gap ~0,02 R2 ternyata SISTEMATIS, bukan noise. Klaim jujur: **ternary QAT bayar ongkos akurasi kecil tapi konsisten (~6% relatif di R2), Pearson praktis sama** (0,60-0,61 vs 0,609). Jauh lebih baik dari post-hoc (R2=-1,497) tapi bukan "gratis".
- [x] **A1g SELESAI 3 SEED — KESIMPULAN FINAL: ternary QAT TIDAK BISA DIBEDAKAN dari FP32.** seed2021 R2=0,3239/Pearson 0,6132; seed2022 0,3201/0,5975; seed2023 **0,3872**/0,6298. **Mean R2=0,3437 (sd 0,0377)** vs FP32 0,3420 → selisih **+0,0017**, sementara sd antar-seed **22x lebih besar** dari selisihnya. Mean Pearson 0,6135 vs 0,609.
  **DUA KALI saya salah di sini, dicatat supaya tidak terulang:** (1) dengan 1 seed saya bilang "setara FP32" padahal varians belum diukur; (2) dengan 2 seed (0,3239 & 0,3201, dua-duanya rendah) saya menyimpulkan gap ~0,02 itu "SISTEMATIS, bukan noise" — seed 2023 di 0,3872 membantah total. Pelajaran sama dengan kasus evidential: **n=2 tidak cukup kalau sd antar-seed sebesar ini**, dan jangan sebut sesuatu "sistematis" sebelum variansnya diukur.
  Klaim jujur final: **ternarisasi via QAT tidak berbiaya akurasi yang terdeteksi** (vs post-hoc yang hancur, R2=-1,497). Hasil: `a1g_test_preds_s{2021,2022,2023}.npz`.

## Pengukuran batch size + AMP (2026-10-09, `guidance/surrogate_data/bench_batch_size.py`)
Diukur sebelum komit jam GPU. **Hipotesis saya bahwa batching memberi ~6x percepatan TERBANTAH:**

| batch | AMP | rows/sec | peak VRAM |
|---|---|---|---|
| 1 | - | 11,6 | 0,96 GB |
| 2 | - | 12,0 | 1,61 GB |
| 4 | - | 11,9 | 3,06 GB |
| 8 | - | **OOM** | - |
| 1 | bf16 | 14,5 | 0,89 GB |
| 2 | bf16 | 15,9 | 1,56 GB |
| **4** | **bf16** | **16,0** | **2,84 GB** |

Throughput **datar** terhadap batch size sementara VRAM naik linear sampai OOM di 8 — sebabnya pocket ~450 atom dengan knn=48 menghasilkan ~23.000 edge per kompleks, jadi satu sampel sudah menjenuhkan GPU; tidak ada under-utilization untuk dieksploitasi (beda dengan model citra). Yang benar-benar membantu cuma **AMP bf16: 1,38x**.

**Konfigurasi arm: `--batch_size 4 --amp`.** Biaya di 16,0 rows/sec (3 seed, ~10 epoch): Arm 0 = **4,1 jam**; Arm A/B @46.964 = **25,5 jam masing-masing**; @20.000 = 11,4 jam; Arm C = 30 jam (1 seed). Total 0+A+B = **55 jam** (27 jam kalau A/B dikecilkan) — bukan 31 jam seperti estimasi optimistis saya.

**AMP mengubah numerik**, jadi **Arm 0 merangkap cek bahwa optimasi ini tidak merusak akurasi** — kalau Arm 0 gagal mereproduksi R2~0,34, masalahnya di AMP/batch bukan di data, dan semua arm sesudahnya tidak bisa dipercaya.
- [x] **Overengineering (sesuai permintaan 2026-10-08, "remember to overengineer")**: `guidance/uncertainty_a1/stats_common.py` — modul statistik bersama (bh, boot_p, partial_spearman, cluster_bootstrap_ci), `mc_dropout_calibration.py` di-refactor untuk pakai ini (diverifikasi byte-identical terhadap hasil A1 yang sudah dilaporkan). Unit test dengan nilai referensi independen (bukan cuma re-run kode sendiri): `guidance/uncertainty_a1/tests/test_stats_common.py`, 11 test lolos — termasuk cross-check partial Spearman terhadap formula closed-form textbook, invariant exact untuk cluster bootstrap dengan grup berukuran sama, dan contoh BH yang dihitung tangan dengan ties.

## Cabang ketidakpastian (dibuka 2026-10-07, lanjut setelah A1c gagal)
- [x] **A1e SELESAI — GATE GAGAL, hasil negatif ke-4.** Spearman(sigma,|err|)=-0,011, CI[-0,134,0,132], p_BH=0,94 (tidak signifikan, bahkan arah point estimate-nya salah). Rasio kalibrasi=1,55, CI[1,31,1,80], p_BH=0,0015 — masih signifikan gagal, TAPI ini yang PALING BAIK dari 4 metode (A1 9,7x -> A1b 64,8x -> A1c 3,59x -> A1e 1,55x, membaik terus tapi belum pernah lolos). R2(mu)=0,291, Pearson=0,593 (sedikit di bawah baseline MSE 0,342). **4 dari 4 metode sigma point-wise gagal** (MC Dropout x2 arsitektur, Deep Ensemble, Heteroscedastic NLL) — cuma conformal prediction (paradigma beda, marginal coverage bukan ranking point-wise) yang lolos. Hasil: `guidance/uncertainty_a1/a1e_results.json`. **Temuan metodologis penting sebelum run ini**: pilot pertama dari scratch (random init) langsung GAGAL dalam <10 step — mu acak jauh dari y bikin loss meledak, log_var langsung mentok di clamp ceiling (variance collapse, sigma~20) dalam 3 step, sebelum mu punya kesempatan belajar. Ini pathology yang sudah dikenal di literatur heteroscedastic NLL. **Fix: warm-start mu dari checkpoint Stage 0 MSE yang sudah ada** (`warm_start()` di `train_egnn_heteroscedastic.py`, cuma transplant bobot, bukan retrain) — re-test pilot setelah fix: mu ikuti y dari step 0, sigma stabil di 1,03-1,28, tidak collapse. **Pelajaran ini dicatat supaya tidak diulang**: training heteroscedastic/evidential head dari scratch tanpa warm-start berisiko tinggi gagal.
- [x] **A1f SELESAI — HASIL PALING MENJANJIKAN dari 5 metode, meski masih belum lolos gate penuh.** Training 15 epoch, best epoch 10 (val loss 1,833, R2 0,333). Beda dari 4 metode lain: **Spearman(sigma,|err|)=0,116, CI[0,002,0,215], p_BH=0,042 — SIGNIFIKAN.** Partial Spearman (kontrol n_lig)=0,146, p_BH=0,012 — juga signifikan, malah lebih kuat. Ini pertama kalinya ranking signal nongol di 5 metode yang dicoba. Masalahnya di skala: rasio kalibrasi mentah=0,040 (sigma ~25x TERLALU BESAR, arah kebalikan dari 4 metode lain yang selalu terlalu kecil) — sigma rata-rata 35 padahal rentang pK cuma ~0-14.
  - **Koreksi skala (post-hoc, fit di val, bukan test)**: karena Spearman invariant terhadap rescaling monoton, saya coba kalikan sigma dengan faktor koreksi dari val (`guidance/uncertainty_a1/analyze_evidential_corrected.py`). Rasio kalibrasi setelah koreksi: **1,178, CI95[1,021, 1,420], p_raw=0,029** — jauh lebih baik (dari 25x jadi ~18% kemelesetan), TAPI CI masih belum menyentuh 1,0, jadi masih gagal secara teknis. Kemungkinan karena val dan test punya target protein yang sama sekali tidak tumpang tindih (shift distribusi ringan antar populasi target).
  - **Kesimpulan**: A1f adalah satu-satunya metode dari 5 yang punya ranking signal asli. Dengan koreksi skala sederhana sudah sangat dekat lolos (bukan 25x lagi, cuma ~18%). Ini kandidat terbaik untuk thesis sebagai "hasil positif parsial", beda dari 4 hasil negatif bersih lainnya. LAMBDA=0,01 (regularizer evidential) dari paper asli, belum di-tuning untuk dataset kita — limitasi yang diakui, bukan diklaim optimal; tuning lambda bisa jadi langkah lanjutan kalau mau dikembangkan.
  - Hasil: `guidance/uncertainty_a1/a1f_results.json`, `a1f_corrected_results.json`.
  - **UPDATE 2026-10-09 — REPLIKASI MEMBANTAH hasil "menjanjikan" di atas.** 2 seed tambahan (2022, 2023) dilatih + dianalisis: Spearman seed2022=0,030 (p_BH=0,61, TIDAK signifikan), seed2023=0,060 (p_BH=0,26, TIDAK signifikan) — beda jauh dari seed2021 (0,116, p_BH=0,042). Rasio kalibrasi juga BERBALIK arah: seed2021 sigma 25x terlalu BESAR, tapi seed2022/2023 sigma malah ~1,8x terlalu KECIL. **Kesimpulan: hasil seed2021 kemungkinan besar fluke statistik, bukan properti asli evidential regression di arsitektur ini — sigma-nya sendiri tidak stabil antar seed.** Ini contoh bagus kenapa replikasi penting: HAMPIR melaporkan hasil false positive kalau cuma pakai 1 seed. **Hasil final: 5 dari 5 metode point-wise (MC Dropout x2, Deep Ensemble, Heteroscedastic, Evidential) GAGAL replikasi ranking signal. Cuma conformal prediction (A1d) yang lolos** — justru karena dia TIDAK butuh ranking point-wise, cuma marginal coverage. Hasil: `guidance/uncertainty_a1/a1f_results_s2022.json`, `a1f_results_s2023.json`.
- [x] **A1d: conformal prediction — LOLOS GATE, hasil pertama yang berhasil dari 4 metode UQ.** Split-conformal pakai checkpoint Stage 0 EGNN (tanpa retrain), kalibrasi di val (n=6.069), target 90% coverage. q=2,196 (unit pK). Test coverage=0,8952, CI95[0,857, 0,925] — nominal 0,90 masuk CI (GATE LOLOS). Sanity kalibrasi: coverage di val sendiri=0,9001 (persis seperti seharusnya). Kekhawatiran exchangeability (target val dan test tidak overlap sama sekali) sudah dicek dan TIDAK bikin coverage rusak secara empiris. Catatan: interval cukup lebar (+-2,2 pK) — valid tapi tidak tajam, trade-off yang memang diharapkan dari conformal. Hasil: `guidance/uncertainty_a1/a1d_conformal_results.json`.
- [x] Deep ensemble: base model diputuskan **EGNN Stage 0** (bukan GIGN). Alasan: dicek dulu log training penuh GIGN (`logs_track_a_full/.../log.txt`) — val-loss-terbaik-di-epoch-1 BUKAN bug, melainkan konsekuensi terdokumentasi dari physics head Track A yang tidak punya learned scale/bias (Pearson/Spearman tetap stabil ~0,58-0,62 di semua epoch, R2 selalu negatif apa pun kriteria checkpoint-nya — dicek juga best_by_pearson.pt, epoch 5, test R2 masih -0,281). GIGN punya ceiling lemah secara struktural; EGNN Stage 0 jauh lebih kuat (R2 0,342) dan lebih murah per run.
- [x] **A1c SELESAI 2026-10-08 — GATE GAGAL, hasil negatif ke-3.** Spearman(sigma,|err|)=0,008, CI[-0,099,0,132], p_BH=0,837 (tidak signifikan — disagreement ensemble sama sekali tidak mengurutkan error). Rasio kalibrasi=3,59, CI[3,14,4,09], p_BH=0,0015 (sigma ~3,6x terlalu kecil — tapi ini yang PALING KECIL dari 3 metode UQ yang dicoba). Partial Spearman (kontrol n_lig)=0,008, p_BH=0,837. Deskriptif: R2 rata-rata ensemble=0,407, Pearson=0,647 (prediksi rata-ratanya sendiri OK, masalahnya cuma sigma-nya tidak bawa informasi error). **MC Dropout (GIGN), MC Dropout (EGNN), Deep Ensemble (EGNN) — 3 dari 3 metode UQ gagal.** Hasil: `guidance/uncertainty_a1/a1c_results.json`. **Keputusan berikutnya**: tutup cabang ketidakpastian dengan hasil negatif (3x), atau coba 1 metode lagi (conformal prediction / heteroscedastic head, lihat bagian "Cabang ketidakpastian" di bawah) — ini perlu keputusan Anda lagi.
- [ ] [opsional] Conformal prediction dengan kalibrasi per cluster target (memberi cakupan interval yang jelas).
- [ ] [opsional] Head heteroscedastic atau evidential, yaitu model yang langsung memprediksi variansinya.
- [ ] [opsional] A3 (akuisisi UCB/EI): hanya masuk akal kalau ada σ yang lolos gate dari deep ensemble di atas.

## Tabel A (roadmap awal)
- [ ] A2: data PDBbind raw belum ada di `data/`. Perlu lisensi PDBbind. Tidak memblokir A1 sampai A6.
- [ ] A4: PCGrad afinitas vs RA-score. Perlu implementasi proyeksi gradien dan uji di banyak pocket (Track D baru punya satu pocket yang powered).
- [ ] A5: gate in silico (ligand efficiency dan jumlah atom berat), memakai hasil yang sudah ada. Murah. Bisa jadi gate untuk A3 sampai A6.
- [ ] A6: loop tertutup dengan pipeline docking Vina. Perlu loop retrain dan pelacakan jumlah atom. Lebih berguna sebagai kontrol daripada sebagai metode.

## Data
- [x] BindingDB diindeks + difilter ulang dari file mentah (`guidance/surrogate_data/build_bindingdb_index.py`, bukan sekadar kutip angka lama): 3.243.660 baris discan, **2.840.436 baris bersih tersisa** (ada afinitas + lolos filter reseptor-PDB-ID dan UniProt accession, dicek di semua 50 kolom chain yang BindingDB sediakan), **8.529 UniProt accession unik** — cocok dengan angka sesi sebelumnya (2,8 juta/8.463), sekarang terverifikasi ulang dan reproducible. Breakdown: IC50 1.930.989, Ki 567.139, EC50 249.701, Kd 97.757 baris (bisa overlap per baris). Disimpan di `guidance/surrogate_data/bindingdb_index_filtered.csv`.
- [ ] [wajib] BindingDB masih hanya berisi ligan + afinitas, BELUM ada struktur pocket. Perlu dipasangkan dengan struktur (CrossDocked yang sudah kita punya, atau BindingNet setelah Crystal_Templates selesai).
- [x] BindingNet v1 diekstrak + diindeks (`guidance/surrogate_data/build_bindingnet_index.py`): 159.061 entri, 7.149 PDB template unik, 802 target ChEMBL, 65.919 compound ChEMBL. Label afinitas (Ki/IC50/Kd/EC50, semua satuan nM) diparse dari REMARK di file .pdb ligand (bukan file index terpisah) — termasuk tangani kasus "<value" (censored, 2.772 baris: afinitas sebenarnya lebih kuat dari angka yang tercatat). Filter ID PDB reseptor vs val+test: 24.830 baris (946 template ID) dibuang karena overlap.
- [x] Filter target ChEMBL-vs-UniProt selesai (`guidance/surrogate_data/build_target_crossref.py`, API publik ChEMBL + UniProt): 176/176 nama target kita (dari 192 dir val+test) berhasil diresolve ke accession UniProt. 74 dari 764 target ChEMBL (yang lolos filter PDB-ID) ternyata protein yang sama dengan target val+test kita lewat entri PDB berbeda — menangkap **8.988 baris tambahan** yang kelewat oleh filter PDB-ID saja. **Total bersih setelah kedua filter: 125.243 / 159.061 baris (78,7%)**, disimpan di `guidance/surrogate_data/bindingnet_v1_index_filtered2.csv`.
- [x] `Crystal_Templates_for_BindingNet1.tar.gz` (2,18GB, 51.538 entri: `{pdb_id}/rec_h_opt.pdb` + `{pdb_id}/cry_lig_opt_converted.sdf` per template) **selesai diunduh 2026-10-08 06:xx WIB** via `guidance/surrogate_data/resume_crystal_templates.py` (chunk 5MB + backoff). Sempat macet berulang di offset 395MB (bukan sekadar lambat — dites langsung, offset itu dan +20MB gagal, -20MB berhasil, indikasi ada zona rusak di CDN Zenodo), tapi resumer berhasil tembus lewat retry sendiri. Arsip sudah diverifikasi utuh (`tar -tzf` baca penuh tanpa error). **Belum**: ekstrak + filter leakage (946 template ID yang sudah diketahui overlap val+test harus dibuang juga dari sini) + pairing ke index ligand yang sudah difilter (`bindingnet_v1_index_filtered2.csv`).
- [ ] [berguna] BDB2020+ (115 kompleks, tanpa overlap dengan split kita): kandidat test eksternal bersih. Perlu preprocessing pocket ke format pocket10 kita dari `protein.pdb` dan `ligand.sdf`.
- [ ] [wajib] LP-PDBBind: metadata dan split tersedia. Struktur harus diunduh dari PDBbind dengan lisensi. Overlap ID PDB tinggi dengan split kita (misalnya 1.059 ID dari test LP-PDBBind ada di train kita). Wajib difilter sebelum training.
- [ ] [berguna] Bandingkan split LP kita dengan split LeakProof di `LP_PDBBind.csv` (kolom `new_split`).
- [ ] [berguna] Tinjau temuan ESM2 yang sudah ada (`logs_lp_split_stage0_esm2`, `logs_lp_split_stage0_esm2pocket`) sebelum mengusulkan cabang sekuens.

## Surrogate (rencana final berurutan, diputuskan 2026-10-07 setelah scoping)
**Temuan scoping penting (sebelum rencana ini ditulis)**: "surrogate pengganti docking di guidance loop" BUKAN pekerjaan baru — sudah ada 3x (`AffinityGuidance`/EGNN, `SynthGuidance`/RA-score, `GIGNPignetGuidance`/Track A), lewat interface `GuidanceModel` yang reusable (`guidance/interfaces.py`, `guidance/guided_sampling.py`). Vina TIDAK PERNAH dipanggil di dalam loop denoising — selalu post-hoc sebagai ground-truth check. Ketiga percobaan FALSIFIED: skor surrogate naik tapi skor Vina asli turun ("Vina-hacking", `DUAL_FALSIFICATION_CONCLUSION.md`, `TRACK_D_SYNTH_GUIDANCE_REPORT.md`). DIAG1 sudah mendiagnosis sebabnya di sisi afinitas: arah gradien surrogate menjauh dari kontak pocket, bukan menuju — ini masalah geometri gradien off-manifold (model dilatih di pose kristal, dipakai di state diffusi yang noisy), BUKAN masalah kurang data. Jadi cuma nambah data ke resep training yang sama (MSE regression di pose kristal) berisiko mengulang kegagalan yang sama untuk ke-4 kalinya.

Rencana (berurutan, bukan pilih satu):
- [ ] [wajib] 1. Bangun surrogate baru dengan data besar: BindingDB + BindingNet v1 + LP-PDBBind, difilter bersih dari semua ID PDB/target di val+test (lihat bagian Data di bawah). Training EGNN/GIGN, dibandingkan ke Stage 0 di LP test yang sama.
- [ ] [wajib] 2. GATE WAJIB sebelum klaim berhasil atau masuk guidance loop: ulangi cek DIAG1 (korelasi arah gradien vs jarak ke kontak pocket) dan Vina-hacking check (apakah menaikkan skor surrogate benar-benar menaikkan skor Vina asli, bukan cuma skor sendiri) pada surrogate baru ini — pre-registrasi kriteria dulu, sebelum lihat hasil, sama seperti A1/A1b/A1c.
- [ ] [wajib, kondisional] 3. Kalau gate #2 GAGAL (skenario realistis — riwayat 3/3 falsified): coba PCGrad (TASKS.md A4, proyeksi gradien afinitas vs RA-score, belum pernah diimplementasi) sebagai usaha spesifik menyasar failure mode DIAG1, diuji di banyak pocket (Track D dulu cuma 1 pocket yang powered). Kalau gate #2 LOLOS, baru masuk ke `GuidanceModel` interface yang sudah ada (tidak perlu wiring baru).
- [x] 4. Peran aman surrogate untuk thesis (tidak bergantung hasil #2-#3): (b) model pelatihan tambahan biasa — prediktor afinitas dibandingkan ke Stage 0, reportable apa pun hasil gate guidance.
- [ ] [berguna] Evaluasi eksternal pada BDB2020+ (115 kompleks).
- [ ] [opsional] Ensemble model surrogate, atau distilasi dari model yang lebih besar.

## Engineering (top-notch)
- [ ] [berguna] Pelacakan eksperimen (MLflow atau W&B). Saat ini hanya log teks dan TensorBoard.
- [ ] [berguna] Pisahkan helper statistik ke modul bersama. Saat ini `mc_dropout_calibration.py` dan `mc_dropout_egnn_a1b.py` saling mengimpor, dan load model masih spesifik per arsitektur.
- [ ] [berguna] Satu skrip per analisis, dengan output JSON dan log yang konsisten.
- [ ] [berguna] README per eksperimen (A1, A1b, dan yang berikutnya).
- [ ] [opsional] Pipeline Makefile atau Snakemake untuk seluruh alur analisis.
- [ ] [opsional] CI untuk unit test dan pemeriksaan kebocoran split otomatis.

## Thesis dan dokumentasi
- [ ] [wajib] Perbarui X.4.1 dengan hasil negatif A1 dan A1b, termasuk figure (scatter σ vs |error|, kurva kalibrasi per desil σ, forest plot CI).
- [ ] [wajib] Perbarui CI Stage 0 di thesis setelah cluster bootstrap selesai.
- [ ] [wajib] Tambahkan lampiran tentang kebocoran data: overlap ID PDB dan target antara split kita dan database eksternal.
- [ ] [wajib] Rebuild PDF dan DOCX setelah semua pembaruan. Target: halaman dan referensi dicek ulang.
- [ ] [wajib] Perbarui README.
- [ ] [berguna] Figure ringkasan semua uji (satu tabel forest plot).

**Antrian rebuild thesis (2026-10-07) — semua ini butuh SATU rebuild PDF/DOCX setelah lengkap, bukan rebuild berkali-kali:**
1. CI Stage 0 (R2, Pearson) diganti ke versi cluster-per-target — lihat hasil di atas (lebar CI naik ~8x).
2. Lampiran/catatan overlap PDB ID train/test (2 ID, 7 kompleks, kasus heterodimer RXRA/PPARG/RARA) — detail presisi di atas.
3. X.4.1 hasil negatif A1 + A1b (sudah ada datanya, belum ditulis ulang ke teks).
4. A1c (deep ensemble EGNN) — MENUNGGU training 4 seed selesai (~6-10 jam, sedang berjalan di background per 2026-10-07 20:57 WIB).
Tunggu training A1c selesai dulu sebelum rebuild, supaya tidak rebuild dua kali.

## Housekeeping
- [x] Hapus folder run gagal `logs_a1b_egnn_mcdrop/crossdocked_affinity_egnn_mcdrop_2026_10_06__21_08_11_a1b` (hanya log 19 detik, tanpa checkpoint) — dihapus 2026-10-07, dikonfirmasi user. Folder `_21_09_08_a1b` (run ulang yang berhasil) tidak disentuh.
- [x] Commit: sudah ada per 2026-10-07 (commit `2055110`, "another commit") — mencakup README.md, PDF/DOCX thesis, `10_conclusion.txt`, `train_egnn_stage0.py`, `prop_egnn.py`, `prop_model.py`, `guidance/uncertainty_a1/`, dan `TASKS.md`. Working tree bersih, branch `main` in sync dengan `origin`. (Baris ini sempat salah/stale — dicek ulang via `git status`/`git log`.)
- [ ] [berguna] Hapus file sementara setelah unduhan selesai: `data/bindingnet_v1/files.json`, `download.log`, `resume.log`.

## Di luar cakupan komputasi
- [ ] [nanti, S3] Cari lab partner biotech untuk validasi eksperimental.
- [ ] Jalur virus-host (VirHostNet 3.0, HPIDB, Viruses.STRING) dan jalur desain agen biologis: tidak dikerjakan.
