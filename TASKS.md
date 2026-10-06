# To-do

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

## Sedang / menunggu
- [ ] BindingNet v1 crystal templates (2,18 GB): unduhan tersendat (106 MB), server menolak sebagian permintaan. Resume berjalan di background (`data/bindingnet_v1/resume.log`). Opsional, tidak wajib untuk arsip utama.
- [ ] Binding MOAD: situs memblokir akses otomatis (403) dan sertifikat HTTPS kedaluwarsa. Perlu unduh manual dari browser, lalu taruh di `data/binding_moad/`. Lisensi belum terverifikasi.

## Keputusan yang perlu diambil (riset)
- [ ] [wajib] Cabang ketidakpastian: tutup dengan hasil negatif (tulis di X.4.1) ATAU lanjut ke deep ensemble (5 seed GIGN, sekitar 12 jam GPU). Rekomendasi saya: tutup dulu, tulis negatif.
- [ ] [wajib] Peran surrogate: (a) pengganti docking di guidance, atau (b) model pelatihan tambahan terintegrasi arsitektur kita. Ini menentukan data dan split yang dipakai.

## Evaluasi dan rigor
- [ ] [wajib] CI Stage 0: ganti bootstrap per kompleks menjadi cluster per target (127 target test, B = 2.000). Bisa dikerjakan di CPU dengan prediksi test yang sudah ada.
- [ ] [wajib] Catat di thesis: split kita punya 2 ID PDB yang tumpang tindih antara train dan test, dan 7 kompleks test berbagi ID PDB dengan train.
- [ ] [berguna] Cek family holdout: `guidance/lp_split/build_family_holdout_split.py` belum saya baca.
- [ ] [berguna] Jalankan ulang kontrol confound ukuran pocket (`analyze_pocket_size_confound.py`) untuk model baru.
- [ ] [berguna] Pre-registrasi tertulis untuk setiap uji baru, sebelum melihat hasil.
- [ ] [berguna] Unit test untuk fungsi statistik (bootstrap cluster, BH, partial Spearman) dengan nilai referensi yang diketahui.

## Cabang ketidakpastian (lanjutan, jika dibuka)
- [ ] [opsional] Deep ensemble (5 seed) sebagai pembanding wajib untuk MC Dropout.
- [ ] [opsional] Conformal prediction dengan kalibrasi per cluster target (memberi cakupan interval yang jelas).
- [ ] [opsional] Head heteroscedastic atau evidential, yaitu model yang langsung memprediksi variansinya.
- [ ] [opsional] A3 (akuisisi UCB/EI): hanya masuk akal kalau ada σ yang lolos gate. Saat ini diblokir oleh A1 dan A1b.

## Tabel A (roadmap awal)
- [ ] A2: data PDBbind raw belum ada di `data/`. Perlu lisensi PDBbind. Tidak memblokir A1 sampai A6.
- [ ] A4: PCGrad afinitas vs RA-score. Perlu implementasi proyeksi gradien dan uji di banyak pocket (Track D baru punya satu pocket yang powered).
- [ ] A5: gate in silico (ligand efficiency dan jumlah atom berat), memakai hasil yang sudah ada. Murah. Bisa jadi gate untuk A3 sampai A6.
- [ ] A6: loop tertutup dengan pipeline docking Vina. Perlu loop retrain dan pelacakan jumlah atom. Lebih berguna sebagai kontrol daripada sebagai metode.

## Data
- [ ] [wajib] BindingDB: setelah filter val dan test (2,8 juta baris, 8.463 entry UniProt), data ini hanya berisi ligan dan afinitas. Belum ada struktur pocket. Perlu dipasangkan dengan struktur (CrossDocked atau BindingNet) sebelum bisa dipakai di model struktur kita.
- [ ] [wajib] BindingNet v1: arsip utama sudah lengkap. Perlu ekstraksi, inspeksi struktur dan label, lalu filter ID PDB dan target yang sama seperti BindingDB.
- [ ] [berguna] BDB2020+ (115 kompleks, tanpa overlap dengan split kita): kandidat test eksternal bersih. Perlu preprocessing pocket ke format pocket10 kita dari `protein.pdb` dan `ligand.sdf`.
- [ ] [wajib] LP-PDBBind: metadata dan split tersedia. Struktur harus diunduh dari PDBbind dengan lisensi. Overlap ID PDB tinggi dengan split kita (misalnya 1.059 ID dari test LP-PDBBind ada di train kita). Wajib difilter sebelum training.
- [ ] [berguna] Bandingkan split LP kita dengan split LeakProof di `LP_PDBBind.csv` (kolom `new_split`).
- [ ] [berguna] Tinjau temuan ESM2 yang sudah ada (`logs_lp_split_stage0_esm2`, `logs_lp_split_stage0_esm2pocket`) sebelum mengusulkan cabang sekuens.

## Surrogate dan model pelatihan tambahan (rencana integrasi)
- [ ] [wajib] Pilih peran surrogate (lihat keputusan di atas).
- [ ] [wajib] Susun split bersih: buang semua ID PDB dan target yang ada di val atau test sebelum training.
- [ ] [wajib] Training EGNN atau GIGN dengan data tambahan, dibandingkan dengan Stage 0 di split LP test yang sama.
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

## Housekeeping
- [ ] [wajib] Hapus folder run gagal `logs_a1b_egnn_mcdrop/crossdocked_affinity_egnn_mcdrop_2026_10_06__21_08_11_a1b` (hanya log 19 detik, tanpa checkpoint). Perlu konfirmasi spesifik.
- [ ] [wajib] Commit: belum ada. Perubahan yang belum di-commit: README.md, PDF dan DOCX thesis, `10_conclusion.txt`, `train_egnn_stage0.py`, `prop_egnn.py`, `prop_model.py`, `guidance/uncertainty_a1/`, dan `TASKS.md`. Folder `data/` dan `logs_*` tidak di-commit. Commit hanya kalau diminta.
- [ ] [berguna] Hapus file sementara setelah unduhan selesai: `data/bindingnet_v1/files.json`, `download.log`, `resume.log`.

## Di luar cakupan komputasi
- [ ] [nanti, S3] Cari lab partner biotech untuk validasi eksperimental.
- [ ] Jalur virus-host (VirHostNet 3.0, HPIDB, Viruses.STRING) dan jalur desain agen biologis: tidak dikerjakan.
