# To-do

## URUTAN KERJA (ditulis 2026-10-10, SATU-SATU, dari atas)

Dasar pemeringkatan ada di `guidance/RESEARCH_PROGRAM.md`. Aturan: satu item sekali jalan, jangan mulai
item GPU berikutnya sebelum yang sekarang selesai + GPU dingin (`scripts/queued_run.sh` menangani itu).

### CARA PAKAI QUEUE (dibangun 2026-10-10)

Satu job sekali jalan, istirahat wajib 20 menit setelah tiap job, lalu gerbang termal (<55C selama 300
detik berturut-turut) sebelum job berikutnya. Resumable: nama step yang selesai dicatat, jadi shutdown
paling buruk cuma kehilangan step yang sedang jalan -- dan step itu sendiri punya `last.pt` per epoch.

```
bash scripts/master_queue.sh status          # apa yang sudah/belum, suhu GPU sekarang
bash scripts/master_queue.sh start           # jalankan, detached, aman kalau terminal ditutup
bash scripts/master_queue.sh stop            # berhenti SETELAH step sekarang beres
bash scripts/master_queue.sh reset <nama>    # lupakan satu step supaya jalan ulang
ONLY=2 bash scripts/master_queue.sh run      # kerjakan maksimal 2 step sesi ini, lalu keluar
REST=2400 bash scripts/master_queue.sh run   # istirahat 40 menit antar step
CPU_PARALLEL=1 ...                           # step CPU tak menunggu GPU (gerbang suhu TETAP berlaku)
CPU_COOL_TO=80 ...                           # ambang suhu CPU; 0 mematikan gerbang CPU (tidak disarankan)
```

### MODE OTONOM — AKTIF SEJAK 2026-10-10 04:59 (atas izin eksplisit)

Queue jalan penuh tanpa pengawasan: 11 step, satu per satu, istirahat 20 menit setelah tiap step, lalu
gerbang termal (GPU <55 C DAN CPU <80 C, keduanya bertahan 300 detik) sebelum step berikutnya. Berhenti
sendiri pada kegagalan pertama. Resumable. Aman kalau terminal ditutup atau laptop dimatikan.

Satu perintah untuk melihat semuanya:
```
bash scripts/master_queue.sh status
```
Menampilkan: step mana sudah/belum, suhu CPU dan GPU sekarang beserta ambangnya, apakah queue hidup, dan
apakah run terakhir GAGAL atau TIMEOUT DI GERBANG (dua hal berbeda, ditandai terpisah).

**Pengaman yang ditambahkan khusus untuk mode tanpa pengawasan:** gerbang termal dulu menunggu TANPA
BATAS. Itu benar saat ada yang mengawasi, tapi tanpa pengawasan ia bisa deadlock -- suhu idle CPU laptop
ini belum pernah terukur, dan kalau ternyata menetap di atas 80 C, gerbangnya tidak akan pernah terbuka
dan tidak ada yang memberi tahu kenapa. Sekarang `GATE_MAX_WAIT=14400` (4 jam): kalau gerbang tidak
terbuka dalam 4 jam, queue berhenti dengan exit 75, menulis penanda `master_queue.gate_timeout`, dan
`status` menyatakannya sebagai "nothing ran and nothing broke" -- bukan sebagai kegagalan. Jalur ini
sudah diuji. Kalau itu terjadi, ukur suhu idle sebenarnya lalu naikkan ambangnya sekali:
`CPU_COOL_TO=85 bash scripts/master_queue.sh start`.

**Batas yang jujur tentang "monitoring":** queue-lah yang memantau dirinya sendiri. Saya tidak bisa
mengawasinya selama ~22 jam di dalam satu percakapan. Yang bisa saya lakukan: melaporkan kapan pun
ditanya, dan membaca `status` + log untuk tahu persis apa yang terjadi -- termasuk kalau ia berhenti.
Jangan andalkan ingatan saya soal apa yang hidup; `status` membacanya dari keadaan nyata.

### PROTOKOL COOL-DOWN — STATUS PER 2026-10-10 04:50

**Tidak ada apa pun yang akan start sendiri.** Queue sudah DIHENTIKAN dengan sengaja (bukan crash), dan
chain Arm B lama sudah dimatikan parent-nya. Yang masih jalan hanya Arm B seed 2021 (pid 42264, sudah
reparented ke init), yang akan selesai sendiri lalu mesin jadi idle.

Kenapa queue dihentikan padahal sudah diset: kalau dibiarkan hidup, setelah rest 20 menit ia akan
OTOMATIS memulai seed 2022. Itu bukan cool-down. Jadi queue ditaruh dalam keadaan siap-tapi-diam, dan
baru dijalankan lagi atas perintah.

Chain Arm B lama (`for s in 2021 2022 2023`) juga dimatikan parent-nya: ia dibuat SEBELUM gerbang termal
ada, jadi akan menjalankan tiga seed berturut-turut tanpa gate dan tanpa jeda -- terukur ~8 jam terus
menerus di CPU 92 C. Seed 2022 dan 2023 sekarang jadi step queue (urutan 1 dan 2), lengkap dengan gate
dan rest. Keduanya pakai `--no_compound_filter` dan itu WAJIB, bukan pilihan: seed 2021 dilatih sebelum
filter senyawa ada, jadi kalau 2022/2023 memakai filter, tiga seed satu arm akan dilatih di data berbeda
dan sebarannya berhenti menaksir varians seed.

**Melanjutkan setelah dingin** -- cek suhu dulu, lalu pilih satu:
```
bash scripts/master_queue.sh status              # lihat suhu CPU/GPU dan sisa step
ONLY=1 bash scripts/master_queue.sh start        # SATU step saja lalu berhenti  <-- disarankan
bash scripts/master_queue.sh start               # kerjakan semuanya, rest 20 menit antar step
```
`ONLY=1` itu yang paling sesuai pola "cicil satu-satu lalu istirahat": queue mengerjakan satu step,
mencatatnya selesai, lalu keluar. Jalankan lagi kapan pun mau lanjut.

**Aman mematikan laptop kapan saja.** Kalau dimatikan saat seed 2021 masih jalan, run itu resumable:
`--resume logs_surrogate_arms/crossdocked_affinity_egnn_2026_10_10__00_32_42_armBmatch_cd6000_bn14000_bs1_s2021/checkpoints/last.pt`
(`last.pt` ditulis setiap epoch validasi; `best.pt` juga sudah ada). Setelah seed 2021 selesai, tidak
ada state apa pun yang perlu dijaga -- queue melacak kemajuan lewat nama step di
`logs_queue/master_queue.done`.

### TEMUAN TERMAL YANG MENGUBAH DESAIN QUEUE (2026-10-10)

Diukur saat HANYA Arm B jalan, load average 1,3 dari 20 thread:

| sensor | suhu |
|---|---|
| GPU | 72 C |
| x86_pkg_temp (paket CPU) | **87-94 C** |
| TCPU / acpitz | 92 C |

**CPU adalah bottleneck termal laptop ini, bukan GPU.** i7-12700H Tjmax 100 C, jadi mesin ini duduk ~8 C
dari throttling dengan satu thread dataloader sibuk. GPU-nya justru santai.

Konsekuensi: gerbang termal versi pertama hanya memantau GPU, yaitu **sensor yang salah** -- ia akan
dengan senang hati memulai pekerjaan baru saat komponen yang panas hampir habis. Bukti yang sempat saya
pakai ("suhu GPU tetap 73 C saat beban CPU jalan") ternyata mengukur hal yang keliru: GPU memang aman
sementara CPU-nya mendidih.

Sudah diperbaiki: `queued_run.sh` sekarang punya `--cpu-cool-to` (default 80 C) dan membaca MAKSIMUM
dari zone x86_pkg_temp / TCPU / acpitz / coretemp (mereka beda sampai 7 C; yang konservatif yang dipakai;
pembacaan <10 C atau >130 C ditolak supaya zone rusak tidak memblokir queue selamanya). Step gpu DAN cpu
dua-duanya lewat gerbang CPU, karena analisis "CPU-only" justru membebani komponen yang sudah panas.

`CPU_PARALLEL=1` hanya menghapus WAIT terhadap trainer lain, **tidak pernah** menghapus gerbang suhu.
Jadi meminta eksekusi paralel tidak mengalahkan pengukuran: step cpu mulai begitu CPU punya headroom,
dan sensornya yang memutuskan kapan. Saat ini (CPU 92 C) itu berarti ia tetap menunggu -- tapi karena
alasan terukur, bukan karena pilihan konfigurasi saya.

Menambah pekerjaan: tambahkan satu baris `nama|gpu-atau-cpu|perintah` di `scripts/queue_steps.sh`.
Runner melewati yang sudah tercatat selesai, jadi menambah ke queue yang sudah jalan separuh itu aman.
JANGAN mengganti nama step yang sudah selesai -- state dicatat per NAMA, jadi ganti nama = jalan ulang.

Verifikasi yang sudah dilakukan (7 jalur diuji, bukan diasumsikan): step gagal MENGHENTIKAN queue dan
melaporkannya (chain lama pakai `set -e` dan TIDAK berhenti -- 3 run gagal, lapor sukses); resume
melewati yang selesai dan mengulang yang gagal; `stop` dihormati di tengah jalan; stopfile basi TIDAK
memblokir sesi baru; step GPU benar-benar menunggu trainer yang jalan; `ONLY=N` membatasi.

Log: `logs_queue/master_queue.log` (queue), `logs_queue/step_<nama>.log` (per step).

### SEDANG JALAN — tidak perlu tindakan
| # | item | di mana | status |
|---|---|---|---|
| 0a | Arm B-matched, 3 seed (6k CD + 14k BN stratified) | GPU | epoch 6, seed 2021/3 |
| 0b | pose sensitivity n=300 (eksperimen kunci) | CPU | **SELESAI** -- hasil di bawah |
| 0c | queue master: 5 run A1e-beta + 3 analisis | GPU/CPU, antre | menunggu 0a |

Hasil 0b (n=300, CI target-clustered): faktorial strip -- antarmuka UTUH 1.2451 pK [1.1021, 1.3694],
antarmuka HANCUR 1.0576 [0.8953, 1.2259], separuh acak 1.2686 [1.1575, 1.3811]. CI tumpang tindih, jadi
isi antarmuka TIDAK berpengaruh; yang berpengaruh jumlah atomnya. Menghancurkan pose (geser 8 A keluar
pocket) hanya 0.2782 pK, dose-response monoton 0.021/0.042/0.086/0.160/0.278 untuk 0.5/1/2/4/8 A.
**Model 4,5x lebih sensitif ke BERAPA BANYAK atom protein daripada ke DI MANA ligannya.** Pilot n=10
terkonfirmasi di n=300.

Pantau: `bash scripts/master_queue.sh status`

---

### 1. KONTROL POSITIF (RQ2) — PRIORITAS TERTINGGI
**Apa:** ulangi ablasi guidance dengan **Vina atau skor PLIP** sebagai pemandu, pada generator dan pocket
yang SAMA. Bukan surrogate terlatih.
**Mengapa ini nomor satu:** tesis belum punya kontrol positif. Semua rute yang diuji gagal, jadi penguji
bisa bertanya "memangnya harness-mu bisa mendeteksi keberhasilan?" dan jawabannya belum ada. Thomas et
al. 2021 menunjukkan guidance docking BERHASIL untuk REINVENT, jadi ada alasan kuat berharap positif.
**Hasil apa pun menentukan:** membaik -> harness valid + M1 (surrogate tidak baca antarmuka) terkonfirmasi
sebagai sebab. Gagal juga -> penjelasan pindah ke M2/M3, sama berharganya.
**Biaya:** GPU, sampling generatif. Perlu di-scope dulu (berapa pocket, berapa molekul) sebelum dilepas.
**Prasyarat:** 0a selesai. Mulai dengan menulis rencana pra-registrasi, BUKAN langsung lari.

### 2. PLIP typed interactions sebagai blok ke-8 tangga representasi
**Apa:** hbond / hidrofobik / pi-stacking / jembatan garam / halogen sebagai fitur, masuk
`representation_ladder.py`. Infrastruktur sudah ada: `guidance/track_e/core.py` `PlipLabelStore`.
**Mengapa:** ECIF (pasangan elemen x shell jarak) itu notasi interaksi yang kasar. Kalau interaksi
BERTIPE pun tidak menambah apa-apa di atas marginal, klaim redundansi jadi sangat kuat. Kalau menambah,
itu hasil POSITIF dan ia menamai perbaikannya.
**Biaya:** CPU saja, bisa paralel dengan GPU. Setengah hari.

### 3. Coverage KONDISIONAL conformal per novelty tier + famili protein
**Apa:** A1d membuktikan conformal satu-satunya UQ yang valid — tapi itu coverage MARGINAL. Uji apakah
coverage tetap valid per tier Tanimoto (tier sudah ada) dan per famili protein.
**Mengapa:** Jeliazkova et al. 2026 memprediksi coverage TURUN di kimia novel; Gibbs et al. 2023 memberi
kerangkanya. Interval yang valid secara marginal tapi diam-diam under-cover justru pada senyawa novel
yang dipedulikan praktisi adalah temuan yang layak dilaporkan. Ini perpanjangan paling menjanjikan dari
satu hasil positif kita.
**Biaya:** CPU saja. Setengah hari.

### 4. y-randomisasi DALAM-TARGET untuk EGNN
**Apa:** permutasi label dalam target, latih ulang EGNN, ukur R2.
**Mengapa:** prior kelas 0.2468 saat ini hanya di-fit untuk pipeline DESKRIPTOR. Supaya jadi kontrol
per-arsitektur (bukan taksiran kanal chemotype), EGNN-nya harus dilatih ulang di label terpermutasi.
Tanpa ini, tabel "di luar prior kelas" punya caveat yang harus selalu dinyatakan.
**Biaya:** GPU, 1 run (~2.5 jam). Murah, dan menutup caveat di temuan terkuat kita.

### 5. Metrik activity-cliff gaya MoleculeACE
**Apa:** identifikasi pasangan activity cliff di test set, ukur performa khusus di sana.
**Mengapa:** pola sudah terlihat di novelty tier (ligand-only kolaps di similarity tinggi) tapi belum
diukur sebagai metrik. van Tilborg et al. 2022 (314 sitasi) menyediakan definisi dan platformnya, dan
temuan mereka (deskriptor > deep learning pada cliff) persis pola kita.
**Biaya:** CPU saja. Setengah hari.

### 6. Analisis matched molecular pair (MMP)
**Apa:** pasangan yang beda satu substituen; apakah model menangkap arah perubahan afinitasnya.
**Mengapa:** cara paling ketat menguji SAR LOKAL, dan pelengkap langsung dekomposisi prior kelas: prior
kelas menangkap antar-seri, MMP menguji dalam-seri. Kwapien et al. 2022 menunjukkan data aditif paling
mudah dan deep learning bukan pengecualian.
**Biaya:** CPU saja. Sehari.

### 7. Stratifikasi error per famili protein / kelas target
**Apa:** kinase vs protease vs nuclear receptor, dsb. Nama entry UniProt sudah ada di anchor table.
**Mengapa:** mengubah satu angka gabungan jadi angka yang bisa dipakai; menunjukkan DI MANA gagalnya.
**Biaya:** CPU saja. Beberapa jam.

### 8. Arm A + Arm C (produksi)
**Apa:** Arm A (20.000 CD, 3 seed). Arm C produksi, WAJIB pakai `bindingnet_v1_clean.csv`.
**Catatan:** pertanyaannya sudah diperbaiki — bukan "apakah R2 naik" tapi "apakah data BN menutup jarak
ke model ligand-only, dan apakah ia melampaui prior kelas 0.2468". Baseline ligand-only jadi komparator
WAJIB tiap arm.
**Prasyarat:** aturan pra-registrasi Arm B (lihat bagian kebocoran senyawa) sudah dievaluasi dulu.
**Biaya:** GPU, ~16 jam per arm.

### 9. Penulisan tesis (satu rebuild di akhir, jangan tiap perubahan)
Urutan: (a) Bab IV seksi baru: audit sumber informasi + noise ceiling + dekomposisi prior kelas;
(b) Bab X revisi kesimpulan (1) — argumen "kegagalan bukan karena akurasi rendah: Pearson 0.58-0.65"
sekarang BOCOR, karena model tanpa input protein mencapai korelasi yang sama; (c) abstrak satu kalimat;
(d) limitasi: gray zone 50-90% identitas (Mattsson et al. bilang identitas sekuens tidak cukup sampai
0.2), split per-target lebih mudah dari leave-superfamily-out, overlap scaffold 40,1%, label campuran
Kd/Ki/IC50 yang komposisinya bergeser antar split; (e) sitasi wajib yang hilang: Volkov 2022, Mattsson
2026, Graber 2025, Bret 2026, van Tilborg 2022, Deng 2023, Hernandez-Garrido 2023, Landrum 2024,
Rucker 2007, Seitzer 2022, Wallace 2023, Gao 2022, Thomas 2021, PLIP; LALU rebuild PDF/DOCX SEKALI.

### 10. RQ4 — conformal-gated guidance (prediksi pra-registrasi)
**Apa:** terapkan gradien guidance / terima kandidat rejection-sampling HANYA di tempat interval conformal
surrogate cukup sempit.
**Mengapa menarik:** kombinasinya untuk SBDD tampak kosong (komponennya ada terpisah: conformal untuk
afinitas, AD-gating anti reward-hacking Yoshizawa 2025, guidance-strength bergantung confidence Azangulov
2025). **Tulis prediksinya DULU:** M3 memprediksi membantu rute rejection; M1 memprediksi TIDAK membantu
rute gradien. Satu eksperimen yang mengkonfirmasi satu dan membantah satunya = bukti kuat untuk seluruh
dekomposisi.
**Biaya:** GPU. Scope dulu.

### 11. RQ3 — DOODL / SVDD (prediksi paling tajam)
**Apa:** gradien lewat denoiser (DOODL, Wallace et al. 2023) atau guidance tanpa turunan (SVDD, Li et al.
2024) dengan surrogate yang ADA.
**Mengapa terakhir:** paling mahal, dan **M1 memprediksi TIDAK akan membantu** padahal literatur
mengharapkan sebaliknya. Nilainya justru dari prediksi itu — jadi pra-registrasikan sebelum jalan.
**Biaya:** GPU, implementasi besar.

---

### TERBLOKIR / RENDAH
* konservasi sekuens & fitur MSA residu pocket
* deskriptor druggability pocket (butuh fpocket/CASTp)
* kurasi "maximal curation" Landrum untuk label BindingNet Arm B/C
* struktur LP-PDBBind (lisensi), Binding MOAD (unduh manual)

### INFRASTRUKTUR — SELESAI, tidak perlu disentuh
7 suite / 91 tes (`python guidance/run_tests.py`), pre-commit hook terpasang, agregator multi-seed dengan
guard daya, runner antrean ber-guard termal, 4 filter kebocoran + guard test-nya.


## KONSOLIDASI: SEMUA TEMUAN, NOVELTY, DAN KESALAHAN (per 2026-10-10)

Daftar lengkap dalam satu tempat. Setiap angka punya skrip dan file hasil di repo ini. Detail dan
sitasi lengkap ada di `guidance/RESEARCH_PROGRAM.md` (1.241 baris, Part I-VIII).

---

### A. TEMUAN EMPIRIS — pengukuran kita sendiri

**A1. Ketidakpastian (A1-A1g)**
1. **5 dari 6 metode sigma point-wise GAGAL** ketiga uji pra-registrasi: MC dropout, deep ensemble,
   heteroscedastic Gaussian NLL, deep evidential, evidential terkoreksi. Hanya **split-conformal** yang
   punya coverage valid.
2. **Kegagalan replikasi evidential**: p=0.042 di satu seed, hilang di tiga seed. sd antar-seed R²
   0.0377 melebihi efek yang diklaim. Ini kandidat kontribusi metodologis.
3. **Ternary QAT (1.58-bit, TWN+STE) tak terbedakan dari FP32**: test R² 0.3437 (sd 0.0377, n=3) vs
   0.3420. Model yang bobotnya bisa direduksi ke tiga nilai tanpa kerugian terukur bukan
   dibatasi kapasitas.

**A2. Audit sumber informasi (tangga representasi)**
4. **desc_ridge (14 deskriptor RDKit, TANPA protein) = 0.3531**, vs EGNN Stage 0 0.3420 / 3-seed 0.3437.
5. **heavy_atoms (SATU fitur, jumlah atom berat) = 0.3069** — 90% performa model struktur.
6. **pocket-only (38 deskriptor, TANPA ligan, target tak terlihat) = 0.2889.**
7. **ligand+pocket vs ligand: SERI.** vs pocket: seri (setelah BH). Kedua sumber **saling redundan**.
8. **ecif_raw 0.3701 vs ecif_norm 0.2077, dR² +0.1625, q=0.013 SIGNIFIKAN** — hitungan kontak mentah
   bekerja terutama dengan menyandikan ulang ukuran ligan. Kontrol konfound yang literatur lewatkan.
9. **Klaim CORDIAL (Brown 2025, PNAS) TIDAK terdukung di split kita**: ecif_norm seri dengan kedua
   marginal.
10. **Uji berpasangan + BH atas 40 pasangan: 12 menang struktur, 28 seri, 0 menang ligand-only.**
    Tiga "kemenangan" marginal (p=0.039/0.044/0.049) **ditarik BH** (q=0.12-0.13).

**A3. Eksperimen kunci: sensitivitas pose (n=300, CI klaster-target)**
11. **Desain faktorial** — ketiganya menghapus TEPAT 50% atom protein:
    | manipulasi | antarmuka | \|Δpred\| |
    |---|---|---|
    | simpan separuh terdekat | **utuh** | 1.2451 [1.1021, 1.3694] |
    | simpan separuh terjauh | **hancur** | 1.0576 [0.8953, 1.2259] |
    | simpan separuh acak | sebagian | 1.2686 [1.1575, 1.3811] |
    | tukar protein lain | diganti | 0.947 |
    | geser ligan 8 A keluar pocket | hancur | **0.2782** |
    **Menghancurkan kontak asli LEBIH MURAH (-15%) daripada menghapus atom yang tak menyentuh apa pun.**
    CI tumpang tindih -> isi antarmuka tidak berpengaruh. **Model ~4,5x lebih sensitif ke BERAPA BANYAK
    atom protein daripada ke DI MANA ligannya.**
12. Dose-response translasi monoton: 0.021/0.042/0.086/0.160/0.278 pK untuk 0.5/1/2/4/8 A.
    Hanya 0.28x RMSE model sendiri.

**A4. Kualitas split (semuanya positif untuk kita)**
13. **Tanimoto 1-NN GAGAL berat** (R² -0.9243); struktur mengalahkannya q=0.002 di SEMUA pasangan.
    Split kita **tidak memberi hadiah untuk menghafal** ligan training terdekat.
14. **64% ligan test di bawah Tanimoto 0.35** ke training (median 0.316, p0 0.186, p90 0.561).
15. **40,1% molekul test di scaffold generik yang muncul di training** (hanya 1.779 scaffold generik di
    46.964 ligan) -- "held-out target" BUKAN "held-out scaffold". Harus didisklos.
16. **Familiaritas scaffold TIDAK membeli akurasi**: diukur RMSE, **0 dari 13 model lebih buruk** di
    scaffold tak-terlihat, 3 justru lebih baik.
17. y-randomisasi global R² +0.0037 (max +0.0141) -> desc_ridge 0.3531 **LOLOS** kontrol
    chance-correlation dengan selisih +0.3390.

**A5. Plafon, dekomposisi, dan daya**
18. **Plafon derau label R² ~0.55-0.60** (dari Pearson 0.76 pengukuran-ulang). Model terbaik kita
    (ensemble 0.4072) = **70,5% dari yang dapat dicapai**, RMSE 1.22x lantai derau.
19. **PRIOR TINGKAT KELAS = R² 0.2468** (sd 0.0019) dari permutasi dalam-target. **70% performa model
    buta-protein terbaik tidak butuh SAR dalam-target sama sekali.** Titik nol yang benar 0.247, bukan 0.
20. Di luar prior kelas: ensemble +0.1604 (48,5% headroom) vs desc_ridge +0.1063 (32,1%) vs vina
    +0.0133 (4,0%). **Diukur dari titik nol yang benar, model struktur terpisah lebih jelas.**
21. **ANALISIS DAYA: jendela 0.2245 R² (lantai 0.3531 -> plafon 0.5776) vs lebar CI rata-rata 0.3871.
    Rasio 0.58 -- lebih sempit dari SATU interval kepercayaan.** Desain ini tidak bisa menentukan posisi
    model di dalamnya, dan begitu juga paper yang melaporkan gain 0.02-0.05 R² di data sejenis.
22. Label campuran **Kd 37% / IC50 37% / Ki 26%**, dan komposisinya **BERGESER** antar split (train Kd
    41%/IC50 33%; test IC50 46%/Kd 26%). IC50 paling bergantung assay dan over-represented di test.

**A6. Kebocoran yang ditemukan**
23. **Kebocoran sekuens**: template `1fm9` 100% identik dengan P19793 (RXRA_HUMAN, target test) --
    lolos dua filter berbasis ID. 19 sekuens / 28 template / 265 baris dibuang.
24. **Kebocoran tingkat senyawa** (sisi ligan, tiga filter sebelumnya semua sisi-protein):
    **0,34% baris training (423/124.973) TAPI 10,12% record evaluasi (1.814/17.924)**, 8,6% ligan
    val/test distinct. Kebocoran sama, diukur dua arah, **selisih 30x**.
25. Eksposur AKTUAL Arm B per seed (replikasi sampling bit-per-bit): 43/58/54 baris = **3,06% / 4,02% /
    4,65% record evaluasi**.
26. Filter senyawa diterapkan: 124.973 -> **124.550 baris** (buang 423, 123 senyawa).

**A7. Temuan operasional**
27. **Arm 0 (AMP bf16 + bs=4) GAGAL gate**: R² 0.299/0.079 vs baseline 0.394. Diagnosis: presisi bf16 +
    4x lebih sedikit step optimizer per epoch sementara patience menghitung epoch.
28. **Hipotesis batch-size saya dibantah pengukuran**: throughput datar, OOM di bs=8.
29. **CPU adalah bottleneck termal, bukan GPU**: 87-94°C vs GPU 72°C, dengan load average 1,3 dari 20
    thread. Tjmax i7-12700H 100°C. Idle CPU terukur 45-73°C (jadi ambang 80°C realistis).
30. Arm B seed 2021: 13 epoch, best val 2.0637 @ epoch 8 (patience reset karena epoch 8 membaik).

---

### B. NOVELTY -- BERJENJANG, JUJUR, setelah 8 ronde pencarian

**TIER 1 -- bisa dipertahankan sebagai BARU (5 item)**
1. **Ablasi faktorial antarmuka-vs-bulk.** Tiga penghapusan 50% atom protein, setara jumlah, beda hanya
   isi antarmuka. Metode atribusi (PointVS, SME, Shapley) bilang atom MANA yang penting; baseline
   bias-only (Durant 2023) bilang model tidak lebih baik dari bias; **tidak satu pun memisahkan JUMLAH
   dari IDENTITAS.** Tidak ditemukan di delapan ronde pencarian.
2. **Dekomposisi prior kelas** via y-randomisasi dalam-target. Titik nol yang benar 0.247, bukan 0.
3. **Analisis daya level-tugas.** Jendela lebih sempit dari satu CI; dikorroborasi ABFEP R²=0.55 dan
   punya mekanisme fisik (kompensasi entalpi-entropi).
4. **Sambungan dari audit informasi ke kegagalan gradien guidance** -- subjek tesis sebenarnya, dan
   tidak satu pun paper yang men-scoop bagian lain membahasnya.
5. **Ternary (1.58-bit) QAT pada GNN afinitas 3D.**

**TIER 2 -- replikasi + rigor. Berharga, JANGAN sebut baru.**
Temuan Volkov di split leakage-controlled target-disjoint, diperluas dengan kedua marginal dan
redundansinya; deskriptor menyamai deep learning; model afinitas tak peduli identitas protein.

**TIER 3 -- BUKAN novel. Sitasi lalu lanjut.**
Perbandingan metode UQ (Rayka 2025, lima metode di LP-PDBBind). Conformal untuk afinitas (Parks 2020;
Rayka 2024). **Baseline ligand-only/bias-only menyamai MLSF berbasis struktur** (Volkov 2022;
**Durant 2023 + ToolBoxSF**; Boyles 2021; Scantlebury 2023; Sieg 2019 -- LIMA demonstrasi independen).
Conditioning pada interaction profile alih-alih guidance (ShEPhERD-2; FLOWR.MULTI). Bias benchmark dan
overfitting (Wallach 2017; Chen 2019; Kapoor 2023).

---

### C. SCOPING YANG DIPAKSA LITERATUR -- masuk ABSTRAK, bukan lampiran

1. **Pose kita DOCKED, bukan kristal** (terverifikasi: `_lig_tt_min_0.sdf`). Boyles et al. 2021
   mengukur efeknya dan **memprediksi temuan utama kita**: pada pose docked kanal struktural melemah
   dan fitur ligan mengambil alih. Ablasi faktorial tetap berdiri; klaim perbandingannya harus di-scope.
2. **Ligand efficiency diperdebatkan secara matematis** (Kenny 2018: "tidak bermakna secara fisis").
   Track C bersandar padanya. **Rumuskan ulang pada atom berat (+12) dan PoseBusters (-29 pp)** yang
   tanpa normalisasi dan tetap memberatkan.
3. **Plafon adalah RENTANG, bukan angka tunggal.** Landrum 2024 (pesimis) vs Kalliokoski 2013 (311
   sitasi, lebih ringan: mixing "hanya menambah derau moderat"). **Saya menyitir selektif.**
4. **Split per-target lebih mudah** dari leave-superfamily-out (standar CORDIAL) dan dari split
   kemiripan-pocket (EPoCS).
5. **Afinitas kesetimbangan mungkin objektif yang SALAH.** Residence time / koff berkorelasi dengan
   efikasi lebih baik (Wang 2022; Bernetti 2019, 159 sitasi; Liu 2026).
6. Eksponen scaling kimia 0.17-0.26 -> pertumbuhan data tidak bisa menutup jarak.
7. Objektif terskalarisasi menyembunyikan trade-off; Pareto alternatif bernama.
8. Harness kita bespoke; CBGBench/MolScore standar komparabilitas bidangnya.

---

### D. KESALAHAN SAYA SENDIRI DALAM SESI INI -- dicatat supaya tidak terulang

1. **QAT: salah dua kali pada pelajaran yang sama.** n=1 -> "menyamai FP32"; n=2 -> "biaya sistematis
   ~0.02". Keduanya dibantah seed 2023 (0.3872). n=2 tidak cukup ketika sd antar-seed 0.0377.
2. **Audit scaffold: draf pertama menyimpulkan KEBALIKANNYA.** "Familiaritas scaffold menggelembungkan
   performa" -- artefak range restriction (varians label 1,77x). Yang membongkarnya: **baseline Vina**
   (tak terlatih) menunjukkan kolaps yang sama.
3. **Menyitir selektif pada derau label** -- hanya Landrum (pesimis), bukan Kalliokoski (311 sitasi,
   lebih ringan), karena yang pertama mendukung argumen saya.
4. **Tiga kali mematikan shell sendiri** dengan `pkill`/`pgrep -f` yang polanya ada di command line
   shell itu sendiri (exit 144). Kerusakan nol tiap kali, tapi dua patch hilang.
5. **Satu proses orphan lolos** dari pembersihan -- gerbang termal sisa yang akan menjalankan step di
   luar urutan tanpa dicatat queue. Ketangkap saat verifikasi, bukan saat pembersihan.
6. **Fixture PDB saya sendiri salah kolom** (lupa field altLoc di indeks 16) -- tes yang saya tulis
   untuk menjaga keselarasan kolom gagal karena alat ukurnya tidak selaras.
7. **Toleransi tes salah** (1e-9 untuk matmul float32) -- kodenya benar, tesnya salah.
8. **Hipotesis batch-size 6x dibantah pengukuran sendiri.**
9. **Terlalu cepat meremehkan MLflow** -- Anda mendorong balik dan Anda benar.

**Pelajaran operasional:** jangan percaya ingatan saya soal apa yang hidup;
`bash scripts/master_queue.sh status` membacanya dari keadaan nyata.

---

### E. INFRASTRUKTUR YANG DIBANGUN -- selesai, jangan disentuh

* **7 suite, 91 tes** (`python guidance/run_tests.py`), pre-commit hook terpasang, jalur FAIL/SKIP diuji
* **4 filter kebocoran** (PDB ID -> UniProt -> identitas sekuens 90% -> skeleton InChIKey ligan) + guard
  test yang menghitung ULANG set eksklusi dari audit, bukan mempercayai summary
* **Queue master ber-gerbang termal** (GPU <55°C DAN CPU <80°C bertahan 300s, istirahat 20 menit,
  resumable, berhenti di kegagalan, timeout gerbang 4 jam) -- 7 jalur kontrol diuji
* **Agregator multi-seed** dengan minimum detectable difference; menolak memberi verdict kalau
  underpowered; sd di n=1 dilaporkan `undefined` bukan 0.0000
* **Modul cheminformatics**: chem_data, ligand_only_baseline, compare_vs_structure, novelty_tiers,
  scaffold_audit, pocket_features, representation_ladder, noise_ceiling
* **pose_sensitivity.py** -- eksperimen kunci, desain faktorial
* **beta-NLL** (`--beta_nll`) + 14 tes yang memverifikasi hukum skala lewat autograd dan detachment
  yang gagal ke DUA arah kesalahan
* `guidance/RESEARCH_PROGRAM.md` -- 1.241 baris, Part I-VIII

---

### F. ENAM ITEM MURAH BERNILAI TINGGI dari rabbit hole (tambahan di luar 11 item utama)

| # | item | biaya | mengapa |
|---|---|---|---|
| F1 | Rumuskan ulang Track C pada atom berat + PoseBusters, turunkan LE | teks saja | menutup kerentanan nyata (Kenny 2018) |
| F2 | Hitung **AVE bias** pada split LP | CPU, jam | klaim POSITIF pertama tentang kualitas split |
| F3 | Terapkan offset Ki->IC50 Kalliokoski (faktor 2) | satu baris | 311 sitasi; plafon naik |
| F4 | Laporkan plafon sebagai rentang dengan kedua sumber | teks saja | memperbaiki sitiran selektif saya |
| F5 | Isi model info sheet Kapoor + checklist Artrith sebagai lampiran | jam | rigor kita jadi bisa DIVERIFIKASI pembaca |
| F6 | Jalankan **ToolBoxSF** (Durant 2023) di data kita | CPU, hari | replikasi pakai tool bidangnya sendiri |

#### STATUS F1-F6 per 2026-10-10 (dikerjakan)

| # | status | hasil |
|---|---|---|
| **F1** | **draf siap** | `guidance/thesis/TRACKC_REFRAMING_DRAFT.md` -- teks pengganti untuk abstrak, bab, dan limitasi. Belum diterapkan ke bab karena rebuild dilakukan sekali di akhir. Kesimpulan Track C BERTAHAN setelah dirumuskan ulang pada +12 atom berat dan -29 pp PoseBusters. |
| **F2** | **SELESAI (pilot) + run penuh di-queue** | `guidance/cheminformatics/ave_bias.py`. **AVE split kita +0.046** (mean 4 ambang pK, CI mengecualikan nol) vs **random ligand split +1.044** -- **23x lebih sedikit redundansi**. Mean NN similarity ke kelas sendiri 0.29-0.31 vs **0.934** di random split. Redundansi ringan SISA, bukan nol; dilaporkan apa adanya. Klaim POSITIF pertama kita tentang kualitas split. |
| **F3** | **SELESAI** | `guidance/cheminformatics/label_harmonisation.py`. Tipe pengukuran direcover per kompleks lewat PDB-id sumber ligan (**coverage 99,7%**), lalu IC50/EC50 dinaikkan log10(2)=0.301 ke sumbu ekuivalen Ki/Kd (Kalliokoski 2013, 311 sitasi; cocok dengan Cheng-Prusoff IC50=2Ki di [S]=Km). **Hasil: kesimpulan BERTAHAN** -- 0 dari 5 model struktur mengalahkan desc_ridge dengan CI berpasangan mengecualikan nol (terdekat ensemble p=0.075, s2023 p=0.091). Urutan sort bergeser di 6 dari 13 posisi tapi semua pergeseran JAUH lebih kecil dari satu CI, jadi itu reorder di antara model yang sudah seri -- bukan temuan. Arah pergeserannya justru **menguntungkan kita** (ligan turun -0.0101, ensemble naik +0.0085), yang berarti campuran tak terkoreksi tadinya **menyanjung model ligand-only**. |
| **F4** | **SELESAI** | `noise_ceiling.py` sekarang melaporkan plafon sebagai **RENTANG R² ~0.55-0.65** dengan KEDUA sisi literatur disebut (Landrum pesimis vs Kalliokoski lebih ringan) plus jangkar fisika ABFEP R²=0.55. Memperbaiki sitiran selektif saya. |
| **F5** | **SELESAI** | `guidance/thesis/REPORTING_CHECKLIST.md` -- model info sheet + checklist 10 bagian, setiap "ya" menyebut skrip yang mengimplementasikannya sehingga tiap baris bisa disalahkan. Mengikuti Kapoor & Narayanan 2023 (1.201 sitasi), Artrith 2021 (419 sitasi), temuan Lu 2022 (model terpasang mendokumentasikan median 39% item). |
| **F6** | **belum** | ToolBoxSF (Durant 2023) butuh instalasi paket eksternal; belum dicoba supaya tidak mengubah env `drugdisc` saat training jalan. Perintahnya dicatat untuk nanti. |

**Juga di-queue:** `ave_bias` (skala penuh, 46.964 fingerprint training, B=2000) dan
`noise_ceiling_refresh`. Keduanya CPU; diantrikan bukan dijalankan inline karena CPU adalah bottleneck
termal mesin ini (92 C saat training jalan) dan disiplin queue-nya harus dipatuhi juga oleh saya.

---

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

## CONFOUND PERGESERAN DISTRIBUSI LABEL BN (2026-10-10 dini hari) — DITEMUKAN SEBELUM MERUSAK ARM B
Audit kualitas data BN sebelum menafsirkan Arm B. **Label BN bergeser sistematis dari CrossDocked:**

| | n | mean | sd | p1 | p99 |
|---|---|---|---|---|---|
| CD train | 46.964 | 6,80 | 1,80 | 2,47 | 10,64 |
| CD test | 11.855 | 6,79 | 1,65 | 2,30 | 10,15 |
| BN final | 122.721 | **7,12** | **1,37** | **3,62** | 10,00 |

Bergeser **+0,32 pK**, sebaran **lebih sempit**, dan **ekor binder-lemah terpotong** (p1 3,62 vs 2,30). KS D=0,099 (p=7e-93) — sekitar **2,2x** variasi alami CD-train vs CD-test (D=0,045). Penyebab: bias publikasi ChEMBL (senyawa aktif dilaporkan, yang lemah tidak).

**Kenapa ini membatalkan tafsiran Arm B:** R2 itu variance-explained. Kalau 14.000 dari 20.000 baris training sebarannya sempit dan tanpa ekor bawah, model tidak pernah belajar memprediksi pK rendah yang ADA di test set.

**Dan Arm B versi random SUDAH membuktikan itu dalam 2 epoch** (sebelum saya hentikan): val R2 0,230 / 0,226, tapi yang menentukan — **std prediksi cuma 0,531 lalu 0,411**, padahal sd label test 1,65. Model memprediksi nyaris konstan ~7,0 (mean BN). Itu variance collapse, bukan "data BN jelek". Run itu diarsipkan sebagai `*_RANDOM_aborted` — dihentikan di 1 jam 15 menit, bukan setelah 16 jam.

**Mitigasi: stratified sampling** (`--bn_match_pk`). Sampel BN per-bin pK supaya histogramnya cocok dengan CD. Feasible dengan longgar — bin tersempit (pK 0-4) punya 2.032 baris untuk kebutuhan 780; maksimum subsample yang bisa match persis = 36.443 baris. Hasil: BN jadi **mean 6,82 / sd 1,76** (acuan CD 6,80/1,80).

**Arm B yang berlaku sekarang adalah versi matched** (`armBmatch`), karena dia menguji kualitas data BN di pijakan yang adil. Versi random menjawab pertanyaan praktis berbeda ("kalau asal dituang, membantu?") — jawabannya sudah diketahui kualitatif (variance collapse), tidak perlu 16 jam untuk mengukurnya lagi.

## AUDIT LEAKAGE BERBASIS SEQUENCE IDENTITY (2026-10-09 malam) — MENEMUKAN LEAKAGE NYATA
`guidance/surrogate_data/leakage_audit.py`. Dibangun karena **semua filter leakage kita sebelumnya berbasis ID** (PDB ID reseptor, lalu ChEMBL target -> UniProt accession), dan itu punya lubang: dua UniProt entry yang BERBEDA bisa protein yang nyaris identik. Standar bidang ini justru sequence identity — persis kriteria "LeakProof" milik LP-PDBBind sendiri (`data/lp_pdbbind/LP_PDBBind.csv` punya kolom `seq` dan `new_split`).

Metode: sekuens UniProt penuh untuk 176 accession target val+test (di-fetch + cache) vs SEQRES semua chain dari 5.786 template BindingNet yang benar-benar dipakai (4.123 sekuens unik). Aligner match/mismatch (match=1, mismatch=0, affine gap, mode local) sehingga SKOR = jumlah residu identik; identity = skor / min(len). Normalisasi min-length dipilih karena konservatif: chain training pendek yang jadi subsekuens sempurna dari protein test tetap terhitung 100%. Biaya: 0,3 ms/pasangan, ~4 menit total.

**HASIL — filter ID memang meloloskan leakage nyata:**

| | unique seq | template | baris training |
|---|---|---|---|
| **>=90% identitas (LEAKAGE)** | 19 | 28 | **265** |
| 50-90% (satu famili, zona abu) | 287 | 383 | 10.435 |

Distribusi: median 20,8%, p90 47,3%, max 100,0%.

**Kasus paling telak: `1fm9` chain A = 100,0% identik dengan P19793 (RXRA_HUMAN), salah satu dari 127 target test kita.** 1FM9 adalah kristal heterodimer PPAR-gamma/RXR-alpha, jadi dia membawa protein test — tapi PDB ID-nya bukan `1rdt` maupun `3a9e` (dua yang ditangkap filter ID), jadi **lolos begitu saja**. Ternyata ada satu keluarga struktur heterodimer nuclear-receptor (1rdt, 3a9e, 1fm9, 3h0a, ...) dan ID-matching cuma menangkap dua. Pelanggar lain: `1iep:A` 98,2% ke P00519 (ABL1), `3uqf/3svv/2qq7/2hwo` 92-96% ke P12931 (SRC), `2ya6:A` 98,9% ke P62576.
Diverifikasi bukan artefak normalisasi: `1fm9:A` 232 residu vs P19793 462 residu -> 232 residu cocok 100% memang subsekuens sempurna (konstruk sebagian dari protein yang sama).

**Kebijakan yang diterapkan** (`apply_sequence_leakage_filter.py`, dan di-enforce ulang di dalam `train_surrogate_arm.py` supaya training tidak bisa memakai index yang belum difilter):
- **>=90% -> HARD EXCLUDE.** 125.238 -> 124.973 baris (-265, 0,21%), template 5.786 -> 5.758.
- **50-90% -> DIPERTAHANKAN tapi WAJIB DIUNGKAP di thesis sebagai limitasi.** Ini homolog satu famili (mis. kinase lain terhadap target test kinase), bukan protein yang sama; generalisasi antar famili justru premis normal model seperti ini, dan membuang 10.435 baris (8,3%) tanpa argumen leakage yang jelas melemahkan data. Dilaporkan, bukan disembunyikan.
- Pool training BN final setelah leakage + censored: **122.721 baris**.

Hasil: `leakage_audit_report.json`, `leakage_audit_report_per_sequence.csv`, `bindingnet_v1_final.csv`, `bindingnet_v1_final_summary.json`.

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

### HASIL: Arm 0 MENGGAGALKAN GERBANG — regime bs=4+AMP DITOLAK (2026-10-09 malam)
Dua seed, keduanya jauh di bawah baseline dan tidak stabil (vs run Stage 0 asli bs=1/no-AMP: best val R2 **0,394** di epoch 4, test 0,342, jalan 9 epoch):

| seed | best val R2 | best di epoch | pola |
|---|---|---|---|
| 2021 | 0,299 | 1 | osilasi 0,299 / -0,656 / 0,295 / -0,459 / 0,213 / -0,085, early-stop epoch 6 |
| 2022 | 0,079 | 3 | osilasi -0,337 / -0,813 / 0,079 / -0,008 |

Seed 2023 dihentikan manual — dua seed sudah cukup, tidak perlu memanaskan laptop untuk konfigurasi yang sudah ditolak.

**Dua sebab yang menumpuk:**
1. **Presisi bf16.** EGNN menjumlahkan ~23.000 edge per kompleks; bf16 cuma 8 bit mantissa, reduksi ribuan suku kehilangan akurasi.
2. **4x lebih sedikit langkah optimizer per epoch** di bs=4 (1.500 vs 6.000), sementara `patience`/`max_epochs` dihitung dalam EPOCH -> early-stop jauh sebelum konvergen. "Epoch" tidak sebanding antar-regime. **Ini yang saya lalai pikirkan saat mengusulkan batching.**

**KEPUTUSAN: kembali ke bs=1 + no-AMP** (regime known-good). Konsekuensi yang justru menguntungkan: **Arm 0 jadi tidak perlu dijalankan** — run Stage 0 yang sudah ada (test R2=0,342) ADALAH anchor di regime itu, jadi perbandingan langsung ke angka yang sudah masuk thesis. Biaya Arm 0 terhapus.

**Anggaran revisi (bs=1, no-AMP, 11,6 rows/sec):** Arm 0 = tidak perlu; Arm A/B @46.964 = 35,5 jam masing-masing (71 jam total); **Arm A/B @20.000 = 16 jam masing-masing (32 jam total)** <- rekomendasi, 20.000 masih 3,3x budget Stage 0 jadi pertanyaan kontrol-jumlah tetap teruji.

Opsi yang tidak diambil (dicatat supaya tidak dibahas ulang): bs=4 + lr dinaikkan 4x + patience disesuaikan bisa menghemat ~22 jam, tapi menambah confound lr yang beda dari baseline dan butuh validasi sendiri. Ditolak karena kita baru saja kena akibat "optimasi" yang belum divalidasi.
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

---

## AUDIT PRIOR ART (2026-10-10) — WAJIB DIBACA SEBELUM MENULIS BAB UQ

Pencarian literatur sistematis (Consensus + web, 8 query) atas dua klaim novelty kita.
Hasilnya mengubah posisi klaim, jadi dicatat di sini supaya tidak terlewat saat menulis.

### Klaim (a) "perbandingan metode UQ untuk afinitas ikatan" -> SUDAH ADA PENDAHULUNYA

* **Rayka et al. 2025, Scientific Reports** -- membandingkan LIMA metode UQ (Deep Ensemble,
  MC Dropout, Laplace, Bayes-by-Backprop, Evidential) untuk afinitas protein-ligan, **di atas
  Leak-Proof PDBBind**. Dataset, pertanyaan, dan 3 metode tumpang-tindih dengan A1-A1f kita.
* **Parks et al. 2020, Frontiers in Molecular Biosciences** -- conformal prediction untuk
  afinitas protein-ligan; melaporkan interval yang terkalibrasi baik. Ini mendahului temuan
  POSITIF kita (A1d).
* **Rayka et al. 2024, Molecular Informatics (ENS-Score)** -- conformal untuk afinitas, CASF-2016.

KONSEKUENSI: klaim "hanya conformal yang terkalibrasi" tidak boleh ditulis sebagai temuan baru.
Ketiga paper di atas WAJIB disitasi di bab related work; tanpa itu bab UQ punya lubang yang
akan dilihat penguji.

Yang MASIH milik kita dari A1-A1f (klaim yang lebih sempit tapi dapat dipertahankan):
1. Kelas model berbeda: Rayka et al. pakai FFNN atas deskriptor ECIF (feature-vector).
   Kita GNN 3D ekuivarian (EGNN/GIGN). "Apakah temuan UQ itu berlaku pada GNN geometrik" terbuka.
2. Kerangka inferensi berbeda: target-clustered bootstrap + BH + pra-registrasi + replikasi
   lintas seed, bukan metrik kalibrasi deskriptif.
3. **KANDIDAT KONTRIBUSI TERKUAT**: kegagalan replikasi evidential. p=0.042 pada satu seed,
   hilang pada 3 seed; sd antar-seed R² = 0.0377 melebihi efek yang diklaim. Ini temuan
   metodologis yang berlaku atas literatur UQ single-seed di domain ini, termasuk paper di atas.

### Klaim (b) "ternary QAT untuk GNN afinitas" -> MASIH KOSONG, tapi sempit

Tidak ada hit untuk ternary/1-bit/low-bit + afinitas ikatan protein-ligan. Prior art terdekat
yang WAJIB disitasi supaya posisi kita jujur:
* **Degree-Quant, Tailor et al. ICLR 2021** -- QAT arsitektur-agnostik untuk GNN, INT8/INT4,
  dievaluasi termasuk pada regresi molekuler (ZINC). Ini prior art terdekat untuk "QAT di GNN
  molekuler". BUKAN ternary, BUKAN afinitas protein-ligan.
* **Zhou et al. 2026, Quantized SO(3)-Equivariant GNN** -- 8-bit, QM9/rMD17, bukan afinitas.
* **Rasool et al. 2025** -- 2-bit, properti molekuler.
* **BitNet b1.58** -- ternary tapi LLM.

Posisi novelty A1g yang bisa dipertahankan, dinyatakan persis sesempit ini:
  "bobot ternary (1.58-bit, TWN+STE, QAT) pada GNN 3D untuk regresi afinitas protein-ligan,
   R² test 0.3437 (sd 0.0377, n=3) vs FP32 0.3420 -- tak terbedakan secara statistik"
Ukuran kontribusi: setara paper workshop. Jangan dibesar-besarkan.

### TEMUAN YANG MENGUBAH PEKERJAAN: A1e memakai loss yang sudah diketahui gagal

**Seitzer et al. 2022, "On the Pitfalls of Heteroscedastic Uncertainty Estimation with
Probabilistic Neural Networks"** (163 sitasi) menunjukkan Gaussian NLL + optimizer berbasis
gradien menghasilkan "estimasi parameter sangat buruk tapi stabil", karena gradien mean
diskalakan oleh variance prediktif. Itu PERSIS variance-collapse yang kita lihat di A1e
cold-start (log_var menabrak clamp dalam 3 step). Mereka mengusulkan **beta-NLL**: kontribusi
tiap titik ke loss dibobot variance^beta. Lihat juga Stirn et al. 2023 (Faithful Heteroscedastic
Regression) dan Immer et al. 2023 (parametrisasi natural + Laplace).

KONSEKUENSI: kesimpulan "heteroscedastic gagal" saat ini TIDAK AMAN -- kita memakai varian loss
yang literatur sudah tahu patologis, dan perbaikan yang dipublikasikan belum diuji.
Dua jalan jujur:
  (i) jalankan A1e-beta (beta-NLL, beta=0.5), warm-start sudah ada, biaya ~1 run x 3 seed; ATAU
  (ii) nyatakan eksplisit di limitasi bahwa yang diuji adalah NLL vanilla dan beta-NLL/natural
       parametrization belum diuji, dengan sitasi Seitzer et al.
REKOMENDASI: (i). Murah, dan menutup pertanyaan penguji yang hampir pasti muncul.
Kalau (i) tetap gagal, temuan kita justru jauh lebih kuat: gagal bahkan dengan perbaikan resmi.

---

## INFRASTRUKTUR TEST & AGREGASI (2026-10-10) — SELESAI

Tiga utang engineering yang saya janjikan sambil Arm B jalan, semuanya CPU-only.

### 1. `guidance/run_tests.py` + `scripts/hooks/pre-commit`
Runner tunggal untuk semua suite (repo ini tidak punya pytest di env `drugdisc`; konvensinya plain
`test_*` function + runner `__main__`, dan runner baru ini menjalankan setiap suite sebagai subprocess).
Status saat ini: **5 suite, 54 tes, semua lolos, ~9 detik.**

Kontrak exit code (penting, jangan diubah tanpa alasan):
  0 = lolos | 1 = gagal | 2 = TIDAK BISA JALAN karena artefak data (CSV multi-MB, gitignored) tidak ada
Exit 2 dilaporkan sebagai SKIP yang TERLIHAT, bukan pass, dan tidak memblokir commit. Guard leakage yang
diam-diam no-op lebih buruk daripada tidak ada guard.

Install hook: `bash scripts/hooks/install.sh`   Bypass satu commit: `git commit --no-verify`
Jalur FAIL dan SKIP sudah diuji dengan suite sementara: FAIL -> hook exit 1 + pesan blokir, SKIP -> exit 0.

### 2. `guidance/surrogate_data/tests/test_leakage_guard.py` (9 tes)
Menegaskan invariant pemisahan train/val/test pada ARTEFAK di disk, setiap kali suite jalan:
  * tidak ada template >=90% identik dengan target val/test yang lolos ke pool training (kelas bug `1fm9`)
  * set eksklusi dihitung ULANG dari audit per-sequence, bukan dipercaya dari summary JSON
  * threshold 90% dipin -- perubahan silent akan melemahkan klaim leakage di tesis
  * filter ID-level lama masih berlaku (menangani target ChEMBL multi-aksesi via irisan himpunan)
  * gray zone 50-90% MASIH ADA -- menegaskan KEPUTUSAN, supaya split yang diam-diam lebih ketat pun
    memicu kegagalan dan mengingatkan untuk memperbarui bagian limitasi
Sengaja TIDAK menjalankan ulang alignment: itu akan menguji kode audit terhadap dirinya sendiri.

### 3. `guidance/surrogate_data/tests/test_pocket_extraction.py` (19 tes)
Proteksi regresi untuk dua hal yang sudah pernah rusak: varian residu AMBER (`KeyError: 'HID'`) dan
H-stripping. Termasuk tes konsekuensi end-to-end: hidrogen yang ditaruh 50 A dari pusat TIDAK boleh
menggeser center_of_mass residu (itu yang di-threshold `query_residues_ligand`), dan HID vs HIS harus
memberi representasi heavy-atom identik. Juga ada `test_fixture_is_column_correct` -- fixture PDB-nya
sendiri salah kolom saat pertama ditulis (lupa field altLoc di indeks 16), jadi alat ukurnya ikut diuji.

### 4. `guidance/aggregate_results.py` — agregator multi-seed dengan guard small-n
Menggantikan tabel multi-seed yang dihitung manual. Membaca `log.txt` (epoch dengan val_loss TERENDAH,
bukan epoch dengan metrik terbaik -- memilih epoch yang memaksimalkan metrik yang lalu dilaporkan adalah
bias seleksi) + `metrics.json` untuk metrik TEST. Seed dibaca dari checkpoint, bukan nama direktori,
karena direktori di proyek ini pernah di-rename tangan (`_RANDOM_aborted`).

Yang membuatnya bukan sekadar mean+-sd:
  * sd di n=1 dilaporkan `undefined`, BUKAN 0.0000 (nol di sana terbaca sebagai presisi sempurna)
  * n<3 memicu WARNING yang menyebut kegagalan konkret proyek ini
  * `--compare A B` mencetak **minimum detectable difference** pada sd teramati (alpha=.05, power=.80).
    Kalau |selisih| < MDE, verdict-nya UNDERPOWERED dan tanda selisihnya dinyatakan tidak didukung.
  * VAL dan TEST diringkas di baris TERPISAH dan berlabel -- angka yang dikutip tesis adalah TEST,
    val hanya yang dioptimasi model selection; mencampur keduanya adalah cara termudah melebih-lebihkan.
  * seed duplikat dalam satu grup memicu WARNING: n-nya palsu, bukan n undian independen.

Validasi: `--metric R2` atas `logs_a1g_qat_ternary` menghasilkan TEST mean 0.3437 sd 0.0377 --
PERSIS angka yang dulu saya hitung manual. Alatnya tervalidasi terhadap hasil yang sudah diketahui.

### 5. `eval_qat_test.py` sekarang mem-persist `metrics.json`
Sebelumnya metrik test hanya di-print ke stdout, jadi tabel test multi-seed harus disusun ulang dari
scrollback terminal -- itulah mekanisme di balik salah-baca n=1 dan n=2 pada eksperimen A1g sendiri.
Metrik 3 seed A1g sudah di-BACKFILL dari file `.npz` prediksi yang tersimpan (CPU saja, GPU tidak
disentuh, Arm B tidak terganggu): s2021 R2=0.3239, s2022 R2=0.3201, s2023 R2=0.3872.

---

## TEMUAN BARU: KEBOCORAN TINGKAT SENYAWA (2026-10-10) — SIGNIFIKAN

Pemeriksaan "overlap tingkat senyawa" yang sudah lama saya tandai belum pernah dikerjakan, akhirnya
dijalankan: `guidance/surrogate_data/compound_overlap_audit.py`.

Pertanyaannya BEDA dari tiga filter sebelumnya. Ketiganya sisi-PROTEIN ("apakah protein training =
protein test?"). Tidak ada yang bertanya "apakah LIGAN training = ligan test?". Itu jalur kebocoran
tersendiri: model yang pernah melihat senyawa X dengan pK terukur saat training bisa mengingat angka itu
saat X muncul di test, bahkan terhadap reseptor berbeda, karena identitas ligan sendiri membawa banyak
sinyal afinitas.

Kriteria: InChIKey SKELETON (14 karakter pertama = blok konektivitas), bukan key penuh. Skeleton
mengabaikan stereokimia dan protonasi, jadi menangkap juga ligan test yang hadir di training sebagai
tautomer/garam/stereoisomer berbeda -- semuanya tetap membocorkan label. Key penuh akan melewatkan 60
dari 101 famili yang bertabrakan.

### HASIL -- dan mengapa framing-nya penting

|                                                  |                            |
|--------------------------------------------------|----------------------------|
| sisi training: baris pool terkontaminasi          | 423 / 124.973 = **0,34%**  |
| sisi EVALUASI: record val+test terdampak          | 1.814 / 17.924 = **10,12%**|
| ligan val/test distinct terdampak                 | 101 / 1.175 = **8,6%**     |

Kebocoran yang SAMA, diukur dua arah, selisih 30x. "0,34%" terbaca seperti derau dan itulah bahayanya:
angka yang membatasi sejauh mana skor test boleh dibaca sebagai generalisasi adalah angka EVALUASI.
Agregator dan guard test sekarang mewajibkan KEDUA arah tercatat (`test_evaluation_side_exposure_...`).

### Eksposur AKTUAL Arm B-matched yang sekarang jalan

Dihitung dengan mereplikasi sampling stratified `build_bn_train_set` bit-per-bit (CSV, filter, rantai
RandomState, seed yang sama) -- jadi ini baris yang BENAR-BENAR ditarik, bukan ekspektasi:

| seed | baris training terkontaminasi | % dari 14.000 | record evaluasi terdampak | % dari 17.924 |
|------|------------------------------|---------------|---------------------------|---------------|
| 2021 | 43                           | 0,31%         | 549 (477 test / 72 val)   | **3,06%**     |
| 2022 | 58                           | 0,41%         | 720 (585 / 135)           | **4,02%**     |
| 2023 | 54                           | 0,39%         | 834 (643 / 191)           | **4,65%**     |

Jadi eksposur Arm B 3-4,7%, bukan 10,12% (itu angka seluruh pool).

### Remediasi: `apply_compound_leakage_filter.py` -- SUDAH DIBUAT DAN DIJALANKAN
`bindingnet_v1_clean.csv`: 124.973 -> **124.550 baris** (buang 423, 0,34%; 123 senyawa; 5.758 -> 5.721
template). Membayar 0,34% data training untuk menghapus kontaminan dari 10% record evaluasi bukan
trade-off yang perlu diperdebatkan.

### KEPUTUSAN PRA-REGISTRASI (ditulis SEBELUM hasil Arm B keluar, supaya bukan rasionalisasi)

Arm B TIDAK dihentikan. Alasannya: restart = 16 jam untuk kontaminasi 3-4,7% record evaluasi, yang
kemungkinan besar menggeser R² kurang dari sd antar-seed yang sudah terukur di proyek ini (0,0377).
Membakar 16 jam untuk efek di bawah lantai deteksi sendiri tidak rasional. Tapi kontaminasinya TIDAK
seragam antar seed (3,06% vs 4,65%), jadi ia menambah varians, dan itu harus disebut.

Aturannya, berlaku apa pun hasilnya:
1. Arm B dilaporkan DENGAN tabel eksposur di atas sebagai limitasi eksplisit, bukan catatan kaki.
2. Kalau Arm B menunjukkan manfaat BN yang KECIL atau nol -> kontaminasi tidak mengubah kesimpulan
   (kontaminasi hanya bisa MENAIKKAN skor BN, jadi batas atas yang bocor tetap mendukung "tidak membantu").
   Kesimpulan aman tanpa run ulang.
3. Kalau Arm B menunjukkan manfaat BN yang BESAR (di luar CI cluster-bootstrap) -> JANGAN dipercaya
   sebelum diulang di atas `bindingnet_v1_clean.csv`. Arah kontaminasi persis arah yang memalsukan
   temuan positif.
4. Arm C (production) WAJIB pakai `bindingnet_v1_clean.csv`, tanpa pengecualian.

### Yang masih harus dilakukan
* `train_surrogate_arm.py` masih membaca `data/bindingnet_pocket10/labels.csv` + filter sekuens, BELUM
  filter senyawa. Harus ditambah SEBELUM Arm C, mengikuti pola penegakan ganda yang sudah ada di
  `build_bn_train_set` (filter ditegakkan di dalam kode training, bukan hanya di CSV upstream).

---

## A1e-BETA: DI-QUEUE DAN BERJALAN (2026-10-10 01:26)

Menindaklanjuti temuan audit prior art bahwa A1e memakai loss yang literatur sudah tahu patologis
(Seitzer et al. 2022). Implementasi, verifikasi, dan antrean sudah selesai; eksekusi menunggu GPU.

### Implementasi: `--beta_nll` di `train_egnn_heteroscedastic.py`
`beta_nll(y, mu, log_var, beta)` -- loss per-contoh dibobot `stopgrad(sigma^2)^beta`. Efeknya pada
gradien mean: `dL/dmu = -(y-mu) * sigma^(2*beta-2)`.
  beta=0   -> sigma^-2  (NLL biasa; rezim patologis)
  beta=0.5 -> sigma^-1  (rekomendasi paper)
  beta=1   -> sigma^0   (independen varians, seperti MSE)

Mekanisme patologinya: di NLL biasa, contoh yang (keliru) diberi sigma besar jadi TIDAK menyumbang ke
fit mean, sehingga error-nya tidak pernah mengecil, sehingga sigma-nya tetap besar. Loop yang
memperkuat diri sendiri -- persis variance collapse A1e saat cold start.

Dua keputusan desain yang menentukan validitas perbandingan:
1. **beta=0.0 adalah default dan mereproduksi loss A1e BIT-PER-BIT** (short-circuit sebelum aritmetika
   apa pun). Tanpa ini, A1e vs A1e-beta bukan perbandingan satu faktor melainkan penulisan ulang.
2. **Validasi SELALU melaporkan dan menyeleksi pada NLL biasa**, apa pun beta saat training. Kalau beta
   ikut mengubah objektif seleksi, perbedaan hasil tidak bisa diatribusikan ke beta. Ini juga menjaga
   setiap angka val tetap sebanding dengan run A1e yang sudah ada.

### Verifikasi (14 tes, `tests/test_beta_nll.py`, semua lolos)
Diuji lewat autograd, bukan lewat membaca ulang rumus -- karena kalau bobotnya tidak di-detach atau
eksponennya kena sigma bukan sigma^2, loss-nya TETAP training dan TETAP menghasilkan angka masuk akal.
Itu tidak terlihat di kurva training.
  * `gaussian_nll` dicek terhadap `scipy.stats.norm.logpdf` (referensi luar, bukan rumus yang sama
    ditulis ulang)
  * hukum skala `sigma^(2beta-2)` diverifikasi autograd untuk beta in {0, 0.25, 0.5, 1}
  * detachment: dibandingkan dengan bentuk tertutup kasus DETACHED, DAN dibuktikan beda dari bentuk
    attached -- jadi tes gagal ke dua arah kesalahan
  * reproduksi patologinya sendiri: dua residual identik, sigma beda e^4 -> di beta=0 pengaruh contoh
    sigma-tinggi ~3000x lebih kecil; beta=0.5 mengurangi ketimpangan itu
  * stabilitas numerik di kedua ujung clamp log_var [-6,6] dalam float32
Plus smoke test integrasi di CPU: model nyata, batch nyata, backward nyata. Rasio loss cocok persis
dengan sigma^(2beta) (0.3349 -> 0.4301 di beta=0.5; -> 0.5525 di beta=1).

### Urutan run: BERGANTIAN, dan itu bukan sembarang
A1e yang ada **hanya 1 seed** (2021). Tiga run beta vs satu run plain = situasi n=3-vs-n=1 yang
`aggregate_results.py` justru menolak beri verdict. Jadi chain-nya menambah A1e ke 3 seed SAMBIL
membangun A1e-beta ke 3, berpasangan per seed:
  step 1: beta 2021   -> 1 v 1
  step 2: plain 2022
  step 3: beta 2022   -> 2 v 2
  step 4: plain 2023
  step 5: beta 2023   -> 3 v 3
Konsekuensinya: perbandingan tetap berimbang di SETIAP titik chain bisa terputus. Berhenti lebih awal
mengorbankan power, tidak pernah validitas.

### `scripts/queued_run.sh` — runner dengan guard termal
Kartu 4GB tidak bisa menampung dua job, jadi pekerjaan lanjutan harus menunggu. Setiap step menunggu
GPU bebas DAN suhu di bawah 55C selama 300 detik berturut-turut (satu sampel dingin tepat setelah job
keluar tidak berarti apa-apa -- kartu masih melepas panas). Exit code diperiksa EKSPLISIT per step;
`set -e` pernah gagal membatalkan chain di sini (3 seed jalan, 3 gagal, chain lapor sukses).
Exit 75 = kondisi start tidak terpenuhi, dibedakan dari command-nya sendiri gagal.

**BUG YANG KETANGKAP SEBELUM MERUSAK (penting, jangan diulang):** `pgrep -f` mencocokkan SELURUH
command line, dan argv `queued_run.sh` sendiri memuat path skrip training sebagai argumen. Jadi
`--wait-for "train_egnn_heteroscedastic[.]py"` mencocokkan wrapper-nya SENDIRI (dan subshell command
substitution yang mewarisi argv sama). Trik bracket mencegah pgrep mencocokkan pgrep, BUKAN mencocokkan
pemanggilnya. Kalau tidak diperbaiki: step pertama yang nama skripnya muncul di pola tunggunya sendiri
akan menunggu dirinya sendiri SELAMANYA -- tanpa error, tanpa output, antrean yang tidak pernah
menyala. Perbaikannya: match hanya dihitung kalau `/proc/<pid>/comm` adalah `python*`, karena `comm`
memuat nama EXECUTABLE bukan argument vector (job training = "python", wrapper = "bash").
Diuji dua arah: masih mendeteksi Arm B (pid python), dan pola deadlock kini langsung menyala.

### Penegakan filter senyawa di `train_surrogate_arm.py` — SELESAI
`build_bn_train_set(..., compound_filter=True)` sekarang menegakkan filter ligan DI DALAM kode training,
pola yang sama dengan filter sekuens -- karena fungsi itu membaca `labels.csv` yang TIDAK terfilter
(125.238 baris) dan menerapkan eksklusi sendiri. Aritmetika terverifikasi: 125.238 -> 124.973 (seq)
-> 124.550 (compound) -> 122.298 (uncensored). Default True; `--no_compound_filter` harus eksplisit dan
mencetak peringatan. Guard test memastikan default-nya tidak bisa diam-diam jadi opt-in.

Status tes proyek: **6 suite, 73 tes, semua lolos.**

---

## LAPISAN CHEMINFORMATICS / BIOINFORMATICS (2026-10-10) — TEMUAN BESAR

Bidang ini menyediakan satu kontrol yang proyek kita **belum pernah jalankan**, dan itu kontrol yang
paling diarahkan ke kelas model kita. Semua CPU-only, tidak mengganggu Arm B.

Prior art yang mendorongnya:
* **Volkov et al. 2022, J. Med. Chem.** (170 sitasi) -- deskripsi eksplisit interaksi nonkovalen
  protein-ligan TIDAK memberi keuntungan dibanding deskriptor ligan atau protein saja; model
  nearest-neighbour sederhana sudah bagus; memorisasi mendominasi pembelajaran.
* **Mattsson et al. 2026, bioRxiv** -- split berbasis identitas sekuens **inheren tidak cukup** karena
  "target mirroring"; leakage bertahan sampai ambang identitas 0.2 (kita pakai 90% + gray zone 50-90%!).
  Model ligand-only mencapai r=0.66 di FEP+. Mengusulkan Novelty-Tiered Affinity Benchmark.
* **Graber et al. 2025, Nature Machine Intelligence** -- PDBbind CleanSplit; melatih ulang model
  terdepan di split bersih membuat angka benchmark mereka jatuh drastis.
* **Jeliazkova et al. 2026** -- conformal sebagai lapisan kalibrasi untuk applicability domain QSAR.
* **Gibbs et al. 2023, JRSS-B** (189 sitasi) -- coverage kondisional eksak itu MUSTAHIL di sampel
  terbatas; ada spektrum antara marginal dan kondisional.

### Modul baru: `guidance/cheminformatics/`
`chem_data.py` (SMILES dari field `ligand_smiles` LMDB + ECFP4 2048-bit + 14 deskriptor RDKit +
scaffold Bemis-Murcko generik; 64.888 molekul, 64.888 ter-parse, 0 gagal; target dan pK di-join dari
anchor table Track E sehingga klaster bootstrap IDENTIK dengan A1-A1g).

### TEMUAN 1: model struktur TIDAK terbedakan dari ridge atas 14 deskriptor
`ligand_only_baseline.py` + `compare_vs_structure.py` (uji BERPASANGAN, bootstrap 127 klaster target
B=2000, koreksi BH atas 40 pasangan).

| model | test R2 | catatan |
|---|---|---|
| EGNN Stage 0 (struktur) | 0.3420 | model yang jadi pokok tesis |
| EGNN ensemble A1c | 0.4153 | |
| EGNN 3 seed (QAT) | 0.3437 +- 0.0377 | |
| **desc_ridge (14 deskriptor, TANPA protein)** | **0.3531** | **mengalahkan Stage 0** |
| heavy_atoms (SATU fitur: jumlah atom berat) | 0.3069 | 90% performa model struktur |
| ecfp_desc_hgb | 0.3211 | |
| vina (fisika) | 0.2601 | |
| tanimoto_1nn | -0.9243 | |

Setelah BH atas 40 pasangan: **12 menang struktur, 28 tak terbedakan, 0 menang ligand-only.**
Tiga "kemenangan" marginal (p=0.039/0.044/0.049) **ditarik** oleh BH (q=0.12-0.13) -- tanpa koreksi
saya akan melaporkan temuan palsu.

Yang BERTAHAN setelah BH:
* struktur mengalahkan **tanimoto nearest-neighbour** secara dominan (dR2 ~ +1.2-1.3, q=0.002 di SEMUA
  pasangan). Ini temuan **POSITIF tentang split kita**: berbeda dari setting PDBbind yang dianalisis
  Volkov et al., split kita TIDAK memberi hadiah untuk menghafal ligan training terdekat.
* ensemble dan seed 2023 mengalahkan **Vina** (q=0.002 / 0.007).
Yang TIDAK: struktur vs desc_ridge tak terbedakan di SEMUA seed dan juga sebagai ensemble.

**Konsekuensi untuk tesis:** klaim "R2 held-out kami menunjukkan pembelajaran pengenalan protein-ligan"
TIDAK didukung. Klaim yang jujur: setara model QSAR ligand-only pada ukuran sampel ini. Ukuran ligan
melakukan sebagian besar pekerjaan.

### TEMUAN 2: 64% test set kita sudah "novel chemistry"
`novelty_tiers.py` (protokol Mattsson; batas 0.35 diambil apa adanya, BUKAN di-tuning -- memilih ambang
setelah melihat mana yang memisahkan model itu menyeleksi hasil).
Distribusi max-Tanimoto test->train: p0 0.186, median **0.316**, p90 0.561. Jadi 64% test di bawah 0.35.
Itu properti BAIK dari split kita.

Hipotesis saya (protein berguna justru di kimia novel) **TIDAK terdukung**: tier novel dR2 +0.0368,
CI [-0.0993, +0.1683], q=0.748. Setelah BH atas 4 tier, TIDAK ADA tier yang signifikan.

Tapi polanya terbalik dan koheren: model ligand-only **KOLAPS** saat similarity naik
(desc_ridge 0.3132 novel -> -0.0211 same-series; heavy_atoms -> -0.4097) sementara model struktur
bertahan 0.17-0.24. Itu rezim **ACTIVITY CLIFF**: dalam satu seri kimia deskriptor hampir tak bergerak
sementara afinitas bergerak. Jelas secara deskriptif, tidak separable secara statistik.

### TEMUAN 3 (metodologis, dan saya hampir melaporkan kebalikannya)
`scaffold_audit.py`. Draf pertama menyimpulkan "familiaritas scaffold menyumbang ke angka yang
dilaporkan" karena SETIAP model kolaps di scaffold tak-terlihat (EGNN 0.3523 -> -0.0240).

**Itu artefak.** R2 = 1 - SSE/SST, dan dua subset punya SST sangat berbeda: sd pK 1.72 (seen) vs 1.29
(unseen), rasio varians **1.77x**. Dengan error absolut sama, subset lebih sempit otomatis dapat R2
lebih rendah. Pearson pun teratenuasi oleh range restriction.

**Yang membongkarnya: baseline Vina.** Vina fungsi skoring fisika yang TIDAK PERNAH dilatih di data
kita, jadi mustahil diuntungkan familiaritas scaffold -- tapi ia menunjukkan kolaps yang sama
(0.1606 -> -0.0181). Pola yang muncul di prediktor tak terlatih adalah properti PARTISI DATA, bukan
pembelajaran.

Setelah diukur dengan RMSE (metrik yang tidak terdistorsi): **0 dari 13 model lebih buruk** di scaffold
tak-terlihat; 3 justru LEBIH BAIK. Familiaritas scaffold tidak membeli akurasi.
Overlap scaffold tetap tinggi dan wajib didisklos: 40,1% molekul test berada di scaffold generik yang
muncul di training (hanya 1.779 scaffold generik unik di 46.964 ligan training -- vokabulari kerangka
CrossDocked memang sempit). "Held-out target" TIDAK berarti "held-out scaffold".

**Aturan yang harus masuk tesis:** R2 tidak boleh dibandingkan antar subset data dengan varians label
berbeda, dan menyimpan baseline TAK TERLATIH di setiap tabel semacam itu adalah cara murah
menangkapnya.

### Tes: `guidance/cheminformatics/tests/test_chem.py` (18 tes)
Yang terpenting: rutin Tanimoto (matmul BLAS; interseksi = Q @ R.T) diuji terhadap
`DataStructs.BulkTanimotoSimilarity` milik RDKit -- implementasi yang benar-benar independen, bukan
penulisan ulang aljabar yang sama. Plus: invariansi chunking, fingerprint nol tidak menghasilkan NaN,
guard keselarasan baris menolak permutasi YANG MEMPERTAHANKAN label (lewat cek string target), dan
R2 subset dihitung atas mean SUBSET.
Catatan presisi: matmul float32 -> kesepakatan dengan RDKit terbatas ~1e-7, jadi toleransi tes 1e-6.
Draf pertama memakai 1e-9 dan gagal; yang salah toleransinya, bukan kodenya.

Status tes proyek: **7 suite, 91 tes, semua lolos.**

### Yang masih terbuka
* Mattsson et al. mengklaim identitas sekuens tidak cukup sampai ambang 0.2. Kita di 90% + menyimpan
  gray zone 50-90%. Harus didisklos sebagai keterbatasan yang kini punya rujukan, bukan lagi sekadar
  judgement call kita.
* Coverage KONDISIONAL conformal per tier kebaruan + per famili protein: belum dikerjakan. Ini
  perpanjangan paling menjanjikan dari satu hasil positif kita (A1d), dan Gibbs et al. 2023 memberi
  kerangka teoretisnya.

---

## TANGGA REPRESENTASI: REPLIKASI VOLKOV LENGKAP + UJI KLAIM CORDIAL (2026-10-10)

Lanjutan `ligand_only_baseline.py`, yang hanya mereplikasi SEPARUH kontrol Volkov et al. 2022 -- mereka
menguji deskriptor PROTEIN juga, bukan cuma ligan. Dua modul baru:
`pocket_features.py` (38 deskriptor pocket + 72 fitur interaksi ECIF, dari field protein LMDB; tidak
butuh file PDB mentah) dan `representation_ladder.py`.

### Tabel: apa nilai masing-masing SUMBER INFORMASI (test, 127 klaster target, B=2000)

| sumber informasi | dim | test R2 | 95% CI |
|---|---|---|---|
| ligand+pocket | 52 | **0.3832** | [+0.2013, +0.4963] |
| ecif_raw (hitungan kontak) | 72 | 0.3701 | [+0.2130, +0.4730] |
| all (ligand+pocket+ecif_norm) | 124 | 0.3639 | [+0.1746, +0.4829] |
| ecif_norm + ukuran | 73 | 0.3547 | [+0.2073, +0.4543] |
| ligand (14 deskriptor RDKit) | 14 | 0.3534 | [+0.2079, +0.4465] |
| **pocket saja -- TANPA ligan** | 38 | **0.2889** | [+0.1525, +0.3765] |
| ecif_norm (densitas interaksi) | 72 | 0.2077 | [+0.0205, +0.3478] |

Model 3D GNN di baris test yang sama: ensemble A1c 0.4072 | qat_s2023 0.3872 | Stage 0 0.3415 |
qat_s2021 0.3239 | qat_s2022 0.3201.

Catatan desain: untuk SETIAP blok di-fit ridge (alpha dipilih di val) DAN gradient boosting, lalu yang
lebih baik di VAL yang dilaporkan -- supaya sebuah blok tidak dirugikan karena cocok untuk satu
regressor. Pertanyaannya nilai INFORMASInya, bukan regressor mana yang kebetulan pas.

### Apa yang ini selesaikan (BH atas 20 uji berpasangan)

**1. Replikasi Volkov sisi protein: TERKONFIRMASI.** Pocket SENDIRIAN -- tanpa ligan sama sekali, di
target yang TIDAK PERNAH DILIHAT -- mencapai R2 0.2889. Pocket vs ligand: **seri**. Jadi kedua marginal
bekerja sendiri-sendiri.

**2. Kedua sumber TIDAK saling menambah.** ligand+pocket vs ligand: seri. vs pocket: seri (setelah BH).
Artinya tidak ada sumber yang membawa informasi yang tidak dimiliki sumber lain di level deskriptor.
Itu persis temuan Volkov et al., sekarang direplikasi di split kita.

**3. Klaim CORDIAL (Brown 2025, PNAS) TIDAK terdukung di sini.** Mereka berargumen representasi
interaksi-saja (tanpa parameterisasi struktur protein/ligan) harus menang di target tak-terlihat.
ecif_norm = 0.2077, vs ligand: seri (q=0.057); vs pocket: seri. Tidak menang.

**4. Konfound ukuran di hitungan interaksi: TERTANGKAP.** ecif_raw (0.3701) vs ecif_norm (0.2077):
dR2 +0.1625, **q=0.013, SIGNIFIKAN**. Jadi hitungan kontak mentah bekerja terutama dengan menyandikan
ulang ukuran ligan. Kalau saya hanya melaporkan ecif_raw, saya akan mengklaim "fitur interaksi
informatif" padahal itu jumlah atom berat. Kontrol ini sengaja dibangun setelah audit scaffold
tertangkap pada kesalahan yang sama.

**5. Plafon level-deskriptor = 0.3832** (ligand+pocket). vs ligand saja: seri.

**6. 3D GNN vs plafon deskriptor: 2 dari 10 pasangan** signifikan setelah BH -- dan keduanya melawan
ecif_norm (blok terlemah), BUKAN melawan plafon. Melawan `all`: **semua seri**.

### Kesimpulan gabungan yang bisa masuk tesis

Di split target-disjoint ini, afinitas dapat diprediksi sampai R2 ~0.35-0.38 dari **sumber informasi
APA SAJA secara terpisah** -- deskriptor ligan, deskriptor pocket, atau hitungan kontak -- sumber-sumber
itu **saling redundan**, dan GNN 3D tidak melampaui plafon deskriptor itu. Ini replikasi lengkap Volkov
et al. 2022 di split yang leakage-controlled, ditambah dua hal yang tidak ada di sana: kontrol konfound
ukuran pada fitur interaksi, dan uji langsung klaim CORDIAL.

### Cakupan bidang: apa yang SUDAH dan BELUM

SUDAH -- cheminformatics: ECFP4, 14 deskriptor RDKit, Tanimoto (diuji vs RDKit BulkTanimotoSimilarity),
scaffold Bemis-Murcko generik, novelty tier (protokol Mattsson), activity cliff, applicability domain
lewat similarity.
SUDAH -- bioinformatics (dasar): komposisi asam amino pocket (20-dim), pengelompokan fisikokimia
(hidrofobik/polar/+/-/aromatik/fleksibel), komposisi elemen, fraksi backbone, geometri (Rg, extent,
sphericity).
SUDAH -- biomolecular informatics (dasar): ECIF-style (elemen protein x elemen ligan x shell jarak).

BELUM, dan ini jujur:
* stratifikasi error per famili protein / kelas target (kinase vs protease vs nuclear receptor)
* konservasi sekuens / fitur MSA residu pocket
* tipe interaksi PLIP (hbond / hidrofobik / pi-stacking / jembatan garam) -- ada di rencana Track E,
  belum dipakai di sini. Ini yang akan membuat blok interaksi jauh lebih bermakna daripada hitungan
  pasangan elemen.
* coverage KONDISIONAL conformal per tier kebaruan dan per famili protein (perpanjangan A1d; kerangka
  dari Gibbs et al. 2023)
* deskriptor druggability pocket (volume, enclosure, buriedness) -- butuh fpocket/CASTp

Status tes proyek: **7 suite, 91 tes, semua lolos.**

---

## PROGRAM RISET TERSATUKAN + EKSPERIMEN KUNCI (2026-10-10)

Dokumen: **`guidance/RESEARCH_PROGRAM.md`** -- menggantikan tumpukan track yang dimotivasi terpisah
dengan SATU pertanyaan, memetakan setiap celah literatur ke cabang kita, dan menyatakan bukti mana yang
sudah dipegang / sedang jalan / masih hilang.

Pertanyaan tunggalnya: ketika guidance gagal, APA tepatnya yang gagal? Tiga penjelasan hidup:
(A) masalah tuning, (B) masalah akurasi, (C) masalah INFORMASI. Tesis sudah menyingkirkan (A) lintas
tiga orde besaran dan tiga varian mekanisme, dan menolak (B) dengan alasan Pearson 0.58-0.65. Kerja
2026-10-10 menegakkan **(C)** -- dan (C) itulah yang membuat penolakan (B) koheren, bukan membingungkan.

### EKSPERIMEN KUNCI: `guidance/pose_sensitivity.py` -- HASIL PILOT SANGAT TAJAM

Semua bukti lain bersifat tidak langsung (tentang apa yang bisa dilakukan model LAIN). Ini menguji
predictor-nya sendiri: tahan molekul dan pocket tetap, ubah HANYA penempatan rigid-body ligan.

Desain faktorial yang memisahkan "pakai antarmuka" dari "pakai statistik bulk protein" -- ketiga varian
menghapus TEPAT 50% atom protein, jadi perubahan bulk identik dan hanya isi antarmuka yang berbeda:

| manipulasi | antarmuka | jumlah atom | mean abs d-prediksi (pilot n=10) |
|---|---|---|---|
| simpan separuh TERDEKAT | **utuh** | -50% | **1.337 pK** |
| simpan separuh TERJAUH | **hancur** | -50% | **1.126 pK** |
| simpan separuh acak | sebagian | -50% | 1.320 pK |
| tukar pocket (protein lain sama sekali) | diganti | ~sama | 0.947 pK |
| geser ligan 8 A KELUAR pocket | hancur | utuh | **0.19-0.25 pK** |

**Menghancurkan kontak asli ligan LEBIH MURAH (-16%) daripada menghapus atom yang tidak menyentuh apa
pun.** Ketiga varian strip praktis sama besar. Dan model **~5x lebih sensitif terhadap BERAPA BANYAK
atom protein yang ada daripada terhadap DI MANA ligannya berada**.

Itu mekanismenya, dan ia menjelaskan seluruh rangkaian temuan sekaligus:
* gradien menjauh dari pocket (r=+0.24) -- tidak ada sinyal antarmuka untuk diikuti
* deskriptor pocket-saja bekerja (0.2889) -- properti bulk pocket
* deskriptor ligan-saja bekerja (0.3534) -- ukuran ligan
* ecif_raw = ukuran ligan menyamar (q=0.013)
* guidance gagal di 5 rute dan 56 perbandingan level-pocket

Dose-response translasi monoton (0.02 -> 0.04 -> 0.08 -> 0.19 -> 0.19 pK untuk 0.5/1/2/4/8 A), jadi
modelnya TIDAK buta-pose sepenuhnya -- hanya lemah: 8 A keluar pocket = 0.28x RMSE-nya sendiri.
Perturbasi hanya rigid-body: konformer ligan tidak pernah diubah, jadi perubahan prediksi tidak bisa
diatribusikan ke geometri internal yang jadi tidak fisis.

Run penuh n=300, B=2000 sedang jalan di CPU (`logs_queue/pose_sensitivity_n300.log`); GPU tidak
disentuh, Arm B aman.

### Celah yang masih hilang, diperingkat (detail di RESEARCH_PROGRAM.md 5)
1. ~~pose sensitivity~~ -- pilot selesai, run penuh jalan
2. **tipe interaksi PLIP** sebagai blok ke-8 tangga representasi (infrastruktur sudah ada di
   `track_e/PlipLabelStore`). Kalau interaksi BERTIPE (hbond/hidrofobik/pi-stacking/jembatan garam) pun
   tidak menambah apa-apa, klaim redundansi jadi sangat kuat. Kalau menambah, itu hasil positif dan ia
   menamai perbaikannya. CPU saja.
3. **coverage KONDISIONAL conformal** per tier kebaruan dan per famili protein (Gibbs et al. 2023 memberi
   kerangka; Jeliazkova et al. 2026 memprediksi coverage turun di kimia novel). Perpanjangan paling
   menjanjikan dari satu hasil positif kita. CPU saja.
4. stratifikasi famili protein / kelas target
5. A1e-beta (jalan, di antrean)

### Dampak ke tesis (RESEARCH_PROGRAM.md 6)
Tidak ada yang dibuang. Bab IV dapat satu seksi; Bab X kesimpulan (1) harus direvisi (argumen
"kegagalan bukan karena akurasi rendah: Pearson 0.58-0.65" sekarang bocor, karena model tanpa input
protein mencapai korelasi yang sama); abstrak dapat satu kalimat. Pernyataan kontribusi naik dari
"negative result yang MELOKALISIR di mana guidance gagal" menjadi "negative result yang MENJELASKAN
MENGAPA guidance gagal, dengan mekanisme diisolasi lewat audit sumber informasi".

---

## NOISE CEILING, DEKOMPOSISI VARIANS, DAN y-RANDOMISASI (2026-10-10) — MENGUBAH INTERPRETASI SEMUA R2

`guidance/cheminformatics/noise_ceiling.py`. Ini analisis yang membingkai ulang SETIAP R2 di tesis, dan
ia hilang sampai sekarang.

### Label kita campuran, dan ketidakpastiannya SUDAH DIUKUR orang
Komposisi LP-PDBBind: **Kd 37% / IC50 37% / Ki 26%**. Dan komposisinya BERGESER antar split --
train Kd 41%/IC50 33%, test IC50 46%/Kd 26%. IC50 paling bergantung assay dan justru over-represented
di test: train dan test tidak mengukur kuantitas yang persis sama.

* **Hernandez-Garrido et al. 2023** (AI in the Life Sciences) memakai record ChEMBL di mana pasangan
  protein-ligan yang sama diukur lebih dari sekali, untuk menaksir ketidakpastian menggabungkan
  Kd+Ki+IC50: **MAE 0.78 log unit, RMSE 1.04, Pearson 0.76**.
* **Landrum et al. 2024** (JCIM, 158 sitasi) dari arah lain: dengan kurasi minimal, **65%** pengukuran
  IC50 berulang untuk senyawa+target yang sama berbeda >0.3 log unit, **27%** berbeda >1 log unit,
  Kendall tau hanya 0.51 -- dan assay Ki ternyata tidak lebih baik. Mereka merilis kode "maximal
  curation".

Kalau dua pengukuran independen atas kuantitas yang sama hanya berkorelasi r=0.76, maka
**tidak ada prediktor yang bisa melampaui R2 = 0.76^2 = 0.5776** atau menekan RMSE di bawah 1.04.
Itu bukan batas pemodelan; itu derau label.

### Performa sebagai fraksi dari yang DAPAT DICAPAI
| model | test R2 | % dari plafon | RMSE/derau |
|---|---|---|---|
| egnn_a1c_ensemble | 0.4072 | **70.5%** | 1.22x |
| egnn_qat_s2023 | 0.3872 | 67.0% | 1.24x |
| desc_ridge (ligan saja) | 0.3531 | 61.1% | 1.27x |
| egnn_stage0 | 0.3415 | 59.1% | 1.29x |
| vina | 0.2601 | 45.0% | 1.36x |

"R2 0.41" terbaca buruk; "70% dari yang dapat dicapai, pada RMSE 1.22x lantai derau eksperimen"
adalah angka yang jujur dan jauh lebih informatif. Ini yang harus dikutip tesis.

### ANALISIS DAYA ATAS TUGASNYA SENDIRI -- dan ini temuan metodologis serius
* lantai (ligand-only terbaik, tanpa protein) : R2 0.3531
* plafon (derau label eksperimen)             : R2 0.5776
* **seluruh jendela** tempat informasi struktural bisa menunjukkan diri: **0.2245 R2**
* lebar rata-rata CI 95% target-clustered kita: **0.3871 R2**
* **jendela / lebar CI = 0.58**

Jendelanya LEBIH SEMPIT dari SATU interval kepercayaan. Desain ini **tidak bisa** menentukan di mana
sebuah model berada di dalamnya -- dan begitu juga perbandingan terpublikasi yang melaporkan gain
0.02-0.05 R2 di atas data sejenis. Laporan yang jujur adalah interval, bukan peringkat. Ini properti
TUGAS dan derau labelnya, bukan properti satu model. (Sebagai pembanding: sd antar-seed arsitektur EGNN
yang sama = 0.0377 R2, yaitu 17% dari jendela.)

### y-RANDOMISASI (Rucker et al. 2007, 897 sitasi) -- kontrol standar QSAR yang belum pernah kita jalankan
20 permutasi, pipeline ridge-14-deskriptor di-fit ulang tanpa diubah:
* **permutasi global**: R2 mean +0.0037, max +0.0141. desc_ridge asli 0.3531 **LOLOS** kontrol
  chance-correlation dengan selisih +0.3390. Hasil ligand-only itu sinyal nyata, bukan artefak 14
  deskriptor yang cukup lentur untuk memuat apa saja.
* **permutasi DALAM-target**: R2 mean **+0.2468**, sd 0.0019.

### DEKOMPOSISI VARIANS -- konsekuensi dari baris terakhir itu, dan ia besar
Permutasi dalam-target menahan himpunan label tiap target tapi mengacak ligan mana membawa label mana.
Jadi ia MENGHANCURKAN hubungan struktur-aktivitas sambil MEMPERTAHANKAN kaitan antara chemotype dan
afinitas tipikal kelas protein yang diikat chemotype itu. Yang bertahan = **prior tingkat kelas**,
bukan SAR.

| | R2 |
|---|---|
| chance (label dipermutasi total) | +0.0037 |
| **PRIOR TINGKAT KELAS (permutasi dalam-target)** | **+0.2468** |
| plafon eksperimen | +0.5776 |
| **sinyal tersedia DI LUAR prior kelas** | **0.3308** |

**70% dari yang dicapai model ligand-only terbaik dapat direproduksi TANPA hubungan
struktur-aktivitas dalam-target sama sekali.** Target test disjoint dari train, jadi ini BUKAN
memorisasi target -- ini chemotype -> afinitas tipikal kelas protein yang diikatnya. Nyata, transferable,
dan bukan arti dari "prediksi afinitas berbasis struktur".

Diukur dari titik nol yang benar (di luar prior kelas), model struktur justru TERPISAH lebih jelas:
| model | di luar prior | % dari headroom |
|---|---|---|
| egnn_a1c_ensemble | +0.1604 | **48.5%** |
| egnn_qat_s2023 | +0.1404 | 42.4% |
| desc_ridge (ligan) | +0.1063 | 32.1% |
| egnn_stage0 | +0.0947 | 28.6% |
| heavy_atoms | +0.0601 | 18.2% |
| vina | +0.0133 | 4.0% |

Caveat yang harus dinyatakan: baseline permutasi hanya di-fit ulang untuk pipeline DESKRIPTOR, karena
mempermutasi label lalu melatih ulang EGNN butuh GPU. Jadi prior itu taksiran kanal chemotype, bukan
kontrol per-arsitektur. Menjalankannya untuk EGNN adalah pekerjaan yang tersisa.

### Prior art lain yang ditemukan dan wajib disitasi
* **van Tilborg et al. 2022** (JCIM, 314 sitasi), MoleculeACE -- benchmark 24 pendekatan ML pada
  activity cliff di 30 target: SEMUA kesulitan, dan **ML berbasis deskriptor MENGALAHKAN deep learning
  yang lebih kompleks**. Itu persis pola kita (desc_ridge >= EGNN), kini dengan rujukan besar.
* **Deng et al. 2023** (Nature Communications, 213 sitasi) -- 62.820 model dilatih: model
  representation-learning menunjukkan performa TERBATAS vs representasi tetap di sebagian besar
  dataset; activity cliff berdampak signifikan; ukuran dataset yang menentukan.
* **Dablander et al. 2023** -- model QSAR memang sering gagal memprediksi activity cliff.
* **Kwapien et al. 2022** -- data aditif paling mudah diprediksi; deep learning bukan pengecualian.
* **Brown 2025** (PNAS, CORDIAL) -- leave-superfamily-out; split kita per-target, bukan per-superfamili,
  jadi masih LEBIH MUDAH dari standar itu. Harus didisklos.
* **PLIP 2021/2025** (Nucleic Acids Research, 1792 + 302 sitasi) -- delapan tipe interaksi nonkovalen.
* **Cheng et al. 2009** (508 sitasi) -- scoring function klasik berkorelasi 0.545-0.644 dengan konstanta
  eksperimen; angka kita setara, dan sekarang bisa dibandingkan terhadap plafon yang benar.

### Pekerjaan yang tersisa, diperbarui
1. **y-randomisasi dalam-target untuk EGNN** (butuh GPU) -- supaya prior kelas jadi kontrol
   per-arsitektur, bukan taksiran.
2. **metrik activity-cliff gaya MoleculeACE** di test set kita -- pola sudah terlihat di novelty tier,
   belum diukur sebagai metrik.
3. **analisis matched molecular pair (MMP)** -- cara ketat menguji apakah model menangkap SAR lokal.
4. tipe interaksi PLIP sebagai blok tangga ke-8.
5. coverage kondisional conformal per tier/famili.
6. kurasi "maximal curation" Landrum untuk label BindingNet Arm B/C.
