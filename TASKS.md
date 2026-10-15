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

**STATUS 2026-10-12: pra-registrasi SELESAI, implementasi SELESAI dan diverifikasi, menunggu keputusan
skala.** Dokumen lengkap: `guidance/RQ2_POSITIVE_CONTROL_PREREGISTRATION.md`. Desain: best-of-N Vina
rerank (bukan backprop lewat Vina -- tidak differentiable) -- oversample pool tanpa guidance, seleksi
murah via `score_only` (~0,57s/molekul, tidak sirkular karena evaluasi akhir pakai pass Vina independen
mode `dock` exhaustiveness=16, SAMA seperti baseline/affinity_only). Kode: variant baru `vina_rerank` di
`guidance/run_task_f.py` (fungsi sibling `run_triple_vina_rerank`, TIDAK mengedit `run_triple` yang
dipakai baseline/affinity_only -- nol risiko regresi). Diverifikasi dua jalur: (1) smoke test mekanis
end-to-end (1 pocket, num_steps=25 sengaja kecil) -- jalur kegagalan pool-kosong bekerja benar, tertangkap
dan tercatat rapi, bukan crash; (2) logika baru (Vina score_only + ranking + mapping kolom
`vina_select_score`) diuji LANGSUNG pakai 5 molekul nyata dari `baseline/pocket0_seed1` yang sudah ada --
skor masuk akal (-5,5 sampai -8,7, searah dengan Vina Dock mean yang sudah dilaporkan -6,8/-7,3), urutan
benar, mapping DataFrame benar. 13 suite / 178 tes lama tetap hijau (tidak ada regresi).
**Skala:** pilot 6 pocket x 1 seed x K=2 ~= 1,1 jam GPU (direkomendasikan untuk dijalankan duluan);
skala penuh (20 pocket x 3 seed, sama seperti baseline/affinity_only) ~= 11,5 jam (K=2) atau 21,5 jam
(K=4). **Ini komitmen GPU/waktu besar -- menunggu persetujuan eksplisit sebelum dilepas**, sesuai
protokol operasional proyek ini.

**HASIL FINAL -- SELESAI 2026-10-13, 60/60 triple (20 pocket x 3 seed), ~22 jam wall-clock total
(pilot+full, lebih lama dari estimasi 12,6 jam karena overhead suspend termal).**

| metrik | vina_rerank | baseline | uji berpasangan (n=20 pocket, 600 vs 597 molekul) |
|---|---|---|---|
| **Vina Dock** | **-8,068 +- 2,660** | -6,813 +- 10,079 | **Wilcoxon p=1,9e-06**, Mann-Whitney p=3,1e-10 |
| delta per-pocket (CI bootstrap klaster-target B=2000) | | | **-1,2668 kcal/mol, CI95 [-2,149, -0,740] -- mengecualikan nol** |
| SA | 0,572 | 0,601 | p=8,2e-05 (SIGNIFIKAN, turun) |
| RA score | 0,353 | 0,435 | p=2,67e-05 (SIGNIFIKAN, turun) |
| QED | 0,462 | 0,449 | p=0,154 (tidak signifikan) |
| ligand efficiency | -0,369 | -0,340 | p=0,756 (tidak signifikan -- FLAT) |
| PoseBusters valid | 0,582 | 0,551 | bertahan/sedikit naik, tidak turun |
| heavy_atoms | 23,09 +- 8,36 | 21,32 +- 9,06 | Mann-Whitney p=0,0002 (SIGNIFIKAN, naik +1,8 atom) |

**Verdict: HARNESS VALID. M1 terkonfirmasi sebagai sebab kegagalan guidance EGNN-surrogate.** Seleksi
berbasis Vina docking ASLI (zero-shot, tidak pernah dilatih di label pK kita, bukan gradien) menggeser
distribusi molekul yang dihasilkan ke arah Vina Dock yang jauh lebih baik -- signifikan secara
statistik, bukan sirkular (skor seleksi `score_only` cepat vs skor evaluasi `dock` mode penuh adalah
dua pass Vina yang berbeda), dan tetap bertahan setelah target-clustered bootstrap. Ini membuktikan
generator + pocket + jalur evaluasi SANGGUP mendeteksi perbaikan ketika sinyal pemandunya genuinely
informatif -- jawaban langsung untuk pertanyaan "bagaimana saya tahu harness-mu bukan yang rusak?"

**Tapi ada confound ukuran yang HARUS dinyatakan, bukan disembunyikan**: molekul vina_rerank
sedikit lebih besar (+1,8 atom berat, p=0,0002) dan kurang synthetically-accessible (SA, RA keduanya
turun signifikan). Ini pola klasik "molekul lebih besar = skor Vina absolut lebih baik". **Tapi bukan
size-inflation SEDERHANA**: ligand efficiency (Vina Dock per atom berat -- metrik yang DIRANCANG
menangkap pola ini) FLAT, tidak memburuk (p=0,756); PoseBusters validitas fisik bertahan/naik, tidak
turun. Jadi perbaikannya sebagian riil (per-atom tidak memburuk), sebagian berasal dari ukuran yang
sedikit lebih besar -- kedua hal ini harus dilaporkan bersama, bukan dipilih salah satu yang nyaman.

**Bug ditemukan dan diperbaiki di tengah jalan**: `rq2_analyze` GAGAL pertama kali (exit 1, IndexError)
karena `run_triple_vina_rerank` menulis marker DONE dengan skema kunci BERBEDA (`n_pool=...`) dari yang
dibaca `load_all_results` (`n_attempted=...`). Verifikasi sintetis sebelumnya tidak pernah memanggil
`load_all_results` sama sekali, jadi ketidakcocokan ini tak terlihat sampai run 60-triple nyata mencapai
tahap analisis. Diperbaiki di kode (tambah `n_attempted=` ke marker baru) DAN di 60 file yang sudah
ditulis di disk (ditambal di tempat, bukan dijalankan ulang -- datanya valid, cuma format markernya
salah). 3 tes regresi baru (`test_analyze_task_f_marker.py`).

### 2. PLIP typed interactions sebagai blok ke-8 tangga representasi -- SELESAI 2026-10-13
**Apa:** hbond / hidrofobik / pi-stacking / jembatan garam / halogen sebagai fitur, masuk
`representation_ladder.py`. Infrastruktur sudah ada: `guidance/track_e/core.py` `PlipLabelStore`.
**Mengapa:** ECIF (pasangan elemen x shell jarak) itu notasi interaksi yang kasar. Kalau interaksi
BERTIPE pun tidak menambah apa-apa di atas marginal, klaim redundansi jadi sangat kuat. Kalau menambah,
itu hasil POSITIF dan ia menamai perbaikannya.

**Catatan skop**: `PlipLabelStore` punya 4 kelas (hbond/hidrofobik/pi-stack/jembatan-garam) -- TIDAK ada
halogen (tak pernah dibangun di cache aslinya). Coverage diverifikasi 100% (64.888/64.888 baris LP-split)
sebelum dikerjakan. Fitur yang dipakai: hitungan BERTIPE dijumlahkan di seluruh atom ligan per kompleks
(4 angka/kompleks) -- representasi paling kasar yang mungkin, BUKAN pernyataan definitif soal semua
representasi interaksi bertipe yang bisa dibuat (versi ber-shell-jarak seperti ECIF tapi dengan tipe PLIP
belum diuji). Ini dinyatakan secara eksplisit sebagai keterbatasan, bukan disembunyikan.

**HASIL** (`guidance/cheminformatics/representation_ladder.py`, blok lama diverifikasi BYTE-EXACT
tak berubah sebelum/sesudah):

| blok | dim | model | val R2 | test R2 | CI95 |
|---|---|---|---|---|---|
| plip_raw | 4 | hgb | 0,1681 | 0,0743 | [-0,093, +0,173] |
| plip_norm | 4 | hgb | 0,2147 | 0,0822 | [-0,043, +0,173] |
| plip_norm+size | 5 | ridge | 0,3654 | 0,3241 | [+0,167, +0,430] |

| perbandingan berpasangan | dR2 | q | verdict |
|---|---|---|---|
| plip_norm - ligand | -0,2712 | 0,003 | ligand MENANG signifikan |
| plip_norm - pocket | -0,2066 | 0,003 | pocket MENANG signifikan |
| plip_norm - ecif_norm | -0,1254 | 0,198 | seri |
| plip_raw - plip_norm | -0,0080 | 0,870 | seri (tak ada confound ukuran ANTARA raw/norm) |
| **plip_norm+size - plip_norm** | **+0,2419** | **<0,001** | **size MENANG telak** |

**Temuan: lebih tajam dari ecif.** Empat hitungan interaksi bertipe, dijumlah mentah-mentah per kompleks,
HAMPIR TAK MEMBAWA INFORMASI (R2 0,07-0,08) -- kalah signifikan dari ligand (0,353) MAUPUN pocket (0,289)
sendirian. Begitu satu fitur ukuran (jumlah atom berat) ditambahkan, melompat ke R2=0,324 -- hampir
seluruh sinyal yang berguna datang dari UKURAN, bukan dari TIPE interaksinya. Ini memperkuat (bukan
cuma mengulang) temuan ecif_raw/ecif_norm: bahkan representasi yang secara eksplisit bertipe kimiawi
pun runtuh jadi proksi ukuran begitu diringkas kasar ke beberapa angka per kompleks.

2 tes baru (`test_chem.py`) memakai contoh hitung-tangan untuk `plip_features` + abort eksplisit untuk
baris yang tak tercakup. **Biaya:** ~15 detik tambahan di atas waktu ladder yang sudah ada.

### 3. Coverage KONDISIONAL conformal per novelty tier + famili protein -- SELESAI 2026-10-12
**Apa:** A1d membuktikan conformal satu-satunya UQ yang valid — tapi itu coverage MARGINAL. Uji apakah
coverage tetap valid per tier Tanimoto (tier sudah ada) dan per famili protein.
**Mengapa:** Jeliazkova et al. 2026 memprediksi coverage TURUN di kimia novel; Gibbs et al. 2023 memberi
kerangkanya. Interval yang valid secara marginal tapi diam-diam under-cover justru pada senyawa novel
yang dipedulikan praktisi adalah temuan yang layak dilaporkan. Ini perpanjangan paling menjanjikan dari
satu hasil positif kita.

**HASIL** (`guidance/uncertainty_a1/conditional_conformal_coverage.py`, B=2000, q=2,1960 dipakai
APA ADANYA dari a1d -- tidak dihitung ulang, supaya pertanyaannya murni "apakah q yang SUDAH dipakai
tetap menepati janjinya per subgrup"):

| tier Tanimoto | n | target | coverage | CI95 | nominal 0,90 di dalam? |
|---|---|---|---|---|---|
| novel (<0,35) | 7577 | 79 | 0,9104 | [0,864, 0,955] | YA |
| distant (0,35-0,50) | 2319 | 67 | 0,8875 | [0,835, 0,935] | YA |
| related (0,50-0,70) | 1526 | 51 | 0,8309 | [0,710, 0,911] | YA (paling dekat gagal) |
| same series (>=0,70) | 433 | 41 | 0,8984 | [0,818, 0,956] | YA |

**0 dari 4 tier novelty gagal.** Bertentangan dengan prediksi umum Jeliazkova et al. 2026 untuk kimia
novel -- tier TERSULIT (<0,35 Tanimoto) justru coverage-nya 0,9104, SEDIKIT DI ATAS nominal, bukan di
bawahnya. Tier "related (0,50-0,70)" paling dekat ke batas gagal (titik 0,83) tapi CI95 [0,710, 0,911]
masih memuat 0,90.

**Per famili Pfam**: 127/127 target test teranotasi (tidak ada yang hilang). Tapi hanya **2 dari 10
strata famili punya >= 10 klaster target** (ambang `MIN_CLUSTERS` yang sama dipakai `novelty_tiers.py`)
-- PF00069/kinase (11 target, CI [0,877, 1,000], LOLOS) dan `other_family` gabungan 83 target (CI
[0,863, 0,940], LOLOS). **7 famili lain (PF01327, PF00169, PF00194, PF00775, PF07714, PF00104, PF00064,
3-7 target masing-masing) ditandai UNDERPOWERED, bukan diberi verdict** -- titik estimasinya melompat
liar (0,52 sampai 0,99) justru karena n sekecil itu, persis contoh kenapa underpowered harus dinyatakan
eksplisit bukan disembunyikan di balik angka titik yang kelihatan meyakinkan.

**Kesimpulan jujur, dua sisi:** (1) tidak ada strata yang cukup bertenaga menunjukkan coverage
kondisional gagal -- hasil positif conformal (A1d) bertahan lebih kuat dari ekspektasi, bahkan di tier
paling novel; (2) tapi mayoritas pemecahan per-famili TIDAK bertenaga untuk mendeteksi apa pun -- ini
keterbatasan daya, bukan bukti tidak ada masalah, dan harus dinyatakan sebagai itu di tesis, bukan
dibiarkan terbaca sebagai "semua famili lolos".

9 tes baru (`guidance/uncertainty_a1/tests/test_conditional_conformal_coverage.py`) menguji logika BARU
(mapping famili, collapse famili jarang via jumlah TARGET bukan jumlah baris, keputusan underpowered) --
`cluster_bootstrap_coverage` dipakai ulang dari `conformal_prediction.py`, tidak diimplementasi ulang.
**Biaya:** CPU saja, ~3,4 detik untuk B=2000 (murni resampling numpy di array yang sudah di-cache,
bukan training ulang) -- dikerjakan paralel sementara GPU dipakai RQ2, tanpa bersaing sumber daya.

### 4. y-randomisasi DALAM-TARGET untuk EGNN
**Apa:** permutasi label dalam target, latih ulang EGNN, ukur R2.
**Mengapa:** prior kelas 0.2468 saat ini hanya di-fit untuk pipeline DESKRIPTOR. Supaya jadi kontrol
per-arsitektur (bukan taksiran kanal chemotype), EGNN-nya harus dilatih ulang di label terpermutasi.
Tanpa ini, tabel "di luar prior kelas" punya caveat yang harus selalu dinyatakan.
**Biaya:** GPU, 1 run (~2.5 jam). Murah, dan menutup caveat di temuan terkuat kita.

**HASIL -- SELESAI 2026-10-13 23:57** (early-stop epoch 7, best val epoch 2, hanya 1j54m -- jauh lebih
cepat dari estimasi karena tidak ada sinyal real untuk terus dipelajari setelah plafon prior tercapai):

| | test R2 | Pearson | Spearman | RMSE |
|---|---|---|---|---|
| EGNN label TERPERMUTASI dalam-target | **0,307** | 0,560 | 0,559 | 1,372 |
| (pembanding) ridge-14-deskriptor, label terpermutasi (sudah ada) | 0,2468 +- 0,0019 | - | - | - |
| (pembanding) EGNN Stage 0, label REAL (sudah ada) | 0,342 | 0,609 | - | - |

**EGNN yang dilatih TANPA sinyal SAR dalam-target sama sekali (label diacak dalam setiap target)
mencapai 0,307 -- LEBIH TINGGI dari benchmark deskriptor (0,2468), dan itu 90% (0,307/0,342) dari
performa EGNN NORMAL yang dilatih di label asli.** Ini menutup caveat yang sudah berulang kali disebut:
klaim prior-kelas sekarang terbukti untuk ARSITEKTUR EGNN sendiri, bukan cuma pipeline deskriptor.

**Kenapa angkanya LEBIH TINGGI dari ridge (bukan kontradiksi, mekanisme beda)**: ridge-deskriptor
mencapai prior kelasnya lewat korelasi TAK LANGSUNG (kemotipe ligan kebetulan berkorelasi dengan target
mana yang diuji). EGNN punya rute LANGSUNG -- ia benar-benar membaca struktur 3D protein, jadi
mengenali "target mana ini" tidak perlu lewat korelasi kemotipe sama sekali. EGNN lebih efisien
MENCAPAI prior yang sama, bukan belajar SAR yang lebih banyak dari itu.

**Implikasi yang lebih tajam dari temuan ligand-only**: untuk ridge, 70% performa adalah prior kelas.
Untuk EGNN, **90%** performa normalnya bisa direproduksi TANPA SATU PUN sinyal SAR dalam-target.
Sisa ruang untuk "benar-benar membaca interaksi" bahkan lebih sempit dari yang sudah disangka.

### 5. Metrik activity-cliff gaya MoleculeACE -- KODE SELESAI, EKSEKUSI DIANTRIKAN 2026-10-12
**Apa:** identifikasi pasangan activity cliff di test set, ukur performa khusus di sana.
**Mengapa:** pola sudah terlihat di novelty tier (ligand-only kolaps di similarity tinggi) tapi belum
diukur sebagai metrik. van Tilborg et al. 2022 (314 sitasi) menyediakan definisi dan platformnya, dan
temuan mereka (deskriptor > deep learning pada cliff) persis pola kita.

**Definisi yang dipakai (DISCLOSED sebagai penyederhanaan, bukan reimplementasi setia)**: konvensi
terpublikasi `calc_cliff_rmse(similarity=0.9, potency_fold=10, in_log10=True)` -- sim>=0,9, |delta pK|
>=1,0 (pK sudah skala log10, tanpa konversi). Kriteria similaritas asli MoleculeACE konsensus TIGA ukuran
(substruktur/scaffold/SMILES); implementasi kita pakai SATU -- ECFP4 Tanimoto yang sudah dipakai di
seluruh proyek ini. Pasangan cliff dihitung HANYA dalam-target (lintas-target tidak bermakna -- split
kita target-disjoint jadi tidak ada partner cliff dari train). Similaritas berpasangan dihitung lewat
matmul matriks bit (intersection = F @ F.T), bukan loop RDKit per-pasangan -- itulah yang membuat target
terbesar (~1 juta pasangan) tangani.

**Kode selesai dan diverifikasi** (`guidance/cheminformatics/activity_cliffs.py`, 4 tes baru di
`test_chem.py` memakai contoh hitung-tangan 4-molekul, termasuk kasus tepat-di-ambang dan kasus yang
sengaja GAGAL salah satu kriteria). **Eksekusi nyata DIANTRIKAN** (`activity_cliffs` di
`queue_steps.sh`, setelah `rq2_analyze`) -- ditulis sementara CPU sudah jadi bottleneck RQ2; estimasi
kasar 10 target terbesar saja ~15 milyar operasi, jadi dijalankan setelah CPU bebas, bukan bersaing
dengan run yang diminta diprioritaskan.
**HASIL NYATA -- SELESAI 2026-10-13 21:11** (11.855 ligan test, 127 target, B=2000):
4.050.366 pasangan dalam-target diperiksa, **2.856 pasangan cliff ditemukan, 344 compound cliff (2,9%)**.

| stratum | n | target | struct_R2 | lig_R2 | dR2 [95% CI] |
|---|---|---|---|---|---|
| cliff | 344 | 11 | 0,2904 | 0,3611 | -0,0708 [-0,428, +0,065] |
| non_cliff | 11.511 | 126 | 0,4111 | 0,3517 | +0,0594 [-0,032, +0,150] |

**CI pada stratum cliff mengecualikan nol? TIDAK** (p=0,285) -- bukan beda signifikan, titik estimasi
sedikit mendukung ligand-only tapi CI-nya lebar dan memuat nol. **Tidak ada klaim "descriptor menang di
cliff" yang dibuat** (skrip sendiri mencetak itu secara eksplisit) -- berbeda dari pola Kwapien et al.
2022 yang dikutip sebagai motivasi, di split DAN sampel ini tidak terbukti signifikan. Konsisten dengan
disiplin proyek: titik estimasi yang menarik tapi CI lebar dilaporkan sebagai itu, bukan dibesar-besarkan.
**Biaya nyata:** ~15 detik (fragmentasi) -- jauh di bawah estimasi awal "sehari".

**PERBAIKAN PREVENTIF 2026-10-13, sebelum skrip ini sempat jalan sekali pun**: audit ulang menemukan
`stratum_r2_comparison`-nya memakai `r2_rows` yang SAMA yang menyebabkan bug +5,8e16 di item 7 (varian
pK lokal mendekati nol tapi tidak persis nol, lolos guard `denom<=0`). Dipasang guard yang sama
(ambang relatif std pK lokal < 1% std pK global) SEBELUM dijalankan di queue tanpa pengawasan --
ditemukan lewat audit, bukan lewat kegagalan nyata. Sekaligus ditemukan DAN diperbaiki kesalahan kedua
di blok "Van Tilborg et al." akhir skrip: perbandingan `champ_cliff > struct_cliff` memakai titik
estimasi TANPA uji signifikansi apa pun -- versi lebih telanjang dari kesalahan yang sama yang baru
diperbaiki di item 6 (MMP). Diganti memakai bootstrap berpasangan yang sudah dihitung di tabel strata
di atasnya. 2 tes baru (`test_chem.py`) memaku guard degenerate dan memastikan strata sehat tidak
ikut tertandai degenerate.

### 6. Analisis matched molecular pair (MMP) -- SELESAI 2026-10-13
**Apa:** pasangan yang beda satu substituen; apakah model menangkap arah perubahan afinitasnya.
**Mengapa:** cara paling ketat menguji SAR LOKAL, dan pelengkap langsung dekomposisi prior kelas: prior
kelas menangkap antar-seri, MMP menguji dalam-seri. Kwapien et al. 2022 menunjukkan data aditif paling
mudah dan deep learning bukan pengecualian.

**Metode** (`guidance/cheminformatics/matched_molecular_pairs.py`): fragmentasi single-cut via
`rdMMPA.FragmentMol` (algoritma standar Hussain & Rea 2010, bukan buatan sendiri) -- fragmen lebih besar
= core, lebih kecil = r-group; dua ligan jadi matched pair kalau core SAMA PERSIS dan core >= 60% berat
molekul. **Dalam-target saja** (lintas-target tidak bermakna, dan split kita target-disjoint jadi tidak
ada partner lintas-target tersedia). Dedup pose-duplikat SEBELUM fragmentasi: 783/861 pasangan
(SMILES,target) punya >1 baris (pose docking berbeda), 26 di antaranya pK sedikit berbeda (rounding,
bukan pengukuran berbeda) -- dirata-rata.

**HASIL: 183 seri matched-pair, 375 pasangan, 360 berdelta-pK taknol.**

| model | sign agreement | vs kebetulan (0,5) | Pearson(aktual,prediksi) |
|---|---|---|---|
| structure (egnn_a1c_ensemble) | 0,461 | CI [0,399, 0,535] memuat 0,5 | -0,0233 |
| ligand_only (desc_ridge) | 0,483 | CI [0,391, 0,596] memuat 0,5 | -0,0592 |

**Uji BERPASANGAN** (bukan membandingkan dua CI marginal yang tumpang-tindih -- lihat catatan kesalahan
di bawah): delta (structure-ligand_only) = -0,0222, CI95 [-0,102, +0,043], p=0,551. **Tidak ada beda
signifikan antar kedua model.**

**Temuan utama, independen dari siapa yang menang**: KEDUA model sign agreement-nya DI BAWAH ATAU DI
TITIK kebetulan (0,461 dan 0,483, keduanya <=0,5). **Tidak satu pun model menangkap arah perubahan
afinitas akibat satu substituen lebih baik dari lempar koin** -- ini SAR paling lokal dan paling mudah
yang bisa diuji (persis yang Kwapien et al. 2022 sebut "kasus termudah untuk model apa pun"), dan
keduanya gagal di situ juga. Konsisten dengan dan memperkuat dekomposisi prior kelas: performa model
bukan dari menangkap SAR, bahkan di tingkat paling lokal sekalipun.

**Kesalahan saya sendiri, ditemukan dan diperbaiki sebelum dicatat**: draf pertama skrip ini
menyimpulkan "ligand-only menang" HANYA dari titik estimasi (0,483 > 0,461) dengan dua CI terpisah yang
jelas tumpang-tindih -- persis kesalahan yang `compare_vs_structure.py` sendiri sudah mendisiplinkan diri
untuk hindari ("overlapping marginal intervals are weak evidence either way, because both models are
evaluated on the SAME complexes"). Diperbaiki dengan bootstrap berpasangan pada pasangan yang SAMA PERSIS
sebelum menyimpulkan apa pun. 5 tes baru di `test_chem.py` memaku fragmentasi, guard lintas-target, arah
delta_pk, ambang core-fraction, dan regresi kesalahan "menang berdasar CI terpisah" ini secara spesifik.

**Biaya terukur:** ~7,5 detik untuk seluruh test set (bukan "sehari" seperti estimasi awal) -- fragmentasi
RDKit jauh lebih murah dari yang diperkirakan; dijalankan langsung `nice -19` paralel dengan RQ2 tanpa
masalah.

### 7. Stratifikasi error per famili protein / kelas target -- SELESAI 2026-10-12
**Apa:** kinase vs protease vs nuclear receptor, dsb. Nama entry UniProt sudah ada di anchor table.
**Mengapa:** mengubah satu angka gabungan jadi angka yang bisa dipakai; menunjukkan DI MANA gagalnya.

**HASIL** (`guidance/cheminformatics/protein_family_stratification.py`, dibangun di atas mapping famili
yang sama dipakai item 3, B=2000):

| famili (confirmatory, >=10 klaster target) | n | target | struct_R2 | lig_R2 | dR2 [95% CI] | BH q |
|---|---|---|---|---|---|---|
| PF00069 (kinase) | 3306 | 11 | 0,3632 | 0,2885 | +0,0747 [-0,264, +0,365] | 0,652 (tidak signifikan) |
| other_family (83 famili digabung) | 6751 | 83 | 0,4550 | 0,4091 | +0,0459 [-0,068, +0,154] | 0,652 (tidak signifikan) |

**Hanya 2 dari 10 strata famili punya >=10 klaster target** (ambang sama seperti item 3) -- 7 famili
lain (3-7 target) ditandai UNDERPOWERED, bukan diberi verdict. Bahkan di famili TUNGGAL terbesar
(kinase, yang paling dipedulikan secara praktis), structure TIDAK mengalahkan ligand-only secara
signifikan -- pola yang sama dengan hasil gabungan (§A.2) bertahan sampai ke level famili yang bisa
diuji.

**Bug nyata ditemukan dan diperbaiki sebelum dilaporkan**: tabel deskriptif (semua ~70 famili, tanpa
CI, "pola untuk dibaca bukan bukti untuk disitir") awalnya mencetak dR2 segila **+5,8e16** untuk famili
berukuran 1-2 target -- bukan temuan, itu `r2_rows` membagi dengan varian pK lokal yang mendekati nol
(terukur: PF08917, 6 baris, varian 1,13e-16) tapi tidak PERSIS nol sehingga lolos guard `denom<=0` yang
sudah ada. Diperbaiki dengan ambang relatif (std pK lokal < 1% std pK global) -- 14 famili kini tertanda
`n/a (no real spread)` eksplisit alih-alih angka yang menyesatkan. **Ditemukan karena saya benar-benar
membaca keluaran angka demi angka, bukan hanya memeriksa skrip tidak crash** -- kelas kesalahan yang
sama dengan "aggregator melaporkan sd `undefined` di n=1, bukan `0.0000`".

2 tes baru (`guidance/cheminformatics/tests/test_chem.py`) memaku ambang degenerate ini dan memverifikasi
tidak ada |dR2| meledak di hasil nyata. **Biaya:** CPU saja, ~3,7 detik untuk B=2000 -- dijalankan
`nice -19` paralel tanpa bersaing sumber daya dengan RQ2 yang sedang berjalan.

### 8. Arm A + Arm C (produksi) -- SELESAI 2026-10-15 19:40 (DIANTRIKAN 2026-10-13, ~32 jam gabungan, dikonfirmasi user)
**Apa:** Arm A (20.000 CD, 3 seed). Arm C produksi, WAJIB pakai `bindingnet_v1_clean.csv`.
**Catatan:** pertanyaannya sudah diperbaiki — bukan "apakah R2 naik" tapi "apakah data BN menutup jarak
ke model ligand-only, dan apakah ia melampaui prior kelas 0.2468". Baseline ligand-only jadi komparator
WAJIB tiap arm.
**Prasyarat:** aturan pra-registrasi Arm B (lihat bagian kebocoran senyawa) sudah dievaluasi dulu.

**Prasyarat DIVERIFIKASI 2026-10-13 (baca kode langsung, bukan diasumsikan)**: `train_surrogate_arm.py`
sudah menegakkan filter senyawa DI DALAM KODE (`compound_filter=True` default, membaca
`bindingnet_v1_clean_summary.json`, raise eksplisit kalau file itu hilang) -- bukan hanya mengandalkan
CSV upstream. Berkas ringkasan filter (sekuens dan senyawa) keduanya ada di disk. Arm C AMAN dijalankan.

**Konfigurasi, mengikuti rezim bs=1/no-AMP yang sudah divalidasi** (bs=4+AMP DITOLAK gerbang Arm 0 --
lihat historis di bawah): Arm A `--cd_rows 20000` (murni CrossDocked, pembanding ukuran-sama terhadap
Arm B/C). Arm C `--cd_rows 6000 --bn_rows 14000 --bn_match_pk` (komposisi IDENTIK dengan Arm B yang
sudah tervalidasi) MINUS `--no_compound_filter` -- flag itu cuma ada supaya seed 2021 Arm B (dilatih
SEBELUM filter ada) tetap sebanding lintas-seed-nya sendiri; Arm C tidak punya kendala itu dan WAJIB
pakai pool bersih. 3 seed (2021/2022/2023) tiap arm, diantrikan di `scripts/queue_steps.sh`.

**Estimasi ~32 jam gabungan (16 jam/arm x 2) -- dikonfirmasi eksplisit dengan user sebelum diantrikan**
(AskUserQuestion, opsi "jalankan keduanya" dipilih), karena jauh lebih besar dari item lain sesi ini.
Menyusul item 4 (y-randomisasi EGNN, ~2,5 jam) yang jalan lebih dulu di GPU yang sama.

**Bug kecil diperbaiki sebelum diantrikan**: draf pertama memanggil `aggregate_results.py` DUA KALI
(`armA_aggregate`, `armC_aggregate`) dengan nama `--json` berbeda -- tapi skrip itu otomatis
mengelompokkan SEMUA run yang ditemukan di `logs_surrogate_arms` (arm0/armBmatch/armA/armC sekaligus),
jadi dua panggilan itu akan menghasilkan DUA FILE IDENTIK yang masing-masing diberi nama menyesatkan
seolah hanya memuat satu arm. Diganti satu panggilan `armAC_aggregate` -> `armAC_table.json`.
**Biaya:** GPU, ~16 jam per arm (32 jam gabungan).

**HASIL -- SELESAI 2026-10-15 19:40, 6 run training (3 seed Arm A + 3 seed Arm C), queue sekarang IDLE
(25/25 langkah). File:** `guidance/surrogate_data/armAC_table.json`.

| arm | komposisi | n seed | val R2 per seed | mean | CI95 |
|---|---|---|---|---|---|
| **Arm A** | 20.000 baris CD murni | 3 | 0,351 / 0,344 / 0,391 (best epoch 4 ketiganya) | **0,362** | [0,299, 0,425] |
| **Arm C** | 6.000 CD + 14.000 BN, pool BERSIH (filter leakage) | 3 | 0,271 / 0,292 / 0,253 (best epoch 4/9/10) | **0,272** | [0,224, 0,320] |
| Arm B-match (pembanding lama, SAMA komposisi TANPA filter) | | 3 | 0,298 / 0,254 / 0,283 | 0,278 | [0,223, 0,334] |

**Menjawab pertanyaan pra-registrasi item ini ("apakah data BN menutup jarak ke ligand-only, dan apakah
ia melampaui prior kelas 0,2468")**: TIDAK jelas melampaui. Pembanding yang sudah ada: ligand-only ridge
desc_ridge = 0,3531 (A2.4), prior kelas (y-randomisasi dalam-target) = 0,2468.
- **Arm C (arm produksi BN yang jadi fokus pertanyaan) CI95 [0,224, 0,320] TIDAK melewati ligand-only
  (0,3531) sama sekali, dan batas bawahnya (0,224) ada DI BAWAH titik estimasi prior kelas (0,2468)** --
  artinya Arm C tidak bisa dibedakan secara meyakinkan dari prior kelas semata tanpa uji formal, apalagi
  dari ligand-only.
- **Arm A (CrossDocked murni, jumlah baris SAMA dengan Arm C) justru mean-nya lebih tinggi** (0,362 vs
  0,272) dan CI-nya [0,299, 0,425] baru menyentuh ligand-only di ujung bawah -- menambah data BindingNet
  (bahkan yang sudah bersih dari kebocoran) TIDAK terlihat membantu dibanding sekadar memakai lebih
  banyak CrossDocked pada jumlah baris yang sama.
- **Arm C vs Arm B-match (efek filter leakage semata, komposisi identik)**: 0,272 vs 0,278, CI tumpang
  tindih besar -- filter kebocoran senyawa (buang 423/124.973 baris) TIDAK menggeser performa secara
  terlihat. Kabar baik metodologis: temuan Arm B sebelumnya tidak banyak digelembungkan oleh kebocoran
  itu, meski tetap benar secara prinsip untuk dibersihkan.

**Disclosure wajib sebelum dipakai di tesis**: perbandingan di atas murni DESKRIPTIF (titik estimasi +
CI marginal per arm, N=3 seed per arm) -- BUKAN uji signifikansi berpasangan. Mengikuti disiplin proyek
ini sendiri ([[project_paired_comparison_pitfall]]), klaim "Arm A > Arm C" tidak boleh dinyatakan sebagai
signifikan tanpa `paired_r2_comparison` (stats_common.py) dijalankan di atas BARIS EVALUASI yang sama
antar arm -- ini belum dikerjakan (n_epochs/val_n per seed konsisten jadi secara teknis bisa dipasangkan,
tapi belum dijalankan). Kalau item 9 (rebuild tesis) perlu klaim perbandingan formal, jalankan uji
berpasangan itu dulu sebelum menulis kesimpulan kuat di bab hasil.

### 9b. RESPONS AUDIT FORENSIK (85 item) -- dikerjakan 2026-10-15, rebuild ke-2 (228 hlm)

User memberi `Audit_Forensik_Tesis_SBDD_Checklist_Detail_2026.pdf`: 85 item (23 confirmed, 54 risk,
6 editorial, 2 verification). Yang dikerjakan, yang ditolak, dan yang diserahkan ke user:

**DUA BUG KODE NYATA ditemukan, keduanya di `build_thesis.py`, bukan di naskah:**
1. **Caption tabel tanpa tab stop** (audit F014/F015 menyebut 2; NYATANYA 7 -- semua tabel Appendix E).
   `table()` memakai pemisah `\t` tapi tidak pernah `add_tab_stop()`, padahal `figure()` melakukannya.
   Label "Table E.1" sedikit lebih lebar dari "Table 3.1" sehingga melewati hanging indent 0.6" dan
   tab-nya kolaps jadi nol -> "Table E.1Test-set...". Diperbaiki: indent 0.6->0.85 + tab stop eksplisit,
   disamakan dengan figure(). Hasil: 0 caption nempel (dari 7).
2. **Pencocokan halaman caption salah** (audit F013, hanya menyebut C.1). Akar sebenarnya LEBIH LUAS dan
   SALING TERKAIT dengan bug 1: `compute_pages()` menyusun needle `f'{label} {num}' + cap_prefix(cap)`
   TANPA pemisah, jadi hanya cocok untuk caption yang RUSAK-nempel. Untuk 66 caption yang benar, needle
   gagal lalu jatuh ke fallback `f'{label} {num}'` yang menangkap penyebutan PROSA lebih dulu -- itulah
   sebabnya Table C.1 tertulis hlm 40 (tempat kalimat "a row of Table C.1, for example") alih-alih 169.
   **Konsekuensi penting: memperbaiki bug 1 saja akan merusak SEMUA nomor halaman tabel/gambar**, karena
   fallback yang keliru itu justru yang selama ini "menyelamatkan" caption rusak. Diperbaiki bersama:
   needle jadi regex `label + \s* + caption_prefix` (pemisah opsional), plus fallback baris-awal.
   Diverifikasi: **95/95 entri List of Tables/Figures kini cocok dengan halaman caption sebenarnya.**

**PENCEGAHAN REGRESI (baru): `verify_output()` di `build_thesis.py`.** Build sekarang GAGAL KERAS
(exit 1) kalau ada caption yang labelnya menempel atau caption yang tidak ditemukan. Dikembangkan
iteratif: versi pertama memberi 25+ false positive karena memeriksa SEMUA kemunculan "Figure 8.5"
termasuk rujukan prosa berkoma; versi kedua masih 3 false positive karena prosa bisa pecah baris tepat
di rujukan ("Figure 5.1). The 24..."). Versi final mengidentifikasi caption lewat TEKS CAPTION-nya,
baru memeriksa spasi -- lolos bersih.

**KALIBRASI KLAIM (P1, inti audit) -- diverifikasi muncul di PDF final:**
- F023 (kontrol positif tidak ortogonal): Bab VI.5 dulu mengklaim "non-circular" terlalu kuat. Sekarang
  dinyatakan: dua mode Vina BERBAGI fungsi skor yang sama, jadi biasnya (preferensi ligan besar) ikut
  dibagi selektor dan evaluator, bukan saling meniadakan. Menyingkirkan sirkularitas definisional, BUKAN
  menjadikannya pengukuran independen. Abstrak + Bab X disesuaikan.
- F017/F018 (klaim menyapu): "failure traces to signal rather than mechanism" dipisah per rute --
  LANGSUNG untuk rute seleksi (mekanisme ditahan, sinyal ditukar), TIDAK LANGSUNG untuk rute gradien
  (Vina tak differentiable). Bab IX ditambahi peringatan eksplisit bahwa ringkasan tanpa kualifikasi itu
  mengklaim lebih dari yang dibawa bukti gradien.
- F020 (null != ekuivalensi): Bab X kini menyatakan tidak ada uji ekuivalensi formal (TOST) yang
  dijalankan dan tidak ada margin indiferensi yang dipra-deklarasikan; yang didukung desain adalah batas
  daya ~1,0-1,4 kcal/mol.
- F021/F072 (docking != binding, trade-off): abstrak kini "predicted docking score", dan trade-off
  (molekul lebih besar + SA/RA turun signifikan) dinaikkan ke abstrak, tidak dikubur.
- F027 ("independent" tak terdefinisi): subbab baru di Bab III mendefinisikan operasional -- soal
  PROVENANS angka, bukan validitas fisik; eksplisit bahwa independen != ortogonal != eksperimental.

**RONDE 2 (user mengirim ulang potongan audit F001-005, F057-064, F072-085) -- rebuild ke-3, 231 hlm.**
Banyak item yang semula saya tandai "butuh eksperimen baru" ternyata meminta PERNYATAAN SCOPE, bukan
compute. Dikerjakan:
- **F057** (norma != arah): Bab V.4 kini menyatakan eksplisit bahwa yang dikorelasikan adalah NORMA,
  yang tidak membawa arah; frasa "strongest pull" dikoreksi jadi "largest magnitude"; cosine similarity
  / finite-difference disebut sebagai pengukuran bertanda yang belum dihitung.
- **F063** (proxy != outcome): dipasang di tempat yang sama, dan diperkuat karena proyek ini PUNYA
  buktinya sendiri -- checkpoint gradient-alignment memperbaiki diagnostiknya tapi docking tetap null.
  Dinyatakan: diagnostik tidak boleh diperlakukan sebagai surrogate endpoint.
- **F059/F060/F061** (Appendix E.5): paragraf limit baru -- n=1 adalah bukti KEBERADAAN bukan laju
  tipikal; gap kecil kompatibel dengan beberapa dinamika (gradien kecil / arah tak konsisten / dilawan
  denoiser), jadi noise-overwhelming adalah hipotesis yang tidak ditolak, BUKAN mekanisme yang
  diisolasi kausal (butuh intervensi pada noise schedule, tidak dijalankan); kanal atom-type tak
  dipandu langsung tapi bisa berubah lewat kopling posisi, dan pergerakan logit-nya tidak diukur.
- **F064**: baris kitchen-sink ditandai agregat eksploratif; 2^4=16 arm di luar budget; tidak ada klaim
  sinergi.
- **F075/F076**: Bab X kini menyatakan tesis murni komputasional (tidak ada molekul disintesis/diuji)
  dan bahwa kriteria yang dipakai TIDAK menyentuh ADMET/toksisitas/off-target -- lolos semua cek di
  tesis ini berarti kandidat untuk triase komputasi lanjutan, bukan "drug-like".
- **F080**: tabel anggaran komputasi NYATA (Tabel A.1) dari log queue. **Mengungkap hal yang memperkuat
  temuan**: Arm C dilatih LEBIH LAMA (38 epoch/24,6 jam) daripada Arm A (27 epoch/17,0 jam) pada
  throughput per-epoch yang sama, tapi hasilnya lebih buruk -> "Arm C kurang training" TERSINGKIR
  sebagai penjelasan. Dicatat juga di Bab IV.7.
- **F082**: ditemukan klaim BASI -- appendix mencatat commit `455bbc08...` sebagai "kode yang
  menghasilkan dokumen ini", padahal HEAD `7e54322` dengan 12 file belum di-commit. Diperbaiki di
  AKARNYA: token `{gitstate}` baru di `build_thesis.py` menuliskan commit + jumlah perubahan
  belum-ter-commit SAAT BUILD, jadi klaim itu tidak bisa basi lagi. PDF sekarang jujur menyatakan
  dirinya dibangun dari working tree yang belum beku, dan meminta tag immutable sebelum submit.
- **F077**: checklist ternyata SUDAH jujur (clean-checkout ditandai "Not verified"). Ditambah baris
  baru yang benar-benar terverifikasi: suite tes otomatis 16 suite / 0 gagal.
- **F085**: diaudit ulang setelah dokumen tumbuh ke 231 hlm -- 95/95 entri daftar tetap cocok.

**RONDE 3 (user: "gas aja, gw izinin semuanya") -- rebuild ke-4, 232 hlm.**
- **F073 SELESAI (verifikasi DOI lewat Crossref/arXiv).** Kelima sitasi `v='kn'` diverifikasi field demi
  field. **Menemukan 2 kesalahan nyata**: (1) BoKDiff -- judul yang saya REKONSTRUKSI salah (sebenarnya
  "BoKDiff: Best-of-K diffusion alignment for target-specific 3D molecule generation") DAN inisial
  penulis pertama salah (A., bukan N.; Garibay sebenarnya Ozmen Garibay); (2) Kwapien -- venue yang
  masih ragu ("ACS Omega / JCIM") ternyata ACS Omega 7(30), 26573-26581. Tiga lainnya terkonfirmasi
  persis seperti ditulis. DOI ditambahkan ke kelimanya. Klaim novelty BoKDiff ("first to apply
  Best-of-K alignment and Best-of-N sampling to SBDD") terkonfirmasi VERBATIM di abstraknya -- ini
  penting karena penyempitan klaim novelty RQ2 di Bab VI.5 bersandar padanya.
  Tier verifikasi baru `v='xr'` ditambahkan ke refs.py. **Sisa entri `v='kn'`: 0.**
  Appendix B kini MENGUNGKAP koreksi ini, bukan menyerapnya diam-diam -- 2 dari 5 sitasi hasil
  rekonstruksi ternyata salah, dan hanya lookup registry yang bisa mengungkapnya.
- **F084 (caption) -- diukur dulu, baru diedit.** Pengecek otomatis pertama menandai 24/33 caption
  "tanpa n"; setelah diperbaiki (angka yang DIEJA seperti "four targets", dan "835 leakage-safe-split
  targets" yang tersandung tanda hubung) tinggal 11, dan 10 di antaranya diagram skematik yang n-nya
  memang tak bermakna. **Premis F084 sebagian besar TIDAK berlaku untuk dokumen ini -- pengecek saya
  yang lemah, bukan caption-nya.** Hanya 2 caption yang benar-benar kosong dan angkanya bisa
  diverifikasi dari tabel ringkasan yang diperbaiki (trackA: 8 pocket/29-30 molekul/24 perbandingan;
  fuquality: 8 checkpoint). Sisanya TIDAK diisi, karena menebak n = menyisipkan angka tak terverifikasi,
  persis yang didisiplinkan tesis ini.
- **F074 SELESAI (verifikasi klaim novelty).** Diuji terarah terhadap klaim Bab IV yang paling
  load-bearing: dekomposisi prior kelas via permutasi label dalam-target. Dua pencarian (sekali lewat
  kontrol permutasinya sendiri, sekali lewat padanan yang mungkin dipakai literatur -- "per-target
  mean" / "target-stratified baseline" di PDBbind) **tidak menemukan** studi yang menerapkan kontrol itu
  pada prediksi afinitas protein-ligan. **Klaim bertahan**, tetap dalam bentuk "we did not find".
  **Tapi pencarian menemukan karya kontemporer yang konvergen**: MIRAGE (arXiv:2609.14491, Sept 2026,
  Yazdani-Jahromi/Padhi/Garibay) -- menyimpulkan performa semu prediksi afinitas melacak *family
  support* historis, bukan kimia ligan; kontrol yang tak bisa mengeksploitasi famili test tetap datar;
  peringkat model BERBALIK di famili baru; dan mereka mengusulkan pelaporan "excess over a
  support-insensitive baseline" -- KUANTITAS YANG SAMA dengan yang dicapai dekomposisi kita, lewat
  konstruksi berbeda (permutasi label vs stratifikasi family support). Disitir di Bab II.4 dan
  dibahas di Bab IV.8 sebagai KONVERGENSI, bukan presedensi (rilis saat tesis ditulis, mengukur hal
  berbeda, tidak saling menggantikan). Pencarian ini didokumentasikan di Appendix B supaya bisa
  diulang pembaca. Refs: 65 (dari 64), nol unused.
- **Exit criteria audit "angka abstrak cocok dengan tabel" -- DIPERIKSA.** Semua angka desimal di
  abstrak ditemukan di badan naskah. Penelusuran lintas-bab untuk 10 angka kunci (RQ2, prior kelas,
  Arm A/C) tidak menemukan varian yang bertentangan. **Satu jebakan keterbacaan ditemukan**: 0,362
  muncul sebagai DUA kuantitas berbeda (Arm A di Bab IV, dan baseline linier dua-fitur Track E di
  Bab VIII), dan keduanya muncul di Bab X. Diverifikasi ke `armAC_table.json` bahwa ini KEBETULAN,
  bukan salah salin -- keduanya benar. Diberi klausa pembeda di Bab X supaya tidak terbaca sebagai
  angka yang sama.
- **F083 SELESAI (audit isolasi test-set per run).** Tabel A.4 baru di Appendix A, disusun dari BUKTI
  bukan asumsi: kriteria seleksi dibaca dari KODE trainer (`best_val_loss` menggerakkan retensi
  checkpoint dan early stopping di `train_egnn_stage0.py` dan `train_surrogate_arm.py`), dan paparan
  test set diverifikasi dari LOG (Stage 0: 8 baris `[Test]`, 18 `[Validate]`; 6 run produksi Arm A/C:
  NOL baris `[Test]`; 5 run A1e: `--skip_test_logging`). Dibedakan tegas antara *selection* (kuantitas
  yang menentukan checkpoint dipertahankan) dan *exposure* (test dievaluasi sama sekali, meski hanya
  untuk log) -- menyamakan keduanya adalah cara klaim semacam ini biasanya runtuh. Satu pengecualian
  yang memang ada DIMASUKKAN ke tabel, bukan disembunyikan: sweep hyperparameter Bab IV.3 memilih
  konfigurasi berdasar metrik TEST, sudah ditandai optimistis di tempatnya, dan tidak ada hasil
  guidance yang bergantung padanya.
- **Pemeriksaan karakter**: ditemukan satu karakter asing (U+51B3) yang tak sengaja masuk saat menulis
  catatan tabel; dibersihkan. Pemindaian non-ASCII seluruh bab: sisanya simbol ilmiah sah.
- **CATATAN OPERASIONAL (kesalahan saya, diulang dari awal sesi)**: rebuild tesis memakai CPU, dan CPU
  adalah bottleneck termal mesin ini. Rebuild berulang sementara DIAG2 berjalan MEMPERLAMBAT DIAG2
  lewat gerbang termal. Build dihentikan sampai DIAG2 selesai; perubahan teks ditumpuk dulu, baru satu
  rebuild di akhir.
- **UPGRADE: audit manual diubah jadi PENJAGA PERMANEN + penjaganya sendiri DIUJI.**
  `verify_output()` di `build_thesis.py` diperluas dari 2 jadi 4 pemeriksaan, semuanya menjaga bug yang
  BENAR-BENAR pernah lolos ke PDF: (1) caption nempel; (2) halaman caption salah resolve; (3) **BARU:**
  setiap entri List of Tables/Figures harus menunjuk halaman caption sebenarnya -- ini audit yang
  sebelumnya saya jalankan MANUAL dua kali (95/95), sekarang jalan tiap build; (4) **BARU:** karakter
  tak terduga (daftar izin simbol ilmiah) -- menjaga kecelakaan U+51B3 yang sempat masuk.
  Build GAGAL KERAS (exit 1) kalau salah satu menyala.
  **Daftar izin karakter dikalibrasi ke dokumen nyata, bukan ditebak**: uji pertama menandai ⁻ (U+207B),
  ∈ (U+2208), µ (U+00B5) -- ketiganya SAH, jadi daftar izinnya yang kurang. Setelah dilengkapi: 0 masalah.
  **Penjaga yang tak pernah berbunyi belum terbukti bekerja**, jadi dibuat
  `guidance/tests/test_thesis_build_verifier.py` (7 tes): tiap penjaga disuntik kerusakan yang dulu
  benar-benar terjadi dan HARUS menyala, plus pasangan bersih yang HARUS diam. Termasuk tes regresi
  untuk false-positive yang sempat membuat versi awal penjaga ini tak terpakai (prosa pecah baris tepat
  di rujukan: "Figure 5.1). The 24..."). Diuji terhadap PDF yang SUDAH ada, tanpa rebuild -- nol beban
  CPU tambahan ke DIAG2.
  Ditemukan juga: runner melaporkan "0 test(s)" untuk suite baru karena menghitung baris `PASS` di
  stdout sementara unittest menulis ke stderr. Suite yang melapor nol bisa menyamarkan masalah, jadi
  diikutkan konvensi proyek (cetak `PASS <nama>` + total). Kini melapor 7 dengan benar.
- **F048 (anggaran tuning baseline) -- DIVERIFIKASI DARI KODE, dan hasilnya MENGUATKAN tesis.**
  `representation_ladder.py` memberi anggaran IDENTIK ke setiap blok (5 alpha ridge + 1 konfigurasi
  gradient boosting, dipilih di VAL saja, test diskor sekali), dengan alasan eksplisit di komentarnya:
  kedua keluarga model dicoba untuk tiap blok supaya sebuah blok tidak dihukum karena kebetulan cocok
  ke satu regressor. Jadi simetris DI DALAM ladder. Terhadap EGNN memang ASIMETRIS -- tapi asimetrinya
  BERPIHAK PADA EGNN (EGNN dapat berjam-jam GPU + early stopping; ridge ligand-only cuma 6 fit murah).
  Ditulis eksplisit di Bab IV.6: keberatan "baseline kurang di-tuning" berjalan ke arah yang salah di
  sini -- baseline yang DI-TUNING LEBIH BAIK hanya akan MELEBARKAN temuan, bukan mempersempitnya.
- **F070 (holdout satu famili)**: Appendix E kini membatasi klaim ke "uji generalisasi SATU-FAMILI
  terkeras yang tersedia di split ini", bukan robustness OOD lintas proteome; ditambah catatan bahwa
  kinase justru famili paling padat dan homogen (kasus paling menguntungkan bagi model yang belajar
  regularitas famili), dan bahwa mengulang dengan beberapa famili adalah penguatan yang jelas tapi di
  luar budget.
- **F069 (hitungan PLIP bukan bukti binding)**: ditambahkan batas eksplisit -- hitungan bisa naik karena
  molekul membesar, atom masuk cut-off jarak, atau pose memenuhi kriteria sudut TANPA perbaikan binding.
  Kenaikan bukan bukti afinitas membaik, penurunan bukan bukti sebaliknya. Dinyatakan bahwa hitungan
  dipakai sebagai cek OUTCOME ORTOGONAL, bukan ukuran binding, dan tidak ada klaim tesis yang
  bersandar pada hitungan interaksi saja.
- **F071 (validitas fisik != dapat diterima secara kimia)**: subbab baru di Bab III -- PoseBusters
  memeriksa POSE, bukan kewajaran senyawa; tidak memeriksa strain di luar tes geometrik, substruktur
  promiscuous/PAINS, atau apa pun yang farmakologis. "Valid" di tesis ini SELALU berarti "lolos
  pemeriksaan yang disebutkan", tidak pernah "layak disintesis atau diuji".
- **F050 (kritik yang TEPAT terhadap tulisan saya sendiri).** Saya sempat menulis bahwa EGNN mencapai
  prior kelasnya "lewat rute lebih langsung: arsitektur membaca geometri target 3-D" -- itu klaim
  MEKANISME, padahal permutasi hanya mengukur BERAPA BANYAK performa yang bertahan, bukan JALUR-nya.
  Diperbaiki: angka 70%/90% dinyatakan sebagai "control-based share of apparent predictive performance";
  penjelasan selisihnya ditandai HIPOTESIS, lengkap dengan ablasi yang akan membuktikannya (melatih
  ulang EGNN dengan kanal protein dibuang/diacak sambil label tetap dipermutasi) yang TIDAK dijalankan.
  Dicek tidak ada instance lain dari klaim itu di bab lain.
- **F058 (rentang λ)**: ditambahkan subbab -- λ adalah pengali tak berdimensi atas gradien yang skalanya
  properti prediktor, jadi "tiga orde besaran" tidak membuktikan regime efektif tercakup. Diikat ke
  pengukuran nyata: magnitudo gradien ~1e-3, dan jejak Appendix E.5 (pada λ=1 jumlah magnitudo nudge
  ~9,5 Å sepanjang 1.000 langkah, tapi perpindahan NETO berakhir <0,1 Å). Kuantitas operatifnya adalah
  perpindahan per-langkah relatif terhadap derau posterior, bukan λ; batas atas ditentukan oleh
  fragmentasi molekul, bukan penilaian kecukupan.
- **F046 (pergeseran domain BindingNet) -- menawarkan PENJELASAN TANDINGAN yang tidak bisa disingkirkan
  desain ini.** Menyamakan jumlah baris TIDAK menyamakan domain: baris BindingNet adalah senyawa yang
  disejajarkan ke template kristal, sedangkan val/test DAN seluruh data Arm A adalah pose CrossDocked
  docked-and-minimised. Jadi "augmentasi BindingNet tidak membantu" dan "melatih sebagian di luar domain
  lebih merugikan daripada manfaat data tambahan" SAMA-SAMA cocok dengan hasilnya, dan itu dua klaim
  berbeda. Dinyatakan sebagai batas atas apa yang dicapai augmentasi DI BAWAH PROTOKOL INI, bukan sifat
  sumber datanya.
- **F067 (RA-score hanya pada molekul yang bisa diskor) -- EFEK SELEKSI STRUKTURAL, bukan sekadar n
  kecil.** Tesis sudah menandai n=6 di λ=0,3, tapi belum menyatakan masalah yang lebih dalam: RA-score
  hanya dihitung pada molekul yang terekonstruksi jadi satu fragmen, jadi setiap mean adalah mean ATAS
  YANG SELAMAT. Itu tidak masalah kalau tingkat kelulusan konstan antar kondisi -- TAPI TIDAK:
  fragmentasi NAIK bersama kekuatan guidance (0 di λ=0 -> 0,21 di λ=0,3). Jadi subset yang dibandingkan
  diseleksi secara berbeda OLEH VARIABEL YANG SEDANG DIUJI, dan arah seleksinya bisa ditebak (yang
  selamat dari guidance kuat adalah yang tahan ditarik pecah). Pergeseran mean RA-score bisa berasal
  dari BERUBAHNYA MOLEKUL MANA YANG DISKOR, bukan dari molekulnya membaik. Dinyatakan, beserta analisis
  yang akan membatasinya (beri nilai kasus-terburuk ke yang tak terskor, atau bandingkan hanya molekul
  yang terskor di SEMUA kondisi) -- tidak dijalankan.
- **F062 (ESM2 global vs pocket-local)**: sudah dihedge sebelumnya; ditambahkan bahwa perbandingan ini
  mengubah DUA hal sekaligus (residu mana yang diwakili, DAN bagaimana embedding direduksi jadi vektor
  tetap), jadi hasilnya membatasi integrasi pocket-restricted YANG INI, bukan nilai fitur protein-LM
  lokal-pocket secara umum.
- **F066, F081 -- SUDAH TERTANGANI, tidak perlu diubah.** E2 ditandai "not run"/"untested" di 4 tempat.
  Checklist reproduktibilitas sudah menyatakan tidak semua ablasi eksploratori mencatat seed, dan
  protokol dua-tier mewajibkan konfirmasi sebelum sesuatu dilaporkan sebagai temuan.
- **PRA-REGISTRASI INTERPRETASI DIAG2 ditulis SAAT run baru 9/20** --
  `guidance/DIAG2_INTERPRETATION_PREREGISTRATION.md`. Mengunci arti tiap kemungkinan hasil SEBELUM
  angka finalnya ada: CI mengecualikan nol + mean NEGATIF -> menguatkan narasi Bab V; CI mengecualikan
  nol + mean POSITIF -> **MELEMAHKAN**, frasa "wrong direction" di Bab V/IX/X WAJIB direvisi; CI memuat
  nol -> konsisten F1 (tak informatif), BUKAN "mendorong menjauh"; kontrol gagal -> hasil utama tidak
  dilaporkan sama sekali. Ditulis eksplisit bahwa smoke test 1 pocket sudah memberi mean cos +0,087
  (58% menuju kontak), jadi baris "positif" adalah kemungkinan NYATA -- dicatat sekarang supaya tidak
  ada godaan mengecilkannya nanti. Ambang final, tidak digeser setelah melihat angka.
- **F078/F079 (provenance + instruksi akses dataset) SELESAI.** Checksum saja membuktikan dua salinan
  identik, bukan dari mana asalnya. Tabel provenance baru (A.x) memasangkan tiap artefak turunan dengan
  SUMBER HULU-nya dan SKRIP PERSIS di repo yang menghasilkannya (`build_split.py`,
  `build_family_holdout_split.py`, `apply_sequence_leakage_filter.py`,
  `apply_compound_leakage_filter.py`, `build_bindingnet_index.py`) -- semua diverifikasi ADA sebelum
  disebut. Catatan: tabel ini DITEGAKKAN TES, bukan cuma didokumentasikan (leakage guard menurunkan
  ulang hitungan penghapusan dan gagal kalau tak cocok).
  **Dua batas dinyatakan, tidak disembunyikan**: (1) LISENSI -- tesis TIDAK menyatakan ulang ketentuan
  lisensi CrossDocked2020/BindingNet/PDB karena TIDAK diverifikasi independen di sesi ini; reproducer
  wajib memeriksa sendiri sebelum meredistribusi. Skrip dan definisi split milik penulis; data
  dasarnya tidak. (2) SKALA -- LMDB-nya saja 3,8 GB, jadi reproduksi ujung-ke-ujung itu usaha besar;
  pembaca yang ingin mengecek SATU angka diarahkan ke skrip per-hasil yang jalan dalam hitungan
  detik-menit.
- **Kesalahan saya sendiri, ditangkap sendiri**: penyisipan seksi provenance membuat `@endtab` YATIM
  (16 `@tab` vs 17 `@endtab`). Terdeteksi lewat pengecekan keseimbangan markup, bukan lewat build yang
  gagal. Dihapus; kini seimbang di SEMUA bab, parse bersih (578 blok, 48 tabel, 33 gambar).
- **UPGRADE LANJUTAN: penjaga keseimbangan markup di titik PARSE (bukan setelah build).**
  Insiden `@endtab` yatim mengungkap sesuatu yang lebih buruk dari sekadar salah ketik: parser
  MENERIMANYA tanpa protes (parse "berhasil" dengan 48 tabel). Artinya kesalahan markup bisa menelan
  atau memindahkan konten DIAM-DIAM, dan build tetap sukses -- mode kegagalan terburuk yang mungkin.
  `check_markup_balance()` baru dipanggil DI AWAL `parse_sources()` dan membatalkan build (SystemExit)
  untuk: @endtab tanpa pasangan, @tab tak pernah ditutup, dan @tab bersarang. Diuji dua arah: naskah
  sekarang LOLOS, dan penjaga MENYALA persis pada kerusakan yang saya buat sendiri tadi.
  Suite tes naik 7 -> 11 (kelas `MarkupBalanceGuard`, 4 tes).
  **Penjaga build kini 5 kelas**: caption nempel, halaman caption, entri daftar vs halaman caption
  sebenarnya, karakter tak terduga, keseimbangan markup.
- **VERIFIKASI ANGKA TERHADAP SUMBER -- MENEMUKAN 2 KESALAHAN DI TULISAN SAYA SENDIRI.**
  Setelah menemukan drift di tabel anggaran komputasi, semua tabel hasil utama dicek terhadap
  JSON/CSV sumbernya.
  1. **Tabel anggaran komputasi (Tabel A.1)**: 4 baris melenceng 0,01 jam karena saya menghitung
     manual dari timestamp (0,98 vs 0,97; 2,83 vs 2,82; 4,15 vs 4,14). Tidak material untuk kesimpulan
     apa pun, TAPI angka tesis harus cocok persis dengan skrip yang menurunkannya. Dibuat
     `guidance/surrogate_data/compute_budget_table.py` + JSON; keempat baris disamakan; catatan tabel
     diubah jadi "Regenerated by ... not estimated or retyped". Diverifikasi: 0 baris tidak cocok.
  2. **Tabel RQ2 (Bab VI.5) SALAH MELABELI UJI.** Empat baris (SA, RA-score, QED, ligand efficiency)
     ditulis memakai "KS", padahal `task_f_analysis.json` TIDAK PERNAH menghitung uji KS sama sekali --
     hanya Mann-Whitney dan Wilcoxon berpasangan. Nilai p yang dikutip BENAR (itu nilai Wilcoxon
     berpasangan), label ujinya yang salah. Ini jenis kesalahan yang akan memalukan di sidang
     ("Anda bilang KS, tapi skrip Anda tidak pernah menghitungnya"). Diperbaiki jadi
     "Wilcoxon (paired by pocket)"; catatan tabel kini menjelaskan kedua uji dihitung dan mengapa yang
     berpasangan yang dikutip (pocket adalah unit replikasi).
  **Yang TIDAK salah, diverifikasi ulang dari data mentah** (`honest_eval.csv`, 597 + 600 molekul):
  heavy atoms 21,323±9,057 vs 23,095±8,364 MW p=0,000206; vina_dock -6,813±10,079 vs -8,068±2,660;
  ligand efficiency -0,340 vs -0,369. SEMUA cocok persis dengan yang tertulis. Tabel Arm A/C juga
  cocok sempurna dengan `armAC_table.json` (mean, CI, dan ketiga nilai seed tiap arm).
- **SAPUAN VERIFIKASI ANGKA DILANJUTKAN -- 3 KESALAHAN LAGI DI TABEL REPRESENTATION LADDER.**
  3. **Label model ECIF-raw SALAH**: ditulis "gradient-boosted tree", padahal JSON mencatat
     `ridge(a=1000.0)` menang di validasi untuk blok itu. (ecif_norm, plip_* memang hgb -- itu benar.)
  4. **Dimensi ECIF ditulis "varies"**, padahal angkanya pasti: 72.
  5. **Baris ligand MENCAMPUR SUMBER**: tabel ladder seharusnya melaporkan angka ladder, tapi barisnya
     memuat 0,3531 yang berasal dari skrip LAIN (`ligand_only_results.json`), sedangkan ladder sendiri
     menghitung 0,3534 untuk blok yang sama. BUKAN kesalahan angka -- keduanya benar dan selisihnya
     cuma detail fitting (ladder me-refit di dalam grid alpha-nya sendiri) -- tapi tabel jadi mengklaim
     sesuatu yang bukan isinya. Diperbaiki: tabel kini melaporkan 0,3534 (angka ladder), dan catatan
     tabel menjelaskan mengapa 0,3531 yang dipakai di seluruh tesis sebagai "ligand-only ceiling"
     (skrip itu juga sumber baseline heavy-atom dan ECFP yang dikutip berdampingan, dan ia yang jadi
     `power.floor_r2` di analisis daya).
  Diverifikasi ulang: ketujuh baris ladder kini cocok dengan JSON pada R2, dimensi, DAN keluarga model.
  Angka lain yang DICEK dan BENAR: activity cliffs (2.856 pasangan cliff, 344 senyawa, 4.050.366
  pasangan diperiksa, n_test 11.855) dan heavy_atoms 0,3069 -- semuanya cocok sumbernya.

  **POLA YANG MUNCUL dari seluruh sapuan ini: SEMUA kesalahan ada di tabel yang saya susun MANUAL
  (anggaran komputasi, RQ2, ladder); tabel yang disalin dari keluaran terstruktur (Arm A/C vs
  armAC_table.json) bersih sempurna.** Ini persis argumen yang dibuat tesis ini sendiri soal angka
  yang diketik tangan, didemonstrasikan pada penulisnya.
- **UPGRADE SISTEMATIS: `guidance/thesis/check_numbers_against_sources.py`.** Pola "semua kesalahan
  ada di tabel manual" menuntut respons sistematis, bukan pemeriksaan sekali jalan. Berkas ini
  memasangkan 22 angka yang dikutip tesis dengan FILE SUMBER + path persisnya (ladder R2 per blok,
  ligand-only ceiling, heavy-atom baseline, activity cliffs, Arm A/C, subgroup M2, anggaran komputasi).
  Dipanggil DI AWAL `build_thesis.py main()` -> angka yang melenceng gagal dalam hitungan detik, bukan
  setelah build dua-pass 4 menit. Kalau file sumber tak ada (checkout tanpa intermediate besar),
  klaim itu DILEWATI dengan pemberitahuan, tidak pernah diam-diam.
  **Versi pertama pemeriksa ini TERLALU LEMAH dan saya menangkapnya dengan mengujinya.** Ia cuma
  menanyakan "apakah angka ini ada di suatu tempat di bab itu" -- jadi saat saya sengaja merusak satu
  sel tabel (0,2889 -> 0,9999), ia TETAP LOLOS, karena angka yang sama masih muncul di prosa
  sekitarnya. Diperkuat: klaim tabel kini wajib cocok DI BARIS YANG SAMA dengan label barisnya.
  Diuji ulang dua arah: lolos pada naskah sekarang, MENYALA pada sel tunggal yang dirusak.
- **KESALAHAN SAYA LAGI, ditangkap suite penuh**: suite tes baru saya LOLOS saat dijalankan sendiri
  (`--pattern`) tapi GAGAL saat suite penuh dijalankan. Sebabnya nyata, bukan flaky: runner memakai
  `sys.executable` = env conda `drugdisc`, yang TIDAK punya python-docx, sedangkan `build_thesis`
  mengimpornya. Jadi suite saya error di sana. Diperbaiki dengan konvensi SKIP_EXIT=2 proyek ini
  (sama seperti `test_lp_split_loader.py` yang skip kalau split JSON tak ada).
  **Konsekuensi yang HARUS dinyatakan, bukan disembunyikan**: suite penjaga tesis ini sekarang
  DILEWATI di env utama, dan runner sendiri memperingatkan "a skipped suite is not a passing suite".
  Ia hanya benar-benar berjalan (a) saat tesis dibangun, dan (b) saat dijalankan dengan interpreter
  yang punya python-docx (`python3`). Status penuh sekarang: **16 passed, 0 failed, 1 skipped**
  (dengan `python3`: 13 tes penjaga semuanya lolos).
  Pelajaran: "lolos saat diuji sendiri" bukan "lolos di suite" -- persis kelas kesalahan yang
  integrasi ada untuk menangkapnya.
- **TABEL RQ2 (hasil terpenting tesis) KINI DITURUNKAN DARI DATA MENTAH, bukan dirakit tangan.**
  `guidance/rq2_table_from_raw.py` menghitung ulang seluruh tabel dari 597+600 CSV per-molekul
  (mean, sd, Mann-Whitney DAN Wilcoxon berpasangan per pocket) -> `rq2_table.json`. Seluruh nilai
  tesis TERVERIFIKASI cocok: sa 0,601/0,572; ra_score 0,435/0,353; qed 0,449/0,462; pb_valid
  0,551/0,582; vina_dock -6,813/-8,068; heavy_atoms 21,32/23,09. Hanya label uji yang tadi salah.
  Dua ketidakkonsistenan internal sekalian dirapikan: (a) baris heavy atoms memakai Mann-Whitney
  sementara lima baris lain Wilcoxon berpasangan, padahal catatan tabel saya SENDIRI berargumen
  pocket adalah unit replikasi -- disamakan ke Wilcoxon (p tetap 0,0002); (b) baris PoseBusters cuma
  menulis "held, did not fall" tanpa angka -- kini memuat p = 0,209.
  8 klaim RQ2 ditambahkan ke manifest penjaga: **30 angka dijaga otomatis** (dari 22).
- **Penjaga angka diperkuat LAGI setelah gagal palsu**: ia melaporkan 2 ketidakcocokan pada nilai
  negatif, padahal naskah benar -- naskah memakai minus tipografis (U+2212), formatter menghasilkan
  hyphen ASCII. Dinormalkan. Diuji ulang: lolos bersih, dan TETAP menyala saat satu sel dirusak.
- **F057 SELESAI 2026-10-16 02:04 -- HASIL MENGOREKSI NASKAH, SESUAI PRA-REGISTRASI.**
  DIAG2 (`guidance/diag_gradient_direction.py`), 20 pocket x 16 sampel = **23.202 atom-state**,
  bootstrap klaster-pocket B=2000. Hasil -> `guidance/diag_gradient_direction_results.json`.
  **Kontrol lolos lebih dulu** (wajib, sesuai pra-registrasi): arah acak +0,0026 CI[-0,0038,+0,0095]
  (memuat nol, seperti seharusnya); beda-hingga mean +0,0012 dengan **80,9% state naik** -> konvensi
  tanda terkonfirmasi empiris. Bahwa 80,9% bukan 100% DILAPORKAN apa adanya, tidak dihaluskan.
  **HASIL UTAMA: arah gradien TIDAK DAPAT DIBEDAKAN DARI ACAK.** mean cos = **-0,019**,
  CI95 **[-0,046, +0,008] MEMUAT NOL**; 48,5% atom didorong menuju kontak terdekat (= kebetulan).
  **Verdict = baris 3 tabel pra-registrasi**: konsisten dengan F1 (sinyal tak informatif),
  dan TIDAK melisensikan klaim lebih kuat bahwa guidance mendorong atom MENJAUHI kontak.
  **KONSEKUENSI: frasa "wrong direction" DITARIK dari naskah.** Diperbaiki di 5 tempat: Bab V.4
  (seksi hasil baru + penarikan eksplisit), tabel mastersum Bab III, tabel objmap Bab I, Bab IX
  (butir bukti iii), Bab X (kesimpulan 4), dan abstrak. Alasannya dinyatakan: NORMA tidak membawa
  arah; saat arah diukur langsung, ia ABSEN, bukan TERBALIK.
  **Pilot 1-pocket memberi +0,087 dan TIDAK bertahan di 20 pocket** -- persis alasan pilot tidak
  pernah dilaporkan sebagai hasil, dan alasan kemungkinan "positif" ditulis eksplisit di pra-registrasi
  sebelum angkanya ada.
  Sekunder (dideklarasikan deskriptif, bukan penentu): paruh dekat +0,034 vs paruh jauh -0,072,
  median jarak kontak 3,57 A -- ada pola bergantung jarak, tapi agregatnya tetap tak terbedakan dari acak.
- **BUILD FINAL: 245 halaman** (dari 148 di awal sesi), 65 referensi, 0 unused, semua penjaga lolos,
  47 angka terverifikasi otomatis terhadap file sumber.

- **(catatan lama) F057 dijalankan**: `guidance/diag_gradient_direction.py` (DIAG2) -- proyeksi BERTANDA gradien
  terhadap vektor menuju atom pocket terdekat, 20 pocket x 16 sampel, bootstrap klaster-pocket B=2000,
  di bawah gerbang termal. Dua kontrol built-in: (a) arah acak harus ~0; (b) uji beda-hingga melangkah
  sepanjang +grad harus MENAIKKAN skor model (validasi konvensi tanda secara empiris, bukan dari baca
  kode). Smoke test: kontrol acak -0,011 (OK), beda-hingga 100% naik (OK) -> konvensi tanda
  TERKONFIRMASI (`guided_sampling.py` menambahkan grad ke posisi, jadi cos>0 = didorong MENUJU kontak).
  Hasil awal 1 pocket: mean cos +0,087, 58% mengarah ke kontak -- **arah POSITIF, bukan negatif**.
  Kalau bertahan di 20 pocket, ini menghaluskan narasi Bab V: magnitudo terbesar memang di atom jauh,
  tapi arahnya rata-rata tetap sedikit menuju pocket. HARUS dilaporkan apa adanya, termasuk kalau
  melemahkan cerita yang sudah ditulis.

**REKAPITULASI TERHADAP 85 ITEM AUDIT (per 2026-10-15, sebelum build final):**

*Diubah di naskah/kode sesi ini (34 item):* F013, F014, F015, F017, F018, F019, F020, F021, F023,
F027, F046, F048, F050, F057(teks; eksperimennya berjalan), F058, F059, F060, F061, F062, F063, F064,
F065, F067, F069, F070, F071, F072, F073, F074, F080, F082, F083, F084, F085.

*Diperiksa, SUDAH tertangani naskah lama, sengaja tidak diubah (6):* F016 (abstrak ditulis ulang di
ronde 1), F052 (CD38 sudah didisklos sebagai diskrepansi nyata), F066 (E2 ditandai "not run" di 4
tempat), F068 (pembalikan tanda PLIP sudah dilaporkan sebagai peringatan selective reporting),
F077 (checklist sudah jujur menandai clean-checkout "Not verified"; ditambah baris suite tes),
F081 (seed eksploratori sudah didisklos; protokol dua-tier mewajibkan konfirmasi).

*Milik user, TIDAK BOLEH saya isi (12):* F001-F012 -- nama, NIM, gelar, tanggal sidang, copyright,
pembimbing, pernyataan orisinalitas, formulir publikasi, formulir pembimbing, laporan similarity,
dedication, acknowledgments. Izin "gas, semuanya" TIDAK memberi saya datanya, dan ini halaman
pernyataan bertanda tangan.

*Butuh komitmen komputasi/pengemasan baru, BELUM dikerjakan (sisanya, ~33):* sebagian besar blok
STATISTIK (F029-F045) meminta analisis ULANG atau sensitivitas tambahan (family-cluster bootstrap,
publikasi semua effect size per-target, definisi family untuk koreksi BH, dll.), plus F078/F079
(manifest provenance + instruksi akses dataset, butuh kerja pengemasan rilis), F053/F054 (AlphaFold),
F051 (perbesar sampel ChEMBL). **Catatan jujur: sebagian item blok statistik kemungkinan sudah
dipenuhi rigor naskah yang ada (Bab III.6 punya seksi unit of analysis, target-clustered bootstrap,
BH, status pra-registrasi, analisis daya), tapi saya TIDAK memverifikasinya satu per satu, jadi tidak
saya klaim selesai.**

*Verifikasi lanjutan (dilakukan setelah menulis catatan di atas, karena "kemungkinan sudah dipenuhi
tapi belum diperiksa" persis jenis klaim yang audit ini ada untuk mencegahnya).* Empat item paling
konsekuensial dari blok statistik DIPERIKSA LANGSUNG di naskah dan semuanya memang sudah tertangani:
- **F029** (unit analisis): ada subbab khusus "Units of analysis" di Bab III.6 yang menyebut eksplisit
  bahwa mencampur level akan menyalahsajikan jumlah bukti independen, lalu mendefinisikan atom ->
  molekul -> pocket -> target.
- **F032** (pasangan Wilcoxon): Bab III.6 menyatakan "Wilcoxon signed-rank tests (paired by pocket)".
- **F041** (hierarki bootstrap): dinyatakan bahwa klaster target adalah unit independen yang tepat
  "whenever several complexes or molecules of the same pocket are compared", plus subbab
  "Target-clustered statistics for predictor comparisons".
- **F030** (definisi keluarga untuk koreksi gabungan 48 uji): Appendix E.6 mendefinisikan keluarga
  secara eksplisit -- SATU variabel outcome per pool -- dan MENGECUALIKAN uji hitungan-interaksi PLIP
  dengan alasan (mengukur kuantitas berbeda, dilaporkan terpisah), dengan Tabel E.7 mendaftar setiap
  sumber yang masuk pool.
Verifikasi dilanjutkan ke empat item lagi, semuanya juga sudah tertangani:
- **F035** (status pra-registrasi): ada subbab khusus "Pre-registration status of each route" yang
  membagi tiap rute ke 4 kategori bukti (pre-registered / planned / post hoc / follow-up).
- **F034** (null underpowered): detectable shift dinyatakan (~1-1,4 kcal/mol), dan Bab X kini menambah
  pernyataan bahwa tidak ada uji ekuivalensi formal yang dijalankan.
- **F043** (target-disjoint != bebas semua kebocoran): Bab X kini menyatakan split ini LEBIH MUDAH
  daripada leave-superfamily-out, plus overlap scaffold 40,1%.
- **F042** (bahasa "0/48"): ditangani SANGAT baik di naskah lama -- klaim 0/48 disertai p mentah
  minimum (0,0306), p terkoreksi minimum (0,525), DAN catatan bahwa satu hit nominal itu cocok dengan
  ~2,4 positif palsu yang diharapkan secara kebetulan. Tidak ada klaim probabilistik atas hipotesis.
  Ditemukan juga penanganan keluarga koreksi yang lebih halus dari yang diminta F030: tier eksploratori
  DIKECUALIKAN dari pool bersama follow-up konfirmatorinya sendiri, "would double-count one question".

- **F031** (KS pada grup yang tumpang tindih): ditangani TELADAN di naskah lama -- Bab VI menyatakan
  sendiri bahwa grup terpilih dan pool tumpang tindih BY CONSTRUCTION, bahwa "a two-sample test is not
  strictly valid on overlapping groups", dan bahwa p-value-nya harus dibaca sebagai deskripsi jarak
  distribusi, bukan tingkat signifikansi eksak. Plus argumen mengapa itu tidak menjelaskan polanya
  (kontrol random-subset dan scrambled-filter punya struktur tumpang tindih IDENTIK dan tidak
  menunjukkan pergeseran sebanding).
- **F039** (varians antar-seed): didisklos di Bab IX -- "seed variance of the guidance predictors was
  therefore not measured".
- **F038 (subgroup M2) -- CELAH NYATA, DITUTUP DENGAN PENGUKURAN, bukan kaveat.** Naskah menyatakan
  aturan inklusi (>=10 kompleks -> 73 dari 127 target) tapi TIDAK pernah membandingkan subgrup itu
  dengan test set penuh. Dihitung dari anchor table (skrip baru
  `guidance/track_e/m2_subgroup_characterisation.py` + JSON, supaya angkanya dapat diturunkan ulang):
  73 target itu memuat **11.652 dari 11.855 kompleks (98,3%)**, jadi cakupan datanya hampir penuh
  meski cuma 57% target. TAPI yang dikecualikan BUKAN sampel acak: median 3 kompleks (vs 39), dan
  **pK mean 6,017 vs 6,807 -- sistematis 0,79 unit LEBIH RENDAH** (sd setara, 1,75 vs 1,64).
  Ditulis di Bab VIII berikut arah biasnya: mengecualikan target berafinitas terlemah justru membuang
  bagian rentang di mana metrik peringkat paling berpeluang gagal, jadi M2 diukur dalam kondisi yang
  sedikit MENGUNTUNGKAN model yang dibandingkan.

**BLOK STATISTIK (F029-F045) KINI TUNTAS DIPERIKSA SELURUHNYA -- 17 item, satu per satu:**
- *Sudah tertangani naskah lama (15)*: F029 (subbab Units of analysis), F030 (satu outcome per pool,
  + eksklusi yang lebih halus: tier eksploratori tak dipool dengan konfirmatorinya sendiri), F031
  (Bab VI menyatakan sendiri uji dua-sampel tak valid pada grup tumpang tindih, plus argumen kontrol),
  F032 (paired by pocket), F033 (44 interval kepercayaan mendampingi p-value), F034 (detectable shift),
  F035 (subbab status pra-registrasi, 4 kategori), F036 (Bab VIII.1.2 "Premise checks that changed the
  plan" -- "Four measurements made BEFORE training", jadi audit bukan tuning pasca-hasil), F037
  (diakui fold/famili bisa dibagi, DAN diuji langsung lewat family-holdout Appendix E.3.2), F039
  (varians seed didisklos tak diukur), F040 (PLIP outcome berbeda, tak dipool), F041 (klaster target),
  F042 (0/48 disertai p mentah/terkoreksi minimum + ekspektasi ~2,4 positif palsu), F043 (diperkuat
  sesi ini), F045 (kebocoran 423 baris/123 senyawa terdokumentasi + ditegakkan di kode).
- *Celah NYATA ditemukan dan ditutup sesi ini (2)*: **F038** (subgroup M2 -- ditutup dengan PENGUKURAN
  + skrip reproduksi, lihat di atas) dan **F044** (atribusi split artefact -- naskah menyebut dua
  perubahan split tapi mengatribusikan bersama; kini dinyatakan eksplisit bahwa split berubah dalam
  LEBIH DARI SATU hal sekaligus, bahwa desain ini tidak memisahkannya, dan bahwa split antara yang
  hanya memperbaiki tumpang-tindih validasi sambil mempertahankan test 27-pocket akan dibutuhkan untuk
  mengapportionkannya -- tidak dibangun. Daftar "Two limits" dinaikkan jadi "Three limits").

**TIDAK DIKERJAKAN, sengaja, dengan alasan:**
- F001-F012 (nama, NIM, gelar, tanggal sidang, pembimbing, tanda tangan, similarity report, dedication,
  acknowledgments): milik user. TIDAK boleh saya isi/karang.
- F073/F074 (verifikasi DOI satu per satu, novelty search independen): butuh lookup eksternal; 5 sitasi
  baru sudah ditandai `v='kn'` di refs.py.
- F077 (clean-room reproducibility run): butuh environment terpisah, keputusan scoping tersendiri.
- Banyak P2 RISK meminta ANALISIS/EKSPERIMEN BARU (F025 signed projection gradien, F037 family-cluster
  bootstrap, F049 publikasi semua effect size, F057 cosine similarity, F060 trace puluhan molekul,
  F065 audit varians residual). Ini komitmen komputasi baru, BUKAN perbaikan teks -- tidak dilepas
  sepihak; perlu keputusan scoping eksplisit lebih dulu.

### 9. Penulisan tesis (satu rebuild di akhir, jangan tiap perubahan) -- SELESAI 2026-10-15 19:58

**Rebuild dieksekusi SEKALI, sesuai rencana.** Output: `Thesis_Coupled_Guidance_SBDD.docx/.pdf`,
**226 halaman** (naik dari 148), **64 referensi tersitasi, nol unused, nol KeyError**. Build berhasil
pass pertama coba (make_figures.py bersih, build_thesis.py 2-pass bersih).

**Judul diganti** (lihat [[project_thesis_document]] -- keputusan user 2026-10-15, tesis belum
terdaftar resmi, "judul ngikut hasil"): dari "Coupled Affinity-Synthesizability Guidance for..."
(menyiratkan metode baru yang berhasil, tidak sesuai hasil) menjadi **"Diagnosing Guidance Failure in
Diffusion-Based Structure-Based Drug Design: A Leakage-Safe, Falsification-Driven Study"**.

**Yang ditambahkan (bab mana, bukan sekadar daftar poin):**
- **Bab IV** (bab terbesar yang ditambah): 4 seksi baru sebelum "Summary of the Chapter" --
  (1) audit sumber informasi 8-blok (ladder representasi: ligand-only 0,353 vs EGNN 0,342/0,344,
  pocket-only 0,289, ecif raw/norm confound, PLIP typed-interaction collapse ke proksi ukuran R²
  0,08->0,32); (2) studi komposisi data BindingNet (Arm A 0,362 vs Arm C 0,272, filter kebocoran
  senyawa sebelum bukan sesudah run produksi); (3) dekomposisi prior kelas (y-randomisasi
  dalam-target: 70% deskriptor, 90% EGNN sendiri, jendela daya 0,22 lebih sempit dari 1 CI);
  (4) diagnosa kegagalan via activity cliffs + MMP + stratifikasi famili protein (semua null/tidak
  signifikan, memperkuat temuan prior kelas). Summary of the Chapter ditulis ulang total.
- **Bab VI** (Track C): seksi baru "A Positive Control" -- hasil RQ2 lengkap (tabel, confound ukuran
  didisclose langsung di tabel yang sama, interpretasi F1/F2/F3, sitasi wajib BoKDiff + Thomas 2021).
- **Bab I**: problem statement, related work, objmap table (+3 baris baru), summary of results,
  organization -- semua diperbarui memasukkan RQ2 + audit representasi.
- **Bab II**: subbab baru "QSAR Controls" (Rucker/van Tilborg/Kwapien), tabel positioning +baris
  BoKDiff, "this thesis" row diperbaiki (PLIP dulu "designed, not tested" sekarang "tested, Ch IV.6").
- **Bab III**: subbab baru "Three candidate mechanisms for a guidance failure" -- memformalkan
  kerangka F1 (sinyal tak informatif) / F2 (mekanisme salah) / F3 (objektif bisa dicurangi).
  **PENTING: diberi label F1/F2/F3, BUKAN M1/M2/M3** -- Bab VIII (Track E) SUDAH memakai M1/M2/M3
  untuk nama metrik lain (pooled R², within-target Spearman, korelasi heavy-atom) sejak sebelum
  sesi ini; dicek via grep SEBELUM nulis bab lain yang memakai kerangka ini, tabrakan dihindari.
- **Bab IX** (Diskusi): paragraf "Consistent" di "Grading the explanations" ditulis ulang -- klaim
  lama "bukti sebab-bersama bersifat tidak langsung" sekarang diganti karena RQ2 memberi bukti
  LANGSUNG. Tabel grand & ces +baris positive control. Tensi BADGER diperkuat (harness terbukti
  sensitif, jadi bukan itu penyebab diskrepansi). +2 methodological finding baru.
- **Bab X** (Kesimpulan): kesimpulan (1) dikualifikasi (Pearson 0,58-0,65 argumen BOCOR, sesuai
  rencana lama); kesimpulan (2) ditulis ulang total (positive control jadi temuan tunggal terpenting);
  (4)(5) diperluas + (5a)(5b) baru (prior kelas, BindingNet); Contributions 5->8 poin; Limitations
  +gray zone scaffold 40,1% + batas positive control (hanya selection-based, bukan gradient).
- **Abstrak**: ditulis ulang, sekarang menyebut RQ2 dan audit representasi sebagai temuan inti.
- **refs.py**: +5 sitasi baru (rucker2007, vantilborg2022, kwapien2022, thomas2021, bokdiff2025),
  ditandai `v='kn'` (dari log riset proyek, belum diverifikasi DOI individual -- SEBELUM submit
  resmi, verifikasi dulu sesuai konvensi refs.py sendiri).

**Yang DIKECUALIKAN dari rebuild ini, sengaja**: suite ketidakpastian A1-A1g, ternary QAT, dan
pengecekan ekuivariansi (Zhou 2026) TIDAK dimasukkan -- itu benang riset terpisah (bukan tentang
guidance), rencana item 9 yang lama (ditulis sebelum sesi ini) juga tidak pernah menyebutkannya.
Kalau nanti mau dimasukkan, itu keputusan scoping baru yang perlu dikonfirmasi eksplisit, bukan
diasumsikan.

**Belum dikerjakan dari rencana lama (poin e)**: beberapa sitasi yang disebut di rencana lama
(Mattsson 2026, Bret 2026, Deng 2023, Hernandez-Garrido 2023, Landrum 2024, Seitzer 2022, Wallace
2023, Gao 2022) TIDAK ditambahkan sesi ini -- tidak ada detail bibliografi yang cukup tersimpan di
TASKS.md/memory untuk mengutipnya tanpa risiko fabrikasi. Kalau dibutuhkan, cari dulu detail lengkapnya
sebelum menambah entri refs.py.

**Verifikasi dijalankan sebelum build** (bukan sesudah, supaya build pertama bersih): semua key tabel
{tab:x} dicek match dengan definisi @tab x (45=45, nol hilang); semua sitasi [@key] dicek ada di
refs.py (nol undefined); sintaks markup dicek konsisten (tidak ada `@{` yang salah tempat tertinggal
dari draf Bab I/II Week 5 yang terpisah).

### 10. RQ4 — conformal-gated guidance -- DISELESAIKAN 2026-10-13 (tanpa GPU, lewat pra-registrasi)
**Apa:** terapkan gradien guidance / terima kandidat rejection-sampling HANYA di tempat interval conformal
surrogate cukup sempit.
**Mengapa menarik:** kombinasinya untuk SBDD tampak kosong (komponennya ada terpisah: conformal untuk
afinitas, AD-gating anti reward-hacking Yoshizawa 2025, guidance-strength bergantung confidence Azangulov
2025). **Tulis prediksinya DULU:** M3 memprediksi membantu rute rejection; M1 memprediksi TIDAK membantu
rute gradien. Satu eksperimen yang mengkonfirmasi satu dan membantah satunya = bukti kuat untuk seluruh
dekomposisi.

**Pra-registrasi lengkap**: `guidance/RQ4_CONFORMAL_GATED_PREREGISTRATION.md`. **Gap desain ditemukan
sebelum kode guidance ditulis**: `q` conformal A1d itu GLOBAL (satu angka untuk semua titik, dibaca
langsung dari `conformal_prediction.py`) -- "terima HANYA di interval sempit" tidak bermakna kalau lebar
interval tidak pernah bervariasi antar molekul. Dua kandidat paling alami untuk membuat lebar bervariasi
secara lokal, DIUJI LANGSUNG (CPU, menit, pakai cache yang sudah ada, tanpa GPU):

| kandidat sinyal lokal | hasil terukur | verdict |
|---|---|---|
| spread ensemble A1c sebagai proksi ketidakpastian | spearman(sigma,error)=0,0084, CI [-0,099, +0,132] | **CI memuat nol -- tidak informatif** |
| novelty tier (item 3) sebagai kategori Mondrian | coverage 0,831-0,910 antar 4 tier, rasio 1,10x | **nyaris flat -- tidak ada variasi bermakna** |

**Kedua kandidat yang paling mungkin dipakai untuk menggerbangi "confidence" sudah terbukti tidak
membawa sinyal yang bisa dipakai** -- bukan dugaan baru, melainkan konfirmasi langsung dari dua temuan
yang SUDAH tercatat (A1c sigma-error, item 3 coverage kondisional). **Kesimpulan: RQ4 seperti yang
dirancang TIDAK BISA diuji secara bermakna dengan infrastruktur UQ yang kita punya** -- bukan karena M1
atau M3 salah, tapi karena tidak ada alat ukur confidence yang terbukti bekerja untuk menggerbanginya.
Ini temuan substantif, bukan jalan buntu: memperkuat kesimpulan UQ-suite (hanya coverage MARGINAL yang
valid, 5/6 metode titik-demi-titik gagal) dengan konsekuensi langsung ke desain guidance generasi
berikutnya -- siapa pun yang mau mencoba conformal-gated guidance di proyek ini harus dulu membangun
metode UQ lokal-adaptif yang BENAR-BENAR terkalibrasi, bukan asumsikan salah satu yang sudah ada cukup.

**Biaya: nol GPU, menit CPU.** Tidak ada komitmen GPU yang dibuat atau diperlukan.

### 11. RQ3 — DOODL / SVDD -- PRA-REGISTRASI SELESAI 2026-10-13, implementasi DITUNDA
**Apa:** gradien lewat denoiser (DOODL, Wallace et al. 2023) atau guidance tanpa turunan (SVDD, Li et al.
2024) dengan surrogate yang ADA.
**Mengapa terakhir:** paling mahal, dan **M1 memprediksi TIDAK akan membantu** padahal literatur
mengharapkan sebaliknya. Nilainya justru dari prediksi itu — jadi pra-registrasikan sebelum jalan.

**Pra-registrasi**: `guidance/RQ3_DOODL_SVDD_PREREGISTRATION.md`. **RQ2 mengubah kalkulusnya**:
affinity_only (gradien dari surrogate) gagal, vina_rerank (seleksi dari Vina asli) berhasil -- kontras
ini sudah memberi bukti tidak langsung kuat bahwa masalahnya di SINYAL (M1), bukan mekanisme gradien
klasik. Menjalankan DOODL/SVDD dengan surrogate yang SAMA (**Jalur A**) kemungkinan besar cuma
mengonfirmasi ulang apa yang sudah bisa disimpulkan, dengan biaya implementasi terbesar di seluruh
daftar. **Jalur B** (SVDD + Vina asli sebagai fungsi nilai, bukan surrogate) adalah pertanyaan BARU yang
baru dibuka oleh RQ2 -- apakah mekanisme guidance canggih menambah nilai DI ATAS reranking sederhana yang
sudah terbukti bekerja -- dan nilainya lebih tinggi. **Keputusan: implementasi DITUNDA** untuk keduanya;
Jalur B dicatat sebagai arah riset terbuka untuk pekerjaan lanjutan, bukan dipaksa masuk sesi ini.
**Biaya:** nol (ditunda dengan alasan tercatat, bukan dikerjakan).

---

### TERBLOKIR / RENDAH
* konservasi sekuens & fitur MSA residu pocket
* deskriptor druggability pocket (butuh fpocket/CASTp)
* kurasi "maximal curation" Landrum untuk label BindingNet Arm B/C
* struktur LP-PDBBind (lisensi), Binding MOAD (unduh manual)

### REFACTOR TERPUSAT 2026-10-13: `paired_r2_comparison` di `stats_common.py`

Setelah kelas kesalahan yang SAMA muncul 3x dalam satu sesi (item 6 MMP: bandingkan dua CI tumpang-tindih
tanpa uji berpasangan; item 7 famili protein: `r2_rows` membelah varian nyaris-nol jadi dR2 +5,8e16;
item 5 activity cliffs: `if champ_cliff > struct_cliff` tanpa uji signifikansi sama sekali) --
alih-alih menambal lagi di tempat ke-4/5/6, logikanya disatukan jadi SATU fungsi teruji:
`guidance/uncertainty_a1/stats_common.py::paired_r2_comparison` (+ `r2_rows` dipindah ke sana sebagai
rumah kanonik, 9 pemanggil lama tetap seperti semula -- lihat di bawah).

**Yang diretrofit (aman, diverifikasi):**
- `activity_cliffs.py::stratum_r2_comparison` -> wrapper tipis di atas fungsi baru. Belum pernah
  dijalankan dengan hasil nyata (masih di queue), jadi nol risiko mengubah angka tercatat; diverifikasi
  dengan data sintetis identik sebelum/sesudah.
- `protein_family_stratification.py`'s tabel CONFIRMATORY -> diverifikasi **byte-exact**: disimpan
  hasil sebelum refactor, jalankan ulang, dibandingkan field demi field (n, n_target_clusters,
  structure_r2, ligand_r2, d_r2, ci95, boot_p, underpowered) -- **nol perbedaan nilai**, hanya field
  `degenerate` baru ditambahkan (penambahan skema, bukan perubahan nilai).

**Yang SENGAJA TIDAK diretrofit** (mengikuti kebijakan tertulis `stats_common.py` sendiri: angka yang
sudah dikutip di TASKS.md tidak disentuh untuk manfaat kebersihan kode semata tanpa verifikasi
byte-exact menyeluruh): `novelty_tiers.py`, `representation_ladder.py`, `noise_ceiling.py`,
`label_harmonisation.py`, `scaffold_audit.py`, `compare_vs_structure.py` -- semua `r2_rows`-nya tetap
seperti semula. Risiko bug yang sama di sana dinilai rendah (strata mereka beroperasi di seluruh test
set atau grup besar, bukan famili 1-2 target) tapi tidak diaudit penuh dalam pass ini.

**AUDIT KEAMANAN (baca-saja) 2026-10-14 -- verifikasi, bukan lagi asumsi**: dibaca langsung tiga dari
enam skrip yang sengaja tidak diretrofit.
- `scaffold_audit.py`: TIDAK ada bug. Perbandingan seen-vs-unseen-scaffold memakai bootstrap TIDAK
  berpasangan SECARA SENGAJA, dan komentarnya sendiri menjelaskan kenapa benar: "the subsets are
  DIFFERENT complexes, so each is resampled over its own target clusters and the two draws are
  independent" -- beda dari kasus model-A-vs-model-B pada baris yang SAMA (itu yang butuh berpasangan).
- `label_harmonisation.py`: TIDAK ada bug. `point = r2_rows(pk_adj,p_,all_rows) - r2_rows(pk_adj,lp_,all_rows)`
  dan bootstrap-nya `[r2_rows(...,d) - r2_rows(...,d) for d in draws]` -- draws YANG SAMA memberi makan
  kedua suku di setiap resample, pola berpasangan yang benar, sudah seperti itu dari awal.
- `noise_ceiling.py`: tidak ditemukan perbandingan model-vs-model (subtraksi dua R2) di jalur yang
  diperiksa -- hanya R2 tunggal per model dengan CI, tidak ada pasangan untuk salah berpasangan.
Kesimpulan: keputusan "tidak retrofit" aman, dikonfirmasi lewat pembacaan langsung, bukan dugaan.

**Bonus temuan saat verifikasi byte-exact**: tabel DESKRIPTIF `protein_family_stratification.py` (yang
TIDAK diretrofit, cuma dibandingkan sebagai kontrol) ternyata **nondeterministik dari sebelumnya** --
kode yang SAMA dijalankan dua kali menghasilkan URUTAN baris berbeda, murni dari iterasi `set` Python
yang tidak stabil antar proses (`{f for f in fam if f != 'no_annotation'}`, bergantung
`PYTHONHASHSEED`). Bukan bug yang saya buat, tapi ketahuan karena saya memverifikasi byte-exact alih-alih
cuma percaya refactor-nya benar. Diperbaiki dengan secondary sort key (nama famili) sehingga tie-break
deterministik -- diverifikasi 2x run berurutan sekarang identik.

8 tes baru untuk `paired_r2_comparison`/`r2_rows` di `test_stats_common.py`, memaku: hitungan tangan di
stratum sehat, guard degenerate, kasus underpowered, properti BERPASANGAN (bukan dua bootstrap
independen), dan resample individual yang mendarat di subset degenerate tidak merusak CI keseluruhan.

---

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

**TAMBAHAN 2026-10-14: RQ2 (kontrol positif) diperiksa literatur -- MEKANISMENYA terscoop, BUKAN
framing diagnostiknya.**

Pencarian terarah (WebSearch extended) menemukan **BoKDiff** (Khodabandeh Yalabadi, Yazdani-Jahromi &
Garibay, arXiv:2501.15631) -- diverifikasi via abstrak: best-of-K/best-of-N SELECTION dari kandidat
diffusion SBDD (dibangun di atas DecompDiff), diranking skor gabungan QED+SA+Vina, TANPA fine-tuning.
Mereka secara eksplisit mengklaim **"first application of Best-of-K alignment and Best-of-N sampling to
SBDD."** Itu persis MEKANISME yang dipakai `vina_rerank`: oversample pool, ranking berdasar skor Vina,
ambil yang terbaik. **Klaim keunggulan atas mekanisme reranking-berbasis-Vina untuk diffusion SBDD TIDAK
BISA dipertahankan** -- BoKDiff lebih dulu, harus disitir bukan diklaim sebagai metode baru.

**Yang diverifikasi TIDAK ada di BoKDiff** (dicek eksplisit via WebFetch terhadap abstraknya): (1) tidak
ada perbandingan terhadap guidance GRADIEN dari surrogate afinitas TERLATIH pada generator dan pocket
yang SAMA; (2) tidak ada framing sebagai KONTROL POSITIF untuk memvalidasi bahwa harness evaluasi mampu
mendeteksi perbaikan; (3) tidak ada dekomposisi M1 (sinyal tak informatif) vs M2 (mekanisme salah) vs M3
(objektif bisa dicurangi) untuk MENJELASKAN kegagalan guidance yang sudah ada.

**Klaim yang bisa dipertahankan, dipersempit jujur**: bukan "kami menemukan reranking berbasis Vina
untuk diffusion SBDD" (BoKDiff lebih dulu, disitir sebagai itu) -- tapi **"memakai reranking (teknik
yang sudah ada) sebagai instrumen DIAGNOSTIK untuk mengisolasi apakah kegagalan guidance gradien berasal
dari sinyal surrogate atau dari mekanisme gradien itu sendiri, lewat perbandingan langsung di generator
dan pocket yang sama."** Itu kontribusi metodologis sempit (framing + dekomposisi), bukan teknik
generasi baru. **Dipindah dari "Tier 1,5 belum diaudit" ke Tier 2** (teknik established, dipakai untuk
tujuan diagnostik yang berbeda dari tujuan asli BoKDiff) -- dengan BoKDiff WAJIB disitir di tesis.

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

**TAMBAHAN 2026-10-11: klaim QAT ternary divalidasi HANYA dengan R2, dan itu tidak cukup.**

*Zhou, Xue, Zhang & Fu 2026* (arXiv:2601.02213, 8-bit SO(3)-ekuivarian, QM9/rMD17) **sudah ada di audit
prior art kita** sebagai prior art terdekat -- jadi makalahnya bukan temuan baru, saya menemukannya ulang.
Yang baru adalah apa yang ia ungkap tentang pekerjaan kita sendiri. Diverifikasi dengan mengambil
abstraknya: 8-bit, energi/gaya, transformer atensi SO(3); **tanpa ternary, tanpa data protein-ligan**,
jadi klaim (b) kita TIDAK terscoop.

Mereka menunjukkan kuantisasi low-bit naif **merusak** struktur ekuivarian, dan melacaknya dengan
*Local Equivariance Error*; menjaganya utuh butuh tiga inovasi terpisah. Klaim kita -- ternary tak
terbedakan dari FP32, R2 0,3437+-0,0377 vs 0,3420 -- berdiri **sepenuhnya di atas R2**. Model bisa
mempertahankan R2 sementara simetrinya merosot. Kita tidak pernah mengukurnya.

**Prediksi, dicatat SEBELUM mengukur** (prediksi yang dibuat sesudahnya bukan uji): EGNN kita seharusnya
imun *secara konstruksi*, karena dua hal yang terlihat di `models/egnn.py` -- geometri masuk ke fitur
skalar hanya lewat `d_sq = sum(rel_x**2)` (baris 42) melalui Gaussian smearing, yakni JARAK antar-atom,
yang tak diubah gerak rigid; dan pembaruan koordinat (baris 63) adalah
`delta_x = scatter_sum((xi-xj)/(|d|+1) * x_mlp(mij))`, vektor posisi relatif dikali **skalar** terpelajar.
Mengkuantisasi `nn.Linear` hanya mengubah skalar itu, jadi pembaruannya tetap kombinasi vektor relatif
yang sama. Kasus Zhou dkk. justru kebalikannya: di sana **fitur vektor ekuivarian itu sendiri** yang
dikuantisasi, dan itulah sebabnya dekopling magnitudo-arah diperlukan.

**Mengapa ini lebih dari menambal lubang:** kalau prediksinya benar, klaimnya naik kelas dari "ternary
kebetulan bekerja untuk kami" menjadi "**apakah kuantisasi bobot bisa mengancam ekuivariansi itu sifat
ARSITEKTUR** -- message passing berbasis jarak aman secara gratis, fitur vektor terkuantisasi tidak".
Lebih sempit, lebih kuat, dan menjelaskan alih-alih sekadar melaporkan.

**HASIL (dijalankan 2026-10-11 17:36, 24 kompleks uji x 8 rotasi, 4 model):** prediksi bertahan.

| model | deviasi gerak rigid (mean / max) | median per-kompleks | >1e-4 | kontrol positif (mean) | rasio |
|---|---|---|---|---|---|
| fp32_stage0 | 7,9e-07 / 1,1e-05 | 9,54e-07 | 0/24 | 7,2e-02 | 8,7e4x |
| ternary s2021 | 1,9e-05 / 1,7e-03 | 9,54e-07 | 1/24 | 1,7e-01 | 8,9e3x |
| ternary s2022 | 6,2e-07 / 3,4e-05 | 9,54e-07 | 0/24 | 7,9e-02 | 1,1e5x |
| ternary s2023 | 2,6e-07 / 4,3e-06 | 4,77e-07 | 0/24 | 1,3e-01 | 4,2e5x |

Tiga hal yang membuat ini bermakna, bukan sekadar "angkanya kecil":
1. **Kontrol positif bekerja.** Transform yang sengaja dirusak menggerakkan prediksi 0,07-0,17 pK rata-rata
   (maks 0,85), empat sampai lima orde lebih besar dari deviasi gerak rigid. Jadi harness-nya MAMPU
   mendeteksi pelanggaran; tidak menemukannya adalah informasi, bukan kebisuan.
2. **Deviasi median ADALAH batas aritmetika.** 9,537e-07 = 2^-20 dan 4,768e-07 = 2^-21; ULP float32 pada
   6,0 adalah 4,77e-07 dan pada 10,0 adalah 9,54e-07. Untuk prediksi berskala 5-7 pK, median itu 1-2 ULP.
   Invariansi bukan "hampir terjaga" -- ia eksak sejauh float32 mampu menyatakannya, dan sama eksaknya
   untuk ternary seperti untuk FP32.
3. **Max s2021 adalah SATU kompleks, bukan degradasi.** Median per-kompleksnya identik dengan FP32 dan
   hanya 1 dari 24 kompleks melewati 1e-4. Dilaporkan apa adanya: 1,7e-03 pK itu 0,14% dari sd prediksi
   (1,23 pK) dan 100x di bawah kontrol positif, jadi verdict tak berubah -- tapi agregat max tanpa
   rincian per-kompleks tidak bisa membedakan satu pencilan dari kemerosotan menyeluruh, dan skripnya
   diperbaiki untuk menyimpan keduanya setelah saya menyadari itu.

**Klaim yang kini boleh dibuat:** bukan "ternary kebetulan bekerja", melainkan *apakah kuantisasi bobot
bisa mengancam ekuivariansi adalah sifat ARSITEKTUR* -- message passing berbasis jarak aman secara
gratis, sedangkan fitur vektor ekuivarian yang dikuantisasi langsung (kasus Zhou dkk.) butuh tiga
inovasi agar aman. Itu menjelaskan, bukan sekadar melaporkan, dan ia memberi klaim QAT kita validasi
simetri yang sebelumnya hanya bersandar pada R2.

**Skrip:** `guidance/uncertainty_a1/equivariance_check.py` (hanya forward pass, tanpa latihan, tanpa komitmen GPU). Ia membandingkan FP32 Stage-0 terhadap ketiga seed QAT
ternary di bawah rotasi proper acak (det=+1, jadi tanpa refleksi -- refleksi bukan simetri molekul kiral)
plus translasi SELURUH kompleks. **Ia membawa kontrol positifnya sendiri**: transform yang sengaja
dirusak, merotasi ligan saja terhadap sentroidnya sendiri, yang mengubah jarak antar-atom nyata dan wajib
menggerakkan prediksi. Skripnya MENOLAK memberi verdict bersih kalau kontrol itu tetap di bawah 1e-3 pK,
karena harness yang tak mampu mendeteksi pelanggaran tidak membuktikan apa pun dengan tidak menemukannya.
Ini juga satu-satunya tempat di proyek ini yang kontrol positifnya sudah terpasang -- lihat item #1 di
URUTAN KERJA, tempat ketiadaannya dicatat sebagai celah teratas.

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
10. **Kemunculan KEEMPAT self-match, kali ini di kode verifikasi saya sendiri.** Pemeriksa "apakah ada
    proses queue yang bertahan" melaporkan shell yang menjalankan pemeriksa itu, karena teks heredoc-nya
    memuat string `queued_run.sh`. Filter `comm` tidak menolong: shell itu memang `bash`. Perbaikan yang
    benar butuh dua hal lagi -- kecualikan pid sendiri **beserta seluruh leluhurnya**, dan cocokkan hanya
    **argv[1]** (skrip yang bash memang disuruh jalankan), bukan string mana pun di argumen berikutnya.
11. **Mem-patch skrip sementara queue-nya jalan TIDAK berefek, dan saya hampir tidak menyadarinya.**
    Watchdog termal sudah saya tulis, uji, dan perbaiki -- tapi queue yang sedang jalan tetap memakai
    versi lama. Sebabnya justru properti yang membuat patch itu aman: `os.replace` menukar entri
    direktori, dan shell yang sedang jalan tetap memegang inode lama yang sudah terhapus
    (`/proc/213447/fd/255 -> master_queue.sh (deleted)`). Buktinya bukan dugaan: gerbang yang ia
    lancarkan untuk `noise_ceiling_refresh` **tidak membawa** `--throttle-at/--throttle-resume/--threads`.
    Dua step terakhir akan jalan tanpa watchdog **dan tanpa thread cap** -- tepat kondisi 92 C/4j42m yang
    watchdog itu dibuat untuk mencegah. **Aturan baru: setiap patch pada `master_queue.sh` atau
    `queued_run.sh` wajib diikuti restart queue, dan diverifikasi dengan membaca cmdline gerbang baru +
    `/proc/<pid>/fd/255` (harus inode hidup, bukan `(deleted)`).**
12. **Lalu saya langsung melanggar sisi lain aturan yang sama.** Saat menambahkan peringatan di atas ke
    header `master_queue.sh`, saya memakai `open(p,'w')` -- truncate di tempat -- sementara bash memegang
    fd 255 pada inode yang **hidup** (queue baru di-restart, jadi tidak lagi inode terhapus). Menambah
    baris menggeser offset baca; bash membaca skrip bertahap dan bisa mengeksekusi sampah. Tidak terjadi
    apa-apa hanya karena body `cmd_run` sudah selesai di-parse. **Aturan: `os.replace` atau `mv`, JANGAN
    `open(...,'w')`, untuk setiap skrip yang mungkin sedang dibaca shell.**

    **KOREKSI 17:34 -- risikonya TERWUJUD, dan saya sudah menyatakan sebaliknya.** Saya memeriksa segera
    setelah mengedit, melihat queue sehat, lalu menulis "risikonya tidak terwujud karena bash sudah
    selesai mem-parse body `cmd_run`". Dua puluh tujuh menit kemudian, saat bash membaca bagian AKHIR
    skrip dari offset yang tergeser:
    `scripts/master_queue.sh: line 290: unexpected EOF while looking for matching '"'`.
    Dampaknya nol -- error muncul SETELAH `queue idle` dicatat dan seluruh 13 step tersimpan, dan file di
    disk sendiri valid (`bash -n` lolos). Tapi pelajaran sebenarnya bukan soal aturan atomic replace,
    melainkan soal cara saya memverifikasi: **"saya cek dan tidak ada yang rusak" bukan bukti aman untuk
    kerusakan yang baru muncul saat kode yang terpengaruh akhirnya dibaca.** Pemeriksaan segera hanya
    menyampel satu titik waktu. Untuk bahaya yang tertunda, pernyataan yang benar adalah "belum terwujud
    sejauh ini", bukan "tidak terwujud". Perhatikan bahwa dua aturan
    ini saling melengkapi, bukan bertentangan: atomic replace aman bagi proses berjalan justru karena ia
    *tidak* mengubah apa yang mereka jalankan -- itulah sebabnya restart tetap wajib.
13. **Guard idempotensi saya sendiri tersandung sebutan yang baru saja saya sisipkan.** Skrip patch untuk
    `test_gradient_normalization.py` memeriksa `'eps_floor_attenuates' not in s` agar tidak menambal dua
    kali -- tapi penggantian sebelumnya di skrip yang sama menyisipkan komentar yang MENYEBUT nama tes
    itu, jadi guard-nya gagal dan tak ada yang tertulis. Keluarga yang sama dengan self-match pgrep:
    mencocokkan sebutan, bukan hal itu sendiri. **Aturan: guard idempotensi harus mencocokkan DEFINISI
    (`def test_...`), bukan namanya saja.**

**Pelajaran operasional:** jangan percaya ingatan saya soal apa yang hidup;
`bash scripts/master_queue.sh status` membacanya dari keadaan nyata. Dan `status` membaca **disk**, bukan
apa yang proses hidup benar-benar jalankan -- untuk itu baca `/proc/<pid>/fd/255` dan cmdline-nya.

---

### E. INFRASTRUKTUR YANG DIBANGUN -- selesai, jangan disentuh

* **13 suite, 178 tes** (`python guidance/run_tests.py`), pre-commit hook terpasang, jalur FAIL/SKIP diuji
* **Watchdog termal** (`queued_run.sh`): gerbang lama hanya mengatur saat MULAI, dan itu terbukti tidak
  cukup -- `ave_bias` lolos di CPU 57°C lalu bertahan di 92°C selama 4j42m. Sekarang ada SIGSTOP/SIGCONT
  (>=88°C bertahan 60s -> suspend; <75°C -> resume; budget 7200s lalu menyerah dan membiarkan jalan) plus
  thread cap 10/20 untuk mengurangi panas di sumbernya. 4 jalur diuji, termasuk satu bug yang ditemukan
  oleh tes saya sendiri (`THROTTLE_AT=0` justru membuat `CT >= THROTTLE_AT` selalu benar). **Terkonfirmasi
  aktif di produksi** pada 2026-10-11 17:08: `thread cap: 10` + `thermal watchdog: suspend above 88C`.
* **`guidance/track_a/tests/test_physics.py` (12 tes) -- menutup celah yang paling berisiko yang ditemukan
  audit:** `interaction_masks` dipanggil dari `track_a/model.py:114`, jadi ia menentukan pasangan atom mana
  yang menyumbang energi vdW/H-bond/metal/hidrofobik pada model TERLATIH -- dan ia tidak punya satu tes pun.
  Tes intinya memanfaatkan pasangan keras/lunak: `soft_interaction_masks` mengaku relaksasi kontinu
  (perkalian untuk AND, a+b-ab untuk OR), jadi diberi probabilitas 0/1 ia WAJIB identik dengan versi keras,
  karena untuk x,y in {0,1}: x*y = x AND y dan x+y-xy = x OR y. 200 undian acak: identik eksak. Nilai
  harapan contoh kecilnya dihitung di atas kertas, bukan dari kode yang diuji.
* **`guidance/lp_split/tests/test_lp_split_loader.py` (15 tes) + penjaga tabrakan indeks.** Audit impor
  menemukan 55 modul kita diimpor kode nyata tapi tak disentuh tes mana pun. Mayoritas driver eksperimen
  (ujinya ya eksperimen itu), tapi satu menonjol: **`build_lp_splits` punya 37 pengimpor** dan ia yang
  menentukan mana data latih dan mana yang ditahan -- cacat di situ tidak menghasilkan eksperimen salah,
  ia menghasilkan tesis yang setiap angkanya tidak sah. 40 baris, penanggung beban, tanpa tes.
  **Bahaya laten yang ditemukan:** `pk_by_idx[idx] = pk` ditulis untuk ketiga partisi tanpa deteksi
  tabrakan, jadi indeks yang hadir di train DAN test akan diciutkan jadi satu label -- kebocoran
  tersamarkan justru oleh struktur yang dibuat untuk mencegahnya. Sekarang `raise` dengan pesan yang
  membedakan kebocoran lintas-partisi dari duplikat dalam satu partisi. Split terkirim diverifikasi
  bersih: 46.964/6.069/11.855 indeks disjoint berpasangan, tanpa duplikat internal, tanpa konflik pK,
  dan 779/65/127 target-nya juga disjoint. Ukuran itu dipaku di tes supaya perubahan senyap pada file
  split jadi kegagalan tes, bukan kejanggalan tabel yang baru disadari berbulan-bulan kemudian.
* **`guidance/tests/test_tnn_qat.py` (18 tes) -- menemukan docstring saya sendiri terlalu kuat.** Modul
  ini menopang klaim "QAT ternary tak terbedakan dari FP32" (0,3437+-0,0377 vs 0,3420, n=3) dan tak punya
  tes. Tes yang menuntut `ste_ternarize(w)` identik bit-per-bit dengan `twn_ternary(w)` **GAGAL**:
  `w + (ternary(w) - w)` sama dalam aritmetika real tapi penjumlahannya membulatkan di float32. Terukur
  atas 16.200 bobot: 0,1% nilai TAK-NOL bergeser maksimum 1,192e-07 = **tepat 1 ULP**. Nolnya **eksak**,
  karena `w + (0 - w)` dievaluasi sebagai `w - w` -- jadi pola sparsity yang diklaim TNN bukan aproksimasi.
  Docstring dikoreksi dari "equals ternary(w) exactly" menjadi pernyataan dua-bagian yang terukur. Satu
  ULP bukan ancaman bagi klaim itu (sd antar-seed 0,0377 R2), tapi docstring tidak boleh melebihkan.
  Ikut dipaku: aturan TWN terhadap contoh hitung-tangan, alpha = mean|w| survivor, gradien STE sampai ke
  bobot yang dinolkan forward (inilah beda QAT vs pruning post-hoc yang mengolapskan R2 ke -1,50), dan
  `convert_to_ternary_qat` MEMBAGI objek nn.Parameter (bukan menyalin) -- salinan akan tetap load tanpa
  error lalu melatih tensor yang tak pernah dibaca forward, kegagalan yang menghasilkan angka masuk akal.
* **`guidance/tests/test_gradient_normalization.py` (14 tes) -- batas validitas yang belum pernah
  dinyatakan, kini terukur.** Transform Track B1 ini ada untuk menghapus SATU ambiguitas: "hasil guidance
  nol itu arah yang tak informatif, atau magnitudo yang salah skala?" Tesnya mengungkap bahwa norma yang
  tercapai bukan 1 melainkan `mean/(mean+eps)`, jadi kekurangannya `eps/(mean+eps)`:

  | mean \|g\| per atom | norma tercapai | kekurangan |
  |---|---|---|
  | 1e-2 | 0,999939 | 0,01% |
  | 1e-3 | 0,999388 | 0,06% |
  | 1e-4 | 0,993915 | 0,61% |
  | 1e-5 | 0,942306 | 5,77% |
  | 1e-6 | 0,620247 | 37,98% |

  Jadi di bawah ~1e-5 transform ini **melemahkan panduan secara diam-diam** -- justru kegagalan yang ia
  dibuat untuk menyingkirkan, hanya dipindah ke skala lain. **Model kita aman:** gradien per-atom terukur
  (`diag_gradient_magnitude_results.json`) 1,011e-3 / 6,53e-4 / 3,37e-4 -> kekurangan 0,10% / 0,15% /
  0,30%, sepele terhadap sd antar-seed 0,0377 R2. **Tidak ada hasil terlapor yang terpengaruh.** Batasnya
  kini ada di docstring sebagai tabel DAN dipaku oleh tes yang membaca JSON terukur itu, sehingga
  menambah model guidance bergradien lebih lemah akan menjatuhkan tes alih-alih kehilangan invariansi
  skala tanpa suara. Ikut diuji: arah terjaga eksak (cosine=1), normalisasi per-SAMPEL bukan per-batch
  (sampel 1e-2 dan 1e4 keduanya mencapai 1), independensi antar-sampel, komponen posisi vs tipe
  dinormalisasi terpisah, dan gradien nol menghasilkan nol bukan NaN.
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

**Audit kode mati (AST, bukan grep) -- 2026-10-11.** Instrumen pertama saya grep dan ia salah: mencari
`nama(` sehingga 18 fungsi yang dirujuk sebagai NILAI di `make_figures.py:487` tampak tak terpakai, dan
mencari `args.flag` sehingga skrip yang memakai `a.flag` tampak mengabaikan semua flag-nya. ~70 positif
palsu. Scan AST yang benar menyisakan 5 kandidat di seluruh repo:
* `scripts/data_preparation/split_pl_dataset.py::get_chain_name` dan
  `scripts/property_prediction/extract_pockets.py --fixed_sdf_dir` -- **upstream TargetDiff**
  (guanjq / Jiaqi Guan, 2023). Tidak disentuh: memangkasnya hanya menciptakan derau diff terhadap upstream.
* `lp_split/ligand_similarity.py::ligand_similarity` dan
  `lp_split/protein_similarity.py::build_category_grouped_pairs` -- artefak REPRODUKSI LP-PDBBind
  (cascade tie-breaking radius 2->4->10->SMILES kanonik). Tak terpakai karena kita memakai split yang
  sudah diterbitkan, bukan menghitungnya ulang; dipertahankan sebagai dokumentasi metode untuk tesis.
* `track_a/physics.py::soft_interaction_masks` -- tak terpakai DAN (dulu) tak teruji, kombinasi paling
  berbahaya. Diselesaikan dengan menguji, bukan menghapus: lihat butir test_physics.py di atas.

Kesimpulan audit: tidak ada overengineering yang perlu dipangkas di kode kita sendiri. Yang ada adalah
kode tak teruji, dan itu sudah ditutup.

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
