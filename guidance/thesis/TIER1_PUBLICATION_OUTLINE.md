# Draf Outline: Publikasi Independen dari Klaster Novelty Tier-1

**Status:** kerangka, belum ditulis penuh. Tujuan: pisahkan 5 kontribusi Tier-1 dari tesis 198 halaman menjadi satu paper yang berdiri sendiri — alasan dijelaskan di §1 `LAPORAN_RISET_LENGKAP.md` §8.3 poin 1.

---

## 1. Kenapa dipisah dari tesis

Tesis menumpuk semuanya (10+ eksperimen, kerangka M1/M2/M3, kritik literatur) dalam satu dokumen 198 halaman. Lima temuan di bawah masing-masing cukup berdiri, tapi tenggelam kalau cuma jadi satu bab di antara banyak bab lain. Dipisah memaksa narasi yang lebih tajam: **bukan "kenapa guidance gagal" (itu untuk tesis), tapi "apa yang sebenarnya sebuah model afinitas protein-ligan 3D pelajari, dan berapa banyak ruang nyata ada untuk membedakan model yang lebih baik dari yang lebih buruk di domain ini".**

## 2. Working title (3 kandidat)

1. *"What Does a 3D Affinity Model Actually Learn? A Decomposition of Ligand, Pocket, and Class-Level Signal"*
2. *"The Resolvable Window is Narrower Than the Noise: Power-Aware Benchmarking for Protein-Ligand Affinity Models"*
3. *"Beyond Ligand-Only Baselines: A Factorial Test of What Structural Information Protein-Ligand Models Use"* — favorit, karena judulnya langsung menyebut kontribusi #1 (ablasi faktorial) yang paling baru, dan menempatkan diri secara eksplisit sebagai KELANJUTAN dari Volkov/Durant/dkk, bukan klaim "menemukan" fenomena yang sudah dikenal.

## 3. Venue yang masuk akal

- **Journal of Cheminformatics** atau **J. Chem. Inf. Model.** — paling cocok untuk kontribusi metodologis (power analysis, dekomposisi) ditambah temuan empiris tunggal yang kuat (ablasi faktorial).
- **Workshop ML4Molecules / MoML @ NeurIPS/ICML** — kalau versi lebih pendek, kerangka 4-8 halaman, untuk submit lebih cepat dan dapat feedback sebelum ke journal.
- **Tidak** ke venue structure-based-drug-design murni (SBDD track) — paper ini BUKAN tentang generator, tentang validitas model afinitas sebagai alat ukur. Audiensnya adalah orang yang mengevaluasi/membangun scoring function, bukan orang yang membangun generator molekul.

## 4. Struktur yang diusulkan (mapping ke 5 temuan Tier-1)

### Abstract (1 paragraf, draf kasar)
> Model afinitas protein-ligan berbasis GNN 3D sering dievaluasi lewat R² agregat tanpa mendekomposisi dari mana performa itu berasal. Kami menunjukkan, di split target-disjoint leakage-controlled, bahwa (1) model GNN 3D penuh tidak terbedakan secara statistik dari ridge regression 14 deskriptor ligan yang tidak pernah melihat protein; (2) 70% dari performa model ligand-only terbaik berasal dari prior tingkat-kelas murni (permutasi label dalam-target R²=0,247), bukan SAR; (3) jendela performa yang tersisa untuk dibedakan antar-model (0,225 R²) lebih sempit dari lebar interval kepercayaan bootstrap standar (0,387 R²) pada ukuran dataset ini — analisis daya yang menyiratkan banyak klaim "model X lebih baik" di literatur serupa secara statistik tidak bisa dibedakan dari derau; dan (4) ablasi faktorial yang menghapus persis 50% atom protein menunjukkan model 4,5× lebih sensitif ke BERAPA BANYAK atom protein ada di input daripada ke APAKAH atom itu benar-benar berkontak dengan ligan — memisahkan efek jumlah dari identitas, sesuatu yang tidak dilakukan metode atribusi maupun baseline bias-only yang ada. Temuan-temuan ini bersama-sama menyarankan protokol evaluasi standar untuk scoring function afinitas 3D.

### Bagian 1 — Pendahuluan
- Konteks: scoring function 3D makin populer (EGNN, SE(3)-transformer, dsb), tapi evaluasi standarnya (R² agregat, split random) tidak mendekomposisi dari mana performa berasal.
- Related work: Volkov 2022, Durant 2023+ToolBoxSF, Boyles 2021, Scantlebury 2023, Sieg 2019 — **sitasi wajib di paragraf pertama**, diposisikan sebagai lima demonstrasi independen dari fenomena yang sama; paper ini menambah MEKANISME (bukan cuma replikasi lagi yang keenam).
- Kontribusi, diurutkan dari paling mapan ke paling baru: replikasi di split leakage-controlled baru → dekomposisi prior kelas → analisis daya level-tugas → ablasi faktorial jumlah-vs-identitas.

### Bagian 2 — Setup
- Dataset: CrossDocked + LP-PDBBind-style split (per-target, leakage-filtered 4 lapis — ringkas metodenya, detail lengkap rujuk SI).
- Model: EGNN 3D (arsitektur, training).
- **Disclosure eksplisit**: pose adalah hasil docking bukan kristal (Boyles 2021 relevan di sini — kaitkan temuan mereka langsung sebagai prediksi yang terverifikasi).

### Bagian 3 — Tangga representasi (kontribusi replikasi + extended)
- Tabel ligand-only vs pocket-only vs ligand+pocket vs EGNN penuh (angka dari §2.3 `LAPORAN_RISET_LENGKAP.md`).
- Uji berpasangan + BH — bukan cuma "seri" tapi KUANTIFIKASI seberapa seri (CI, bukan cuma p-value).

### Bagian 4 — Dekomposisi prior kelas (kontribusi baru #2)
- Metodologi y-randomisasi DALAM-target vs GLOBAL — jelaskan bedanya secara eksplisit (global tidak mengontrol untuk "target mana", dalam-target melakukannya).
- Hasil: 0,247 R² dari permutasi dalam-target. Implikasi: titik nol yang benar untuk SEMUA perbandingan model afinitas bukan 0.

### Bagian 5 — Analisis daya (kontribusi baru #3)
- Hitung jendela yang bisa diselesaikan = plafon derau label − lantai ligand-only.
- Bandingkan ke lebar CI bootstrap standar pada n yang sebanding dengan benchmark publik (PDBBind test set, CASF, dsb).
- Pesan utama: **ini bukan kritik ke satu paper, ini kritik ke desain eksperimen level-bidang.** Beri rekomendasi konkret (ukuran sampel minimum, atau pelaporan CI wajib, bukan cuma titik estimasi).

### Bagian 6 — Ablasi faktorial jumlah-vs-identitas (kontribusi baru #1, paling kuat)
- Desain: tiga manipulasi 50%-penghapusan-atom-protein (interface utuh / interface hancur / acak).
- Hasil: CI tumpang tindih → jumlah, bukan identitas, yang menentukan.
- Bandingkan eksplisit ke metode atribusi (PointVS dkk — bilang atom MANA penting) dan baseline bias-only (Durant/ToolBoxSF — bilang model tak lebih baik dari bias): **tidak satu pun memisahkan jumlah dari identitas**, dan itulah gap yang diisi bagian ini.

### Bagian 7 — Diskusi
- Sintesis: performa model afinitas 3D itu terurai jadi (a) prior kelas besar, (b) sinyal ligan intrinsik besar, (c) sinyal protein yang ADA tapi responsnya ke jumlah bukan identitas, (d) jendela sisa yang sempit secara statistik.
- Rekomendasi protokol: laporkan R² relatif ke prior kelas, laporkan CI bukan cuma titik, sertakan ablasi faktorial jumlah-vs-identitas sebagai pemeriksaan standar sebelum klaim "model membaca interaksi".
- Keterbatasan: pose docked bukan kristal (Boyles 2021); satu arsitektur (EGNN) — generalisasi ke SE(3)-transformer dkk belum diuji.

### SI (Supporting Information)
- Detail filter kebocoran (4 lapis), detail bootstrap (B, seed, target-clustering), kode+data tersedia (link repo).

## 5. Yang masih kurang sebelum submission-ready

1. **Ablasi faktorial baru dijalankan di SATU model (EGNN Stage 0).** Untuk klaim yang generalizable, idealnya direplikasi di minimal satu arsitektur lain (misal Track A GIGN+PIGNet2, yang sudah ada infrastrukturnya) — kalau pola yang sama muncul di dua arsitektur berbeda, klaimnya jauh lebih kuat daripada "kami amati ini di satu model".
2. **Related work perlu dibaca lebih dekat** — khususnya memastikan tidak ada paper ke-6 yang sudah melakukan versi ablasi faktorial jumlah-vs-identitas ini (8 ronde pencarian literatur yang sudah dilakukan untuk tesis cukup sebagai basis, tapi pencarian khusus untuk FRAMING paper ini — "amount vs identity ablation", "atom count confound structure-based" — belum dilakukan secara terpisah).
3. **Gambar/figure belum dibuat** untuk versi paper-standalone ini (figure tesis ada tapi formatnya untuk thesis, bukan 2-column journal).
4. **Draft bagian 1-7 belum ditulis sebagai teks utuh** — ini kerangka, bukan draf.

## 6. Langkah selanjutnya kalau mau dikerjakan

Item ini TIDAK dikerjakan sekarang (murni teks/kerangka, menunggu keputusan Anda kapan mau mulai menulis penuh) — tapi kalau mau jalan, urutan yang disarankan:
1. Replikasi ablasi faktorial di Track A (butuh GPU tapi jauh lebih murah dari RQ2 — model sudah terlatih, cuma perlu forward pass seperti §2.6, bukan sampling generatif).
2. Tulis draf lengkap bagian 1-7 dari kerangka di atas.
3. Cari 3-5 paper tambahan khusus untuk framing "amount vs identity" (pencarian terpisah dari audit 8-ronde yang sudah ada).
4. Buat figure versi 2-column.
