# Laporan Lengkap Riset: Coupled-Guidance Structure-Based Drug Design

**Disusun:** 2026-10-12 · **Lingkup:** seluruh isi repo `targetdiff` sampai titik ini · **Status tesis:** draf 198 halaman (build terakhir 2026-10-06), rebuild final belum dilakukan

Ini bukan ringkasan dari ingatan — setiap angka di bawah dibaca ulang dari file hasil (`*_results.json`), `TASKS.md`, dan kode sumber saat laporan ini ditulis. Kalau ada yang kedengaran janggal, itu lebih mungkin kesalahan pencatatan daripada kesalahan pengukuran; silakan tunjuk, saya cek ulang ke sumbernya.

---

## 1. Apa proyek ini sebenarnya

Tesis S2 tentang **coupled-guidance structure-based drug design**: memandu model diffusi molekul (TargetDiff) dengan gradien dari model afinitas terlatih, supaya molekul yang dihasilkan lebih mengikat target protein tertentu. Pertanyaan risetnya bergeser selama pengerjaan — dari "apakah guidance bekerja" menjadi **"kalau guidance tidak bekerja, di titik mana rantai kausalnya putus, dan bisakah itu dibuktikan bukan cuma didiagnosis"**. Itulah kerangka tiga-mekanisme (M1/M2/M3) yang sekarang menjadi tulang punggung tesis.

Dikerjakan di laptop pribadi: RTX 3050 4GB VRAM, i7-12700H. Itu bukan detail sepele — ia menentukan desain eksperimen (subsample train 6000, batch size kecil, kuantisasi ternary punya alasan nyata bukan cuma gimmick) dan menjadi alasan ada 1.400+ baris infrastruktur termal (lihat §6).

**Ukuran codebase kita** (tidak termasuk `guidance/reference_code/`, yang itu kode vendor pihak ketiga — PIGNet, GIGN, EIGN, eMOSAIC, LP-PDBBind, 173 file, tidak kita sentuh):

| | jumlah |
|---|---|
| file Python kita | 187 |
| baris kode kita | 30.320 |
| file hasil (`*_results.json`/`*_report.json`/`*_table.json`) | 67 |
| suite tes | 13 |
| jumlah tes | 178 (0 gagal) |
| direktori run terlatih (`logs_*`) | 24, total 1,4 GB, 46 checkpoint `best.pt` |

---

## 2. Temuan empiris — yang sudah terbukti dengan angka

Ini inti riset: bukan "guidance gagal" sebagai pernyataan tunggal, tapi rantai bukti yang saling menopang tentang **kenapa**.

### 2.1 Ketidakpastian (UQ) — hampir semuanya gagal, dan itu temuannya

| metode | hasil |
|---|---|
| MC Dropout | spearman(σ, error) = 0,0021, CI [−0,074, 0,099] → **tidak informatif** |
| Deep Ensemble | spearman = 0,0084, CI [−0,099, 0,132] → **tidak informatif** |
| Heteroscedastic Gaussian NLL (vanilla) | gagal pra-registrasi |
| Deep Evidential | p=0,042 di **satu** seed, hilang total di 3 seed (sd antar-seed 0,0377 > efek yang diklaim) → **kegagalan replikasi** |
| β-NLL (perbaikan resmi dari Seitzer et al. 2022) | **masih gagal** — underpowered, p=0,757, jadi kesimpulan "heteroscedastic gagal" bertahan bahkan setelah memakai perbaikan yang diterbitkan, bukan cuma varian yang sudah diketahui buruk |
| **Split-conformal** | coverage 0,8952 (CI [0,857, 0,925]) vs nominal 0,90 → **satu-satunya yang valid** |

Lima dari enam gagal. Yang valid cuma yang paling sederhana secara teori (conformal punya garansi coverage marginal tanpa asumsi distribusi). Ini pola yang konsisten, bukan kebetulan satu metode.

### 2.2 Kuantisasi ternary (QAT) — model afinitas bisa dikompres ke 1,58-bit tanpa rugi

- Test R² ternary (TWN+STE, 3 seed) = **0,3437 ± 0,0377** vs FP32 **0,3420** — tak terbedakan.
- **Baru dikerjakan (11 Okt):** validasi tambahan di luar R² — apakah kuantisasi merusak invariansi E(3) model (rotasi/translasi protein+ligan tidak boleh mengubah prediksi afinitas). **Deviasi gerak rigid median = 1–2 ULP float32 (batas presisi aritmetika itu sendiri)**, untuk FP32 maupun ternary. Kontrol positif (merotasi ligan saja, merusak jarak nyata) menggerakkan prediksi 4–5 orde lebih besar — jadi uji ini terbukti mampu mendeteksi pelanggaran, dan tidak menemukannya adalah informasi nyata.
- Klaim yang kini bisa dibuat: bukan "ternary kebetulan bekerja", tapi **ancaman kuantisasi terhadap ekuivariansi adalah sifat arsitektur** — message-passing berbasis jarak-antar-atom (EGNN kita) aman secara konstruksi; fitur vektor ekuivarian yang dikuantisasi langsung (kasus pembanding terdekat di literatur) butuh rekayasa tambahan untuk tetap aman.

### 2.3 Audit sumber informasi — temuan paling mengubah arah tesis

Pertanyaan: dari mana performa model afinitas 3D sebenarnya berasal?

| representasi | dim | test R² | CI95 |
|---|---|---|---|
| **ligand-only** (14 deskriptor RDKit, TANPA protein) | 14 | **0,3534** | [0,208, 0,447] |
| pocket-only (38 deskriptor, TANPA ligan) | 38 | 0,2889 | [0,153, 0,377] |
| ligand+pocket | 52 | 0,3832 | [0,201, 0,496] |
| EGNN 3D penuh (Stage 0, FP32) | — | 0,3420 | — |
| EGNN 3D penuh (ensemble 3-model) | — | **0,4072** | — |
| ecif_raw (hitungan kontak mentah) | 72 | 0,3701 | [0,213, 0,473] |
| ecif_norm (dinormalisasi ukuran) | 72 | 0,2077 | [0,021, 0,348] |

**Model afinitas 3D penuh (EGNN) tidak terbedakan secara statistik dari ridge regression 14 deskriptor molekul yang tidak pernah melihat protein.** Uji berpasangan + Benjamini-Hochberg atas 40 pasangan: 12 menang struktur, 28 seri, **0 menang ligand-only**; tiga "kemenangan" marginal ditarik setelah koreksi multiple-testing.

Konfirmasi tambahan: `heavy_atoms` — **satu** fitur, jumlah atom berat — sendirian mencapai R²=0,3069, **90% dari performa model struktur**. Dan `ecif_raw` vs `ecif_norm`: dR²=+0,1625 (q=0,013, signifikan) murni dari menghapus normalisasi ukuran — hitungan kontak "bekerja" terutama dengan menyandikan ulang ukuran ligan, bukan geometri interaksi.

Ini bukan berarti model struktur tidak belajar apa-apa — ia belajar sesuatu, tapi sinyal protein tidak menambah banyak **di atas** apa yang sudah bisa ditebak dari ligan sendiri.

### 2.4 Dekomposisi: berapa dari performa itu cuma "menebak kelas target"?

- Permutasi label **dalam-target** (bukan global): R² = **0,2468 ± 0,0019**. Ini prior tingkat kelas murni — tidak butuh SAR apa pun di dalam target, cuma "target mana" yang sudah membawa 0,2468 R².
- **70% dari performa model ligand-only terbaik tidak butuh SAR dalam-target sama sekali.** Titik nol yang benar untuk membandingkan model adalah 0,247, bukan 0.
- Setelah dikoreksi ke titik nol itu: ensemble +0,1604 di atas prior (48,5% dari sisa ruang), desc_ridge +0,1063 (32,1%), Vina +0,0133 (4,0%) — model struktur jadi lebih jelas terpisah, tapi ruang yang tersisa untuk dipisahkan sangat sempit.

### 2.5 Plafon derau label — kenapa ruang itu sempit

- Dari pengukuran-ulang eksperimental (Pearson 0,76) → **plafon R² ≈ 0,55–0,65** (dilaporkan sebagai rentang, bukan angka tunggal, setelah saya menyitir selektif di draf pertama — lihat §5).
- Jendela yang bisa diselesaikan: plafon (0,578) dikurangi lantai ligand-only (0,353) = **0,225 R²**. Lebar CI bootstrap rata-rata kita sendiri: **0,387 R²**. **Rasio 0,58 — jendela riset ini lebih sempit dari satu interval kepercayaan.** Desain eksperimen manapun di data jenis ini (termasuk paper lain yang melaporkan gain 0,02–0,05 R²) secara statistik tidak bisa menentukan posisi model di dalam jendela itu.

### 2.6 Eksperimen kunci: sensitivitas pose (desain faktorial, n=300)

Tiga manipulasi yang **masing-masing menghapus TEPAT 50% atom protein**, beda hanya kontennya:

| manipulasi | antarmuka ligan | \|Δprediksi\| (CI95) |
|---|---|---|
| simpan separuh terdekat ligan | **utuh** | 1,2451 [1,102, 1,369] |
| simpan separuh terjauh | **hancur** | 1,0576 [0,895, 1,226] |
| simpan separuh acak | sebagian | 1,2686 [1,158, 1,381] |

CI-nya tumpang-tindih semua → **isi antarmuka tidak berpengaruh ke prediksi**. Bandingkan dengan menggeser ligan 8 Å keluar pocket: Δ=0,278 — jauh lebih kecil. **Model ~4,5× lebih sensitif ke BERAPA BANYAK atom protein ada di input daripada ke APAKAH atom itu benar-benar menyentuh ligan.** Ini bukti langsung untuk M1 (surrogate afinitas tidak membaca antarmuka interaksi secara bermakna) dan inilah yang menjelaskan kenapa guidance gagal: gradiennya datang dari model yang reaksinya ke protein adalah soal volume, bukan kontak.

### 2.7 Kualitas split — satu-satunya bagian yang sepenuhnya positif

- **AVE bias split kita: +0,0495** vs random ligand split: **+1,1726** (23× lebih sedikit redundansi).
- Tanimoto 1-NN (hafalan murni) **gagal berat**: R²=−0,924. Struktur mengalahkannya di semua pasangan (q=0,002). **Split kita tidak memberi hadiah untuk menghafal ligan training terdekat.**
- Tapi: 40,1% molekul test berada di scaffold generik yang juga ada di training (held-out-target ≠ held-out-scaffold) — harus didisklos, bukan disembunyikan. Diuji dampaknya: **0 dari 13 model lebih buruk** di scaffold tak-terlihat (diukur RMSE, setelah audit pertama saya salah arah karena artefak range-restriction — lihat §5).

### 2.8 Kebocoran data yang ditemukan dan ditutup

- **Kebocoran sekuens**: template `1fm9` 100% identik dengan target test P19793 — lolos dua filter berbasis ID database. 19 sekuens/28 template/265 baris training dibuang.
- **Kebocoran tingkat senyawa** (BindingNet, ditemukan lewat InChIKey skeleton): hanya 0,34% baris training (423/124.973) tapi **10,12% record evaluasi** (1.814/17.924) — rasio 30×, karena satu senyawa populer menyumbang banyak baris tapi ligan test jarang.

---

## 3. Novelty & kontribusi — dinilai jujur setelah 8+ ronde pencarian literatur

### Tier 1 — bisa dipertahankan sebagai baru
1. **Ablasi faktorial antarmuka-vs-bulk** (§2.6) — memisahkan efek JUMLAH atom protein dari IDENTITAS/posisinya. Metode atribusi yang ada (PointVS, SME, Shapley) bilang atom mana yang penting; baseline bias-only (Durant 2023) bilang model tak lebih baik dari bias. Tidak satu pun memisahkan jumlah dari identitas — tidak ditemukan di 8 ronde pencarian.
2. **Dekomposisi prior tingkat kelas** via y-randomisasi dalam-target — titik nol yang benar 0,247 bukan 0 (§2.4).
3. **Analisis daya level-tugas** — jendela yang bisa diselesaikan lebih sempit dari satu CI, dikorroborasi oleh literatur ABFEP independen dan punya mekanisme fisik (kompensasi entalpi-entropi).
4. **Sambungan dari audit sumber informasi ke kegagalan gradien guidance** — ini subjek tesis yang sesungguhnya, dan tidak satu pun paper yang menyentuh bagian lain membahas sambungan ini.
5. **Ternary (1,58-bit) QAT pada GNN afinitas 3D, divalidasi lewat R² DAN ekuivariansi** (§2.2).

### Tier 2 — replikasi + rigor tambahan. Berharga, tapi jangan sebut "baru"
Temuan gaya Volkov (ligand-only ≈ model struktur) direplikasi di split leakage-controlled target-disjoint, diperluas dengan kedua arah marginal dan uji redundansinya secara eksplisit.

### Tier 3 — sudah ada di literatur. Sitasi, lalu lanjut
- Perbandingan metode UQ untuk afinitas protein-ligan **sudah dikerjakan Rayka et al. 2025** (5 metode, Leak-Proof PDBBind) — klaim kita dipersempit ke kelas model (3D-GNN), kerangka inferensi (bootstrap klaster-target + BH + pra-registrasi), dan temuan kegagalan-replikasi evidential secara spesifik.
- "Model ligand-only ≈ model struktur" sudah ditunjukkan **lima kali independen** (Volkov 2022, Durant 2023 + ToolBoxSF, Boyles 2021, Scantlebury 2023, Sieg 2019).
- QAT 8-bit pada GNN ekuivarian **sudah ada** (Zhou et al. 2026, arXiv:2601.02213) — tapi ternary + protein-ligan tidak disentuh, jadi klaim kita selamat secara sempit.

---

## 4. Keterbatasan yang dipaksa literatur — harus masuk abstrak, bukan lampiran

1. **Pose kita adalah hasil docking, bukan kristal** (`_lig_tt_min_*.sdf`, terverifikasi). Boyles et al. 2021 mengukur efek ini dan *memprediksi* temuan utama kita — pada pose docked, kanal struktural melemah dan fitur ligan mengambil alih.
2. **Ligand efficiency diperdebatkan secara matematis** (Kenny 2018: "tidak bermakna secara fisis") — Track C bersandar padanya; sudah ada draf perumusan ulang (`TRACKC_REFRAMING_DRAFT.md`) pada atom berat + PoseBusters, kesimpulannya bertahan.
3. **Plafon derau label adalah rentang**, bukan angka tunggal — koreksi atas sitiran selektif saya sendiri di draf pertama.
4. **Split per-target** (yang kita pakai) secara struktural lebih mudah dari leave-superfamily-out atau split kemiripan-pocket.
5. **Afinitas kesetimbangan mungkin objektif yang salah** — residence time/koff berkorelasi lebih baik dengan efikasi in vivo di sejumlah studi.
6. Eksponen scaling kimia (0,17–0,26) menyiratkan menambah data mentah tidak akan menutup jendela sempit di §2.5.
7. Harness kita bespoke; CBGBench/MolScore adalah standar komparabilitas bidang ini yang tidak kita pakai.

---

## 5. Kesalahan yang pernah terjadi — dan sudah ditutup

Dicatat bukan untuk self-flagellation, tapi karena pola kesalahannya berulang dan sekali dikenali, bisa dicegah secara sistematis:

- **Audit scaffold pertama menyimpulkan KEBALIKAN dari kebenarannya** — "familiaritas scaffold menggelembungkan performa" ternyata artefak range-restriction (rasio varians label 1,77×). Terbongkar oleh baseline Vina (tak terlatih) yang menunjukkan kolaps identik.
- **Menyitir selektif** pada derau label (hanya sumber pesimis) — dikoreksi jadi rentang.
- **Prior-art scoop** pada perbandingan UQ (Rayka 2025) — dilaporkan jujur, klaim dipersempit alih-alih dipertahankan secara tidak jujur.
- **Docstring sendiri melebih-lebihkan** — "equals ternary(w) exactly" ternyata meleset 1 ULP; ditemukan oleh tes yang saya tulis sendiri dan gagal, bukan oleh audit eksternal.
- **Bug performa O(B·n·m)** di `ave_bias.py`: 4 jam 42 menit di 92°C tanpa selesai satu threshold, karena similarity matrix dihitung ulang di setiap draw bootstrap padahal sisi training-nya tetap. Diperbaiki → 1 menit 27 detik, hasil identik bit-per-bit.
- **Tiga kali self-match `pgrep`/`pkill`** mematikan shell sendiri karena pola pencarian proses cocok dengan command line shell itu sendiri — dan kemunculan keempat muncul lagi di kode *verifikasi* saya sendiri. Pelajarannya ditulis jadi aturan permanen: selalu filter via `/proc/<pid>/comm`, dan kecualikan pid sendiri + leluhurnya.
- **Watchdog termal sempat ditulis, diuji, diperbaiki — tapi queue yang sedang jalan tidak memakainya** karena `os.replace()` yang membuat patch aman untuk shell berjalan juga berarti shell itu tetap memegang inode lama. Butuh restart + verifikasi `/proc/<pid>/fd/255` secara eksplisit; sekarang jadi aturan tertulis di header skrip.

Daftar lengkap (13 item) ada di `TASKS.md` bagian D.

---

## 6. Infrastruktur yang dibangun — ini juga bagian dari kontribusi

Untuk riset yang separuh isinya adalah "buktikan angka ini valid secara statistik", infrastrukturnya bukan hal sampingan:

- **13 suite, 178 tes**, nol gagal. Termasuk tes yang memaku statistik terhadap rumus tangan (bukan terhadap kode itu sendiri), tes yang memverifikasi harness power analysis, dan — baru — tes untuk dua modul penanggung-beban yang sebelumnya tidak teruji sama sekali: `build_lp_splits` (37 pengimpor; menentukan data latih/tahan untuk SELURUH proyek) dan `interaction_masks` di model Track A yang terlatih.
- **Queue otonom ber-gerbang termal**: menjalankan backlog tanpa pengawasan, dengan watchdog yang men-SIGSTOP/SIGCONT job kalau CPU >88°C selama 60s dan resume di <75°C, budget total 7200s sebelum menyerah. Resumable, fail-closed, exit code gerbang timeout berbeda dari exit code kegagalan asli.
- **4 filter kebocoran berlapis** (PDB ID → UniProt → identitas sekuens 90% → skeleton InChIKey ligan), masing-masing dengan guard test yang menghitung ULANG set eksklusi dari audit mentah, bukan mempercayai ringkasannya.
- **Agregator multi-seed** yang menolak memberi verdict kalau underpowered (minimum detectable difference dihitung eksplisit), melaporkan sd `undefined` bukan `0.0000` di n=1, dan tidak pernah menelan metrics.json yang korup secara diam-diam.

---

## 7. Status tesis

Draf **198 halaman** (`.docx` + `.pdf`), build terakhir **6 Oktober** — artinya **belum memasukkan** semua temuan dari §2.2–2.8 dan bagian kuantisasi-ekuivariansi yang baru. Keputusan yang sudah diambil (dan sebaiknya dipegang): **satu rebuild di akhir**, bukan setiap perubahan, supaya versi draf tidak bercabang-cabang. Urutan yang sudah disiapkan untuk rebuild itu (di `TASKS.md` item 9):
1. Bab IV: seksi baru audit sumber informasi + noise ceiling + dekomposisi prior kelas.
2. Bab X revisi — argumen lama "kegagalan guidance bukan karena akurasi rendah" sekarang bocor, karena model TANPA protein mencapai korelasi yang sama; perlu direformulasi.
3. Limitasi: gray-zone identitas sekuens 50–90%, split per-target lebih mudah dari leave-superfamily-out, overlap scaffold 40,1%, komposisi label Kd/Ki/IC50 yang bergeser antar split.
4. ~15 sitasi wajib yang masih hilang (Volkov 2022, Mattsson 2026, van Tilborg 2022, Seitzer 2022, Thomas 2021, Zhou 2026, dll — daftar lengkap di `TASKS.md`).

---

## 8. Apa yang bisa dikembangkan/di-upgrade/di-overengineer/dioptimasi

Dipisah tiga lapis: **riset inti** (paling bernilai, paling sejalan dengan tesis), **penguatan teknis/infrastruktur** (opsional, nilainya buat ketahanan bukan buat tesis langsung), dan **ambisi jangka panjang** (kalau proyek ini mau dilanjut melampaui S2).

### 8.1 Riset inti — urutan prioritas yang sudah disepakati

| # | item | biaya | kenapa ini penting |
|---|---|---|---|
| **1** | **Kontrol positif (RQ2)** | GPU, perlu di-scope | **Ini gap paling besar di tesis.** Semua rute guidance yang diuji gagal — tapi tesis belum punya satu kasus pun yang membuktikan harness-nya *bisa* mendeteksi keberhasilan. Ulangi ablasi guidance dengan Vina atau skor PLIP (bukan surrogate terlatih) sebagai pemandu. Membaik → M1 terkonfirmasi sebagai sebab. Tetap gagal → penjelasan pindah ke M2/M3, sama berharganya. Thomas et al. 2021 menunjukkan docking-guidance berhasil untuk model lain, jadi ada alasan nyata berharap positif. |
| 2 | PLIP typed interactions sebagai blok ke-8 tangga representasi | CPU, ½ hari | ECIF cuma pasangan elemen×jarak — kasar. Kalau fitur interaksi BERTIPE (h-bond, hidrofobik, π-stacking, jembatan garam, halogen) pun tidak menambah apa-apa, klaim redundansi protein-ligan jadi jauh lebih kuat. Kalau menambah, itu hasil positif baru dan ia menamai perbaikannya secara spesifik. |
| 3 | Coverage conformal **kondisional** per novelty tier + famili protein | CPU, ½ hari | Satu-satunya hasil UQ yang valid (conformal) itu coverage-nya MARGINAL. Literatur (Jeliazkova 2026) memprediksi coverage turun di kimia novel — kalau benar, itu temuan penting: interval yang "valid" secara agregat tapi diam-diam under-cover justru di senyawa novel yang paling dipedulikan praktisi. |
| 4 | y-randomisasi dalam-target untuk EGNN (bukan cuma pipeline deskriptor) | GPU, 1 run ~2,5 jam | Menutup satu-satunya caveat di temuan terkuat kita (§2.4): prior kelas 0,2468 saat ini cuma dibuktikan untuk model deskriptor, belum untuk arsitektur 3D itu sendiri. |
| 5 | Metrik activity-cliff gaya MoleculeACE | CPU, ½ hari | Pola sudah terlihat (ligand-only kolaps di similarity tinggi) tapi belum diukur sebagai metrik baku bidang ini. |
| 6 | Matched molecular pair (MMP) analysis | CPU, 1 hari | Uji SAR paling ketat; melengkapi dekomposisi prior kelas — prior kelas menangkap antar-seri, MMP menguji dalam-seri. |
| 7 | Stratifikasi error per famili protein | CPU, beberapa jam | Mengubah satu angka gabungan jadi peta "di mana model gagal" — kinase vs protease vs nuclear receptor dst. |
| 8 | Arm A + Arm C produksi (data BindingNet skala penuh) | GPU, ~16 jam/arm | Pertanyaan sudah diperbaiki: bukan "apakah R² naik" tapi "apakah data tambahan menutup jarak ke ligand-only DAN melampaui prior kelas 0,247". |
| 10 | RQ4 — conformal-gated guidance | GPU, perlu scope | Kombinasi yang tampak belum ada di literatur SBDD. Prediksi pra-registrasi dulu: M3 memprediksi rute rejection-sampling terbantu; M1 memprediksi rute gradien tidak. Satu eksperimen, dua prediksi berlawanan — kalau keduanya benar, itu bukti kuat untuk seluruh dekomposisi M1/M2/M3. |
| 11 | RQ3 — DOODL/SVDD | GPU, implementasi besar | Paling mahal dan paling berisiko diremehkan reviewer kalau tanpa pra-registrasi — M1 memprediksi TIDAK membantu padahal literatur umum mengharapkan sebaliknya. Nilainya justru ada di prediksi yang berlawanan dengan ekspektasi itu. |

### 8.2 Overengineering/penguatan teknis — bernilai tapi opsional

Ini murni soal ketahanan kode, bukan isi tesis. Jangan dikerjakan kalau menyita waktu dari §8.1, tapi kalau ada waktu longgar:

- **Audit impor menemukan 55 modul kita** yang diimpor kode nyata tapi tak disentuh tes mana pun. Dua yang paling berisiko (`build_lp_splits`, `interaction_masks`) sudah ditutup. Sisanya kebanyakan *driver eksperimen* (`train_*.py`, `diag_*.py`) yang ujinya adalah eksperimen itu sendiri — tapi beberapa modul library kecil (`atom_features.py`, `interfaces.py`, `bond_estimator.py`) masih layak diperiksa dengan pola yang sama: baca kode, cari properti yang bisa diuji (bukan cuma re-implementasi), jalankan, lihat apa yang gagal.
- **Jalankan ToolBoxSF (Durant 2023)** di data kita (F6, satu-satunya item dari rabbit hole yang belum dikerjakan) — replikasi "ligand-only ≈ structure-based" pakai tool milik bidang ini sendiri, bukan skrip buatan sendiri. Butuh instalasi paket eksternal; sengaja ditunda supaya tidak mengganggu env `drugdisc` saat training jalan.
- **Scan performa gaya `ave_bias`**: sudah dilakukan sekali (bersih — tidak ada recompute-in-loop lain yang ketemu), tapi scan itu sendiri cuma menyasar bootstrap. Loop pelatihan epoch-per-epoch atau loop ekstraksi pocket belum pernah di-profile dengan cara yang sama.
- **CI/pre-commit yang lebih keras**: hook pre-commit sudah ada untuk test suite; bisa diperluas menolak commit yang menambah `except: pass` tanpa komentar, atau `assert` di jalur yang bisa kena `python -O`.
- **Reproducibility packaging**: `environment_actual.yml` dan `requirements-lock.txt` sudah ada di `guidance/thesis/`, tapi belum ada skrip tunggal yang membangun ulang semua `*_results.json` dari nol untuk audit independen — saat ini masing-masing skrip dijalankan manual lewat queue.

### 8.3 Kalau proyek ini mau dibawa ke ranah lebih tinggi (melampaui S2)

Ini bukan rekomendasi untuk dikerjakan sekarang — ini peta kalau suatu saat ada waktu/sumber daya lebih (publikasi jurnal, lanjut S3, atau proyek pribadi lanjutan):

1. **Publikasi Tier 1 sebagai paper terpisah dari tesis.** Item 1–5 di §3 (ablasi faktorial, dekomposisi prior kelas, analisis daya, sambungan ke kegagalan guidance, QAT+ekuivariansi) masing-masing cukup berdiri sebagai kontribusi workshop/short paper. Menyatukan semuanya dalam satu venue (bukan cuma satu bab tesis) memaksa kejelasan naratif yang kadang hilang kalau semua ditumpuk di satu dokumen 198 halaman.
2. **Pindah dari CrossDocked/LP-PDBBind ke benchmark yang punya leaderboard aktif** (CBGBench, MolScore) supaya angka-angka kita bisa dibandingkan langsung dengan metode SBDD lain tanpa klaim "kami lebih baik" yang harus dijelaskan ulang setiap kali — komparabilitas langsung, bukan komparabilitas yang perlu footnote.
3. **Uji M2 (misaligned gradient) secara eksplisit**, bukan cuma lewat eliminasi. Sejauh ini M1 punya bukti positif langsung (§2.6), M3 punya jalur uji di RQ4 (§8.1 item 10) — tapi M2 belum punya eksperimen yang ditarget khusus ke situ (misalnya mengukur korelasi arah gradien guidance terhadap arah perubahan afinitas sungguhan pada pasangan molekul kecil yang diketahui).
4. **Perluasan ke residence time / kinetic selectivity** sebagai objektif guidance alternatif, bukan cuma catatan limitasi (§4.5) — ini perubahan besar (butuh data kinetik, jauh lebih jarang tersedia dari data afinitas kesetimbangan) tapi langsung menjawab kritik paling tajam terhadap keseluruhan premis "afinitas kesetimbangan sebagai target optimasi".
5. **Conditional/adaptive guidance strength berbasis confidence surrogate** (disinggung di RQ4) sebagai arah metodologis umum, dilepas dari SBDD — relevan untuk guidance diffusion model apa pun yang punya surrogate dengan uncertainty quantification yang tervalidasi (bukan cuma yang gagal seperti §2.1), bukan cuma untuk afinitas protein-ligan.
6. **Skala data**: eksponen scaling kimia yang ditemukan (0,17–0,26) menyiratkan jendela sempit di §2.5 TIDAK akan ditutup dengan menambah data CrossDocked/BindingNet linear — perlu order-of-magnitude lebih besar atau sumber data dengan derau lebih rendah (misal eksperimen biofisik presisi tinggi seperti ITC/SPR yang dikurasi khusus), bukan cuma "scraping lebih banyak".

---

## 9. Kalau harus dipilih satu langkah berikutnya

**Item 1 di §8.1 — kontrol positif.** Semua temuan lain di laporan ini kuat secara statistik, tapi mereka semua menjelaskan kegagalan. Tesis yang isinya murni "ini gagal dan inilah sebabnya" tanpa satu pun demonstrasi bahwa harness pengukurannya memang bisa mendeteksi keberhasilan, akan selalu kena pertanyaan penguji yang sama: *"bagaimana saya tahu alat ukurmu bukan yang rusak?"* Item 1 adalah satu-satunya eksperimen di daftar ini yang menjawab pertanyaan itu secara langsung, apa pun hasilnya.

---

*Sumber data laporan ini: `TASKS.md` (1.495 baris), 67 file `*_results.json`, kode di `guidance/` dan `scripts/`. Dibuat oleh asisten riset (Claude) atas permintaan pengguna untuk meninjau keseluruhan hasil proyek — bukan dokumen tesis resmi, dan belum di-review manusia.*
