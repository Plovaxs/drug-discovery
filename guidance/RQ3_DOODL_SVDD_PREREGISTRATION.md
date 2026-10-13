# Pra-registrasi: RQ3 — DOODL / SVDD

**Ditulis:** 2026-10-13, SEBELUM kode ditulis. **Status:** prediksi dicatat, implementasi DITUNDA dengan
alasan eksplisit (lihat §3) -- bukan dikerjakan, bukan dibatalkan.

---

## 1. Rencana asli

Gradien lewat denoiser (DOODL, Wallace et al. 2023) atau guidance tanpa turunan (SVDD, Li et al. 2024)
dengan surrogate EGNN yang SUDAH ada. **M1 (surrogate tidak informatif) memprediksi TIDAK akan membantu**
-- bertentangan dengan ekspektasi umum literatur bahwa mekanisme guidance yang lebih canggih (DOODL/SVDD
dirancang untuk mengoreksi masalah gradien kasar/noisy classifier guidance biasa) akan memperbaiki hasil.
Nilai eksperimen ini ada di prediksi yang BERTENTANGAN dengan harapan umum, bukan di harapan hasil positif.

## 2. Bagaimana RQ2 mengubah kalkulus nilai eksperimen ini

RQ2 (kontrol positif, selesai 2026-10-13) sudah memberi bukti langsung yang relevan:
- `affinity_only` (guidance GRADIEN dari surrogate EGNN yang sama) **gagal** -- tidak beda signifikan
  dari baseline tak-terpandu.
- `vina_rerank` (SELEKSI best-of-N dari sinyal GENUIN -- Vina asli, bukan gradien, bukan surrogate kita)
  **berhasil** -- p=1,9e-06, CI mengecualikan nol.

Perbandingan ini sudah menjawab pertanyaan "apakah MEKANISME guidance yang salah, atau SINYALNYA" secara
substansial: ketika sinyal diganti (surrogate -> Vina asli) dengan mekanisme sederhana (rerank), hasilnya
berbalik total. DOODL/SVDD mengusulkan mekanisme yang LEBIH CANGGIH (particle guidance / backprop lewat
sampler) tapi TETAP memakai surrogate EGNN yang SAMA yang sudah terbukti tidak membaca antarmuka (§2.6
ablasi faktorial, 4,5x lebih sensitif ke jumlah atom daripada identitas kontak). **M1 memprediksi mekanisme
secanggih apa pun tidak menyelamatkan sinyal yang memang tidak membawa informasi** -- dan itu sudah
didukung kuat secara tidak langsung oleh kontras affinity_only-vs-vina_rerank: masalahnya terbukti di
SINYAL (M1), bukan hanya di mekanisme gradien klasik (yang akan jadi argumen kalau M2 benar).

**Implikasi untuk prioritas**: menjalankan DOODL/SVDD dengan surrogate EGNN yang sama punya nilai
informasi MARJINAL YANG LEBIH RENDAH dari yang dibayangkan saat item ini pertama ditulis (sebelum RQ2
ada) -- kita sudah punya bukti kuat (walau tidak identik) bahwa gantinya mekanisme tidak akan menolong
kalau sinyalnya sendiri yang bermasalah. Pertanyaan yang BENAR-BENAR masih terbuka, dan yang justru RQ2
buka sebagai pertanyaan baru, adalah: **apakah mekanisme SVDD/DOODL yang canggih menambah sesuatu DI ATAS
reranking sederhana, KETIKA dipasangkan dengan sinyal yang genuinely informatif (Vina asli)?** Itu bukan
"M1 vs M3" lagi -- itu pertanyaan "berapa nilai tambah mekanisme guidance yang canggih, dipisah dari
nilai sinyal yang baik", dan itu TIDAK TERJAWAB oleh RQ2 (RQ2 cuma menguji reranking, bukan particle
guidance/backprop sampler).

## 3. Keputusan: implementasi DITUNDA, dua jalur dicatat untuk nanti

**Jalur A (prediksi konfirmatori, nilai informasi rendah setelah RQ2)**: DOODL/SVDD + surrogate EGNN kita.
M1 memprediksi gagal; RQ2 sudah memberi bukti tidak langsung kuat ke arah itu. Menjalankan ini sekarang
sebagian besar mengulang kesimpulan yang sudah bisa ditarik, dengan biaya implementasi PALING BESAR di
seluruh daftar todo ("implementasi besar" -- DOODL butuh backprop lewat seluruh sampler, SVDD butuh
particle filter yang benar). Rasio biaya/informasi-baru-nya buruk dibanding sebelum RQ2 ada.

**Jalur B (pertanyaan baru, dibuka OLEH RQ2, nilai informasi lebih tinggi)**: DOODL/SVDD + Vina asli
sebagai fungsi nilai (bukan surrogate EGNN). Ini menguji apakah mekanisme canggih menambah nilai DI ATAS
reranking sederhana yang sudah terbukti bekerja -- pertanyaan yang genuinely belum terjawab yang BARU
lahir dari hasil RQ2 ini sendiri. Vina tidak bisa diturunkan, jadi DOODL (butuh gradien lewat sampler)
tidak langsung cocok untuk Jalur B; SVDD (Li et al. 2024, importance-sampling/particle guidance TANPA
gradien) lebih cocok karena cuma butuh fungsi nilai yang bisa dievaluasi, bukan diturunkan -- sama seperti
Vina dipakai di RQ2's seleksi.

**Rekomendasi**: kalau proyek ini punya waktu untuk satu dari dua jalur, **Jalur B (SVDD + Vina) lebih
berharga** daripada Jalur A (DOODL/SVDD + surrogate) -- Jalur A kemungkinan besar mengonfirmasi ulang apa
yang RQ2 sudah tunjukkan secara tidak langsung, Jalur B menjawab pertanyaan baru yang RQ2 buka. Keduanya
TETAP "implementasi besar" dan butuh scoping GPU terpisah sebelum dilepas -- tidak dikerjakan di pass ini.

## 4. Status akhir item ini untuk sekarang

**Tidak diimplementasikan.** Dicatat sebagai: (a) prediksi M1 untuk Jalur A, sudah didukung tidak
langsung oleh RQ2 -- bisa dilaporkan di tesis SEBAGAI PREDIKSI DAN PENALARAN, bukan sebagai eksperimen
yang dijalankan, kalau waktu tidak cukup; (b) Jalur B sebagai arah riset terbuka yang baru, bernilai
untuk pekerjaan lanjutan di luar S2 ini (lihat `guidance/thesis/TIER1_PUBLICATION_OUTLINE.md` §6 untuk
pola yang sama -- arah lanjutan dicatat, tidak dipaksa dikerjakan dalam sesi yang sudah padat).
